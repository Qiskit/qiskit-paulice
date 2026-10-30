// This code is part of Qiskit.
//
// (C) Copyright IBM 2026
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

//! Gamma and Monte Carlo LER scoring through a separately installed engine.
//!
//! An engine registers a `PauliceEngine` table (`include/paulice_engine.h`),
//! and is used only once registered. The engine takes only batches it can
//! score bit-identically to the host; for anything else `gamma_scores` returns `None` and the caller
//! scores on the host. `device::accepts` spells out what that rules out
//! (non-empty stabilizers, noise models other than uniform depolarizing and
//! readout, more than `MAX_QUBITS` qubits, an engine that opens no device), and
//! `device::gamma_scores` also turns away any malformed candidate.
//!
//! For the LER the device draws the sample, transposes it and scores it, so
//! the sample never crosses the bus. The draw is the host's draw: the same
//! Philox streams, the same Poisson tables and the same rejection loop, so a
//! job gets the same integer counts on either side. `ler_accepts_deferred`
//! spells out which jobs qualify, and a job the device fails on at dispatch is
//! scored on the host instead.

use super::ler::{LerJob, LerTables, ShotCounts};
use super::noise_model::UNoiseModel;
use super::pauli::Pauli;
use super::poisson_table::PoissonTables;
use rustiq_core::structures::CliffordCircuit;

/// The state every candidate in a batch shares.
pub struct GammaContext<'a> {
    pub noise_models: &'a [UNoiseModel],
    pub stabilizers: &'a [Pauli],
    pub measured_qubits: &'a [usize],
}

/// What distinguishes one candidate: the circuit with its check already
/// inserted, and the check rows to post-select on.
///
/// Candidates are described rather than pre-built: the device derives each
/// candidate's generators and cumulants itself, so the host never
/// materialises them for the whole batch.
pub struct GammaCandidate<'a> {
    pub circuit: &'a CliffordCircuit,
    pub check_qubits: &'a [usize],
    pub virtual_zs: &'a [Vec<usize>],
}

/// Scores a batch on the device, one score per candidate in input order, or
/// returns `None` if the device is unavailable or declines the batch.
pub fn gamma_scores(ctx: &GammaContext, candidates: &[GammaCandidate]) -> Option<Vec<f64>> {
    if candidates.is_empty() {
        return Some(Vec::new());
    }
    device::gamma_scores(ctx, candidates)
}

/// Whether the device would take a batch sharing `ctx` on `nqubits`-qubit
/// circuits. Cheap, so callers can check it before building any candidate.
pub fn accepts(ctx: &GammaContext, nqubits: usize) -> bool {
    device::accepts(ctx, nqubits)
}

/// Whether the device would draw and score an LER job with these tables, so
/// that its preparation can skip the host draw.
pub fn ler_accepts_deferred(
    tables: &LerTables,
    poisson: &PoissonTables,
    rates: &[f64],
    nshots: usize,
) -> bool {
    device::ler_accepts_deferred(tables, poisson, rates, nshots)
}

/// Draws and scores every deferred job on the device, returning
/// `(counts, nonempty)` per job in input order, or `None` for a job that was
/// drawn on the host or that the device failed on.
pub fn ler_counts_batch(jobs: &[LerJob]) -> Vec<Option<(ShotCounts, usize)>> {
    device::ler_counts_batch(jobs).0
}

/// Whether a scoring engine is registered. Cheap: this sits on the batch path.
pub fn engine_enabled() -> bool {
    device::ffi::registered().is_some()
}

/// Installs a scoring engine for the rest of the process.
///
/// # Safety
///
/// As for `device::ffi::register`.
pub unsafe fn register_engine(table: *const std::os::raw::c_void) -> Result<(), String> {
    unsafe { device::ffi::register(table.cast()) }
}

/// The registered engine's name.
pub fn engine_name() -> Option<String> {
    device::ffi::name()
}

/// Whether the registered engine opened a device session.
pub(crate) fn session_available() -> bool {
    device::session_available()
}

/// How one conformance check went.
pub enum CheckOutcome {
    Passed,
    Failed(String),
    Skipped(String),
}

/// Runs every conformance check against the registered engine, in order.
///
/// The checks compare each engine entry point, and the evaluator paths that
/// use them, with the host bit for bit. They are skipped when no engine is
/// registered or no device session opens, and a decline fails them, since
/// every input is well inside an engine's documented limits.
pub fn run_conformance() -> Vec<(&'static str, CheckOutcome)> {
    let checks = gamma_conformance::CHECKS
        .iter()
        .chain(ler_conformance::CHECKS)
        .chain(crate::check_picker::engine_conformance::CHECKS);
    let ready = if !engine_enabled() {
        Some("no scoring engine is registered".to_string())
    } else if !session_available() {
        Some(format!(
            "the {} engine opened no device session",
            engine_name().unwrap_or_default()
        ))
    } else {
        None
    };
    checks
        .map(|&(name, check)| {
            if let Some(why) = &ready {
                return (name, CheckOutcome::Skipped(why.clone()));
            }
            let outcome = match std::panic::catch_unwind(check) {
                Ok(()) => CheckOutcome::Passed,
                Err(payload) => CheckOutcome::Failed(
                    payload
                        .downcast_ref::<String>()
                        .cloned()
                        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
                        .unwrap_or_else(|| "panicked".to_string()),
                ),
            };
            (name, outcome)
        })
        .collect()
}

/// Number of devices the engine sees, or a negative value if there is no
/// engine or its runtime cannot be queried.
pub fn device_count() -> i32 {
    unsafe { device::ffi::paulice_gpu_device_count() }
}

/// Gamma conformance: the engine's scores against the host's, bit for bit.
/// This is the guard against an engine's propagation, counting or reduction,
/// or its generator enumeration, drifting from the host walk.
mod gamma_conformance {
    use super::*;
    use crate::noise_model::{Readout, UNoiseModel, UniformDepolarizing};
    use rustiq_core::structures::{CliffordCircuit, CliffordGate};

    fn models() -> Vec<UNoiseModel> {
        vec![
            UNoiseModel::UniformDepolarizing(UniformDepolarizing::new(0.01)),
            UNoiseModel::Readout(Readout::new(0.02)),
        ]
    }

    /// The host score for each candidate, computed the way
    /// `CheckEvaluator::evaluate` does under `Metric::Gamma`.
    fn host_gamma_scores(ctx: &GammaContext, candidates: &[GammaCandidate]) -> Vec<f64> {
        candidates
            .iter()
            .map(|candidate| {
                let mut coverage =
                    crate::coverage::Coverage::new(candidate.circuit, ctx.noise_models);
                coverage.set_check_cumulants(candidate.check_qubits, candidate.virtual_zs);
                coverage.set_logical_cumulants(ctx.stabilizers, ctx.measured_qubits);
                coverage.gamma_apx()
            })
            .collect()
    }

    /// A small deterministic pseudo-random circuit, so the test pulls in no rng
    /// dependency and reproduces the same batch on every run.
    fn sample_circuit(nqbits: usize, ngates: usize, seed: u64) -> CliffordCircuit {
        let mut state = seed;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state >> 33
        };
        let other = |q: usize, r: u64, nqbits: usize| {
            let q2 = (r as usize) % nqbits;
            if q2 == q { (q + 1) % nqbits } else { q2 }
        };
        let mut c = CliffordCircuit::new(nqbits);
        for _ in 0..ngates {
            let q = (next() as usize) % nqbits;
            match next() % 7 {
                0 => c.gates.push(CliffordGate::CZ(q, other(q, next(), nqbits))),
                1 => c
                    .gates
                    .push(CliffordGate::CNOT(q, other(q, next(), nqbits))),
                2 => c.gates.push(CliffordGate::H(q)),
                3 => c.gates.push(CliffordGate::S(q)),
                4 => c.gates.push(CliffordGate::SqrtX(q)),
                5 => c.gates.push(CliffordGate::SqrtXd(q)),
                _ => c.gates.push(CliffordGate::Sd(q)),
            }
        }
        c
    }

    /// Scores one batch of random circuits both ways and asserts equality.
    /// `nmeasured` sets how many logical (and, minus one, post-selected) rows
    /// each candidate carries, which is what packs into a propagation
    /// wavefront -- taking it past 32 and up to 64 is what exercises the full
    /// 64-lane ballot rather than just its first few bits.
    fn assert_parity(nqubits: usize, ngates: usize, nmeasured: usize, salt: u64) -> bool {
        let models = models();
        let measured: Vec<usize> = (0..nmeasured.min(nqubits)).collect();
        let ctx = GammaContext {
            noise_models: &models,
            stabilizers: &[],
            measured_qubits: &measured,
        };

        let circuits: Vec<CliffordCircuit> = (0..8)
            .map(|s| sample_circuit(nqubits, ngates, 0x9E37_79B9_7F4A_7C15 ^ salt ^ s))
            .collect();
        // Several check rows so the post-selected table also spans many rows.
        let check_qubits: Vec<usize> = (0..measured.len().saturating_sub(1).max(1)).collect();
        // Give every other row one or two virtual Zs, varying by candidate, so
        // the post-selected seeds carry more than the check qubit's own bit.
        let virtual_zs: Vec<Vec<Vec<usize>>> = (0..circuits.len())
            .map(|s| {
                check_qubits
                    .iter()
                    .map(|&q| match (q + s) % 4 {
                        1 => vec![(q + 1 + s) % nqubits],
                        3 => vec![(q + 2 + s) % nqubits, (q + 3 * s + 5) % nqubits],
                        _ => Vec::new(),
                    })
                    .collect()
            })
            .collect();
        let candidates: Vec<GammaCandidate> = circuits
            .iter()
            .zip(virtual_zs.iter())
            .map(|(circuit, vzs)| GammaCandidate {
                circuit,
                check_qubits: &check_qubits,
                virtual_zs: vzs,
            })
            .collect();

        let cpu = host_gamma_scores(&ctx, &candidates);
        let gpu = match device::gamma_scores(&ctx, &candidates) {
            Some(scores) => scores,
            // Device present but declined the batch (or the launch failed); the
            // CPU path already covers this, so there is nothing to compare.
            None => return false,
        };

        assert_eq!(cpu.len(), gpu.len());
        for (i, (a, b)) in cpu.iter().zip(gpu.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "nqubits={nqubits} nmeasured={nmeasured} candidate {i}: cpu {a} vs gpu {b}"
            );
        }
        true
    }

    fn device_matches_cpu_bit_for_bit() {
        if device_count() <= 0 {
            return;
        }
        // Sweep widths and row counts to cover the shapes the kernels
        // specialise on:
        //
        //   * width picks the propagate specialisation -- `QW` is 1 up to 64
        //     qubits, then 2, 3 and 4 at the 128, 192 and 256 boundaries, so
        //     each register-blocking variant gets exercised.
        //   * row count drives the wavefront packing. Past 32 rows a wave32
        //     device would diverge, and past 64 the columns need more than one
        //     word, which brings in the multi-row-block path.
        for (nqubits, ngates, nmeasured) in [
            (4, 24, 3),
            (16, 60, 16),
            (40, 120, 40),
            (64, 200, 64),
            (128, 300, 128),
            (192, 360, 192),
            (256, 420, 256),
        ] {
            assert!(
                assert_parity(nqubits, ngates, nmeasured, nqubits as u64),
                "the device declined the {nqubits}-qubit batch"
            );
        }
    }

    pub(super) const CHECKS: &[(&str, fn())] = &[(
        "device_matches_cpu_bit_for_bit",
        device_matches_cpu_bit_for_bit,
    )];
}

/// LER conformance, stage by stage and end to end. The device counts shots in
/// integers, so every comparison is exact. Each test skips with a note when no device
/// session can be opened, and fails if the device declines: every input here
/// is well inside the device's limits, and a pass on a GPU machine must mean
/// the device ran.
mod ler_conformance {
    use super::*;
    use crate::coverage::Coverage;
    use crate::ler::{
        LerTables, ShotCounts, sample_shot_positions, sample_shots, score_shots, transpose_to_shots,
    };
    use crate::noise_model::{Readout, UNoiseModel, UniformDepolarizing};
    use crate::philox::GeneratorStream;
    use crate::poisson_table::PoissonTables;
    use rustiq_core::structures::{CliffordCircuit, CliffordGate};

    fn device_or_note(test: &str) -> bool {
        if !device::session_available() {
            eprintln!("note: no engine device session; {test} was NOT checked here");
            return false;
        }
        true
    }

    fn declined(test: &str) -> ! {
        panic!("the device declined; {test} was not checked");
    }

    /// A CZ-H chain measured on every qubit, post-selected on qubit `check`.
    fn chain_coverage<'a>(
        circuit: &'a CliffordCircuit,
        models: &'a [UNoiseModel],
        check: usize,
    ) -> Coverage<'a> {
        let measured: Vec<usize> = (0..circuit.nqbits).collect();
        let mut coverage = Coverage::new(circuit, models);
        coverage.set_check_cumulants(&[check], &[vec![]]);
        coverage.set_logical_cumulants(&[], &measured);
        coverage
    }

    fn chain(nqbits: usize) -> CliffordCircuit {
        let mut circuit = CliffordCircuit::new(nqbits);
        for q in 0..nqbits - 1 {
            circuit.gates.push(CliffordGate::CZ(q, q + 1));
            circuit.gates.push(CliffordGate::H(q));
        }
        circuit
    }

    fn pack(
        coverage: &Coverage,
        nqbits: usize,
    ) -> (Vec<crate::noise_model::NoiseGenerator>, LerTables) {
        let generators = coverage.get_generator_errors();
        let tables = LerTables::pack(
            &generators,
            coverage.post_selected_cumulants(),
            coverage.logical_cumulants(),
            nqbits,
        );
        (generators, tables)
    }

    fn device_scores_a_host_sample_like_the_host() {
        let test = "LER scoring of a host-drawn sample";
        if !device_or_note(test) {
            return;
        }
        let models = [
            UNoiseModel::UniformDepolarizing(UniformDepolarizing::new(0.01)),
            UNoiseModel::Readout(Readout::new(0.02)),
        ];
        for (nqbits, nshots) in [(4usize, 1024usize), (12, 8192), (32, 4096), (64, 2048)] {
            let circuit = chain(nqbits);
            let coverage = chain_coverage(&circuit, &models, 0);
            let (generators, tables) = pack(&coverage, nqbits);
            let sample = sample_shots(&generators, nshots, Some(20260903));
            let Some(device) = device::ler_counts(&tables, &sample) else {
                declined(test);
            };
            assert_eq!(
                device,
                score_shots(&tables, &sample),
                "{nqbits} qubits, {nshots} shots (stride {}, {} generators)",
                tables.stride(),
                tables.ngen
            );
        }
    }

    /// Gate noise 0.003 and readout 0.01 over 64 generators, so 64
    /// generators share 2 Poisson tables and the generator-to-table map is
    /// not the identity.
    fn two_rate_generators() -> Vec<f64> {
        (0..64)
            .map(|i| if i % 2 == 0 { 0.003 } else { 0.01 })
            .collect()
    }

    /// 200k shots gives tables about three times wider than 20k, so the
    /// binary search runs deeper and each draw asks for ten times more
    /// positions.
    const SHOT_COUNTS: [usize; 2] = [20_000, 200_000];

    fn host_draws(seed: u64, tables: &PoissonTables, nshots: usize, ngen: usize) -> Vec<Vec<u32>> {
        (0..ngen)
            .map(|g| {
                let mut stream = GeneratorStream::new(seed, g);
                sample_shot_positions(&mut stream, tables.for_generator(g), nshots)
            })
            .collect()
    }

    fn device_stream_words_match_the_host() {
        let test = "the device Philox stream";
        if !device_or_note(test) {
            return;
        }
        let rates = two_rate_generators();
        let tables = PoissonTables::build(&rates, SHOT_COUNTS[0]);
        // Twelve words span three Philox blocks, so a buffering error shows
        // up at a block boundary.
        const PROBE: usize = 12;
        let seed = 0x5EED_D0D0_u64;
        let Some(got) = device::ler_draw_parity(seed, SHOT_COUNTS[0], &tables, PROBE) else {
            declined(test);
        };
        for g in 0..rates.len() {
            let mut host = GeneratorStream::new(seed, g);
            for i in 0..PROBE {
                assert_eq!(
                    got.words[g * PROBE + i],
                    host.next_u64(),
                    "generator {g} word {i}"
                );
            }
        }
    }

    /// The device keeps the same set of positions as the host; the order
    /// within a generator may differ, so positions are compared sorted and
    /// checked for duplicates first.
    fn device_draw_matches_the_host_draw() {
        let test = "the device draw";
        if !device_or_note(test) {
            return;
        }
        let rates = two_rate_generators();
        let seed = 0x0DDB_A110_u64;
        for nshots in SHOT_COUNTS {
            let tables = PoissonTables::build(&rates, nshots);
            assert_eq!(tables.num_tables(), 2);
            let Some(got) = device::ler_draw_parity(seed, nshots, &tables, 0) else {
                declined(test);
            };
            for (g, want) in host_draws(seed, &tables, nshots, rates.len())
                .iter()
                .enumerate()
            {
                let count = got.counts[g];
                assert_ne!(
                    count,
                    device::ffi::LER_DRAW_OVERFLOW,
                    "{nshots} shots, generator {g}"
                );
                assert_eq!(count as usize, want.len(), "{nshots} shots, generator {g}");
                let mut have = got.positions[g * got.pos_stride..][..want.len()].to_vec();
                have.sort_unstable();
                assert!(
                    have.windows(2).all(|w| w[0] < w[1]),
                    "{nshots} shots, generator {g}: duplicate position"
                );
                assert_eq!(&have, want, "{nshots} shots, generator {g}");
            }
        }
    }

    /// Equal positions do not show that the device consumed as many stream
    /// words as the host, since a collision costs an extra draw. The next
    /// word after the draw does.
    fn device_leaves_the_stream_where_the_host_does() {
        let test = "the device stream position after the draw";
        if !device_or_note(test) {
            return;
        }
        let rates = two_rate_generators();
        let nshots = SHOT_COUNTS[1];
        let tables = PoissonTables::build(&rates, nshots);
        let seed = 0xC0FF_EE00_u64;
        let Some(got) = device::ler_draw_parity(seed, nshots, &tables, 0) else {
            declined(test);
        };
        for g in 0..rates.len() {
            let mut host = GeneratorStream::new(seed, g);
            let _ = sample_shot_positions(&mut host, tables.for_generator(g), nshots);
            assert_eq!(got.after[g], host.next_u64(), "generator {g}");
        }
    }

    fn device_transpose_matches_the_host_transpose() {
        let test = "the device transpose";
        if !device_or_note(test) {
            return;
        }
        let rates = two_rate_generators();
        for nshots in SHOT_COUNTS {
            let tables = PoissonTables::build(&rates, nshots);
            let fired = host_draws(0x7A5A_5B05, &tables, nshots, rates.len());
            let want = transpose_to_shots(&fired, nshots);
            let Some(got) = device::ler_transpose(&fired, nshots) else {
                declined(test);
            };
            assert_eq!(got.offsets, want.offsets, "{nshots} shots: offsets");
            assert_eq!(got.indices, want.indices, "{nshots} shots: indices");
            assert_eq!(got.nonempty, want.nonempty, "{nshots} shots: nonempty");
        }
    }

    /// Positions handed to the transpose in reverse order, as the device
    /// draw may emit them, still come out in the host's order, and every
    /// firing is kept.
    fn device_transpose_sorts_each_shot_and_keeps_every_firing() {
        let test = "the device transpose's per-shot sort";
        if !device_or_note(test) {
            return;
        }
        let rates = two_rate_generators();
        let nshots = SHOT_COUNTS[1];
        let tables = PoissonTables::build(&rates, nshots);
        let fired = host_draws(0x0C0F_5E12, &tables, nshots, rates.len());
        let reversed: Vec<Vec<u32>> = fired
            .iter()
            .map(|shots| shots.iter().rev().copied().collect())
            .collect();
        let drawn: usize = fired.iter().map(Vec::len).sum();
        let Some(got) = device::ler_transpose(&reversed, nshots) else {
            declined(test);
        };
        assert_eq!(got.offsets[nshots] as usize, drawn);
        let want = transpose_to_shots(&fired, nshots);
        assert_eq!(got.indices, want.indices);
    }

    fn realistic_models() -> [UNoiseModel; 2] {
        [
            UNoiseModel::UniformDepolarizing(UniformDepolarizing::new(0.003)),
            UNoiseModel::Readout(Readout::new(0.01)),
        ]
    }

    /// What the host gets for a job: drawn and scored on the host.
    fn host_counts(coverage: &Coverage, nshots: usize, seed: u64) -> (ShotCounts, usize) {
        let job = coverage.prepare_psr_ler(nshots, Some(seed));
        crate::ler::score_batch(std::slice::from_ref(&job))
            .pop()
            .expect("one result per job")
    }

    fn device_draws_and_scores_like_the_host() {
        let test = "LER drawn and scored on the device";
        if !device_or_note(test) {
            return;
        }
        let models = realistic_models();
        let seed = 0x5A3F_1E00_u64;
        for (nqbits, nshots) in [(12usize, 8192usize), (32, 20_000), (64, 20_000)] {
            let circuit = chain(nqbits);
            let coverage = chain_coverage(&circuit, &models, 0);
            let job = coverage.prepare_psr_ler_deferred(nshots, Some(seed));
            if job.deferred().is_none() {
                declined(test);
            }
            let (results, draws) = device::ler_counts_batch(std::slice::from_ref(&job));
            let Some(device) = results[0] else {
                declined(test);
            };
            assert_eq!(draws, 1);
            assert_eq!(
                device,
                host_counts(&coverage, nshots, seed),
                "{nqbits} qubits, {nshots} shots"
            );
        }
    }

    /// Candidates that differ only in their check share generators, so the
    /// device draws once for the round. Each must still get exactly what it
    /// gets alone, and the candidates must differ, or one answer returned for
    /// the whole round would pass.
    fn a_round_sharing_one_sample_scores_like_each_candidate_alone() {
        let test = "a round sharing one device sample";
        if !device_or_note(test) {
            return;
        }
        let models = realistic_models();
        let seed = 0x0A0B_0C0D_u64;
        for (nqbits, nshots) in [(32usize, 20_000usize), (64, 20_000), (64, 200_000)] {
            let circuit = chain(nqbits);
            let coverages: Vec<Coverage> = [0usize, 5, 11, 23]
                .into_iter()
                .map(|check| chain_coverage(&circuit, &models, check))
                .collect();
            let jobs: Vec<_> = coverages
                .iter()
                .map(|c| c.prepare_psr_ler_deferred(nshots, Some(seed)))
                .collect();
            if jobs.iter().any(|job| job.deferred().is_none()) {
                declined(test);
            }
            let (results, draws) = device::ler_counts_batch(&jobs);
            if results.iter().any(Option::is_none) {
                declined(test);
            }
            assert_eq!(draws, 1, "{nqbits} qubits, {nshots} shots");
            let alone: Vec<_> = coverages
                .iter()
                .map(|c| host_counts(c, nshots, seed))
                .collect();
            for (i, (got, want)) in results.iter().zip(&alone).enumerate() {
                assert_eq!(
                    got.as_ref(),
                    Some(want),
                    "{nqbits} qubits, {nshots} shots, candidate {i}"
                );
            }
            assert!(
                alone.iter().any(|r| r.0 != alone[0].0),
                "{nqbits} qubits, {nshots} shots: every candidate scored the same"
            );
        }
    }

    /// A round mixing jobs drawn on the host, jobs left to the device, and a
    /// deferred job the device turns down at dispatch: each gets what the
    /// host gives it, in input order.
    fn a_mixed_round_scores_every_job_like_the_host() {
        let test = "a round mixing host and device jobs";
        if !device_or_note(test) {
            return;
        }
        let models = realistic_models();
        let nshots = 20_000;
        let circuit = chain(32);
        let coverages: Vec<Coverage> = [0usize, 7, 19, 30]
            .into_iter()
            .map(|check| chain_coverage(&circuit, &models, check))
            .collect();
        let seeds = [11u64, 12, 11, 13];
        let mut jobs = vec![
            coverages[0].prepare_psr_ler_deferred(nshots, Some(seeds[0])),
            coverages[1].prepare_psr_ler(nshots, Some(seeds[1])),
            coverages[2].prepare_psr_ler_deferred(nshots, Some(seeds[2])),
            coverages[3].prepare_psr_ler_deferred(nshots, Some(seeds[3])),
        ];
        if [0, 2, 3].iter().any(|&i| jobs[i].deferred().is_none()) {
            declined(test);
        }
        // A trailing word breaks the table's shape, which the dispatch-time
        // check refuses; the host scorer only reads whole rows.
        jobs[3].tables.words.push(0);

        let (device, _) = device::ler_counts_batch(&jobs);
        assert!(device[1].is_none(), "a host-drawn job went to the device");
        assert!(device[3].is_none(), "the malformed job went to the device");
        if device[0].is_none() || device[2].is_none() {
            declined(test);
        }
        let got = crate::ler::score_batch(&jobs);
        for (i, (coverage, seed)) in coverages.iter().zip(seeds).enumerate() {
            assert_eq!(got[i], host_counts(coverage, nshots, seed), "job {i}");
        }
    }

    /// The resident sample outlives a dispatch, as it does between the chunks
    /// of a round, and is replaced when the key changes: A, A, B, then A
    /// again. Draw counts are not asserted, since tests share the session.
    fn the_resident_sample_is_reused_and_replaced_across_dispatches() {
        let test = "the resident sample across dispatches";
        if !device_or_note(test) {
            return;
        }
        let models = realistic_models();
        let nshots = 20_000;
        let circuit = chain(32);
        let coverage = chain_coverage(&circuit, &models, 5);
        let (a, b) = (0x00A1_u64, 0x00B2_u64);
        assert_ne!(
            host_counts(&coverage, nshots, a).0,
            host_counts(&coverage, nshots, b).0,
            "the two keys must give different counts"
        );
        for seed in [a, a, b, a] {
            let job = coverage.prepare_psr_ler_deferred(nshots, Some(seed));
            if job.deferred().is_none() {
                declined(test);
            }
            let (results, _) = device::ler_counts_batch(std::slice::from_ref(&job));
            let Some(device) = results[0] else {
                declined(test);
            };
            assert_eq!(
                device,
                host_counts(&coverage, nshots, seed),
                "seed {seed:#x}"
            );
        }
    }

    /// An unseeded job resolves its seed when it is prepared, and the device
    /// draws from that seed.
    fn an_unseeded_job_draws_from_its_resolved_seed() {
        let test = "an unseeded job on the device";
        if !device_or_note(test) {
            return;
        }
        let models = realistic_models();
        let nshots = 20_000;
        let circuit = chain(32);
        let coverage = chain_coverage(&circuit, &models, 11);
        let job = coverage.prepare_psr_ler_deferred(nshots, None);
        let Some(draw) = job.deferred() else {
            declined(test);
        };
        let seed = draw.seed;
        let (results, _) = device::ler_counts_batch(std::slice::from_ref(&job));
        let Some(device) = results[0] else {
            declined(test);
        };
        assert_eq!(device, host_counts(&coverage, nshots, seed));
    }

    pub(super) const CHECKS: &[(&str, fn())] = &[
        (
            "device_scores_a_host_sample_like_the_host",
            device_scores_a_host_sample_like_the_host,
        ),
        (
            "device_stream_words_match_the_host",
            device_stream_words_match_the_host,
        ),
        (
            "device_draw_matches_the_host_draw",
            device_draw_matches_the_host_draw,
        ),
        (
            "device_leaves_the_stream_where_the_host_does",
            device_leaves_the_stream_where_the_host_does,
        ),
        (
            "device_transpose_matches_the_host_transpose",
            device_transpose_matches_the_host_transpose,
        ),
        (
            "device_transpose_sorts_each_shot_and_keeps_every_firing",
            device_transpose_sorts_each_shot_and_keeps_every_firing,
        ),
        (
            "device_draws_and_scores_like_the_host",
            device_draws_and_scores_like_the_host,
        ),
        (
            "a_round_sharing_one_sample_scores_like_each_candidate_alone",
            a_round_sharing_one_sample_scores_like_each_candidate_alone,
        ),
        (
            "a_mixed_round_scores_every_job_like_the_host",
            a_mixed_round_scores_every_job_like_the_host,
        ),
        (
            "the_resident_sample_is_reused_and_replaced_across_dispatches",
            the_resident_sample_is_reused_and_replaced_across_dispatches,
        ),
        (
            "an_unseeded_job_draws_from_its_resolved_seed",
            an_unseeded_job_draws_from_its_resolved_seed,
        ),
    ];
}

/// Engine dispatch. Builds the flat batch description the engine consumes,
/// runs it, and turns the returned rate sums into scores.
mod device {
    use super::{GammaCandidate, GammaContext};
    use crate::ler::{LerJob, LerTables, ShotCounts};
    use crate::noise_model::UNoiseModel;
    use crate::poisson_table::{FlatTables, PoissonTables};
    use rayon::prelude::*;
    use rustiq_core::structures::{CliffordCircuit, CliffordGate};
    use std::sync::{Mutex, OnceLock};

    /// Matches `MAX_QWORDS * 64` in the kernels.
    const MAX_QUBITS: usize = 256;

    const GATE_CNOT: u32 = 0;
    const GATE_CZ: u32 = 1;
    const GATE_H: u32 = 2;
    const GATE_S: u32 = 3;
    const GATE_SD: u32 = 4;
    const GATE_SQRTX: u32 = 5;
    const GATE_SQRTXD: u32 = 6;

    const NOISE_DEPOLARIZING: i32 = 0;
    const NOISE_READOUT: i32 = 1;

    pub mod ffi {
        use std::os::raw::{c_char, c_int, c_void};

        #[repr(C)]
        pub struct NoiseBlock {
            pub kind: i32,
            pub rate: f64,
        }

        #[repr(C)]
        pub struct GammaBatch {
            pub n_candidates: i32,
            pub nqubits: i32,
            pub gates: *const u32,
            pub gate_offset: *const i32,
            pub n_two_qubit: *const i32,
            pub post_seed: *const u64,
            pub post_row_offset: *const i32,
            pub logical_seed: *const u64,
            pub logical_row_offset: *const i32,
            pub row_words: i32,
            pub readout_column: *const i32,
            pub blocks: *const NoiseBlock,
            pub n_blocks: i32,
            pub out_rates: *mut f64,
        }

        /// Matches `PAULICE_LER_MAX_WORDS`.
        pub const LER_MAX_WORDS: usize = 1024;

        /// Matches `PAULICE_LER_MAX_GENERATORS`.
        pub const LER_MAX_GENERATORS: usize = 65535;

        #[repr(C)]
        pub struct LerSampleScore {
            pub gen_words: *const u64,
            pub n_generators: i32,
            pub stride: i32,
            pub planes_end: i32,
            pub post_end: i32,
            pub n_shots: i32,
            pub seed: u64,
            pub cdf_all: *const f64,
            pub n_cdf: u32,
            pub table_offset: *const u32,
            pub table_len: *const u32,
            pub table_klo: *const u32,
            pub n_tables: u32,
            pub table_of: *const u32,
            pub pos_stride: u32,
            pub reuse_sample: i32,
            pub out_accepted: *mut u64,
            pub out_logical: *mut u64,
            pub out_nonempty: *mut u64,
        }

        #[repr(C)]
        pub struct LerBatch {
            pub gen_words: *const u64,
            pub n_generators: i32,
            pub stride: i32,
            pub planes_end: i32,
            pub post_end: i32,
            pub offsets: *const u32,
            pub indices: *const u32,
            pub n_shots: i32,
            pub n_indices: i32,
            pub out_accepted: *mut u64,
            pub out_logical: *mut u64,
        }

        /// Matches `PAULICE_LER_DRAW_OVERFLOW`.
        pub const LER_DRAW_OVERFLOW: u32 = u32::MAX;

        #[repr(C)]
        pub struct LerDrawParity {
            pub seed: u64,
            pub n_generators: u32,
            pub nshots: u32,
            pub cdf_all: *const f64,
            pub n_cdf: u32,
            pub table_offset: *const u32,
            pub table_len: *const u32,
            pub table_klo: *const u32,
            pub n_tables: u32,
            pub table_of: *const u32,
            pub probe_words: u32,
            pub out_words: *mut u64,
            pub out_counts: *mut u32,
            pub out_pos: *mut u32,
            pub pos_stride: u32,
            pub out_after: *mut u64,
        }

        #[repr(C)]
        pub struct LerTranspose {
            pub pos: *const u32,
            pub pos_stride: u32,
            pub gen_counts: *const u32,
            pub n_generators: u32,
            pub n_shots: u32,
            pub out_offsets: *mut u32,
            pub out_indices: *mut u32,
            pub n_indices_capacity: u32,
            pub out_nonempty: *mut u64,
        }

        /// Matches `PAULICE_ENGINE_ABI_VERSION`.
        pub const ABI_VERSION: u32 = 1;

        /// Matches `PauliceEngine` in `include/paulice_engine.h`. The entries
        /// are `Option`s so that a null pointer from a broken engine is caught
        /// at registration rather than called.
        #[repr(C)]
        pub struct Engine {
            pub abi_version: u32,
            pub size: u32,
            pub name: *const c_char,
            pub device_count: Option<unsafe extern "C" fn() -> c_int>,
            pub create: Option<unsafe extern "C" fn(c_int) -> *mut c_void>,
            pub destroy: Option<unsafe extern "C" fn(*mut c_void)>,
            pub gamma_batch: Option<unsafe extern "C" fn(*mut c_void, *const GammaBatch) -> c_int>,
            pub ler_sample_score:
                Option<unsafe extern "C" fn(*mut c_void, *const LerSampleScore) -> c_int>,
            pub ler_score: Option<unsafe extern "C" fn(*mut c_void, *const LerBatch) -> c_int>,
            pub ler_draw_parity:
                Option<unsafe extern "C" fn(*mut c_void, *const LerDrawParity) -> c_int>,
            pub ler_transpose:
                Option<unsafe extern "C" fn(*mut c_void, *const LerTranspose) -> c_int>,
        }

        struct Registered(&'static Engine);
        // The table is immutable, process-lifetime data in the engine's
        // library, and its entries are plain C functions.
        unsafe impl Send for Registered {}
        unsafe impl Sync for Registered {}

        static ENGINE: std::sync::OnceLock<Registered> = std::sync::OnceLock::new();

        /// Installs the engine for the rest of the process.
        ///
        /// # Safety
        ///
        /// `engine` must be null or point to a table, at least as large as it
        /// says, that stays valid and unchanged for the rest of the process.
        pub unsafe fn register(engine: *const Engine) -> Result<(), String> {
            if engine.is_null() {
                return Err("the engine table is null".into());
            }
            // Only the two leading words are known to exist until they have
            // been checked.
            let (version, size) = unsafe {
                (
                    std::ptr::addr_of!((*engine).abi_version).read(),
                    std::ptr::addr_of!((*engine).size).read(),
                )
            };
            if version != ABI_VERSION {
                return Err(format!(
                    "the engine was built for scoring-engine ABI {version}, and this \
                     qiskit-paulice speaks ABI {ABI_VERSION}"
                ));
            }
            if (size as usize) < std::mem::size_of::<Engine>() {
                return Err(format!(
                    "the engine table is {size} bytes, too small for ABI {version}"
                ));
            }
            let table = unsafe { &*engine };
            if table.device_count.is_none()
                || table.create.is_none()
                || table.destroy.is_none()
                || table.gamma_batch.is_none()
                || table.ler_sample_score.is_none()
                || table.ler_score.is_none()
                || table.ler_draw_parity.is_none()
                || table.ler_transpose.is_none()
            {
                return Err("the engine table has a null entry".into());
            }
            ENGINE
                .set(Registered(table))
                .map_err(|_| "a scoring engine is already registered".to_string())
        }

        pub fn registered() -> Option<&'static Engine> {
            ENGINE.get().map(|r| r.0)
        }

        /// The engine's name, if one is registered.
        pub fn name() -> Option<String> {
            let table = registered()?;
            if table.name.is_null() {
                return Some(String::new());
            }
            Some(
                unsafe { std::ffi::CStr::from_ptr(table.name) }
                    .to_string_lossy()
                    .into_owned(),
            )
        }

        fn table() -> &'static Engine {
            registered().expect("a device context exists only with a registered engine")
        }

        // The wrappers keep the names the rest of this module calls; every
        // entry was checked to be non-null at registration.

        pub unsafe fn paulice_gpu_device_count() -> c_int {
            match registered() {
                Some(table) => unsafe { table.device_count.unwrap()() },
                None => -1,
            }
        }
        pub unsafe fn paulice_gpu_create(device_id: c_int) -> *mut c_void {
            unsafe { table().create.unwrap()(device_id) }
        }
        pub unsafe fn paulice_gpu_destroy(ctx: *mut c_void) {
            unsafe { table().destroy.unwrap()(ctx) }
        }
        pub unsafe fn paulice_gpu_gamma_batch(ctx: *mut c_void, batch: *const GammaBatch) -> c_int {
            unsafe { table().gamma_batch.unwrap()(ctx, batch) }
        }
        pub unsafe fn paulice_gpu_ler_sample_score(
            ctx: *mut c_void,
            req: *const LerSampleScore,
        ) -> c_int {
            unsafe { table().ler_sample_score.unwrap()(ctx, req) }
        }
        pub unsafe fn paulice_gpu_ler_score(ctx: *mut c_void, batch: *const LerBatch) -> c_int {
            unsafe { table().ler_score.unwrap()(ctx, batch) }
        }
        pub unsafe fn paulice_gpu_ler_draw_parity(
            ctx: *mut c_void,
            req: *const LerDrawParity,
        ) -> c_int {
            unsafe { table().ler_draw_parity.unwrap()(ctx, req) }
        }
        pub unsafe fn paulice_gpu_ler_transpose(
            ctx: *mut c_void,
            req: *const LerTranspose,
        ) -> c_int {
            unsafe { table().ler_transpose.unwrap()(ctx, req) }
        }
    }

    /// The device context is a plain handle. Calls on it are serialised by the
    /// mutex around the session, but may come from any thread.
    struct Session {
        ctx: *mut std::os::raw::c_void,
        /// The inputs of the LER sample left on the device by the last
        /// successful draw, and its non-empty shot count.
        resident: Option<(SampleKey, usize)>,
    }
    unsafe impl Send for Session {}

    impl Drop for Session {
        fn drop(&mut self) {
            unsafe { ffi::paulice_gpu_destroy(self.ctx) };
        }
    }

    /// Everything an LER sample is a function of. Rates are compared by their
    /// bits, so two keys are equal only if the draws are.
    #[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
    struct SampleKey {
        nshots: usize,
        seed: u64,
        rates: Vec<u64>,
    }

    /// Whether the registered engine opened a device session.
    pub fn session_available() -> bool {
        session().is_some()
    }

    fn session() -> Option<&'static Mutex<Session>> {
        static SESSION: OnceLock<Option<Mutex<Session>>> = OnceLock::new();
        SESSION
            .get_or_init(|| {
                if super::device_count() <= 0 {
                    return None;
                }
                let device = std::env::var("PAULICE_ENGINE_DEVICE")
                    .ok()
                    .and_then(|v| v.parse::<i32>().ok())
                    .unwrap_or(0);
                let ctx = unsafe { ffi::paulice_gpu_create(device) };
                if ctx.is_null() {
                    None
                } else {
                    Some(Mutex::new(Session {
                        ctx,
                        resident: None,
                    }))
                }
            })
            .as_ref()
    }

    fn opcode(gate: &CliffordGate) -> (u32, u32, u32) {
        match gate {
            CliffordGate::CNOT(i, j) => (GATE_CNOT, *i as u32, *j as u32),
            CliffordGate::CZ(i, j) => (GATE_CZ, *i as u32, *j as u32),
            CliffordGate::H(i) => (GATE_H, *i as u32, 0),
            CliffordGate::S(i) => (GATE_S, *i as u32, 0),
            CliffordGate::Sd(i) => (GATE_SD, *i as u32, 0),
            CliffordGate::SqrtX(i) => (GATE_SQRTX, *i as u32, 0),
            CliffordGate::SqrtXd(i) => (GATE_SQRTXD, *i as u32, 0),
        }
    }

    /// The noise blocks, if every model is one the device knows how to
    /// enumerate. `None` means "score this batch on the CPU".
    ///
    /// Only models whose generators all share a single rate qualify: that is
    /// what lets the device collapse the gamma sum to a count and still
    /// reproduce the host's summation order bit for bit.
    fn blocks(models: &[UNoiseModel]) -> Option<Vec<ffi::NoiseBlock>> {
        if models.is_empty() {
            return None;
        }
        models
            .iter()
            .map(|model| match model {
                UNoiseModel::UniformDepolarizing(m) => Some(ffi::NoiseBlock {
                    kind: NOISE_DEPOLARIZING,
                    rate: m.rate(),
                }),
                UNoiseModel::Readout(m) => Some(ffi::NoiseBlock {
                    kind: NOISE_READOUT,
                    rate: m.rate(),
                }),
                _ => None,
            })
            .collect()
    }

    /// Column of each qubit's readout wire, or -1 when the wire is not recorded.
    ///
    /// Mirrors the host walk's column numbering: the backward walk visits
    /// 2-qubit gates last-first, so the gate with ascending rank `r` owns
    /// columns `2 * (n2q - 1 - r)` and the next one, and the input wires follow
    /// all of them.
    ///
    /// `n2q` and `rank_of_gate` come from the caller's walk over the gates,
    /// and `out` is this candidate's slice of the shared buffer.
    fn readout_columns(
        circuit: &CliffordCircuit,
        n2q: usize,
        rank_of_gate: &[i32],
        out: &mut [i32],
    ) {
        let wires = crate::utils::last_wires(circuit);
        debug_assert_eq!(out.len(), wires.len());
        for (column, wire) in out.iter_mut().zip(wires.iter()) {
            *column = match wire {
                crate::wire::Wire::Input(q) => (2 * n2q + q) as i32,
                crate::wire::Wire::GateWire(gate_index, slot) => {
                    let rank = rank_of_gate[*gate_index];
                    if rank < 0 {
                        // A single-qubit gate's wire is never snapshotted.
                        -1
                    } else {
                        2 * (n2q as i32 - 1 - rank) + *slot as i32
                    }
                }
            };
        }
    }

    fn set_bit(mask: &mut [u64], bit: usize) {
        mask[bit / 64] |= 1u64 << (bit % 64);
    }

    /// Disjoint per-candidate views into one preallocated buffer.
    ///
    /// `split_at_mut` in a loop, so the borrow checker proves the pieces do not
    /// overlap and rayon can hand them to separate threads.
    fn split_by<'a, T>(mut buf: &'a mut [T], lens: &[usize]) -> Vec<&'a mut [T]> {
        let mut out = Vec::with_capacity(lens.len());
        for &len in lens {
            let (head, tail) = buf.split_at_mut(len);
            out.push(head);
            buf = tail;
        }
        out
    }

    pub fn accepts(ctx: &GammaContext, nqubits: usize) -> bool {
        // Forward-propagated stabilizer rows would need a second walk direction
        // the kernels do not implement; the measured-qubit mode is what the
        // search actually uses.
        ctx.stabilizers.is_empty()
            && !ctx.measured_qubits.is_empty()
            && (1..=MAX_QUBITS).contains(&nqubits)
            && ctx.measured_qubits.iter().all(|&q| q < nqubits)
            && blocks(ctx.noise_models).is_some()
            && session().is_some()
    }

    pub fn gamma_scores(ctx: &GammaContext, candidates: &[GammaCandidate]) -> Option<Vec<f64>> {
        let nqubits = candidates[0].circuit.nqbits;
        if !accepts(ctx, nqubits) {
            return None;
        }
        let blocks = blocks(ctx.noise_models)?;
        if candidates.iter().any(|c| c.circuit.nqbits != nqubits) {
            return None;
        }

        // The device path runs ahead of the CPU fallback, so a malformed batch
        // has to bail out to `None` here rather than silently truncating a
        // mismatched zip or panicking in `set_bit` on an out-of-range index.
        for candidate in candidates {
            if candidate.check_qubits.len() != candidate.virtual_zs.len() {
                return None;
            }
            if candidate.check_qubits.iter().any(|&q| q >= nqubits)
                || candidate.virtual_zs.iter().flatten().any(|&q| q >= nqubits)
            {
                return None;
            }
        }

        let row_words = nqubits.div_ceil(64);
        let measured_qubits = ctx.measured_qubits;
        let logical_stride = measured_qubits.len() * row_words;

        // Every per-candidate length is known before any work starts, so the
        // offsets are built up front and each candidate writes straight into
        // its own slice of one preallocated buffer. A `map`/`collect` would
        // have to join per-thread buffers afterwards, which is a large copy at
        // realistic batch sizes.
        let gate_lens: Vec<usize> = candidates.iter().map(|c| c.circuit.gates.len()).collect();
        let post_lens: Vec<usize> = candidates
            .iter()
            .map(|c| c.check_qubits.len() * row_words)
            .collect();

        let mut gate_offset: Vec<i32> = Vec::with_capacity(candidates.len() + 1);
        let mut post_row_offset: Vec<i32> = Vec::with_capacity(candidates.len() + 1);
        let mut logical_row_offset: Vec<i32> = Vec::with_capacity(candidates.len() + 1);
        gate_offset.push(0);
        post_row_offset.push(0);
        logical_row_offset.push(0);
        let mut total_gates = 0usize;
        let mut total_post_rows = 0usize;
        for (index, candidate) in candidates.iter().enumerate() {
            total_gates += gate_lens[index];
            total_post_rows += candidate.check_qubits.len();
            // The kernels index with i32, so a batch too large for that is
            // declined rather than handed over with wrapped offsets.
            gate_offset.push(i32::try_from(total_gates).ok()?);
            post_row_offset.push(i32::try_from(total_post_rows).ok()?);
            logical_row_offset.push(i32::try_from((index + 1) * measured_qubits.len()).ok()?);
        }

        let mut gates: Vec<u32> = vec![0; total_gates];
        let mut n_two_qubit: Vec<i32> = vec![0; candidates.len()];
        let mut post_seed: Vec<u64> = vec![0; total_post_rows * row_words];
        let mut logical_seed: Vec<u64> = vec![0; candidates.len() * logical_stride];
        let mut readout_column: Vec<i32> = vec![0; candidates.len() * nqubits];

        {
            let jobs: Vec<_> = candidates
                .iter()
                .zip(split_by(&mut gates, &gate_lens))
                .zip(split_by(&mut post_seed, &post_lens))
                .zip(logical_seed.chunks_mut(logical_stride))
                .zip(readout_column.chunks_mut(nqubits))
                .zip(n_two_qubit.iter_mut())
                .map(|(((((c, g), p), l), r), n)| (c, g, p, l, r, n))
                .collect();

            jobs.into_par_iter().for_each(
                |(candidate, gate_out, post_out, logical_out, readout_out, n2q_out)| {
                    let circuit = candidate.circuit;

                    // One walk yields the gate encoding, the 2-qubit count and
                    // the 2-qubit ranks `readout_columns` needs.
                    let mut rank_of_gate = vec![-1i32; circuit.gates.len()];
                    let mut n2q = 0i32;
                    for ((gate, word), rank) in circuit
                        .gates
                        .iter()
                        .zip(gate_out.iter_mut())
                        .zip(rank_of_gate.iter_mut())
                    {
                        let (op, q0, q1) = opcode(gate);
                        *word = (op << 24) | (q0 << 12) | q1;
                        if gate.arity() == 2 {
                            *rank = n2q;
                            n2q += 1;
                        }
                    }
                    *n2q_out = n2q;

                    // Post-selected rows: Z on the check qubit and on each virtual Z.
                    for ((row, check), vzs) in post_out
                        .chunks_mut(row_words)
                        .zip(candidate.check_qubits.iter())
                        .zip(candidate.virtual_zs.iter())
                    {
                        set_bit(row, *check);
                        for qubit in vzs {
                            set_bit(row, *qubit);
                        }
                    }

                    // Logical rows: Z on each measured qubit. They are the same
                    // for every candidate, but each candidate's circuit is
                    // walked over its own copy of the seeds.
                    for (row, qubit) in logical_out
                        .chunks_mut(row_words)
                        .zip(measured_qubits.iter())
                    {
                        set_bit(row, *qubit);
                    }

                    readout_columns(circuit, n2q as usize, &rank_of_gate, readout_out);
                },
            );
        }

        let mut rates = vec![0.0f64; candidates.len()];
        let batch = ffi::GammaBatch {
            n_candidates: candidates.len() as i32,
            nqubits: nqubits as i32,
            gates: gates.as_ptr(),
            gate_offset: gate_offset.as_ptr(),
            n_two_qubit: n_two_qubit.as_ptr(),
            post_seed: post_seed.as_ptr(),
            post_row_offset: post_row_offset.as_ptr(),
            logical_seed: logical_seed.as_ptr(),
            logical_row_offset: logical_row_offset.as_ptr(),
            row_words: row_words as i32,
            readout_column: readout_column.as_ptr(),
            blocks: blocks.as_ptr(),
            n_blocks: blocks.len() as i32,
            out_rates: rates.as_mut_ptr(),
        };

        let session = session()?;
        let guard = session.lock().ok()?;
        let status = unsafe { ffi::paulice_gpu_gamma_batch(guard.ctx, &batch) };
        drop(guard);
        if status != 0 {
            return None;
        }
        // `exp` is applied here, not on the device: the summed rate is
        // bit-identical either way, but two libm implementations need not agree
        // on the last ulp of the transcendental.
        Some(rates.into_iter().map(|acc| (2.0 * acc).exp()).collect())
    }

    /// Room per generator for its draw. No generator can draw more than its
    /// table's top count.
    fn pos_stride(poisson: &PoissonTables) -> usize {
        poisson.max_count() as usize + 1
    }

    /// Checked when a job is prepared and again at dispatch, so a job that
    /// reaches the device always fits what the kernels index with: the score
    /// kernel's accumulator, the transpose's grid, `i32` shot and generator
    /// counts, and `u32` position offsets. Memory is checked by the device
    /// itself, which declines a draw whose bitsets would not fit.
    pub fn ler_accepts_deferred(
        tables: &LerTables,
        poisson: &PoissonTables,
        rates: &[f64],
        nshots: usize,
    ) -> bool {
        let ngen = tables.ngen;
        let stride = tables.stride();
        (1..=ffi::LER_MAX_WORDS).contains(&stride)
            && (1..=ffi::LER_MAX_GENERATORS).contains(&ngen)
            && tables.words.len() == ngen * stride
            && rates.len() == ngen
            && poisson.num_generators() == ngen
            && (1..=i32::MAX as usize).contains(&nshots)
            && pos_stride(poisson)
                .checked_mul(ngen)
                .is_some_and(|positions| positions <= u32::MAX as usize)
            && session().is_some()
    }

    /// A run of deferred jobs that share one sample, with the tables
    /// flattened once for all of them.
    struct Run {
        key: SampleKey,
        flat: FlatTables,
        pos_stride: u32,
        jobs: Vec<usize>,
    }

    /// `super::ler_counts_batch`, also returning how many draws the device
    /// made, which is what tells sharing a sample apart from drawing it per
    /// job.
    pub fn ler_counts_batch(jobs: &[LerJob]) -> (Vec<Option<(ShotCounts, usize)>>, usize) {
        let mut out = vec![None; jobs.len()];
        let keyed: Vec<(usize, SampleKey)> = jobs
            .iter()
            .enumerate()
            .filter_map(|(index, job)| {
                let draw = job.deferred()?;
                Some((
                    index,
                    SampleKey {
                        nshots: job.nshots,
                        seed: draw.seed,
                        rates: draw.rates.iter().map(|r| r.to_bits()).collect(),
                    },
                ))
            })
            .collect();
        let mut order: Vec<usize> = (0..keyed.len()).collect();
        order.sort_by(|a, b| keyed[*a].1.cmp(&keyed[*b].1));

        // Flattened before the session is locked, since it does not need the
        // device.
        let mut runs: Vec<Run> = Vec::new();
        for position in order {
            let (index, key) = &keyed[position];
            match runs.last_mut() {
                Some(run) if run.key == *key => run.jobs.push(*index),
                _ => {
                    let poisson = &jobs[*index]
                        .deferred()
                        .expect("keyed jobs are deferred")
                        .poisson;
                    let Ok(pos_stride) = u32::try_from(pos_stride(poisson)) else {
                        continue;
                    };
                    runs.push(Run {
                        key: key.clone(),
                        flat: poisson.flatten(),
                        pos_stride,
                        jobs: vec![*index],
                    });
                }
            }
        }

        let Some(session) = session() else {
            return (out, 0);
        };
        let Ok(mut guard) = session.lock() else {
            return (out, 0);
        };
        let mut draws = 0;
        for run in &runs {
            for &index in &run.jobs {
                let job = &jobs[index];
                let draw = job.deferred().expect("keyed jobs are deferred");
                if !ler_accepts_deferred(&job.tables, &draw.poisson, &draw.rates, job.nshots) {
                    continue;
                }
                let reuse = guard
                    .resident
                    .as_ref()
                    .is_some_and(|(key, _)| *key == run.key);
                if let Some(counted) = sample_score(guard.ctx, job, draw.seed, run, reuse) {
                    let nonempty = if reuse {
                        guard.resident.as_ref().map_or(0, |(_, nonempty)| *nonempty)
                    } else {
                        draws += 1;
                        guard.resident = Some((run.key.clone(), counted.1));
                        counted.1
                    };
                    out[index] = Some((counted.0, nonempty));
                } else {
                    guard.resident = None;
                }
            }
        }
        (out, draws)
    }

    /// One `paulice_gpu_ler_sample_score` dispatch. The non-empty count is
    /// only meaningful when the sample was drawn, not reused.
    fn sample_score(
        ctx: *mut std::os::raw::c_void,
        job: &LerJob,
        seed: u64,
        run: &Run,
        reuse: bool,
    ) -> Option<(ShotCounts, usize)> {
        let tables = &job.tables;
        let flat = &run.flat;
        let (mut accepted, mut logical, mut nonempty) = (0u64, 0u64, 0u64);
        let req = ffi::LerSampleScore {
            gen_words: tables.words.as_ptr(),
            n_generators: i32::try_from(tables.ngen).ok()?,
            stride: i32::try_from(tables.stride()).ok()?,
            planes_end: i32::try_from(2 * tables.wire_words).ok()?,
            post_end: i32::try_from(2 * tables.wire_words + tables.post_words).ok()?,
            n_shots: i32::try_from(job.nshots).ok()?,
            seed,
            cdf_all: flat.cdf.as_ptr(),
            n_cdf: u32::try_from(flat.cdf.len()).ok()?,
            table_offset: flat.offset.as_ptr(),
            table_len: flat.len.as_ptr(),
            table_klo: flat.k_lo.as_ptr(),
            n_tables: u32::try_from(flat.offset.len()).ok()?,
            table_of: flat.table_of.as_ptr(),
            pos_stride: run.pos_stride,
            reuse_sample: i32::from(reuse),
            out_accepted: &mut accepted,
            out_logical: &mut logical,
            out_nonempty: &mut nonempty,
        };
        if unsafe { ffi::paulice_gpu_ler_sample_score(ctx, &req) } != 0 {
            return None;
        }
        Some((
            ShotCounts { accepted, logical },
            usize::try_from(nonempty).ok()?,
        ))
    }

    /// Scores a host-drawn LER sample on the device, or `None` if the device
    /// is unavailable or the tables do not fit the kernel.
    pub fn ler_counts(
        tables: &crate::ler::LerTables,
        sample: &crate::ler::ShotSample,
    ) -> Option<crate::ler::ShotCounts> {
        let stride = tables.stride();
        if stride == 0 || stride > ffi::LER_MAX_WORDS || tables.ngen == 0 || sample.nshots == 0 {
            return None;
        }
        let (mut accepted, mut logical) = (0u64, 0u64);
        let batch = ffi::LerBatch {
            gen_words: tables.words.as_ptr(),
            n_generators: i32::try_from(tables.ngen).ok()?,
            stride: stride as i32,
            planes_end: (2 * tables.wire_words) as i32,
            post_end: (2 * tables.wire_words + tables.post_words) as i32,
            offsets: sample.offsets.as_ptr(),
            indices: sample.indices.as_ptr(),
            n_shots: i32::try_from(sample.nshots).ok()?,
            n_indices: i32::try_from(sample.indices.len()).ok()?,
            out_accepted: &mut accepted,
            out_logical: &mut logical,
        };
        let session = session()?;
        let guard = session.lock().ok()?;
        let status = unsafe { ffi::paulice_gpu_ler_score(guard.ctx, &batch) };
        drop(guard);
        (status == 0).then_some(crate::ler::ShotCounts { accepted, logical })
    }

    /// The device draw of every generator, step by step.
    pub struct DeviceDraw {
        /// `probe_words` raw stream words per generator.
        pub words: Vec<u64>,
        /// Per generator; `ffi::LER_DRAW_OVERFLOW` if the draw did not fit.
        pub counts: Vec<u32>,
        /// `pos_stride` per generator, of which `counts[g]` are live.
        pub positions: Vec<u32>,
        pub pos_stride: usize,
        /// The next stream word after each generator's draw.
        pub after: Vec<u64>,
    }

    pub fn ler_draw_parity(
        seed: u64,
        nshots: usize,
        tables: &crate::poisson_table::PoissonTables,
        probe_words: usize,
    ) -> Option<DeviceDraw> {
        let flat = tables.flatten();
        let n_generators = flat.table_of.len();
        let pos_stride = tables.max_count() as usize + 1;
        let mut words = vec![0u64; probe_words * n_generators];
        let mut counts = vec![0u32; n_generators];
        let mut positions = vec![0u32; pos_stride * n_generators];
        let mut after = vec![0u64; n_generators];
        let req = ffi::LerDrawParity {
            seed,
            n_generators: u32::try_from(n_generators).ok()?,
            nshots: u32::try_from(nshots).ok()?,
            cdf_all: flat.cdf.as_ptr(),
            n_cdf: u32::try_from(flat.cdf.len()).ok()?,
            table_offset: flat.offset.as_ptr(),
            table_len: flat.len.as_ptr(),
            table_klo: flat.k_lo.as_ptr(),
            n_tables: u32::try_from(flat.offset.len()).ok()?,
            table_of: flat.table_of.as_ptr(),
            probe_words: u32::try_from(probe_words).ok()?,
            out_words: words.as_mut_ptr(),
            out_counts: counts.as_mut_ptr(),
            out_pos: positions.as_mut_ptr(),
            pos_stride: u32::try_from(pos_stride).ok()?,
            out_after: after.as_mut_ptr(),
        };
        let session = session()?;
        let guard = session.lock().ok()?;
        let status = unsafe { ffi::paulice_gpu_ler_draw_parity(guard.ctx, &req) };
        drop(guard);
        (status == 0).then_some(DeviceDraw {
            words,
            counts,
            positions,
            pos_stride,
            after,
        })
    }

    /// Runs the device transpose on host-drawn positions.
    pub fn ler_transpose(fired: &[Vec<u32>], nshots: usize) -> Option<crate::ler::ShotSample> {
        let n_generators = fired.len();
        let pos_stride = fired.iter().map(Vec::len).max().unwrap_or(0).max(1);
        let total: usize = fired.iter().map(Vec::len).sum();
        let mut pos = vec![0u32; pos_stride * n_generators];
        let mut gen_counts = vec![0u32; n_generators];
        for (g, shots) in fired.iter().enumerate() {
            pos[g * pos_stride..][..shots.len()].copy_from_slice(shots);
            gen_counts[g] = shots.len() as u32;
        }
        let mut offsets = vec![0u32; nshots + 1];
        let mut indices = vec![0u32; total];
        let mut nonempty = 0u64;
        let req = ffi::LerTranspose {
            pos: pos.as_ptr(),
            pos_stride: u32::try_from(pos_stride).ok()?,
            gen_counts: gen_counts.as_ptr(),
            n_generators: u32::try_from(n_generators).ok()?,
            n_shots: u32::try_from(nshots).ok()?,
            out_offsets: offsets.as_mut_ptr(),
            out_indices: indices.as_mut_ptr(),
            n_indices_capacity: u32::try_from(total).ok()?,
            out_nonempty: &mut nonempty,
        };
        let session = session()?;
        let mut guard = session.lock().ok()?;
        guard.resident = None;
        let status = unsafe { ffi::paulice_gpu_ler_transpose(guard.ctx, &req) };
        drop(guard);
        (status == 0).then_some(crate::ler::ShotSample {
            nshots,
            offsets,
            indices,
            nonempty: nonempty as usize,
        })
    }
}

#[cfg(test)]
mod registration_tests {
    use super::device::ffi::{ABI_VERSION, Engine, register};
    use super::*;

    fn table(abi_version: u32, size: usize) -> Engine {
        Engine {
            abi_version,
            size: size as u32,
            name: std::ptr::null(),
            device_count: None,
            create: None,
            destroy: None,
            gamma_batch: None,
            ler_sample_score: None,
            ler_score: None,
            ler_draw_parity: None,
            ler_transpose: None,
        }
    }

    /// No table here is valid, so none is installed and the other tests
    /// still run without an engine.
    #[test]
    fn registration_refuses_a_table_it_cannot_use() {
        let size = std::mem::size_of::<Engine>();
        let refusals = [
            unsafe { register(std::ptr::null()) },
            unsafe { register(&table(ABI_VERSION + 1, size)) },
            unsafe { register(&table(ABI_VERSION, size - 8)) },
            unsafe { register(&table(ABI_VERSION, size)) },
        ];
        for (refusal, needle) in refusals
            .iter()
            .zip(["null", "ABI", "too small", "null entry"])
        {
            let why = refusal
                .as_ref()
                .expect_err("an unusable table was accepted");
            assert!(why.contains(needle), "{why}");
        }
        assert!(!engine_enabled());
    }

    #[test]
    fn conformance_skips_every_check_without_an_engine() {
        let outcomes = run_conformance();
        assert_eq!(outcomes.len(), 14);
        for (name, outcome) in &outcomes {
            assert!(matches!(outcome, CheckOutcome::Skipped(_)), "{name} ran");
        }
    }
}
