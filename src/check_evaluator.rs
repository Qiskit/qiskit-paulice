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

use super::circuit_building::add_check_no_allocate;
use super::coverage::Coverage;
use super::metric::Metric;
use super::noise_model::UNoiseModel;
use super::pauli::Pauli;
use super::sparse_pauli::SparsePauli;
use super::stabilizer_group::StabilizerGroup;
use rayon::prelude::*;
use rustiq_core::structures::{CliffordCircuit, PauliLike, PauliSet};

#[derive(Clone, Debug)]
pub struct CheckEvaluator {
    base_circuit: CliffordCircuit,
    metric: Metric,
    noise_models: Vec<UNoiseModel>,
    stabilizers: Vec<Pauli>,
    measured_qubits: Vec<usize>,
    current_checks: Vec<usize>,
    current_virtual_zs: Vec<Vec<usize>>,
    ancilla: usize,
}

impl CheckEvaluator {
    #![allow(clippy::too_many_arguments)]
    pub fn new(
        base_circuit: CliffordCircuit,
        metric: Metric,
        noise_models: Vec<UNoiseModel>,
        stabilizer_group: StabilizerGroup,
        measured_qubits: Vec<usize>,
        current_checks: Vec<usize>,
        current_virtual_zs: Vec<Vec<usize>>,
        ancilla: usize,
    ) -> Self {
        Self {
            base_circuit,
            metric,
            noise_models,
            stabilizers: stabilizer_group.generators,
            measured_qubits,
            current_checks,
            current_virtual_zs,
            ancilla,
        }
    }

    pub fn get_ancilla(&self) -> usize {
        self.ancilla
    }
    /// Evaluates some check (described by a SparsePauli object and a collection of virtual Zs at the end of the circuit)
    pub fn evaluate(&self, check: &SparsePauli, virtual_zs: &[usize]) -> f64 {
        self.with_coverage(check, virtual_zs, |coverage| match self.metric {
            Metric::LogicalErrorRate(nshots, seed) => coverage.approximate_psr_ler(nshots, seed).1,
            Metric::Gamma => coverage.gamma_apx(),
            Metric::BalancedGamma => coverage.balanced_gamma_apx(),
        })
    }

    /// Runs `f` on the coverage of the base circuit with `check` added.
    fn with_coverage<R>(
        &self,
        check: &SparsePauli,
        virtual_zs: &[usize],
        f: impl FnOnce(&Coverage) -> R,
    ) -> R {
        let checked_circuit = add_check_no_allocate(self.base_circuit.clone(), check, self.ancilla);
        let mut all_check_qubits = self.current_checks.clone();
        all_check_qubits.push(self.ancilla);
        let mut all_virtual_zs = self.current_virtual_zs.clone();
        all_virtual_zs.push(virtual_zs.to_vec());
        let mut coverage = Coverage::new(&checked_circuit, &self.noise_models);
        coverage.set_check_cumulants(&all_check_qubits, &all_virtual_zs);
        coverage.set_logical_cumulants(&self.stabilizers, &self.measured_qubits);
        f(&coverage)
    }

    /// Scores each candidate in parallel; equivalent to calling `evaluate` on each.
    pub fn evaluate_batch(&self, items: &[(SparsePauli, Vec<usize>)]) -> Vec<f64> {
        if crate::engine::engine_enabled() {
            if let Some(scores) = self.evaluate_gamma_on_device(items) {
                return scores;
            }
            if let Metric::LogicalErrorRate(nshots, seed) = self.metric {
                return self.evaluate_ler_in_chunks(items, nshots, seed);
            }
        }
        items
            .par_iter()
            .map(|(check, vzs)| self.evaluate(check, vzs))
            .collect()
    }

    /// Candidates per LER dispatch: enough that a chunk takes the device
    /// session once for several candidates, few enough that chunks still
    /// prepare in parallel.
    pub(crate) const LER_CHUNK: usize = 8;

    /// The LER of each candidate, prepared per candidate and scored a chunk
    /// at a time, so the device can draw a chunk's shared sample once. Equal
    /// to `evaluate` on each, since a job's result does not depend on the
    /// jobs scored beside it.
    pub(crate) fn evaluate_ler_in_chunks(
        &self,
        items: &[(SparsePauli, Vec<usize>)],
        nshots: usize,
        seed: Option<u64>,
    ) -> Vec<f64> {
        self.evaluate_ler_in_chunks_counting_device(items, nshots, seed)
            .0
    }

    /// `evaluate_ler_in_chunks`, also returning how many of the candidates
    /// the device scored.
    pub(crate) fn evaluate_ler_in_chunks_counting_device(
        &self,
        items: &[(SparsePauli, Vec<usize>)],
        nshots: usize,
        seed: Option<u64>,
    ) -> (Vec<f64>, usize) {
        let chunks: Vec<(Vec<f64>, usize)> = items
            .par_chunks(Self::LER_CHUNK)
            .map(|chunk| {
                let jobs: Vec<crate::ler::LerJob> = chunk
                    .par_iter()
                    .map(|(check, vzs)| {
                        self.with_coverage(check, vzs, |coverage| {
                            coverage.prepare_psr_ler_deferred(nshots, seed)
                        })
                    })
                    .collect();
                let (counts, on_device) = crate::ler::score_batch_counting_device(&jobs);
                let rates = counts
                    .into_iter()
                    .zip(&jobs)
                    .map(|((counts, nonempty), job)| job.finish(counts, nonempty).1)
                    .collect();
                (rates, on_device)
            })
            .collect();
        let on_device = chunks.iter().map(|(_, n)| n).sum();
        (
            chunks.into_iter().flat_map(|(rates, _)| rates).collect(),
            on_device,
        )
    }

    /// Scores the batch on the device, or returns `None` for the caller to
    /// score on the host: the metric is not `Gamma` or the device declined the
    /// batch. `evaluate_batch` only calls it when a scoring engine is registered.
    pub(crate) fn evaluate_gamma_on_device(
        &self,
        items: &[(SparsePauli, Vec<usize>)],
    ) -> Option<Vec<f64>> {
        use super::engine::{GammaCandidate, GammaContext};

        if !matches!(self.metric, Metric::Gamma) {
            return None;
        }
        let ctx = GammaContext {
            noise_models: &self.noise_models,
            stabilizers: &self.stabilizers,
            measured_qubits: &self.measured_qubits,
        };
        if !crate::engine::accepts(&ctx, self.base_circuit.nqbits) {
            return None;
        }

        let circuits: Vec<CliffordCircuit> = items
            .par_iter()
            .map(|(check, _)| add_check_no_allocate(self.base_circuit.clone(), check, self.ancilla))
            .collect();
        let mut all_check_qubits = self.current_checks.clone();
        all_check_qubits.push(self.ancilla);
        let virtual_zs: Vec<Vec<Vec<usize>>> = items
            .iter()
            .map(|(_, vzs)| {
                let mut all = self.current_virtual_zs.clone();
                all.push(vzs.clone());
                all
            })
            .collect();
        let candidates: Vec<GammaCandidate> = circuits
            .iter()
            .zip(virtual_zs.iter())
            .map(|(circuit, vzs)| GammaCandidate {
                circuit,
                check_qubits: &all_check_qubits,
                virtual_zs: vzs,
            })
            .collect();
        crate::engine::gamma_scores(&ctx, &candidates)
    }

    /// Utility method to infer the virtual Zs for a given check.
    /// Also checks that the check ends up diagonal when pulled to the end.
    pub fn compute_vzs(&self, check: &SparsePauli) -> Vec<usize> {
        let checked_circuit = add_check_no_allocate(self.base_circuit.clone(), check, self.ancilla);
        let mut pset = PauliSet::new_empty(checked_circuit.nqbits, 1);
        pset.set_entry(0, self.ancilla, false, true);
        pset.conjugate_with_circuit(&checked_circuit);
        let final_pauli = pset.get_as_vec_bool(0).1;
        if final_pauli
            .iter()
            .take(checked_circuit.nqbits)
            .enumerate()
            .filter(|(i, _)| *i != self.ancilla)
            .any(|(_, b)| *b)
        {
            panic!(
                "Unexpected X component in forward propagatated check in the presence of measured qubits."
            );
        }
        final_pauli
            .iter()
            .enumerate()
            .filter(|(i, b)| **b && *i != self.ancilla)
            .map(|(i, _)| i - checked_circuit.nqbits)
            .collect()
    }
    /// Returns the current energy of the base circuit
    pub fn get_current_energy(&self) -> f64 {
        let mut coverage = Coverage::new(&self.base_circuit, &self.noise_models);

        coverage.set_check_cumulants(&self.current_checks, &self.current_virtual_zs);
        coverage.set_logical_cumulants(&self.stabilizers, &self.measured_qubits);
        match self.metric {
            Metric::LogicalErrorRate(nshots, seed) => coverage.approximate_psr_ler(nshots, seed).1,
            Metric::Gamma => coverage.gamma_apx(),
            Metric::BalancedGamma => coverage.balanced_gamma_apx(),
        }
    }
}
