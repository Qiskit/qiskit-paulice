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

//! Bit-plane form of the Monte Carlo logical-error-rate scorer.
//!
//! Scoring a shot means multiplying the generators that fired into one Pauli
//! error and then asking two `CumulantTable` parity questions about it. Every
//! step of that is linear over GF(2), which is what this module exploits:
//!
//! * `_prod_pauli` is the phase-free Pauli product, so on the two-bit codes it
//!   is exactly XOR -- multiplying generators is XOR of bit planes.
//! * `pauli_xz_bits` is XOR-linear on those codes, and `parity_word_at` is
//!   linear in `(xe, ze)`, so a shot's parity accumulator is the XOR of a
//!   per-generator *signature* that can be computed once per evaluation instead
//!   of once per shot.
//!
//! So a shot becomes: XOR a handful of fixed-width bit vectors, then test three
//! of them for zero. No hash maps, no per-shot allocation, and nothing to
//! lock. The answer is counted in integers and only divided at the end, so the
//! packed scorer reproduces the hash-map scorer exactly rather than
//! approximately.

use super::cumulant_table::{CumulantTable, pauli_xz_bits};
use super::noise_model::NoiseGenerator;

use crate::philox::GeneratorStream;
use crate::poisson_table::{PoissonTable, PoissonTables};
use rand::Rng;
use rayon::prelude::*;

use std::collections::HashMap;

const WORD: usize = 64;

/// Draws which of `nshots` shots one error generator fires in, ascending.
///
/// The count comes from a tabulated `Poisson(lambda)` and the positions from
/// rejection sampling without replacement, so `lambda` well below `nshots` is
/// assumed -- it is, since `proba_from_rate` is bounded by 1/2. The count is
/// capped at `nshots`, since there are only that many distinct positions to
/// draw and a larger count would never leave the loop.
///
/// Every operation here is chosen so that a seeded sample is the same on every
/// platform and at every thread count, which is why neither `StdRng` nor
/// `rand_distr` is used:
///
///   - The stream is Philox, keyed on `(seed, index)`, so generator `index` can
///     be drawn on any thread without replaying anything. `StdRng` is a
///     stream cipher, so its n'th output is only reachable by generating the
///     first n.
///   - The count is a binary search in a precomputed CDF table, so the draw
///     calls no transcendental. Every O(1) Poisson sampler for large lambda
///     needs `log` of a random uniform and a log-gamma of a random count, and
///     two libm implementations need not agree on the last ulp of either.
///   - The positions come from a multiply-shift reduction of a 64-bit word,
///     which is integer-only.
///
/// So the sample is bit-identical across platforms by construction rather than
/// statistically equivalent, which is what keeps exact-cost tests usable.
///
/// Draws in deficit-sized batches and dedups instead of testing membership one
/// draw at a time, which consumes the randomness identically: the i'th
/// `next_below` call yields the i'th value either way, and both stop at the
/// first draw count that has accumulated `num_successes` distinct values.
/// Neither can overshoot, because `d` draws add at most `d` distinct values, so
/// a batch sized to the exact deficit can only just reach the target. That
/// equivalence is what `batched_draw_matches_the_rejection_loop_exactly`
/// pins down, stream position included.
fn sample_shot_positions(
    stream: &mut GeneratorStream,
    table: &PoissonTable,
    nshots: usize,
) -> Vec<u32> {
    let num_successes = (table.sample(stream.next_f64()) as usize).min(nshots);
    let mut pos: Vec<u32> = Vec::with_capacity(num_successes);
    while pos.len() < num_successes {
        // Read before the pushes: sizing the batch off a length that the loop
        // is changing would draw randomness the rejection loop never would.
        let deficit = num_successes - pos.len();
        for _ in 0..deficit {
            pos.push(stream.next_below(nshots as u64) as u32);
        }
        pos.sort_unstable();
        pos.dedup();
    }
    pos
}

/// Which generators fired in which shot, as a compressed row per shot.
///
/// `indices[offsets[s]..offsets[s + 1]]` lists the generators that fired in
/// shot `s`. The draw is per generator and the transpose to per shot is a
/// counting pass, so no shot is ever written to concurrently.
pub struct ShotSample {
    pub nshots: usize,
    pub offsets: Vec<u32>,
    pub indices: Vec<u32>,
    /// Shots in which at least one generator fired. The rest contribute an
    /// identity error, which the caller accounts for without scoring.
    pub nonempty: usize,
}

/// Draws the sample. With `seed = Some(_)` the result is fully determined by
/// the seed, including which slot of a shot each generator lands in, since the
/// transpose below runs in generator order.
///
/// `seed = None` draws one random base seed for the whole call, from an
/// OS-seeded generator, and keys every generator's stream on it. The unseeded
/// sample is therefore different on every call but still independent of how
/// rayon splits the work, and there is one draw path to keep bit-exact.
pub fn sample_shots(generators: &[NoiseGenerator], nshots: usize, seed: Option<u64>) -> ShotSample {
    assert!(
        nshots <= u32::MAX as usize,
        "shot positions are stored as u32"
    );
    // One table per distinct lambda, shared by every generator that has it.
    // `lambda = nshots * proba_from_rate(rate)` carries no candidate-dependent
    // term, so for a uniform noise model this is two tables of a few hundred
    // entries however many generators there are.
    let rates: Vec<f64> = generators.iter().map(|(_, rate)| *rate).collect();
    let tables = PoissonTables::build(&rates, nshots);

    let base = seed.unwrap_or_else(|| rand::rng().random::<u64>());

    let fired: Vec<Vec<u32>> = generators
        .par_iter()
        .enumerate()
        .map(|(index, _)| {
            let mut stream = GeneratorStream::new(base, index);
            sample_shot_positions(&mut stream, tables.for_generator(index), nshots)
        })
        .collect();

    let mut offsets = vec![0u32; nshots + 1];
    for shots in fired.iter() {
        for &shot in shots.iter() {
            offsets[shot as usize + 1] += 1;
        }
    }
    let mut nonempty = 0;
    for s in 0..nshots {
        if offsets[s + 1] > 0 {
            nonempty += 1;
        }
        offsets[s + 1] += offsets[s];
    }

    let mut indices = vec![0u32; offsets[nshots] as usize];
    let mut cursor = offsets.clone();
    for (generator, shots) in fired.iter().enumerate() {
        for &shot in shots.iter() {
            indices[cursor[shot as usize] as usize] = generator as u32;
            cursor[shot as usize] += 1;
        }
    }

    ShotSample {
        nshots,
        offsets,
        indices,
        nonempty,
    }
}

/// Per-generator bit planes and cumulant parity signatures.
///
/// One generator's words are contiguous, so scoring a shot walks a handful of
/// contiguous rows: `[X plane | Z plane | post-selected signature | logical
/// signature]`, of widths `[wire_words, wire_words, post_words, log_words]`.
/// The planes are indexed over only the wires some generator touches, since no
/// other wire can appear in an accumulated error.
pub struct LerTables {
    pub wire_words: usize,
    pub post_words: usize,
    pub log_words: usize,
    pub words: Vec<u64>,
}

impl LerTables {
    #[inline]
    pub fn stride(&self) -> usize {
        2 * self.wire_words + self.post_words + self.log_words
    }

    pub fn pack(
        generators: &[NoiseGenerator],
        post: &CumulantTable,
        logical: &CumulantTable,
        nqbits: usize,
    ) -> Self {
        let mut active: Vec<usize> = generators
            .iter()
            .flat_map(|(pauli, _)| pauli.paulis.keys().map(|w| w.dense_id(nqbits)))
            .collect();
        active.sort_unstable();
        active.dedup();
        let slot: HashMap<usize, usize> =
            active.iter().enumerate().map(|(i, id)| (*id, i)).collect();

        let wire_words = active.len().div_ceil(WORD);
        // A table with no rows answers "covered" with `false`, and zero words
        // is how that falls out here: the accumulator is empty, so nothing can
        // be non-zero.
        let post_words = post.nwords();
        let log_words = logical.nwords();
        let stride = 2 * wire_words + post_words + log_words;

        let mut words = vec![0u64; generators.len() * stride];
        if stride > 0 {
            words
                .par_chunks_mut(stride)
                .zip(generators.par_iter())
                .for_each(|(row, (pauli, _))| {
                    let (planes, rest) = row.split_at_mut(2 * wire_words);
                    let (x_plane, z_plane) = planes.split_at_mut(wire_words);
                    let (post_sig, log_sig) = rest.split_at_mut(post_words);
                    for (wire, p) in pauli.paulis.iter() {
                        let (xe, ze) = pauli_xz_bits(*p);
                        let bit = slot[&wire.dense_id(nqbits)];
                        if xe {
                            x_plane[bit / WORD] |= 1u64 << (bit % WORD);
                        }
                        if ze {
                            z_plane[bit / WORD] |= 1u64 << (bit % WORD);
                        }
                        for (k, word) in post_sig.iter_mut().enumerate() {
                            *word ^= post.covered_parity_word(wire, xe, ze, k);
                        }
                        for (k, word) in log_sig.iter_mut().enumerate() {
                            *word ^= logical.covered_parity_word(wire, xe, ze, k);
                        }
                    }
                });
        }

        Self {
            wire_words,
            post_words,
            log_words,
            words,
        }
    }
}

/// What the scorer counts: shots that survive post-selection, and how many of
/// those carry a logical error. Integers, so the reduction is order-free and
/// exact however the partials are combined.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ShotCounts {
    pub accepted: u64,
    pub logical: u64,
}

impl ShotCounts {
    fn merge(self, other: Self) -> Self {
        Self {
            accepted: self.accepted + other.accepted,
            logical: self.logical + other.logical,
        }
    }
}

/// Scores every non-empty shot.
///
/// Parallel over shots rather than over per-thread buckets of shots, so the
/// work splits evenly without a redistribution pass, and the XOR scratch is
/// reused for every shot a thread handles instead of allocated per shot.
pub fn score_shots(tables: &LerTables, sample: &ShotSample) -> ShotCounts {
    let stride = tables.stride();
    if stride == 0 {
        return ShotCounts::default();
    }
    let planes_end = 2 * tables.wire_words;
    let post_end = planes_end + tables.post_words;

    (0..sample.nshots)
        .into_par_iter()
        .fold(
            || (vec![0u64; stride], ShotCounts::default()),
            |(mut acc, counts), shot| {
                let lo = sample.offsets[shot] as usize;
                let hi = sample.offsets[shot + 1] as usize;
                if lo == hi {
                    return (acc, counts);
                }
                acc.iter_mut().for_each(|w| *w = 0);
                for generator in &sample.indices[lo..hi] {
                    let base = *generator as usize * stride;
                    for (a, w) in acc.iter_mut().zip(&tables.words[base..base + stride]) {
                        *a ^= *w;
                    }
                }
                // An identity error is not an error, and one the checks detect
                // is thrown away by post-selection. Note that a shot whose
                // generators cancel is dropped rather than counted as accepted,
                // which is what the hash-map scorer did.
                let identity = acc[..planes_end].iter().all(|w| *w == 0);
                let detected = acc[planes_end..post_end].iter().any(|w| *w != 0);
                let logical = acc[post_end..].iter().any(|w| *w != 0);
                let counts = if identity || detected {
                    counts
                } else {
                    ShotCounts {
                        accepted: counts.accepted + 1,
                        logical: counts.logical + u64::from(logical),
                    }
                };
                (acc, counts)
            },
        )
        .map(|(_, counts)| counts)
        .reduce(ShotCounts::default, ShotCounts::merge)
}

#[cfg(test)]
mod tests {
    use super::*;
    // `sample_shots` gets lambda from `PoissonTables::build`; the tests
    // derive it by hand.
    use crate::coverage::proba_from_rate;

    /// Mirrors the per-generator draw performed by `sample_shots`.
    fn draw(seed: u64, index: usize, lambda: f64, nshots: usize) -> Vec<u32> {
        let table = PoissonTable::new(lambda);
        sample_shot_positions(&mut GeneratorStream::new(seed, index), &table, nshots)
    }

    /// The one-at-a-time `BTreeSet` draw-and-reject loop, kept as the
    /// reference the batched version has to reproduce.
    fn draw_reference(
        stream: &mut GeneratorStream,
        table: &PoissonTable,
        nshots: usize,
    ) -> Vec<u32> {
        let num_successes = (table.sample(stream.next_f64()) as usize).min(nshots);
        let mut pos = std::collections::BTreeSet::new();
        while pos.len() < num_successes {
            pos.insert(stream.next_below(nshots as u64) as u32);
        }
        pos.into_iter().collect()
    }

    #[test]
    fn batched_draw_matches_the_rejection_loop_exactly() {
        // The equivalence is a property of the value sequence, not of the
        // generator producing it: a chunk sized to the exact deficit adds at
        // most one distinct value per draw, so it can only reach the target on
        // its final element and can never draw past the point one-at-a-time
        // rejection would stop.
        //
        // Both halves of the assertion matter. Equal samples alone would not
        // catch a version that drew one value too many: the set would agree
        // and every later draw in the process would be shifted, so the stream
        // position is compared after the draw as well. The dense cases are the
        // ones that exercise it -- at lambda 300 of 512 shots most draws
        // collide and the batch loop goes round many times, where at lambda 1
        // it never repeats.
        for (lambda, nshots) in [(1.0, 64), (12.0, 4096), (200.0, 512), (300.0, 512)] {
            let table = PoissonTable::new(lambda);
            for seed in 0..16u64 {
                let mut batched = GeneratorStream::new(seed, 0);
                let mut reference = GeneratorStream::new(seed, 0);
                let got = sample_shot_positions(&mut batched, &table, nshots);
                let want = draw_reference(&mut reference, &table, nshots);
                assert_eq!(
                    got, want,
                    "sample differs at lambda={lambda} nshots={nshots} seed={seed}"
                );
                assert_eq!(
                    batched.next_u64(),
                    reference.next_u64(),
                    "same sample but different randomness consumed at \
                     lambda={lambda} nshots={nshots} seed={seed}"
                );
            }
        }
    }

    /// The `rand_distr`/`StdRng` draw, kept as a reference for the
    /// distribution checks below: `rand_distr`'s Poisson over a seeded
    /// `StdRng`, with the same deficit-batched position loop.
    fn draw_old_algorithm<R: rand::Rng>(rng: &mut R, lambda: f64, nshots: usize) -> Vec<u32> {
        use rand::distr::Distribution;
        let num_successes = rand_distr::Poisson::new(lambda).unwrap().sample(rng) as usize;
        let mut pos: Vec<u32> = Vec::with_capacity(num_successes);
        while pos.len() < num_successes {
            let deficit = num_successes - pos.len();
            for _ in 0..deficit {
                pos.push(rng.random_range(0..nshots) as u32);
            }
            pos.sort_unstable();
            pos.dedup();
        }
        pos
    }

    /// Checks the Philox-driven draw against the `rand_distr`/`StdRng` one.
    ///
    /// Philox and a tabulated CDF produce a different stream from `StdRng` and
    /// `rand_distr`, so the two draw different samples. What they must share
    /// is the distribution being sampled, and that is not something
    /// the component tests establish on their own: the table is verified
    /// against scipy and the generator against numpy, but the wiring between
    /// them could still be biased. So this compares the two algorithms head to
    /// head on the quantities the estimate actually depends on.
    ///
    /// Detectable effect, stated so that passing means something. With 50k
    /// draws at lambda 12 the standard error on each mean is sqrt(12/50000),
    /// about 0.0155, so a relative bias in the fired-count above roughly 0.7%
    /// fails. The position histogram expects about 146 per shot, which catches
    /// a mis-scaled range but says nothing about the multiply-shift's bias of
    /// at most about 2^-40, or 1e-12 -- no feasible sample size would see
    /// that, and it is about nine orders of magnitude under the 1e-3 Monte
    /// Carlo error of a million-shot estimate.
    #[test]
    fn the_new_draw_samples_the_same_distribution_as_the_old_one() {
        use rand::SeedableRng;
        const N: usize = 50_000;
        let lambda = 12.0;
        let nshots = 4096usize;
        let table = PoissonTable::new(lambda);

        let mut old_rng = rand::rngs::StdRng::seed_from_u64(0x01D_5EED);
        let mut old_counts = 0.0f64;
        let mut old_sq = 0.0f64;
        let mut old_hist = vec![0u64; nshots];
        for _ in 0..N {
            let d = draw_old_algorithm(&mut old_rng, lambda, nshots);
            old_counts += d.len() as f64;
            old_sq += (d.len() * d.len()) as f64;
            for p in d {
                old_hist[p as usize] += 1;
            }
        }

        let mut new_counts = 0.0f64;
        let mut new_sq = 0.0f64;
        let mut new_hist = vec![0u64; nshots];
        for i in 0..N {
            let mut stream = GeneratorStream::new(0x5EED_0002, i);
            let d = sample_shot_positions(&mut stream, &table, nshots);
            new_counts += d.len() as f64;
            new_sq += (d.len() * d.len()) as f64;
            for p in d {
                new_hist[p as usize] += 1;
            }
        }

        let (old_mean, new_mean) = (old_counts / N as f64, new_counts / N as f64);
        let se = (lambda / N as f64).sqrt();
        assert!(
            (old_mean - new_mean).abs() < 4.0 * se * std::f64::consts::SQRT_2,
            "fired-count means disagree: old {old_mean:.4} vs new {new_mean:.4}, \
             more than 4 combined standard errors ({se:.4} each)"
        );
        // Variance too: a sampler can centre correctly and still get the
        // spread wrong, which would change the estimate's error bars without
        // moving its mean.
        let var = |sum: f64, sq: f64| sq / N as f64 - (sum / N as f64).powi(2);
        let (old_var, new_var) = (var(old_counts, old_sq), var(new_counts, new_sq));
        let var_se = (2.0 * lambda * lambda / N as f64).sqrt();
        assert!(
            (old_var - new_var).abs() < 4.0 * var_se * std::f64::consts::SQRT_2,
            "fired-count variances disagree: old {old_var:.4} vs new {new_var:.4}"
        );

        // Positions: both must be uniform over the shots. Compared against
        // uniform rather than against each other, since that is the property
        // either one could get wrong in the same direction.
        let chi2 = |hist: &[u64], total: f64| -> f64 {
            let expected = total / nshots as f64;
            assert!(expected > 20.0, "histogram bins too thin to test");
            hist.iter()
                .map(|&o| {
                    let d = o as f64 - expected;
                    d * d / expected
                })
                .sum()
        };
        let df = nshots as f64 - 1.0;
        let bound = df + 5.0 * (2.0 * df).sqrt();
        for (name, hist, total) in [
            ("old", &old_hist, old_counts),
            ("new", &new_hist, new_counts),
        ] {
            let x = chi2(hist, total);
            assert!(
                x < bound,
                "{name} position histogram is not uniform: chi-square {x:.1} \
                 over {df} degrees of freedom exceeds {bound:.1}"
            );
        }
    }

    /// The same comparison at realistic lambdas, with enough samples to
    /// resolve a bias far smaller than an end-to-end comparison of check
    /// searches could. A search also commits different checks once the stream
    /// changes, and different checks have genuinely different error rates;
    /// this test has no search in it, so it attributes any shift to the draw
    /// alone.
    ///
    /// `#[ignore]` because a million draws at each lambda takes tens of
    /// seconds even in release, which is too slow for every `cargo test`. Run
    /// with `cargo test --release -- --ignored`.
    #[test]
    #[ignore]
    fn the_new_draw_matches_the_old_one_at_realistic_lambdas() {
        use rand::SeedableRng;
        const N: usize = 1_000_000;
        // Gate noise 0.003 and readout 0.01 at 20k shots; see
        // proba_from_rate.
        for lambda in [59.82f64, 198.01] {
            let nshots = 20_000usize;
            let table = PoissonTable::new(lambda);

            let mut rng = rand::rngs::StdRng::seed_from_u64(0x0_1d);
            let mut old_sum = 0.0f64;
            for _ in 0..N {
                old_sum += draw_old_algorithm(&mut rng, lambda, nshots).len() as f64;
            }
            let mut new_sum = 0.0f64;
            for i in 0..N {
                let mut stream = GeneratorStream::new(0x5EED_0003, i);
                new_sum += sample_shot_positions(&mut stream, &table, nshots).len() as f64;
            }

            let (old_mean, new_mean) = (old_sum / N as f64, new_sum / N as f64);
            let se = (lambda / N as f64).sqrt();
            let combined = se * std::f64::consts::SQRT_2;
            let t = (new_mean - old_mean) / combined;
            let mde = 2.0 * combined / lambda;
            println!(
                "lambda={lambda}: old {old_mean:.4} new {new_mean:.4} \
                 t={t:+.2} detectable={:.4}%",
                mde * 100.0
            );
            assert!(
                t.abs() < 4.0,
                "lambda={lambda}: fired-count means disagree, old {old_mean:.4} \
                 vs new {new_mean:.4}, t={t:+.2}. A bias above {:.4}% would \
                 show, so this is the draw and not the search.",
                mde * 100.0
            );
        }
    }

    #[test]
    fn draw_is_sorted_and_distinct() {
        // The transpose writes each generator into a shot's slot in the order
        // it walks these, and the scorer's parity signatures are order-free
        // only because no generator appears twice in a shot.
        for index in 0..8 {
            let got = draw(0x5EED, index, 40.0, 1024);
            assert!(
                got.windows(2).all(|w| w[0] < w[1]),
                "not strictly ascending: {got:?}"
            );
        }
    }

    #[test]
    fn seeded_draw_repeats_exactly() {
        for index in 0..8 {
            assert_eq!(
                draw(0xABCD, index, 12.0, 4096),
                draw(0xABCD, index, 12.0, 4096),
                "generator {index} drew a different sample on a repeat call"
            );
        }
    }

    #[test]
    fn adjacent_indices_get_unrelated_streams() {
        // Adjacent indices differ in a single bit. Philox takes the index as a
        // counter word, and decorrelating nearby inputs is its whole contract,
        // so no extra mixing step is needed; this pins the property down.
        let samples: Vec<_> = (0..16).map(|i| draw(7, i, 20.0, 4096)).collect();
        for i in 0..samples.len() {
            for j in (i + 1)..samples.len() {
                assert_ne!(
                    samples[i], samples[j],
                    "generators {i} and {j} drew the same shots"
                );
            }
        }
    }

    #[test]
    fn seeded_draw_does_not_depend_on_thread_count() {
        // The point of keying each stream on the generator index: the sample is
        // a pure function of (seed, index), so rayon's scheduling cannot reach
        // it. This is the property that lets an LER change be checked against a
        // reference instead of against a noisy re-estimate.
        let lambdas: Vec<f64> = (0..64).map(|i| 1.0 + (i as f64) * 0.5).collect();
        let reference: Vec<Vec<u32>> = lambdas
            .iter()
            .enumerate()
            .map(|(i, l)| draw(99, i, *l, 8192))
            .collect();

        for threads in [1usize, 2, 7, 16] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            let got: Vec<Vec<u32>> = pool.install(|| {
                lambdas
                    .par_iter()
                    .enumerate()
                    .map(|(i, l)| draw(99, i, *l, 8192))
                    .collect()
            });
            assert_eq!(got, reference, "sample changed at {threads} threads");
        }
    }

    #[test]
    fn generators_in_a_shot_multiply_in_any_order() {
        // The packed scorer XORs generator rows in whatever order the shot's
        // row happens to list them, and the hash-map scorer it has to match
        // multiplies them in yet another order. Both are only well defined
        // because the phase-free Pauli product commutes.
        use crate::sparse_pauli::SparsePauli;
        use crate::wire::Wire;

        let g = |wires: &[(usize, u8)]| {
            let mut p = SparsePauli::new();
            for (w, pa) in wires {
                p.update(Wire::Input(*w), *pa);
            }
            p
        };
        let gens = [
            g(&[(0, 1), (3, 2)]),
            g(&[(1, 3), (3, 2)]),
            g(&[(0, 2), (2, 1)]),
            g(&[(2, 1), (3, 1)]),
        ];

        let mut forward = SparsePauli::new();
        for x in gens.iter() {
            forward.mult_inplace(x);
        }
        let mut backward = SparsePauli::new();
        for x in gens.iter().rev() {
            backward.mult_inplace(x);
        }
        assert_eq!(forward.paulis, backward.paulis);
    }

    /// The packed scorer has to agree with the hash-map scorer, kept as a
    /// reference, *exactly*, not closely: both count integers over the same seeded sample,
    /// so any difference is a bug in the GF(2) formulation rather than Monte
    /// Carlo noise. This is the check that licenses the packed form.
    #[test]
    fn packed_scorer_matches_the_hash_map_scorer() {
        use crate::coverage::Coverage;
        use crate::noise_model::{Readout, UNoiseModel, UniformDepolarizing};
        use rustiq_core::structures::{CliffordCircuit, CliffordGate};

        let models = vec![
            UNoiseModel::UniformDepolarizing(UniformDepolarizing::new(0.01)),
            UNoiseModel::Readout(Readout::new(0.02)),
        ];
        for nqbits in [4usize, 7, 12] {
            let mut circuit = CliffordCircuit::new(nqbits);
            for q in 0..nqbits - 1 {
                circuit.gates.push(CliffordGate::CZ(q, q + 1));
                circuit.gates.push(CliffordGate::H(q));
            }
            let measured: Vec<usize> = (0..nqbits).collect();
            let mut coverage = Coverage::new(&circuit, &models);
            coverage.set_check_cumulants(&[0], &[vec![]]);
            coverage.set_logical_cumulants(&[], &measured);

            for seed in [1u64, 12345] {
                let packed = coverage.approximate_psr_ler(4096, Some(seed));
                let reference = coverage.approximate_psr_ler_reference(4096, Some(seed));
                assert_eq!(
                    packed.0.to_bits(),
                    reference.0.to_bits(),
                    "acceptance differs at {nqbits} qubits, seed {seed}: {} vs {}",
                    packed.0,
                    reference.0
                );
                assert_eq!(
                    packed.1.to_bits(),
                    reference.1.to_bits(),
                    "LER differs at {nqbits} qubits, seed {seed}: {} vs {}",
                    packed.1,
                    reference.1
                );
            }
        }
    }

    #[test]
    fn sample_transpose_keeps_every_draw() {
        // The CSR transpose is the one place a generator could go missing.
        let rates: Vec<f64> = (0..48).map(|i| 0.001 + (i as f64) * 0.0005).collect();
        let generators: Vec<NoiseGenerator> = rates
            .iter()
            .map(|r| (crate::sparse_pauli::SparsePauli::new(), *r))
            .collect();
        let nshots = 2048;
        let sample = sample_shots(&generators, nshots, Some(4242));

        let mut expected = 0usize;
        let mut expected_nonempty = 0usize;
        let mut per_shot: Vec<Vec<u32>> = vec![Vec::new(); nshots];
        for (index, (_, rate)) in generators.iter().enumerate() {
            let lambda = nshots as f64 * proba_from_rate(*rate);
            let shots = draw(4242, index, lambda, nshots);
            expected += shots.len();
            for s in shots {
                per_shot[s as usize].push(index as u32);
            }
        }
        for row in per_shot.iter() {
            if !row.is_empty() {
                expected_nonempty += 1;
            }
        }

        assert_eq!(sample.indices.len(), expected, "lost or duplicated a draw");
        assert_eq!(sample.nonempty, expected_nonempty);
        assert_eq!(sample.offsets.len(), nshots + 1);
        for (shot, row) in per_shot.iter().enumerate() {
            let lo = sample.offsets[shot] as usize;
            let hi = sample.offsets[shot + 1] as usize;
            assert_eq!(&sample.indices[lo..hi], row.as_slice());
        }
    }
}
