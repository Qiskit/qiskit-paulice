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

//! Poisson counts by table lookup, so the draw calls no transcendental.
//!
//! O(1) Poisson samplers for large lambda, such as Hoermann's PTRS, are
//! rejection schemes whose accept test needs `log` of a random uniform and a
//! log-gamma of a random count. `rand_distr` and numpy use samplers of that
//! kind, and neither specifies its output bit for bit. Those arguments are not
//! known ahead of time, so they cannot be precomputed, and two libm
//! implementations need not agree on the last ulp of either. A seeded sample
//! drawn that way is statistically correct everywhere and reproducible only on
//! the platform it was drawn on.
//!
//! Inverting a tabulated CDF instead moves every hard function into building
//! the table and leaves the draw as a binary search over `f64` comparisons. It
//! is viable here because of two facts about how rates enter:
//!
//!   - `UniformDepolarizing` and `Readout` each compute one rate and emit it
//!     for every generator they produce, so a typical model has a handful of
//!     distinct lambdas rather than one per generator. They are deduplicated
//!     below.
//!   - `lambda = nshots * proba_from_rate(rate)` holds no candidate-dependent
//!     term, so it is fixed for an entire search.
//!
//! A table spans about 20 standard deviations, so it is O(sqrt(lambda)) and
//! tiny: 797 entries at lambda 1500, under 7 KB.

/// The inverse CDF of `Poisson(lambda)` over a truncated window.
#[derive(Clone, Debug)]
pub struct PoissonTable {
    /// The count that `cdf[0]` corresponds to.
    k_lo: u32,
    /// `cdf[i] = P(X <= k_lo + i)`, renormalised over the window so the last
    /// entry is exactly 1.0. That is not cosmetic: it means a uniform drawn
    /// from `[0, 1)` always lands inside the table, so the lookup has no
    /// out-of-range branch to get wrong and no clamp that would silently pile
    /// tail mass onto the endpoint.
    cdf: Vec<f64>,
}

impl PoissonTable {
    /// How many standard deviations the window covers on each side.
    ///
    /// Ten puts the discarded mass below 1e-20 -- for reference, the Monte
    /// Carlo error on an estimate from a million shots is around 1e-3, some
    /// seventeen orders of magnitude larger. Renormalising over the window makes the
    /// sampled distribution differ from a true Poisson by that much in total
    /// variation, which is the one deliberate approximation here. It is a
    /// modelling choice and not a reproducibility one: every draw reads the
    /// same table.
    const SIGMAS: f64 = 10.0;

    pub fn new(lambda: f64) -> Self {
        assert!(
            lambda.is_finite() && lambda >= 0.0,
            "lambda must be finite and non-negative, got {lambda}"
        );
        // A zero rate fires in no shots. Worth special-casing because the
        // recurrence below divides by lambda.
        if lambda == 0.0 {
            return Self {
                k_lo: 0,
                cdf: vec![1.0],
            };
        }

        let sd = lambda.sqrt();
        // The `+ 10` keeps the window sensible when sqrt(lambda) is small: at
        // lambda 0.5 ten sigmas is only 7 counts, but the tail there is fat
        // relative to the mean.
        let k_lo = (lambda - Self::SIGMAS * sd - 10.0).floor().max(0.0) as u32;
        let k_hi = (lambda + Self::SIGMAS * sd + 10.0).ceil() as u32;
        let len = (k_hi - k_lo + 1) as usize;

        // Build the unnormalised pmf outward from the mode by recurrence:
        //
        //     p(k+1) = p(k) * lambda / (k+1)      going up
        //     p(k-1) = p(k) * k / lambda          going down
        //
        // rather than from `exp(-lambda) * lambda^k / k!` directly, which
        // underflows hard -- exp(-1500) is zero in f64 -- and rather than in
        // log space, which would need lgamma. Both ratios are below one moving
        // away from the mode, so every value is at most the 1.0 we start from
        // and the smallest is around 1e-23. Nothing overflows, nothing
        // underflows, and the whole construction is multiplication and
        // division, so there is no transcendental even here.
        let mode = lambda.floor().max(k_lo as f64) as u32;
        let mode_at = (mode - k_lo) as usize;
        let mut pmf = vec![0.0f64; len];
        pmf[mode_at] = 1.0;
        for i in (mode_at + 1)..len {
            let k = k_lo as f64 + i as f64;
            pmf[i] = pmf[i - 1] * lambda / k;
        }
        for i in (0..mode_at).rev() {
            let k = k_lo as f64 + (i + 1) as f64;
            pmf[i] = pmf[i + 1] * k / lambda;
        }

        // Normalise via the cumulative sum, so the CDF is monotone by
        // construction rather than by hoping the divisions cooperate.
        let total: f64 = pmf.iter().sum();
        let mut cdf = Vec::with_capacity(len);
        let mut running = 0.0f64;
        for p in &pmf {
            running += p;
            cdf.push(running / total);
        }
        // Round-off can leave the last entry a few ulps under one, which would
        // put a uniform in that sliver outside the table.
        *cdf.last_mut().unwrap() = 1.0;

        Self { k_lo, cdf }
    }

    /// The count for a uniform `u` in `[0, 1)`.
    ///
    /// Comparisons only, which is the entire point: nothing here can round
    /// differently from one platform to another.
    #[inline]
    pub fn sample(&self, u: f64) -> u32 {
        debug_assert!((0.0..1.0).contains(&u), "u must be in [0,1), got {u}");
        // First index whose CDF exceeds u. The last entry is exactly 1.0 and
        // u is strictly below 1.0, so this is always a valid index.
        self.k_lo + self.cdf.partition_point(|&c| c <= u) as u32
    }

    /// Smallest and largest count the table can return, for tests.
    pub fn window(&self) -> (u32, u32) {
        (self.k_lo, self.k_lo + self.cdf.len() as u32 - 1)
    }

    pub fn len(&self) -> usize {
        self.cdf.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cdf.is_empty()
    }
}

/// One table per distinct lambda, plus the per-generator index into them.
///
/// Deduplicated on the bit pattern rather than with a tolerance: two rates
/// that differ in the last ulp are different lambdas and get different tables,
/// which keeps this from quietly changing anyone's distribution to save a few
/// kilobytes.
pub struct PoissonTables {
    tables: Vec<PoissonTable>,
    /// `table_of[g]` indexes `tables` for generator `g`.
    table_of: Vec<u32>,
}

impl PoissonTables {
    pub fn build(rates: &[f64], nshots: usize) -> Self {
        let mut tables: Vec<PoissonTable> = Vec::new();
        let mut seen: std::collections::HashMap<u64, u32> = std::collections::HashMap::new();
        let mut table_of = Vec::with_capacity(rates.len());
        for rate in rates {
            let lambda = nshots as f64 * crate::coverage::proba_from_rate(*rate);
            let slot = *seen.entry(lambda.to_bits()).or_insert_with(|| {
                tables.push(PoissonTable::new(lambda));
                (tables.len() - 1) as u32
            });
            table_of.push(slot);
        }
        Self { tables, table_of }
    }

    #[inline]
    pub fn for_generator(&self, index: usize) -> &PoissonTable {
        &self.tables[self.table_of[index] as usize]
    }

    /// Number of distinct lambdas, i.e. tables built.
    pub fn num_tables(&self) -> usize {
        self.tables.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::philox::GeneratorStream;

    /// Reference values from `scipy.stats.poisson.cdf`, an independent
    /// implementation. The recurrence above is a different algorithm from
    /// scipy's, so agreeing with it to twelve digits says the construction is
    /// right rather than merely self-consistent -- the same reason the Philox
    /// vectors come from numpy.
    ///
    /// ```python
    /// from scipy.stats import poisson
    /// poisson.cdf(1500, 1500.0)
    /// ```
    ///
    /// The literals are scipy's output transcribed verbatim, one digit past
    /// what an f64 can hold. Rounding them to fit would make the file no
    /// longer show what the oracle actually printed, and the comparison below
    /// is against a 1e-12 tolerance, so the last digit is decoration either
    /// way.
    #[test]
    #[allow(clippy::excessive_precision)]
    fn matches_scipy_cdf() {
        let cases: &[(f64, &[(u32, f64)])] = &[
            (
                0.5,
                &[
                    (0, 6.06530659712633424e-01),
                    (1, 9.09795989568950136e-01),
                    (2, 9.85612322033029287e-01),
                ],
            ),
            (
                12.0,
                &[
                    (5, 2.03410294169283981e-02),
                    (12, 5.75965248573064525e-01),
                    (15, 8.44415652450183241e-01),
                    (22, 9.96952628855827694e-01),
                ],
            ),
            (
                1500.0,
                &[
                    (1422, 2.19883461471864043e-02),
                    (1500, 5.06866511984763624e-01),
                    (1538, 8.39923522811066747e-01),
                    (1616, 9.98532850103907577e-01),
                ],
            ),
        ];
        for (lambda, points) in cases {
            let table = PoissonTable::new(*lambda);
            let (k_lo, _) = table.window();
            for (k, want) in *points {
                let got = table.cdf[(k - k_lo) as usize];
                let rel = (got - want).abs() / want;
                assert!(
                    rel < 1e-12,
                    "lambda={lambda} k={k}: table cdf {got:.17e} vs scipy \
                     {want:.17e}, relative error {rel:.3e}"
                );
            }
        }
    }

    #[test]
    fn cdf_is_monotone_and_ends_at_exactly_one() {
        for lambda in [0.0, 0.5, 1.0, 12.0, 300.0, 1500.0, 30_000.0] {
            let table = PoissonTable::new(lambda);
            for i in 1..table.len() {
                assert!(
                    table.cdf[i] >= table.cdf[i - 1],
                    "cdf decreased at lambda={lambda} index {i}"
                );
            }
            // Exact, not approximate: `sample` relies on it to guarantee a
            // uniform below 1.0 always lands in the table.
            assert_eq!(
                *table.cdf.last().unwrap(),
                1.0,
                "cdf must end at exactly 1.0 for lambda={lambda}"
            );
        }
    }

    #[test]
    fn a_uniform_just_under_one_still_lands_in_the_table() {
        // The boundary the exact 1.0 above protects. `next_f64` can return
        // 1.0 - 2^-53, and a clamp or an off-by-one here would show up as a
        // rare wrong count, which is the kind of bug a mean test never sees.
        for lambda in [0.5, 12.0, 1500.0] {
            let table = PoissonTable::new(lambda);
            let (lo, hi) = table.window();
            let biggest = f64::from_bits(1.0f64.to_bits() - 1);
            assert!(biggest < 1.0);
            let k = table.sample(biggest);
            assert!(
                (lo..=hi).contains(&k),
                "lambda={lambda}: u just under 1 gave {k}, outside [{lo},{hi}]"
            );
            assert_eq!(table.sample(0.0), lo, "u=0 must give the low end");
        }
    }

    #[test]
    fn zero_lambda_is_a_point_mass_at_zero() {
        let table = PoissonTable::new(0.0);
        assert_eq!(table.window(), (0, 0));
        for u in [0.0, 0.5, 0.999_999] {
            assert_eq!(table.sample(u), 0);
        }
    }

    /// Statistical check of the *sampler*, as distinct from the table: the
    /// table is pinned against scipy above, so what is left to verify is that
    /// driving it with Philox uniforms reproduces the distribution.
    #[test]
    fn sampled_mean_and_variance_match_lambda() {
        // Poisson has variance equal to its mean, so a sampler that got the
        // spread wrong while keeping the centre right -- the failure a
        // mean-only test misses -- is caught here too.
        const N: usize = 200_000;
        for lambda in [0.5f64, 12.0, 1500.0] {
            let table = PoissonTable::new(lambda);
            let mut stream = GeneratorStream::new(0x51EED, 0);
            let mut sum = 0.0f64;
            let mut sumsq = 0.0f64;
            for _ in 0..N {
                let k = table.sample(stream.next_f64()) as f64;
                sum += k;
                sumsq += k * k;
            }
            let mean = sum / N as f64;
            let var = sumsq / N as f64 - mean * mean;
            // Standard error of the mean is sqrt(lambda/N); allow five of
            // them, so this fails on bias but not on honest noise.
            let se = (lambda / N as f64).sqrt();
            assert!(
                (mean - lambda).abs() < 5.0 * se.max(1e-3),
                "lambda={lambda}: mean {mean:.4} off by more than 5 se ({se:.4})"
            );
            // The variance of a sample variance is about 2*lambda^2/N here.
            let var_se = (2.0 * lambda * lambda / N as f64).sqrt();
            assert!(
                (var - lambda).abs() < 5.0 * var_se.max(1e-2),
                "lambda={lambda}: variance {var:.4} should be near lambda"
            );
        }
    }

    /// Chi-square goodness of fit against the table's own pmf.
    ///
    /// Deterministic by construction -- a fixed Philox seed -- so this is a
    /// regression test rather than a flaky one.
    #[test]
    fn sampled_counts_pass_a_chi_square_fit() {
        const N: usize = 500_000;
        let lambda = 12.0;
        let table = PoissonTable::new(lambda);
        let (lo, hi) = table.window();
        let mut observed = vec![0u64; (hi - lo + 1) as usize];
        let mut stream = GeneratorStream::new(0xC415, 0);
        for _ in 0..N {
            observed[(table.sample(stream.next_f64()) - lo) as usize] += 1;
        }
        // Pool bins with a small expectation, or chi-square is not valid there.
        let mut chi2 = 0.0f64;
        let mut bins = 0usize;
        let mut pooled_obs = 0u64;
        let mut pooled_exp = 0.0f64;
        let mut prev = 0.0f64;
        for (&obs, &cum) in observed.iter().zip(table.cdf.iter()) {
            let exp = (cum - prev) * N as f64;
            prev = cum;
            if exp < 5.0 {
                pooled_obs += obs;
                pooled_exp += exp;
                continue;
            }
            let d = obs as f64 - exp;
            chi2 += d * d / exp;
            bins += 1;
        }
        if pooled_exp > 5.0 {
            let d = pooled_obs as f64 - pooled_exp;
            chi2 += d * d / pooled_exp;
            bins += 1;
        }
        // Roughly `bins` degrees of freedom; the 0.999 quantile of chi-square
        // sits near df + 3.5*sqrt(2*df), so this is a loose bound that still
        // rejects a mis-scaled or shifted sampler decisively.
        let df = bins as f64;
        let bound = df + 3.5 * (2.0 * df).sqrt();
        assert!(
            chi2 < bound,
            "chi-square {chi2:.2} over {bins} bins exceeds {bound:.2}; the \
             Philox-driven sampler does not reproduce the tabulated pmf"
        );
    }

    #[test]
    fn tables_are_deduplicated_by_lambda() {
        // The property that keeps the tables small: a uniform noise model
        // emits one rate for every generator it produces.
        let rates = vec![0.003; 1000];
        let tables = PoissonTables::build(&rates, 1_000_000);
        assert_eq!(tables.num_tables(), 1, "one rate should give one table");

        let mixed: Vec<f64> = (0..1000)
            .map(|i| if i % 2 == 0 { 0.003 } else { 0.01 })
            .collect();
        let tables = PoissonTables::build(&mixed, 1_000_000);
        assert_eq!(tables.num_tables(), 2, "two rates should give two tables");
        // And every generator must still reach the table for its own rate.
        let lambda_of = |rate: f64| 1e6 * crate::coverage::proba_from_rate(rate);
        for (i, rate) in mixed.iter().enumerate() {
            let (lo, hi) = tables.for_generator(i).window();
            let lambda = lambda_of(*rate);
            assert!(
                (lo as f64) <= lambda && lambda <= (hi as f64),
                "generator {i} got a table whose window [{lo},{hi}] excludes \
                 its own lambda {lambda}"
            );
        }
    }

    #[test]
    #[allow(clippy::excessive_precision)]
    fn a_shifted_table_fails_the_scipy_check() {
        // Mutation check on the oracle itself: if `matches_scipy_cdf` would
        // pass against a table built for the wrong lambda, it is not testing
        // anything. One percent off is far smaller than any plausible bug.
        let table = PoissonTable::new(1515.0);
        let (k_lo, _) = table.window();
        let got = table.cdf[(1500 - k_lo) as usize];
        let want = 5.06866511984763624e-01;
        assert!(
            (got - want).abs() / want > 1e-12,
            "a table built for lambda=1515 agreed with scipy at lambda=1500, \
             so the comparison is insensitive"
        );
    }
}
