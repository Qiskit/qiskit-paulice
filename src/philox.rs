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

//! Philox4x64-10: the counter-based generator the LER draw is keyed on.
//!
//! Counter-based rather than stateful, and that is the whole point. A stream
//! cipher like `StdRng` carries state, so reproducing a particular generator's
//! stream means replaying it from the start. Here the n'th output of generator
//! `g` is a pure function of `(seed, g, n)`, so each generator's stream can be
//! drawn on any thread, in any order, and still come out the same.
//!
//! Every operation below is a 64-bit integer multiply, add or xor. There is no
//! floating point in the generator at all, so there is no libm to disagree
//! about and nothing for a compiler to contract into an fma: the same three
//! numbers give the same words on every platform.
//!
//! Verified against numpy's `numpy.random.Philox`, an independent
//! implementation of the same generator, rather than against constants
//! transcribed from a paper -- see `matches_numpy_philox`.

/// The two round multipliers, from Salmon et al., "Parallel Random Numbers: As
/// Easy as 1, 2, 3" (SC11), as implemented by Random123 and numpy.
const M0: u64 = 0xD2E7_470E_E14C_6C93;
const M1: u64 = 0xCA5A_8263_9512_1157;

/// Per-round key bumps: the golden ratio minus one (phi - 1) and sqrt(3) - 1,
/// in 64-bit fixed point.
const W0: u64 = 0x9E37_79B9_7F4A_7C15;
const W1: u64 = 0xBB67_AE85_84CA_A73B;

/// Ten rounds is the standard parameterisation, and the one numpy uses. Seven
/// already passes the usual test batteries; the margin is cheap here because
/// one call yields four 64-bit words.
const ROUNDS: usize = 10;

#[inline(always)]
fn mulhilo(a: u64, b: u64) -> (u64, u64) {
    let product = (a as u128) * (b as u128);
    ((product >> 64) as u64, product as u64)
}

#[inline(always)]
fn round(ctr: [u64; 4], key: [u64; 2]) -> [u64; 4] {
    let (hi0, lo0) = mulhilo(M0, ctr[0]);
    let (hi1, lo1) = mulhilo(M1, ctr[2]);
    [hi1 ^ ctr[1] ^ key[0], lo1, hi0 ^ ctr[3] ^ key[1], lo0]
}

/// Four random 64-bit words from a counter and a key. No state, no allocation.
#[inline]
pub fn philox4x64_10(mut ctr: [u64; 4], mut key: [u64; 2]) -> [u64; 4] {
    ctr = round(ctr, key);
    for _ in 1..ROUNDS {
        key = [key[0].wrapping_add(W0), key[1].wrapping_add(W1)];
        ctr = round(ctr, key);
    }
    ctr
}

/// One error generator's random stream.
///
/// The counter is `[n, 0, index, 0]`: the generator's index sits in a different
/// counter word from the draw number, so generator `index` gets its own
/// subsequence and cannot run into its neighbour's however many draws it takes.
/// Adjacent indices need no extra mixing: a counter-based generator's whole
/// contract is that nearby inputs give unrelated outputs.
///
/// Keying on the index rather than on draw order is what makes the sample
/// independent of how rayon schedules the generators, which is the property
/// `seeded_draw_does_not_depend_on_thread_count` pins and the reason this is
/// worth preserving exactly.
pub struct GeneratorStream {
    key: [u64; 2],
    index: u64,
    /// Which block of four words comes next.
    block: u64,
    buf: [u64; 4],
    /// How many of `buf` are still unread, so a draw never wastes three words.
    live: usize,
}

impl GeneratorStream {
    pub fn new(seed: u64, index: usize) -> Self {
        Self {
            // The high key word is fixed: the seed is 64 bits, and spreading it
            // over both would make two different seeds share a key.
            key: [seed, 0],
            index: index as u64,
            block: 0,
            buf: [0; 4],
            live: 0,
        }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        if self.live == 0 {
            self.buf = philox4x64_10([self.block, 0, self.index, 0], self.key);
            self.block = self.block.wrapping_add(1);
            self.live = 4;
        }
        let word = self.buf[4 - self.live];
        self.live -= 1;
        word
    }

    /// A double in `[0, 1)` with 53 bits of mantissa.
    ///
    /// The shift takes the top 53 bits and the scale is a power of two, so the
    /// conversion is exact on any IEEE-754 machine and needs no rounding
    /// decision -- there is nothing here for two platforms to disagree about.
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// A uniform integer in `[0, bound)`.
    ///
    /// Multiply-shift (Lemire) rather than a modulo: integer-only, one word per
    /// draw, and no rejection loop, so a stream's counter advances by a fixed
    /// amount per position drawn. The bias is at most `bound / 2^64`, which for
    /// the shot counts here is under 2^-40 and is far below the Monte Carlo
    /// error of the estimate it feeds. It is also the same on every platform,
    /// so it costs nothing in reproducibility.
    #[inline]
    pub fn next_below(&mut self, bound: u64) -> u64 {
        (((self.next_u64() as u128) * (bound as u128)) >> 64) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Known answers from numpy 2.2.6's `numpy.random.Philox`, an independent
    /// implementation of the same generator. This is what makes the constants
    /// and the round structure above trustworthy: taking them from a paper and
    /// asserting self-consistency would only prove the code agrees with
    /// itself.
    ///
    /// Two conventions of numpy's matter when reproducing these vectors:
    ///
    ///   - numpy advances its counter *before* emitting a block, so its first
    ///     `random_raw(4)` is this function at counter `[1, 0, 0, 0]`, not
    ///     `[0, 0, 0, 0]`.
    ///   - numpy's `key=[a, b]` does not map `b` onto `key[1]` here. The
    ///     vectors below therefore all have a zero high key word, which is
    ///     the only shape `GeneratorStream` produces. Anything that starts
    ///     using `key[1]` needs its own vectors, not these.
    ///
    /// Reproduce with:
    /// ```python
    /// from numpy.random import Philox
    /// Philox(key=[0xdeadbeef, 0], counter=[0, 0, 0, 0]).random_raw(4)
    /// ```
    #[test]
    fn matches_numpy_philox() {
        // Zero key, first block.
        assert_eq!(
            philox4x64_10([1, 0, 0, 0], [0, 0]),
            [
                0x02f4_ba64_08e4_d89b,
                0x3dd6_2b0b_9ca8_c5b2,
                0x1c86_67a5_5d90_2e79,
                0x907d_7a05_2fd5_b4dc
            ],
            "disagrees with numpy at key=0, first block"
        );
        // Second block, so a counter that does not advance cannot pass.
        assert_eq!(
            philox4x64_10([2, 0, 0, 0], [0, 0]),
            [
                0x809b_f322_8839_87c3,
                0x4711_28b9_e807_f7dd,
                0xf250_ba0d_bec0_65b7,
                0xfc6e_d667_67a4_57bc
            ],
            "disagrees with numpy on the second block"
        );
        // Non-zero key, so a dropped key word cannot pass.
        assert_eq!(
            philox4x64_10([1, 0, 0, 0], [0xdead_beef, 0]),
            [
                0xa4e9_30dc_738a_caf1,
                0xb1c7_ecc6_484e_9cf0,
                0x6b88_a411_9092_98aa,
                0x66f3_c896_201f_7262
            ],
            "disagrees with numpy on a non-zero key"
        );
        // Non-zero third counter word: this is the shape `GeneratorStream`
        // uses to separate generators, so it is the one that matters most.
        assert_eq!(
            philox4x64_10([1, 0, 0x1000, 0], [0xabcd, 0]),
            [
                0x0ff8_a07a_7264_00d9,
                0xb7d7_4b83_2c73_8c15,
                0x5279_24ef_73c8_0a37,
                0x27e4_7bcc_cb48_7a40
            ],
            "disagrees with numpy when the generator index word is set"
        );
    }

    #[test]
    fn the_high_key_word_stays_unused() {
        // The vectors above only cover `key = [seed, 0]`, because that is all
        // this type builds. If that ever changes, `matches_numpy_philox` would
        // still pass while validating nothing about the new shape, so the
        // constraint is asserted here rather than left as a comment.
        for seed in [0u64, 1, 0xdead_beef, u64::MAX] {
            assert_eq!(
                GeneratorStream::new(seed, 7).key,
                [seed, 0],
                "GeneratorStream must key only the low word; new vectors are \
                 needed before using key[1]"
            );
        }
    }

    #[test]
    fn stream_is_a_pure_function_of_seed_and_index() {
        // The property reproducibility rests on: nothing carried, so the same
        // three numbers give the same word wherever they are evaluated.
        for index in [0usize, 1, 2, 63, 4096] {
            let mut a = GeneratorStream::new(0xABCD, index);
            let mut b = GeneratorStream::new(0xABCD, index);
            let first: Vec<u64> = (0..37).map(|_| a.next_u64()).collect();
            let again: Vec<u64> = (0..37).map(|_| b.next_u64()).collect();
            assert_eq!(first, again, "stream for index {index} did not repeat");
        }
    }

    #[test]
    fn adjacent_indices_and_seeds_give_unrelated_streams() {
        // Adjacent indices differ in one bit. Neighbouring indices, and seeds
        // one apart, must still not share a prefix.
        let words = |seed: u64, index: usize| -> Vec<u64> {
            let mut s = GeneratorStream::new(seed, index);
            (0..8).map(|_| s.next_u64()).collect()
        };
        for index in 0..16usize {
            assert_ne!(words(7, index), words(7, index + 1));
            assert_ne!(words(7, index), words(8, index));
        }
    }

    #[test]
    fn buffered_words_match_unbuffered_blocks() {
        // The stream hands out four words per Philox call, so an off-by-one in
        // the buffer would silently drop or repeat one. Compare against the
        // blocks computed directly.
        let mut stream = GeneratorStream::new(0x5EED, 11);
        let got: Vec<u64> = (0..12).map(|_| stream.next_u64()).collect();
        let mut want = Vec::new();
        for block in 0..3u64 {
            want.extend_from_slice(&philox4x64_10([block, 0, 11, 0], [0x5EED, 0]));
        }
        assert_eq!(got, want, "buffering reordered or dropped a word");
    }

    #[test]
    fn doubles_are_in_range_and_below_bound_holds() {
        let mut stream = GeneratorStream::new(3, 3);
        for _ in 0..100_000 {
            let x = stream.next_f64();
            assert!((0.0..1.0).contains(&x), "f64 out of [0,1): {x}");
            assert!(stream.next_below(1_000_000) < 1_000_000);
        }
        // A bound of 1 has exactly one legal answer, which catches a
        // multiply-shift that lost the high half.
        assert_eq!(stream.next_below(1), 0);
    }
}
