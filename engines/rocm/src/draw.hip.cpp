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

// The LER Monte Carlo draw, bit-identical to the host.
//
// A transcription of `sample_shot_positions` in src/ler.rs:
//
//   - Philox4x64-10 is 64-bit integer multiply, add and xor. `__umul64hi`
//     gives the same high word as Rust's u128 multiply.
//   - The Poisson count is a binary search over a CDF the host built, so the
//     device compares doubles it was given and never computes one.
//   - Positions are a multiply-shift reduction, integer-only.
//
// The one floating-point operation is scaling a 53-bit integer by 2^-53 in
// `next_f64`, which is exact.

#include <hip/hip_runtime.h>
#include <stdint.h>

#include "paulice_rocm.h"

namespace {

// Philox4x64-10, as in src/philox.rs.

__device__ constexpr uint64_t M0 = 0xD2E7470EE14C6C93ULL;
__device__ constexpr uint64_t M1 = 0xCA5A826395121157ULL;
__device__ constexpr uint64_t W0 = 0x9E3779B97F4A7C15ULL;
__device__ constexpr uint64_t W1 = 0xBB67AE8584CAA73BULL;
constexpr int ROUNDS = 10;

struct Ctr4 {
    uint64_t v[4];
};

__device__ __forceinline__ Ctr4 philox_round(Ctr4 c, uint64_t k0, uint64_t k1) {
    const uint64_t hi0 = __umul64hi(M0, c.v[0]);
    const uint64_t lo0 = M0 * c.v[0];
    const uint64_t hi1 = __umul64hi(M1, c.v[2]);
    const uint64_t lo1 = M1 * c.v[2];
    Ctr4 out;
    out.v[0] = hi1 ^ c.v[1] ^ k0;
    out.v[1] = lo1;
    out.v[2] = hi0 ^ c.v[3] ^ k1;
    out.v[3] = lo0;
    return out;
}

__device__ __forceinline__ Ctr4 philox4x64_10(Ctr4 c, uint64_t k0, uint64_t k1) {
    c = philox_round(c, k0, k1);
    for (int r = 1; r < ROUNDS; ++r) {
        k0 += W0;
        k1 += W1;
        c = philox_round(c, k0, k1);
    }
    return c;
}

/// `GeneratorStream` in src/philox.rs: counter `[block, 0, index, 0]`, key
/// `[seed, 0]`, and the same four-word buffering.
struct GeneratorStream {
    uint64_t key0;
    uint64_t index;
    uint64_t block;
    Ctr4 buf;
    int live;

    __device__ __forceinline__ void init(uint64_t seed, uint64_t gen_index) {
        key0 = seed;
        index = gen_index;
        block = 0;
        live = 0;
    }

    __device__ __forceinline__ uint64_t next_u64() {
        if (live == 0) {
            Ctr4 c;
            c.v[0] = block;
            c.v[1] = 0;
            c.v[2] = index;
            c.v[3] = 0;
            buf = philox4x64_10(c, key0, 0);
            block += 1;
            live = 4;
        }
        const uint64_t word = buf.v[4 - live];
        live -= 1;
        return word;
    }

    __device__ __forceinline__ double next_f64() {
        return static_cast<double>(next_u64() >> 11) * (1.0 / 9007199254740992.0);
    }

    __device__ __forceinline__ uint64_t next_below(uint64_t bound) {
        return __umul64hi(next_u64(), bound);
    }
};

/// One table of the concatenated CDF upload.
struct TableView {
    const double *cdf;
    uint32_t len;
    uint32_t k_lo;
};

/// `PoissonTable::sample`: the window's low count plus the number of entries
/// at or below `u`. The last entry is exactly 1.0 and `u < 1.0`, so the search
/// stays inside the table.
__device__ __forceinline__ uint32_t poisson_sample(const TableView &t, double u) {
    uint32_t lo = 0;
    uint32_t hi = t.len;
    while (lo < hi) {
        const uint32_t mid = lo + (hi - lo) / 2;
        if (t.cdf[mid] <= u) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return t.k_lo + lo;
}

/// One generator's distinct shot positions, in draw order. Returns
/// `PAULICE_LER_DRAW_OVERFLOW` without drawing if the count exceeds
/// `capacity`, since a truncated draw would consume a different number of
/// stream words than the host.
///
/// Batches are sized to the deficit read before the batch, as on the host, so
/// the device consumes exactly the host's stream words. Duplicates are dropped
/// with a bitset of one bit per shot instead of the host's sort; that keeps the
/// same set, and only the order within a generator differs, which the
/// transpose's per-shot sort removes. `bits` must be all zero on entry and is
/// left all zero.
__device__ uint32_t draw_positions(uint64_t seed, uint32_t generator, const double *cdf_all,
                                   const uint32_t *table_offset, const uint32_t *table_len,
                                   const uint32_t *table_klo, const uint32_t *table_of,
                                   uint32_t nshots, uint32_t *out, uint32_t capacity,
                                   uint64_t *bits, uint64_t *after) {
    const uint32_t t = table_of[generator];
    const TableView table{cdf_all + table_offset[t], table_len[t], table_klo[t]};
    GeneratorStream stream;
    stream.init(seed, generator);

    uint32_t want = poisson_sample(table, stream.next_f64());
    if (want > nshots) {
        want = nshots;
    }
    if (want > capacity) {
        return PAULICE_LER_DRAW_OVERFLOW;
    }
    uint32_t have = 0;
    while (have < want) {
        const uint32_t deficit = want - have;
        for (uint32_t i = 0; i < deficit; ++i) {
            const uint32_t p = static_cast<uint32_t>(stream.next_below(nshots));
            const uint64_t mask = 1ULL << (p & 63u);
            uint64_t &word = bits[p >> 6];
            if ((word & mask) == 0) {
                word |= mask;
                out[have++] = p;
            }
        }
    }
    for (uint32_t i = 0; i < have; ++i) {
        bits[out[i] >> 6] = 0;
    }
    if (after != nullptr) {
        *after = stream.next_u64();
    }
    return have;
}

/// One thread per generator, reporting the raw stream words, the draw and the
/// next stream word after it, so each can be checked against the host.
__global__ void ler_draw_parity_kernel(
    const uint64_t seed, const uint32_t n_generators, const uint32_t nshots,
    const double *__restrict__ cdf_all, const uint32_t *__restrict__ table_offset,
    const uint32_t *__restrict__ table_len, const uint32_t *__restrict__ table_klo,
    const uint32_t *__restrict__ table_of, const uint32_t probe_words,
    uint64_t *__restrict__ out_words, uint32_t *__restrict__ out_counts,
    uint32_t *__restrict__ out_pos, const uint32_t pos_stride, uint64_t *__restrict__ bitsets,
    const uint32_t bitset_words, uint64_t *__restrict__ out_after) {
    const uint32_t g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n_generators) {
        return;
    }
    GeneratorStream probe;
    probe.init(seed, g);
    for (uint32_t i = 0; i < probe_words; ++i) {
        out_words[static_cast<size_t>(g) * probe_words + i] = probe.next_u64();
    }
    out_counts[g] = draw_positions(seed, g, cdf_all, table_offset, table_len, table_klo, table_of,
                                   nshots, out_pos + static_cast<size_t>(g) * pos_stride,
                                   pos_stride, bitsets + static_cast<size_t>(g) * bitset_words,
                                   out_after + g);
}

/// A pool of `n_lanes` threads, each drawing generator after generator with
/// one bitset, so the bitset memory is bounded by the pool rather than by the
/// generator count. Every generator's draw depends only on its own stream, so
/// the pool size changes nothing in the sample.
__global__ void ler_draw_kernel(const uint64_t seed, const uint32_t n_generators,
                                const uint32_t nshots, const double *__restrict__ cdf_all,
                                const uint32_t *__restrict__ table_offset,
                                const uint32_t *__restrict__ table_len,
                                const uint32_t *__restrict__ table_klo,
                                const uint32_t *__restrict__ table_of,
                                uint32_t *__restrict__ out_counts, uint32_t *__restrict__ out_pos,
                                const uint32_t pos_stride, uint64_t *__restrict__ bitsets,
                                const uint32_t bitset_words, const uint32_t n_lanes,
                                unsigned long long *__restrict__ out_overflow) {
    const uint32_t lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= n_lanes) {
        return;
    }
    uint64_t *bits = bitsets + static_cast<size_t>(lane) * bitset_words;
    for (uint32_t g = lane; g < n_generators; g += n_lanes) {
        const uint32_t n =
            draw_positions(seed, g, cdf_all, table_offset, table_len, table_klo, table_of, nshots,
                           out_pos + static_cast<size_t>(g) * pos_stride, pos_stride, bits,
                           nullptr);
        if (n == PAULICE_LER_DRAW_OVERFLOW) {
            // The transpose still reads this count, so it must stay in range;
            // the host fails the whole dispatch on the flag.
            atomicAdd(out_overflow, 1ULL);
            out_counts[g] = 0;
        } else {
            out_counts[g] = n;
        }
    }
}

} // namespace

extern "C" hipError_t paulice_launch_ler_draw(uint64_t seed, uint32_t n_generators, uint32_t nshots,
                                              const double *cdf_all, const uint32_t *table_offset,
                                              const uint32_t *table_len, const uint32_t *table_klo,
                                              const uint32_t *table_of, uint32_t *out_counts,
                                              uint32_t *out_pos, uint32_t pos_stride,
                                              uint64_t *bitsets, uint32_t bitset_words,
                                              uint32_t n_lanes, unsigned long long *out_overflow,
                                              hipStream_t stream) {
    if (n_generators == 0 || n_lanes == 0) {
        return hipSuccess;
    }
    constexpr int BLOCK = 256;
    const int blocks = static_cast<int>((n_lanes + BLOCK - 1) / BLOCK);
    ler_draw_kernel<<<blocks, BLOCK, 0, stream>>>(seed, n_generators, nshots, cdf_all,
                                                  table_offset, table_len, table_klo, table_of,
                                                  out_counts, out_pos, pos_stride, bitsets,
                                                  bitset_words, n_lanes, out_overflow);
    return hipGetLastError();
}

extern "C" hipError_t paulice_launch_ler_draw_parity(
    uint64_t seed, uint32_t n_generators, uint32_t nshots, const double *cdf_all,
    const uint32_t *table_offset, const uint32_t *table_len, const uint32_t *table_klo,
    const uint32_t *table_of, uint32_t probe_words, uint64_t *out_words, uint32_t *out_counts,
    uint32_t *out_pos, uint32_t pos_stride, uint64_t *bitsets, uint32_t bitset_words,
    uint64_t *out_after, hipStream_t stream) {
    if (n_generators == 0) {
        return hipSuccess;
    }
    constexpr int BLOCK = 256;
    const int blocks = static_cast<int>((n_generators + BLOCK - 1) / BLOCK);
    ler_draw_parity_kernel<<<blocks, BLOCK, 0, stream>>>(
        seed, n_generators, nshots, cdf_all, table_offset, table_len, table_klo, table_of,
        probe_words, out_words, out_counts, out_pos, pos_stride, bitsets, bitset_words, out_after);
    return hipGetLastError();
}
