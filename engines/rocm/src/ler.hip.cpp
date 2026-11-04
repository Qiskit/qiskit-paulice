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

// Monte Carlo logical-error-rate scoring.
//
// Each error generator is packed into a row of bit planes and cumulant parity
// signatures (see `LerTables` in src/ler.rs), so scoring a shot is GF(2) work:
// XOR the rows of the generators that fired, then ask whether three ranges of
// the result are zero -- is the error the identity, do the checks detect it,
// does it touch a logical operator. The kernel counts shots in integers and the
// host does the one division, so the device agrees with the host exactly.
//
// A shot's accumulator is spread across the lanes of one wavefront and the
// zero tests are wave-wide votes, so the kernel needs a 64-wide wavefront;
// `paulice_rocm_create` refuses any other device.

#include <hip/hip_runtime.h>
#include <stdint.h>

#include "paulice_rocm.h"

namespace {

constexpr int WAVE = 64;

/// Accumulator words per lane. The whole accumulator lives in registers, so
/// this fixes the widest row the kernel handles.
constexpr int MAX_PER_LANE = PAULICE_ROCM_LER_MAX_WORDS / WAVE;

/// One wavefront per shot, striding over shots. Lane `l` owns accumulator
/// words `l, l + 64, ...`, so the lanes of a wave read consecutive words of a
/// generator row.
__global__ __launch_bounds__(WAVE) void ler_score_kernel(
    const uint64_t *__restrict__ gen_words, const int stride, const int planes_end,
    const int post_end, const uint32_t *__restrict__ offsets,
    const uint32_t *__restrict__ indices, const int n_shots,
    unsigned long long *__restrict__ out_accepted, unsigned long long *__restrict__ out_logical) {
    const int lane = static_cast<int>(threadIdx.x);
    unsigned long long accepted = 0;
    unsigned long long logical = 0;

    // Unsigned, so the last step past an n_shots near INT_MAX cannot overflow.
    for (uint32_t shot = blockIdx.x; shot < static_cast<uint32_t>(n_shots); shot += gridDim.x) {
        // Every lane reads the same offsets, so the skip and the votes below
        // stay uniform across the wave.
        const uint32_t lo = offsets[shot];
        const uint32_t hi = offsets[shot + 1];
        if (lo == hi) {
            // Nothing fired; the host counts identity shots without scoring.
            continue;
        }

        uint64_t acc[MAX_PER_LANE];
#pragma unroll
        for (int i = 0; i < MAX_PER_LANE; ++i) {
            acc[i] = 0;
        }
        for (uint32_t k = lo; k < hi; ++k) {
            const uint64_t *__restrict__ row =
                gen_words + static_cast<size_t>(indices[k]) * static_cast<size_t>(stride);
            for (int i = 0, w = lane; w < stride; ++i, w += WAVE) {
                acc[i] ^= row[w];
            }
        }

        uint64_t any_plane = 0;
        uint64_t any_post = 0;
        uint64_t any_logical = 0;
        for (int i = 0, w = lane; w < stride; ++i, w += WAVE) {
            if (w < planes_end) {
                any_plane |= acc[i];
            } else if (w < post_end) {
                any_post |= acc[i];
            } else {
                any_logical |= acc[i];
            }
        }
        const bool identity = __any(any_plane != 0) == 0;
        const bool detected = __any(any_post != 0) != 0;
        const bool touches_logical = __any(any_logical != 0) != 0;

        // As on the host, a shot whose generators cancel is dropped rather
        // than counted as accepted.
        if (!identity && !detected) {
            accepted += 1;
            if (touches_logical) {
                logical += 1;
            }
        }
    }

    // Integer counts, so the order of the atomics cannot change the total.
    if (lane == 0) {
        if (accepted != 0) {
            atomicAdd(out_accepted, accepted);
        }
        if (logical != 0) {
            atomicAdd(out_logical, logical);
        }
    }
}

} // namespace

extern "C" hipError_t paulice_launch_ler_score(const uint64_t *gen_words, int stride, int planes_end,
                                               int post_end, const uint32_t *offsets,
                                               const uint32_t *indices, int n_shots,
                                               unsigned long long *out_accepted,
                                               unsigned long long *out_logical,
                                               hipStream_t stream) {
    if (n_shots <= 0) {
        return hipSuccess;
    }
    constexpr int MAX_BLOCKS = 16384;
    const int blocks = n_shots < MAX_BLOCKS ? n_shots : MAX_BLOCKS;
    ler_score_kernel<<<blocks, WAVE, 0, stream>>>(gen_words, stride, planes_end, post_end, offsets,
                                                  indices, n_shots, out_accepted, out_logical);
    return hipGetLastError();
}
