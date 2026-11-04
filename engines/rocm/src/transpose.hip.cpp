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

// Per-generator draws to per-shot rows, as `transpose_to_shots` in src/ler.rs.
//
// A histogram of how many generators fired in each shot, an exclusive prefix
// sum over it, and a scatter. Integer work only. The scatter's atomics fill a
// shot's slots in arbitrary order, so each shot's slice is then sorted to
// restore the ascending generator order the host produces.

#include <hip/hip_runtime.h>
#include <stdint.h>

#include "paulice_rocm.h"

namespace {

constexpr int BLOCK = 256;
/// Elements per thread in the scan.
constexpr int PER_THREAD = 8;
constexpr uint32_t CHUNK = BLOCK * PER_THREAD;

/// `blockIdx.y` is the generator, so a generator's positions are spread over
/// a row of blocks.
__global__ void histogram_kernel(const uint32_t *__restrict__ pos, const uint32_t pos_stride,
                                 const uint32_t *__restrict__ gen_counts,
                                 uint32_t *__restrict__ shot_counts) {
    const uint32_t g = blockIdx.y;
    const uint32_t n = gen_counts[g];
    const uint32_t *__restrict__ p = pos + static_cast<size_t>(g) * pos_stride;
    const uint32_t stride = gridDim.x * blockDim.x;
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        atomicAdd(&shot_counts[p[i]], 1u);
    }
}

/// Inclusive Hillis-Steele scan of `vals` across one block, in place.
__device__ __forceinline__ uint32_t block_inclusive_scan(uint32_t *vals, const int lane) {
    for (int offset = 1; offset < BLOCK; offset <<= 1) {
        const uint32_t add = lane >= offset ? vals[lane - offset] : 0u;
        __syncthreads();
        vals[lane] += add;
        __syncthreads();
    }
    return vals[BLOCK - 1];
}

/// Scan pass 1: the total of each chunk.
__global__ void scan_chunk_sums(const uint32_t *__restrict__ counts, const uint32_t n,
                                uint32_t *__restrict__ chunk_sums) {
    __shared__ uint32_t shared[BLOCK];
    const int lane = static_cast<int>(threadIdx.x);
    const uint32_t base = blockIdx.x * CHUNK + lane * PER_THREAD;
    uint32_t sum = 0;
    for (int k = 0; k < PER_THREAD; ++k) {
        const uint32_t i = base + k;
        if (i < n) {
            sum += counts[i];
        }
    }
    shared[lane] = sum;
    __syncthreads();
    const uint32_t total = block_inclusive_scan(shared, lane);
    if (lane == 0) {
        chunk_sums[blockIdx.x] = total;
    }
}

/// Scan pass 2: exclusive scan of the chunk totals by one thread.
/// `chunk_sums[n_chunks]` receives the grand total.
__global__ void scan_chunk_offsets(uint32_t *__restrict__ chunk_sums, const uint32_t n_chunks) {
    uint32_t running = 0;
    for (uint32_t i = 0; i < n_chunks; ++i) {
        const uint32_t v = chunk_sums[i];
        chunk_sums[i] = running;
        running += v;
    }
    chunk_sums[n_chunks] = running;
}

/// Scan pass 3: write `offsets[0..n]`, and count the shots in which anything
/// fired.
__global__ void scan_apply(const uint32_t *__restrict__ counts, const uint32_t n,
                           const uint32_t *__restrict__ chunk_offsets,
                           uint32_t *__restrict__ offsets,
                           unsigned long long *__restrict__ out_nonempty) {
    __shared__ uint32_t shared[BLOCK];
    __shared__ unsigned long long block_nonempty;
    const int lane = static_cast<int>(threadIdx.x);
    const uint32_t base = blockIdx.x * CHUNK + lane * PER_THREAD;

    uint32_t mine[PER_THREAD];
    uint32_t sum = 0;
    uint32_t nonempty = 0;
    for (int k = 0; k < PER_THREAD; ++k) {
        const uint32_t i = base + k;
        mine[k] = i < n ? counts[i] : 0u;
        sum += mine[k];
        nonempty += mine[k] != 0 ? 1u : 0u;
    }
    shared[lane] = sum;
    if (lane == 0) {
        block_nonempty = 0;
    }
    __syncthreads();
    (void)block_inclusive_scan(shared, lane);
    uint32_t running = chunk_offsets[blockIdx.x] + shared[lane] - sum;
    for (int k = 0; k < PER_THREAD; ++k) {
        const uint32_t i = base + k;
        if (i < n) {
            offsets[i] = running;
            running += mine[k];
        }
    }

    atomicAdd(&block_nonempty, static_cast<unsigned long long>(nonempty));
    __syncthreads();
    if (lane == 0 && block_nonempty != 0) {
        atomicAdd(out_nonempty, block_nonempty);
    }
}

/// Writes `offsets[n]`, the end of the last shot.
__global__ void scan_tail(const uint32_t *__restrict__ chunk_offsets, const uint32_t n_chunks,
                          const uint32_t n, uint32_t *__restrict__ offsets) {
    offsets[n] = chunk_offsets[n_chunks];
}

/// `cursor` starts as a copy of `offsets` and is bumped atomically.
__global__ void scatter_kernel(const uint32_t *__restrict__ pos, const uint32_t pos_stride,
                               const uint32_t *__restrict__ gen_counts,
                               uint32_t *__restrict__ cursor, uint32_t *__restrict__ indices) {
    const uint32_t g = blockIdx.y;
    const uint32_t n = gen_counts[g];
    const uint32_t *__restrict__ p = pos + static_cast<size_t>(g) * pos_stride;
    const uint32_t stride = gridDim.x * blockDim.x;
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        const uint32_t slot = atomicAdd(&cursor[p[i]], 1u);
        indices[slot] = g;
    }
}

/// Insertion sort of each shot's generator list; a shot holds few generators.
__global__ void sort_shot_slices(const uint32_t *__restrict__ offsets,
                                 uint32_t *__restrict__ indices, const uint32_t n) {
    const uint32_t stride = gridDim.x * blockDim.x;
    for (uint32_t s = blockIdx.x * blockDim.x + threadIdx.x; s < n; s += stride) {
        const uint32_t lo = offsets[s];
        const uint32_t hi = offsets[s + 1];
        for (uint32_t i = lo + 1; i < hi; ++i) {
            const uint32_t x = indices[i];
            uint32_t j = i;
            while (j > lo && indices[j - 1] > x) {
                indices[j] = indices[j - 1];
                --j;
            }
            indices[j] = x;
        }
    }
}

} // namespace

/// Entries `chunk_sums` needs for `n_shots` shots.
extern "C" size_t paulice_ler_transpose_chunk_sums(uint32_t n_shots) {
    return (static_cast<size_t>(n_shots) + CHUNK - 1) / CHUNK + 1;
}

/// `shot_counts` (n_shots entries) and `*out_nonempty` must be zero on entry,
/// and `offsets` and `cursor` hold `n_shots + 1` entries. `n_generators` is
/// at most `PAULICE_ROCM_LER_MAX_GENERATORS`, since it is the grid's y extent.
extern "C" hipError_t paulice_launch_ler_transpose(
    const uint32_t *pos, uint32_t pos_stride, const uint32_t *gen_counts, uint32_t n_generators,
    uint32_t n_shots, uint32_t *shot_counts, uint32_t *chunk_sums, uint32_t *offsets,
    uint32_t *cursor, uint32_t *indices, unsigned long long *out_nonempty, hipStream_t stream) {
    if (n_generators == 0 || n_shots == 0 || n_generators > PAULICE_ROCM_LER_MAX_GENERATORS) {
        return hipErrorInvalidValue;
    }
    const uint32_t n_chunks = (n_shots + CHUNK - 1) / CHUNK;
    // Enough blocks per generator to cover its largest possible draw, capped.
    const uint32_t per_gen = (pos_stride + BLOCK - 1) / BLOCK;
    const uint32_t blocks_x = per_gen == 0 ? 1u : (per_gen > 64u ? 64u : per_gen);
    const dim3 grid(blocks_x, n_generators);

    histogram_kernel<<<grid, BLOCK, 0, stream>>>(pos, pos_stride, gen_counts, shot_counts);
    if (hipError_t e = hipGetLastError(); e != hipSuccess) {
        return e;
    }
    scan_chunk_sums<<<n_chunks, BLOCK, 0, stream>>>(shot_counts, n_shots, chunk_sums);
    if (hipError_t e = hipGetLastError(); e != hipSuccess) {
        return e;
    }
    scan_chunk_offsets<<<1, 1, 0, stream>>>(chunk_sums, n_chunks);
    if (hipError_t e = hipGetLastError(); e != hipSuccess) {
        return e;
    }
    scan_apply<<<n_chunks, BLOCK, 0, stream>>>(shot_counts, n_shots, chunk_sums, offsets,
                                               out_nonempty);
    if (hipError_t e = hipGetLastError(); e != hipSuccess) {
        return e;
    }
    scan_tail<<<1, 1, 0, stream>>>(chunk_sums, n_chunks, n_shots, offsets);
    if (hipError_t e = hipGetLastError(); e != hipSuccess) {
        return e;
    }
    if (hipError_t e = hipMemcpyAsync(cursor, offsets,
                                      (static_cast<size_t>(n_shots) + 1) * sizeof(uint32_t),
                                      hipMemcpyDeviceToDevice, stream);
        e != hipSuccess) {
        return e;
    }
    scatter_kernel<<<grid, BLOCK, 0, stream>>>(pos, pos_stride, gen_counts, cursor, indices);
    if (hipError_t e = hipGetLastError(); e != hipSuccess) {
        return e;
    }
    const uint32_t sort_blocks = (n_shots + BLOCK - 1) / BLOCK;
    sort_shot_slices<<<sort_blocks < 16384 ? sort_blocks : 16384, BLOCK, 0, stream>>>(
        offsets, indices, n_shots);
    return hipGetLastError();
}
