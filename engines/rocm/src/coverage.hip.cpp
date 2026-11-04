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

// Gamma scoring kernels.
//
// The work splits into three passes, one kernel each:
//
//   1. `propagate_kernel`  -- walk a candidate's circuit backwards, snapshotting
//                             the cumulant columns. One wavefront per candidate
//                             per table, one lane per cumulant row, so a whole
//                             packed column word falls out of a single ballot.
//   2. `count_kernel`      -- for each candidate, count the generators that
//                             contribute to gamma, per noise block.
//   3. `reduce_kernel`     -- turn those counts back into a summed rate.
//
// The split exists for bit-exactness. The host sums rates sequentially in
// generator order, and floating-point addition is not associative, so a tree
// reduction would not agree with it. Every generator inside a noise block
// carries the same rate, so "which generators contributed" collapses to a
// count -- an integer reduction, which *is* order-free and exact -- and pass 3
// replays exactly the host's sequence of additions from those counts.

#include <hip/hip_runtime.h>
// Only the primitive that is used: the rocPRIM umbrella header drags in
// `texture_cache_iterator.hpp`, which does not compile under ROCm 7.2 because
// its host-side `memset` resolves to HIP's `__device__` overload.
#include <rocprim/block/block_reduce.hpp>
#include <stdint.h>

#include "paulice_rocm.h"

namespace {

constexpr int WAVE = 64;
constexpr int COUNT_BLOCK = 256;
/// Gates pulled into LDS per round by the propagation walk.
constexpr int GATE_CHUNK = 256;

__device__ inline void pauli_xz(uint8_t p, bool *xe, bool *ze) {
    *xe = (p == 1) || (p == 2);
    *ze = (p == 2) || (p == 3);
}

// The row state is a handful of 64-bit words held in registers. Indexing it by
// a runtime qubit would spill it to scratch and put two memory round-trips on
// every gate, so the word is picked by an unrolled compare chain instead: with
// `QW` a compile-time constant those collapse to register selects.
template <int QW> __device__ inline bool get_bit(const uint64_t (&words)[QW], int q) {
    const int word = q >> 6;
    uint64_t value = 0;
#pragma unroll
    for (int i = 0; i < QW; ++i) {
        if (i == word) {
            value = words[i];
        }
    }
    return (value >> (q & 63)) & 1ull;
}

template <int QW> __device__ inline void xor_bit(uint64_t (&words)[QW], int q, bool value) {
    const uint64_t mask = static_cast<uint64_t>(value) << (q & 63);
    const int word = q >> 6;
#pragma unroll
    for (int i = 0; i < QW; ++i) {
        if (i == word) {
            words[i] ^= mask;
        }
    }
}

template <int QW> __device__ inline void set_bit(uint64_t (&words)[QW], int q, bool value) {
    const uint64_t mask = 1ull << (q & 63);
    const int word = q >> 6;
#pragma unroll
    for (int i = 0; i < QW; ++i) {
        if (i == word) {
            words[i] = value ? (words[i] | mask) : (words[i] & ~mask);
        }
    }
}

/// Symplectic (x, z) action of a Clifford gate on one row.
///
/// Phases are never read back -- the cumulant table only stores x/z bits -- so
/// they are not tracked. That also makes the backward walk use these same
/// rules: S and S-dagger have identical symplectic action, as do SqrtX and its
/// dagger, and H/CZ/CNOT are self-inverse here.
template <int QW>
__device__ inline void apply_gate(uint64_t (&x)[QW], uint64_t (&z)[QW], uint32_t packed) {
    const uint32_t op = packed >> 24;
    const int a = static_cast<int>((packed >> 12) & 0xfff);
    const int b = static_cast<int>(packed & 0xfff);
    switch (op) {
    case PAULICE_GATE_H: {
        const bool xa = get_bit(x, a);
        const bool za = get_bit(z, a);
        set_bit(x, a, za);
        set_bit(z, a, xa);
        break;
    }
    case PAULICE_GATE_S:
    case PAULICE_GATE_SD:
        xor_bit(z, a, get_bit(x, a));
        break;
    case PAULICE_GATE_SQRTX:
    case PAULICE_GATE_SQRTXD:
        xor_bit(x, a, get_bit(z, a));
        break;
    case PAULICE_GATE_CZ:
        xor_bit(z, b, get_bit(x, a));
        xor_bit(z, a, get_bit(x, b));
        break;
    case PAULICE_GATE_CNOT:
        xor_bit(x, b, get_bit(x, a));
        xor_bit(z, a, get_bit(z, b));
        break;
    default:
        break;
    }
}

/// Packs the calling wavefront's bit for `q` into one column word. Lane `r`
/// holds cumulant row `r`, so the ballot mask *is* the packed column.
template <int QW> __device__ inline uint64_t ballot_bit(const uint64_t (&words)[QW], int q) {
    return __ballot(get_bit(words, q));
}

/// Writes the staged columns out, one per lane.
__device__ inline void flush(const uint64_t *staged_x, const uint64_t *staged_z, int &staged,
                             int first_column, int lane, size_t base, size_t stride,
                             uint64_t *x_out, uint64_t *z_out) {
    __builtin_amdgcn_wave_barrier();
    if (lane < staged) {
        const size_t at = base + static_cast<size_t>(first_column + lane) * stride;
        x_out[at] = staged_x[lane];
        z_out[at] = staged_z[lane];
    }
    __builtin_amdgcn_wave_barrier();
    staged = 0;
}

/// Parks one column in LDS, flushing once a wavefront's worth has piled up.
__device__ inline void stage(uint64_t *staged_x, uint64_t *staged_z, int &staged,
                             int &first_column, int column, uint64_t x_word, uint64_t z_word,
                             int lane, size_t base, size_t stride, uint64_t *x_out,
                             uint64_t *z_out) {
    if (staged == 0) {
        first_column = column;
    }
    if (lane == staged) {
        staged_x[lane] = x_word;
        staged_z[lane] = z_word;
    }
    ++staged;
    if (staged == WAVE) {
        flush(staged_x, staged_z, staged, first_column, lane, base, stride, x_out, z_out);
    }
}

/// One wavefront per (candidate, table, row block). Walks the circuit backwards
/// and writes the packed cumulant columns.
///
/// Column order mirrors the host walk: the two wires of each 2-qubit gate from
/// the last gate down to the first, then one column per input wire.
template <int QW>
__global__ __launch_bounds__(WAVE) void propagate_kernel(
    const uint32_t *gates, const int32_t *gate_offset, const int32_t *n_two_qubit,
    const uint64_t *seeds, const int32_t *row_offset, int row_words, int nqubits,
    const int64_t *word_offset, const int32_t *nwords_per_candidate, uint64_t *x_out,
    uint64_t *z_out) {
    const int candidate = blockIdx.x;
    const int row_block = blockIdx.y;
    const int lane = threadIdx.x;

    const int row_begin = row_offset[candidate];
    const int nrows = row_offset[candidate + 1] - row_begin;
    const int nwords = nwords_per_candidate[candidate];
    if (row_block >= nwords) {
        return;
    }

    const int row = row_block * WAVE + lane;
    uint64_t x[QW];
    uint64_t z[QW];
#pragma unroll
    for (int w = 0; w < QW; ++w) {
        x[w] = 0;
        // Rows past the end stay identity, which contributes nothing.
        z[w] = (row < nrows && w < row_words)
                   ? seeds[static_cast<size_t>(row_begin + row) * row_words + w]
                   : 0ull;
    }

    const int gate_begin = gate_offset[candidate];
    const int gate_end = gate_offset[candidate + 1];
    const int n2q = n_two_qubit[candidate];
    // Where this candidate's columns start, and the word stride within one.
    // The offset is precomputed on the host because the word count per column
    // differs per candidate, so it is not just a column count times a stride.
    const size_t base = static_cast<size_t>(word_offset[candidate]) + row_block;
    const size_t stride = nwords;

    // A ballot leaves the whole column word in every lane, so storing it
    // directly would be one single-lane store per column -- thousands of them,
    // each moving 8 bytes. Instead the columns are parked in LDS until a full
    // wavefront's worth has accumulated, then written out with one store per
    // lane. Consecutive columns are `stride` words apart, so for the common
    // single-word case that flush is fully coalesced.
    __shared__ uint64_t staged_x[WAVE];
    __shared__ uint64_t staged_z[WAVE];
    int staged = 0;
    int first_staged_column = 0;

    // The walk is one long dependency chain over the gates, and with a single
    // wavefront per candidate there is no other wave on the SIMD to hide a
    // global load behind. Pull the gates in a chunk at a time with all lanes
    // cooperating, then read them back out of LDS.
    __shared__ uint32_t gate_buf[GATE_CHUNK];

    int column = 0;
    for (int chunk_end = gate_end; chunk_end > gate_begin; chunk_end -= GATE_CHUNK) {
        const int chunk_begin = max(gate_begin, chunk_end - GATE_CHUNK);
        const int count = chunk_end - chunk_begin;
        for (int i = lane; i < count; i += WAVE) {
            gate_buf[i] = gates[chunk_begin + i];
        }
        __builtin_amdgcn_wave_barrier();
        for (int i = count - 1; i >= 0; --i) {
            const uint32_t packed = gate_buf[i];
            const uint32_t op = packed >> 24;
            const bool two_qubit = (op == PAULICE_GATE_CZ) || (op == PAULICE_GATE_CNOT);
            if (two_qubit) {
                const int a = static_cast<int>((packed >> 12) & 0xfff);
                const int b = static_cast<int>(packed & 0xfff);
                // Snapshot before undoing the gate, exactly like the host walk.
                const uint64_t xa = ballot_bit(x, a);
                const uint64_t za = ballot_bit(z, a);
                const uint64_t xb = ballot_bit(x, b);
                const uint64_t zb = ballot_bit(z, b);
                stage(staged_x, staged_z, staged, first_staged_column, column, xa, za, lane, base,
                      stride, x_out, z_out);
                stage(staged_x, staged_z, staged, first_staged_column, column + 1, xb, zb, lane,
                      base, stride, x_out, z_out);
                column += 2;
            }
            apply_gate(x, z, packed);
        }
        __builtin_amdgcn_wave_barrier();
    }
    // Input wires, recorded last.
    for (int q = 0; q < nqubits; ++q) {
        const uint64_t xq = ballot_bit(x, q);
        const uint64_t zq = ballot_bit(z, q);
        stage(staged_x, staged_z, staged, first_staged_column, 2 * n2q + q, xq, zq, lane, base,
              stride, x_out, z_out);
    }
    flush(staged_x, staged_z, staged, first_staged_column, lane, base, stride, x_out, z_out);
}

/// XOR-parity of one generator against one cumulant table: does any row
/// anticommute with it.
__device__ inline bool covered(const uint64_t *x_words, const uint64_t *z_words, int nwords,
                               int nrows, const int *columns, const uint8_t *paulis, int nterms) {
    if (nrows <= 0) {
        return false;
    }
    for (int k = 0; k < nwords; ++k) {
        uint64_t acc = 0;
        for (int t = 0; t < nterms; ++t) {
            const int col = columns[t];
            if (col < 0) {
                continue;
            }
            bool xe = false;
            bool ze = false;
            pauli_xz(paulis[t], &xe, &ze);
            if (ze) {
                acc ^= x_words[static_cast<size_t>(col) * nwords + k];
            }
            if (xe) {
                acc ^= z_words[static_cast<size_t>(col) * nwords + k];
            }
        }
        if (acc != 0) {
            return true;
        }
    }
    return false;
}

/// One workgroup per candidate. Counts, per noise block, how many generators
/// are invisible to the checks but visible to the logicals.
__global__ __launch_bounds__(COUNT_BLOCK) void count_kernel(
    const int32_t *n_two_qubit, int nqubits, const int32_t *readout_column,
    const PauliceNoiseBlock *blocks, int n_blocks, const uint64_t *post_x, const uint64_t *post_z,
    const int64_t *post_word_offset, const int32_t *post_nwords, const int32_t *post_row_offset,
    const uint64_t *log_x, const uint64_t *log_z, const int64_t *log_word_offset,
    const int32_t *log_nwords, const int32_t *log_row_offset, int32_t *counts_out) {
    using BlockReduce = rocprim::block_reduce<int, COUNT_BLOCK>;
    __shared__ typename BlockReduce::storage_type storage;

    const int candidate = blockIdx.x;
    const int n2q = n_two_qubit[candidate];

    const int post_nw = post_nwords[candidate];
    const int log_nw = log_nwords[candidate];
    const int post_rows = post_row_offset[candidate + 1] - post_row_offset[candidate];
    const int log_rows = log_row_offset[candidate + 1] - log_row_offset[candidate];
    const uint64_t *px = post_x + static_cast<size_t>(post_word_offset[candidate]);
    const uint64_t *pz = post_z + static_cast<size_t>(post_word_offset[candidate]);
    const uint64_t *lx = log_x + static_cast<size_t>(log_word_offset[candidate]);
    const uint64_t *lz = log_z + static_cast<size_t>(log_word_offset[candidate]);
    const int32_t *ro_col = readout_column + static_cast<size_t>(candidate) * nqubits;

    for (int b = 0; b < n_blocks; ++b) {
        const int kind = blocks[b].kind;
        const int n_gen = (kind == PAULICE_NOISE_DEPOLARIZING) ? 15 * n2q : nqubits;
        int local = 0;
        for (int g = threadIdx.x; g < n_gen; g += COUNT_BLOCK) {
            int columns[2];
            uint8_t paulis[2];
            int nterms = 0;
            if (kind == PAULICE_NOISE_DEPOLARIZING) {
                // Generators run 15 per 2-qubit gate, in ascending gate order;
                // within a gate the Pauli pairs run row-major over (p0, p1)
                // skipping identity, so index i maps to the pair i + 1.
                const int rank = g / 15;
                const int pair = (g % 15) + 1;
                const uint8_t p0 = static_cast<uint8_t>(pair >> 2);
                const uint8_t p1 = static_cast<uint8_t>(pair & 3);
                // The backward walk visits 2-qubit gates last-first, so the
                // gate of ascending rank `rank` owns columns 2*(n2q-1-rank){,+1}.
                const int col0 = 2 * (n2q - 1 - rank);
                if (p0 != 0) {
                    columns[nterms] = col0;
                    paulis[nterms] = p0;
                    ++nterms;
                }
                if (p1 != 0) {
                    columns[nterms] = col0 + 1;
                    paulis[nterms] = p1;
                    ++nterms;
                }
            } else {
                columns[0] = ro_col[g];
                paulis[0] = 1; // X on the readout wire
                nterms = 1;
            }
            const bool post = covered(px, pz, post_nw, post_rows, columns, paulis, nterms);
            const bool logical = covered(lx, lz, log_nw, log_rows, columns, paulis, nterms);
            local += (!post && logical) ? 1 : 0;
        }
        int total = 0;
        BlockReduce().reduce(local, total, storage, rocprim::plus<int>());
        if (threadIdx.x == 0) {
            counts_out[static_cast<size_t>(candidate) * n_blocks + b] = total;
        }
        __syncthreads();
    }
}

/// One thread per candidate. Replays the host's summation: the contributing
/// generators of a block all share its rate, so adding that rate `count` times
/// in block order reproduces the host's sequential accumulation exactly.
///
/// Returns the accumulated rate, not the score. IEEE-754 pins double addition
/// down to the bit, so this matches the host exactly -- but `exp` is a library
/// function with no such guarantee, and the device libm need not agree with
/// the host's last ulp. The host applies `exp` itself.
__global__ void reduce_kernel(const int32_t *counts, const PauliceNoiseBlock *blocks, int n_blocks,
                              int n_candidates, double *out_rates) {
    const int candidate = blockIdx.x * blockDim.x + threadIdx.x;
    if (candidate >= n_candidates) {
        return;
    }
    double acc = 0.0;
    for (int b = 0; b < n_blocks; ++b) {
        const int count = counts[static_cast<size_t>(candidate) * n_blocks + b];
        const double rate = blocks[b].rate;
        for (int i = 0; i < count; ++i) {
            acc += rate;
        }
    }
    out_rates[candidate] = acc;
}

} // namespace

extern "C" hipError_t paulice_launch_propagate(
    const uint32_t *gates, const int32_t *gate_offset, const int32_t *n_two_qubit,
    const uint64_t *seeds, const int32_t *row_offset, int row_words, int nqubits,
    const int64_t *word_offset, const int32_t *nwords_per_candidate, uint64_t *x_out,
    uint64_t *z_out, int n_candidates, int max_row_blocks, hipStream_t stream) {
    const dim3 grid(n_candidates, max_row_blocks, 1);
    // Specialise on the number of 64-bit words a row needs, so the state stays
    // in registers. `paulice_rocm_gamma_batch` declines batches wider than
    // `PAULICE_ROCM_MAX_QUBITS`.
#define PAULICE_LAUNCH(QW)                                                                         \
    propagate_kernel<QW><<<grid, WAVE, 0, stream>>>(gates, gate_offset, n_two_qubit, seeds,        \
                                                    row_offset, row_words, nqubits, word_offset,   \
                                                    nwords_per_candidate, x_out, z_out)
    if (nqubits <= 64) {
        PAULICE_LAUNCH(1);
    } else if (nqubits <= 128) {
        PAULICE_LAUNCH(2);
    } else if (nqubits <= 192) {
        PAULICE_LAUNCH(3);
    } else if (nqubits <= PAULICE_ROCM_MAX_QUBITS) {
        PAULICE_LAUNCH(4);
    } else {
        return hipErrorInvalidValue;
    }
#undef PAULICE_LAUNCH
    return hipGetLastError();
}

extern "C" hipError_t paulice_launch_count(
    const int32_t *n_two_qubit, int nqubits, const int32_t *readout_column,
    const PauliceNoiseBlock *blocks, int n_blocks, const uint64_t *post_x, const uint64_t *post_z,
    const int64_t *post_word_offset, const int32_t *post_nwords, const int32_t *post_row_offset,
    const uint64_t *log_x, const uint64_t *log_z, const int64_t *log_word_offset,
    const int32_t *log_nwords, const int32_t *log_row_offset, int32_t *counts_out,
    int n_candidates, hipStream_t stream) {
    count_kernel<<<n_candidates, COUNT_BLOCK, 0, stream>>>(
        n_two_qubit, nqubits, readout_column, blocks, n_blocks, post_x, post_z, post_word_offset,
        post_nwords, post_row_offset, log_x, log_z, log_word_offset, log_nwords, log_row_offset,
        counts_out);
    return hipGetLastError();
}

extern "C" hipError_t paulice_launch_reduce(const int32_t *counts, const PauliceNoiseBlock *blocks,
                                            int n_blocks, int n_candidates, double *out_rates,
                                            hipStream_t stream) {
    const int block = 64;
    const int grid = (n_candidates + block - 1) / block;
    reduce_kernel<<<grid, block, 0, stream>>>(counts, blocks, n_blocks, n_candidates, out_rates);
    return hipGetLastError();
}
