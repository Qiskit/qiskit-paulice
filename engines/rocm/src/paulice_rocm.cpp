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

// Host side of the HIP backend: device context, scratch buffers, and the
// upload/launch/download sequence for each entry point.

#include "paulice_rocm.h"

#include <hip/hip_runtime.h>
#include <algorithm>
#include <math.h>
#include <new>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

extern "C" hipError_t paulice_launch_propagate(
    const uint32_t *gates, const int32_t *gate_offset, const int32_t *n_two_qubit,
    const uint64_t *seeds, const int32_t *row_offset, int row_words, int nqubits,
    const int64_t *word_offset, const int32_t *nwords_per_candidate, uint64_t *x_out,
    uint64_t *z_out, int n_candidates, int max_row_blocks, hipStream_t stream);

extern "C" hipError_t paulice_launch_count(
    const int32_t *n_two_qubit, int nqubits, const int32_t *readout_column,
    const PauliceNoiseBlock *blocks, int n_blocks, const uint64_t *post_x, const uint64_t *post_z,
    const int64_t *post_word_offset, const int32_t *post_nwords, const int32_t *post_row_offset,
    const uint64_t *log_x, const uint64_t *log_z, const int64_t *log_word_offset,
    const int32_t *log_nwords, const int32_t *log_row_offset, int32_t *counts_out,
    int n_candidates, hipStream_t stream);

extern "C" hipError_t paulice_launch_reduce(const int32_t *counts, const PauliceNoiseBlock *blocks,
                                            int n_blocks, int n_candidates, double *out_rates,
                                            hipStream_t stream);

extern "C" hipError_t paulice_launch_ler_score(const uint64_t *gen_words, int stride, int planes_end,
                                               int post_end, const uint32_t *offsets,
                                               const uint32_t *indices, int n_shots,
                                               unsigned long long *out_accepted,
                                               unsigned long long *out_logical,
                                               hipStream_t stream);

extern "C" hipError_t paulice_launch_ler_draw_parity(
    uint64_t seed, uint32_t n_generators, uint32_t nshots, const double *cdf_all,
    const uint32_t *table_offset, const uint32_t *table_len, const uint32_t *table_klo,
    const uint32_t *table_of, uint32_t probe_words, uint64_t *out_words, uint32_t *out_counts,
    uint32_t *out_pos, uint32_t pos_stride, uint64_t *bitsets, uint32_t bitset_words,
    uint64_t *out_after, hipStream_t stream);

extern "C" hipError_t paulice_launch_ler_draw(uint64_t seed, uint32_t n_generators, uint32_t nshots,
                                              const double *cdf_all, const uint32_t *table_offset,
                                              const uint32_t *table_len, const uint32_t *table_klo,
                                              const uint32_t *table_of, uint32_t *out_counts,
                                              uint32_t *out_pos, uint32_t pos_stride,
                                              uint64_t *bitsets, uint32_t bitset_words,
                                              uint32_t n_lanes, unsigned long long *out_overflow,
                                              hipStream_t stream);

extern "C" size_t paulice_ler_transpose_chunk_sums(uint32_t n_shots);

extern "C" hipError_t paulice_launch_ler_transpose(
    const uint32_t *pos, uint32_t pos_stride, const uint32_t *gen_counts, uint32_t n_generators,
    uint32_t n_shots, uint32_t *shot_counts, uint32_t *chunk_sums, uint32_t *offsets,
    uint32_t *cursor, uint32_t *indices, unsigned long long *out_nonempty, hipStream_t stream);

namespace {

/// A device allocation that grows but never shrinks, so a steady-state batch
/// loop stops calling hipMalloc entirely.
class DeviceBuffer {
  public:
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;

    ~DeviceBuffer() { release(); }

    void release() {
        if (ptr_ != nullptr) {
            (void)hipFree(ptr_);
            ptr_ = nullptr;
        }
        capacity_ = 0;
    }

    /// Ensures room for `bytes`, returning false if the allocation failed.
    bool ensure(size_t bytes) {
        if (bytes <= capacity_) {
            return true;
        }
        release();
        if (hipMalloc(&ptr_, bytes) != hipSuccess) {
            ptr_ = nullptr;
            capacity_ = 0;
            return false;
        }
        capacity_ = bytes;
        return true;
    }

    bool holds(size_t bytes) const { return bytes <= capacity_; }
    size_t capacity() const { return capacity_; }

    template <typename T> T *as() { return static_cast<T *>(ptr_); }
    template <typename T> const T *as() const { return static_cast<const T *>(ptr_); }

  private:
    void *ptr_ = nullptr;
    size_t capacity_ = 0;
};

/// Uploads `count` elements, growing `buffer` first.
template <typename T>
bool upload(DeviceBuffer &buffer, const T *src, size_t count, hipStream_t stream) {
    if (!buffer.ensure(count * sizeof(T))) {
        return false;
    }
    if (count == 0) {
        return true;
    }
    return hipMemcpyAsync(buffer.as<T>(), src, count * sizeof(T), hipMemcpyHostToDevice, stream) ==
           hipSuccess;
}

/// Sizes the draw's dedup bitsets, one bit per shot per lane, and makes sure
/// they are zero. A draw leaves its bitset zero, so they are only cleared when
/// the buffer grows; a dispatch that fails after the draw started releases the
/// buffer instead, so a dirty bitset is never reused.
bool ensure_draw_bitsets(DeviceBuffer &bits, size_t n_shots, size_t lanes, size_t &words_out,
                         hipStream_t stream) {
    const size_t words = (n_shots + 63) / 64;
    const size_t bytes = words * lanes * sizeof(uint64_t);
    words_out = words;
    const bool grew = !bits.holds(bytes);
    if (!bits.ensure(bytes)) {
        return false;
    }
    return !grew || hipMemsetAsync(bits.as<void>(), 0, bytes, stream) == hipSuccess;
}

/// Per-stage timing, printed when PAULICE_ROCM_PROFILE is set. Off by default:
/// it forces a synchronise between stages.
bool profiling() {
    static const bool on = [] {
        const char *v = getenv("PAULICE_ROCM_PROFILE");
        return v != nullptr && v[0] == '1';
    }();
    return on;
}

class StageTimer {
  public:
    explicit StageTimer(hipStream_t stream) : stream_(stream) {
        if (!profiling()) {
            return;
        }
        (void)hipEventCreate(&start_);
        (void)hipEventCreate(&stop_);
        (void)hipEventRecord(start_, stream_);
    }

    ~StageTimer() {
        if (!profiling()) {
            return;
        }
        (void)hipEventDestroy(start_);
        (void)hipEventDestroy(stop_);
    }

    void report(const char *name) {
        if (!profiling()) {
            return;
        }
        (void)hipEventRecord(stop_, stream_);
        (void)hipEventSynchronize(stop_);
        float ms = 0.0f;
        (void)hipEventElapsedTime(&ms, start_, stop_);
        fprintf(stderr, "[paulice-rocm] %-12s %8.3f ms\n", name, ms);
        (void)hipEventRecord(start_, stream_);
    }

  private:
    hipStream_t stream_;
    hipEvent_t start_ = nullptr;
    hipEvent_t stop_ = nullptr;
};

} // namespace

struct PauliceEngineCtx {
    int device_id = 0;
    hipStream_t stream = nullptr;
    /// The two cumulant tables are independent walks, so the second one runs on
    /// its own stream. Each is only a few hundred wavefronts -- far short of
    /// filling the device -- so overlapping them is close to free parallelism.
    hipStream_t alt_stream = nullptr;
    hipEvent_t fork = nullptr;
    hipEvent_t join = nullptr;

    DeviceBuffer gates;
    DeviceBuffer gate_offset;
    DeviceBuffer n_two_qubit;
    DeviceBuffer post_seed;
    DeviceBuffer post_row_offset;
    DeviceBuffer logical_seed;
    DeviceBuffer logical_row_offset;
    DeviceBuffer readout_column;
    DeviceBuffer blocks;

    DeviceBuffer post_x;
    DeviceBuffer post_z;
    DeviceBuffer post_word_offset;
    DeviceBuffer post_nwords;
    DeviceBuffer log_x;
    DeviceBuffer log_z;
    DeviceBuffer log_word_offset;
    DeviceBuffer log_nwords;
    DeviceBuffer counts;
    DeviceBuffer rates;

    DeviceBuffer ler_gen_words;
    DeviceBuffer ler_offsets;
    DeviceBuffer ler_indices;
    DeviceBuffer ler_counts;

    DeviceBuffer draw_cdf;
    DeviceBuffer draw_table_offset;
    DeviceBuffer draw_table_len;
    DeviceBuffer draw_table_klo;
    DeviceBuffer draw_table_of;
    DeviceBuffer draw_words;
    DeviceBuffer draw_counts;
    DeviceBuffer draw_pos;
    DeviceBuffer draw_after;
    DeviceBuffer draw_bits;

    DeviceBuffer tr_shot_counts;
    DeviceBuffer tr_chunk_sums;
    DeviceBuffer tr_offsets;
    DeviceBuffer tr_cursor;
    DeviceBuffer tr_indices;
    DeviceBuffer tr_nonempty;

    /// Whether `tr_offsets` and `tr_indices` hold a sample drawn by
    /// `paulice_rocm_ler_sample_score`, and its shape.
    bool sample_resident = false;
    uint32_t resident_shots = 0;
    uint32_t resident_generators = 0;
};

int paulice_rocm_device_count(void) {
    int count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess) {
        return -1;
    }
    return count;
}

PauliceEngineCtx *paulice_rocm_create(int device_id) {
    if (hipSetDevice(device_id) != hipSuccess) {
        return nullptr;
    }
    // The kernels are written for a 64-wide wavefront: a ballot packs one
    // cumulant row per lane into a 64-bit column word, and a workgroup is one
    // wavefront. On a 32-wide device (e.g. RDNA) that packing is wrong and the
    // scores come out silently incorrect, so refuse the device here and let the
    // caller fall back to the CPU rather than return garbage.
    hipDeviceProp_t props;
    if (hipGetDeviceProperties(&props, device_id) != hipSuccess || props.warpSize != 64) {
        return nullptr;
    }
    PauliceEngineCtx *ctx = new (std::nothrow) PauliceEngineCtx();
    if (ctx == nullptr) {
        return nullptr;
    }
    ctx->device_id = device_id;
    if (hipStreamCreateWithFlags(&ctx->stream, hipStreamNonBlocking) != hipSuccess ||
        hipStreamCreateWithFlags(&ctx->alt_stream, hipStreamNonBlocking) != hipSuccess ||
        hipEventCreateWithFlags(&ctx->fork, hipEventDisableTiming) != hipSuccess ||
        hipEventCreateWithFlags(&ctx->join, hipEventDisableTiming) != hipSuccess) {
        paulice_rocm_destroy(ctx);
        return nullptr;
    }
    return ctx;
}

void paulice_rocm_destroy(PauliceEngineCtx *ctx) {
    if (ctx == nullptr) {
        return;
    }
    (void)hipSetDevice(ctx->device_id);
    if (ctx->join != nullptr) {
        (void)hipEventDestroy(ctx->join);
    }
    if (ctx->fork != nullptr) {
        (void)hipEventDestroy(ctx->fork);
    }
    if (ctx->alt_stream != nullptr) {
        (void)hipStreamDestroy(ctx->alt_stream);
    }
    if (ctx->stream != nullptr) {
        (void)hipStreamDestroy(ctx->stream);
    }
    delete ctx;
}

int paulice_rocm_gamma_batch(PauliceEngineCtx *ctx, const PauliceGammaBatch *batch) {
    if (ctx == nullptr || batch == nullptr || batch->n_candidates <= 0 || batch->n_blocks <= 0 ||
        batch->nqubits <= 0 || batch->nqubits > PAULICE_ROCM_MAX_QUBITS) {
        return -1;
    }
    if (hipSetDevice(ctx->device_id) != hipSuccess) {
        return -1;
    }

    const int n = batch->n_candidates;
    const int nq = batch->nqubits;
    const hipStream_t stream = ctx->stream;
    StageTimer timer(stream);
    // Copies and launches are asynchronous, so on failure wait for both
    // streams before returning: the caller frees the host inputs, and the next
    // batch reuses the device buffers.
    const auto fail = [ctx]() {
        (void)hipStreamSynchronize(ctx->alt_stream);
        (void)hipStreamSynchronize(ctx->stream);
        return -1;
    };

    // Every candidate records two columns per 2-qubit gate plus one per input
    // wire, and both tables share that column layout since both walk the same
    // circuit. The number of *words* per column differs though -- it follows
    // each table's row count -- so the word offsets are accumulated explicitly
    // rather than derived from a column count and a single stride.
    std::vector<int64_t> post_word_offset(n + 1, 0);
    std::vector<int64_t> log_word_offset(n + 1, 0);
    std::vector<int32_t> post_nwords(n, 1);
    std::vector<int32_t> log_nwords(n, 1);
    int max_post_blocks = 1;
    int max_log_blocks = 1;
    for (int c = 0; c < n; ++c) {
        const int64_t columns = 2 * batch->n_two_qubit[c] + nq;
        const int post_rows = batch->post_row_offset[c + 1] - batch->post_row_offset[c];
        const int log_rows = batch->logical_row_offset[c + 1] - batch->logical_row_offset[c];
        post_nwords[c] = std::max(1, (post_rows + 63) / 64);
        log_nwords[c] = std::max(1, (log_rows + 63) / 64);
        post_word_offset[c + 1] = post_word_offset[c] + columns * post_nwords[c];
        log_word_offset[c + 1] = log_word_offset[c] + columns * log_nwords[c];
        max_post_blocks = std::max(max_post_blocks, post_nwords[c]);
        max_log_blocks = std::max(max_log_blocks, log_nwords[c]);
    }

    const size_t n_gates = static_cast<size_t>(batch->gate_offset[n]);
    const size_t n_post_rows = static_cast<size_t>(batch->post_row_offset[n]);
    const size_t n_log_rows = static_cast<size_t>(batch->logical_row_offset[n]);
    const size_t row_words = static_cast<size_t>(batch->row_words);

    if (!upload(ctx->gates, batch->gates, n_gates, stream) ||
        !upload(ctx->gate_offset, batch->gate_offset, static_cast<size_t>(n) + 1, stream) ||
        !upload(ctx->n_two_qubit, batch->n_two_qubit, static_cast<size_t>(n), stream) ||
        !upload(ctx->post_seed, batch->post_seed, n_post_rows * row_words, stream) ||
        !upload(ctx->post_row_offset, batch->post_row_offset, static_cast<size_t>(n) + 1, stream) ||
        !upload(ctx->logical_seed, batch->logical_seed, n_log_rows * row_words, stream) ||
        !upload(ctx->logical_row_offset, batch->logical_row_offset, static_cast<size_t>(n) + 1,
                stream) ||
        !upload(ctx->readout_column, batch->readout_column, static_cast<size_t>(n) * nq, stream) ||
        !upload(ctx->blocks, batch->blocks, static_cast<size_t>(batch->n_blocks), stream) ||
        !upload(ctx->post_word_offset, post_word_offset.data(), static_cast<size_t>(n) + 1,
                stream) ||
        !upload(ctx->log_word_offset, log_word_offset.data(), static_cast<size_t>(n) + 1, stream) ||
        !upload(ctx->post_nwords, post_nwords.data(), static_cast<size_t>(n), stream) ||
        !upload(ctx->log_nwords, log_nwords.data(), static_cast<size_t>(n), stream)) {
        return fail();
    }
    timer.report("upload");

    // Fan the two propagations out across both streams, then join before the
    // counting pass, which reads from both tables.
    if (hipEventRecord(ctx->fork, stream) != hipSuccess ||
        hipStreamWaitEvent(ctx->alt_stream, ctx->fork, 0) != hipSuccess) {
        return fail();
    }

    const size_t post_words = static_cast<size_t>(post_word_offset[n]);
    const size_t log_words = static_cast<size_t>(log_word_offset[n]);
    if (!ctx->post_x.ensure(post_words * sizeof(uint64_t)) ||
        !ctx->post_z.ensure(post_words * sizeof(uint64_t)) ||
        !ctx->log_x.ensure(log_words * sizeof(uint64_t)) ||
        !ctx->log_z.ensure(log_words * sizeof(uint64_t)) ||
        !ctx->counts.ensure(static_cast<size_t>(n) * batch->n_blocks * sizeof(int32_t)) ||
        !ctx->rates.ensure(static_cast<size_t>(n) * sizeof(double))) {
        return fail();
    }

    if (paulice_launch_propagate(ctx->gates.as<uint32_t>(), ctx->gate_offset.as<int32_t>(),
                                 ctx->n_two_qubit.as<int32_t>(), ctx->post_seed.as<uint64_t>(),
                                 ctx->post_row_offset.as<int32_t>(), batch->row_words, nq,
                                 ctx->post_word_offset.as<int64_t>(),
                                 ctx->post_nwords.as<int32_t>(), ctx->post_x.as<uint64_t>(),
                                 ctx->post_z.as<uint64_t>(), n, max_post_blocks,
                                 stream) != hipSuccess) {
        return fail();
    }
    if (paulice_launch_propagate(ctx->gates.as<uint32_t>(), ctx->gate_offset.as<int32_t>(),
                                 ctx->n_two_qubit.as<int32_t>(), ctx->logical_seed.as<uint64_t>(),
                                 ctx->logical_row_offset.as<int32_t>(), batch->row_words, nq,
                                 ctx->log_word_offset.as<int64_t>(), ctx->log_nwords.as<int32_t>(),
                                 ctx->log_x.as<uint64_t>(), ctx->log_z.as<uint64_t>(), n,
                                 max_log_blocks, ctx->alt_stream) != hipSuccess) {
        return fail();
    }
    if (hipEventRecord(ctx->join, ctx->alt_stream) != hipSuccess ||
        hipStreamWaitEvent(stream, ctx->join, 0) != hipSuccess) {
        return fail();
    }
    timer.report("propagate");
    if (paulice_launch_count(ctx->n_two_qubit.as<int32_t>(), nq, ctx->readout_column.as<int32_t>(),
                             ctx->blocks.as<PauliceNoiseBlock>(), batch->n_blocks,
                             ctx->post_x.as<uint64_t>(), ctx->post_z.as<uint64_t>(),
                             ctx->post_word_offset.as<int64_t>(), ctx->post_nwords.as<int32_t>(),
                             ctx->post_row_offset.as<int32_t>(), ctx->log_x.as<uint64_t>(),
                             ctx->log_z.as<uint64_t>(), ctx->log_word_offset.as<int64_t>(),
                             ctx->log_nwords.as<int32_t>(), ctx->logical_row_offset.as<int32_t>(),
                             ctx->counts.as<int32_t>(), n, stream) != hipSuccess) {
        return fail();
    }
    timer.report("count");
    if (paulice_launch_reduce(ctx->counts.as<int32_t>(), ctx->blocks.as<PauliceNoiseBlock>(),
                              batch->n_blocks, n, ctx->rates.as<double>(), stream) != hipSuccess) {
        return fail();
    }

    if (hipMemcpyAsync(batch->out_rates, ctx->rates.as<double>(),
                       static_cast<size_t>(n) * sizeof(double), hipMemcpyDeviceToHost,
                       stream) != hipSuccess ||
        hipStreamSynchronize(stream) != hipSuccess) {
        return fail();
    }
    timer.report("reduce+dl");
    return 0;
}

int paulice_rocm_ler_score(PauliceEngineCtx *ctx, const PauliceLerBatch *batch) {
    if (ctx == nullptr || batch == nullptr || batch->n_shots <= 0 || batch->n_generators <= 0 ||
        batch->n_indices < 0 || batch->stride <= 0 || batch->stride > PAULICE_ROCM_LER_MAX_WORDS) {
        return -1;
    }
    if (hipSetDevice(ctx->device_id) != hipSuccess) {
        return -1;
    }

    const hipStream_t stream = ctx->stream;
    StageTimer timer(stream);
    // As for gamma: wait for the stream before returning on failure, since the
    // copies read caller memory and write `host_counts` below.
    const auto fail = [ctx]() {
        (void)hipStreamSynchronize(ctx->alt_stream);
        (void)hipStreamSynchronize(ctx->stream);
        return -1;
    };

    const size_t n_words =
        static_cast<size_t>(batch->n_generators) * static_cast<size_t>(batch->stride);
    const size_t n_shots = static_cast<size_t>(batch->n_shots);
    if (!upload(ctx->ler_gen_words, batch->gen_words, n_words, stream) ||
        !upload(ctx->ler_offsets, batch->offsets, n_shots + 1, stream) ||
        !upload(ctx->ler_indices, batch->indices, static_cast<size_t>(batch->n_indices), stream) ||
        !ctx->ler_counts.ensure(2 * sizeof(unsigned long long)) ||
        hipMemsetAsync(ctx->ler_counts.as<void>(), 0, 2 * sizeof(unsigned long long), stream) !=
            hipSuccess) {
        return fail();
    }
    timer.report("ler upload");

    unsigned long long *counters = ctx->ler_counts.as<unsigned long long>();
    unsigned long long host_counts[2] = {0, 0};
    if (paulice_launch_ler_score(ctx->ler_gen_words.as<uint64_t>(), batch->stride,
                                 batch->planes_end, batch->post_end,
                                 ctx->ler_offsets.as<uint32_t>(), ctx->ler_indices.as<uint32_t>(),
                                 batch->n_shots, counters, counters + 1, stream) != hipSuccess ||
        hipMemcpyAsync(host_counts, counters, sizeof(host_counts), hipMemcpyDeviceToHost,
                       stream) != hipSuccess ||
        hipStreamSynchronize(stream) != hipSuccess) {
        return fail();
    }
    timer.report("ler score");

    *batch->out_accepted = host_counts[0];
    *batch->out_logical = host_counts[1];
    return 0;
}

int paulice_rocm_ler_draw_parity(PauliceEngineCtx *ctx, const PauliceLerDrawParity *req) {
    if (ctx == nullptr || req == nullptr || req->n_generators == 0 || req->n_tables == 0 ||
        req->n_cdf == 0 || req->pos_stride == 0 || req->nshots == 0) {
        return -1;
    }
    if (hipSetDevice(ctx->device_id) != hipSuccess) {
        return -1;
    }
    const hipStream_t stream = ctx->stream;
    StageTimer timer(stream);
    const auto fail = [ctx]() {
        (void)hipStreamSynchronize(ctx->alt_stream);
        (void)hipStreamSynchronize(ctx->stream);
        ctx->draw_bits.release();
        return -1;
    };

    const size_t n_gen = req->n_generators;
    const size_t words = static_cast<size_t>(req->probe_words) * n_gen;
    const size_t positions = static_cast<size_t>(req->pos_stride) * n_gen;
    // One bitset per generator: the parity kernel runs a thread per generator.
    size_t bitset_words = 0;
    if (!upload(ctx->draw_cdf, req->cdf_all, req->n_cdf, stream) ||
        !upload(ctx->draw_table_offset, req->table_offset, req->n_tables, stream) ||
        !upload(ctx->draw_table_len, req->table_len, req->n_tables, stream) ||
        !upload(ctx->draw_table_klo, req->table_klo, req->n_tables, stream) ||
        !upload(ctx->draw_table_of, req->table_of, n_gen, stream) ||
        !ctx->draw_counts.ensure(n_gen * sizeof(uint32_t)) ||
        !ctx->draw_pos.ensure(positions * sizeof(uint32_t)) ||
        !ctx->draw_words.ensure(std::max<size_t>(words, 1) * sizeof(uint64_t)) ||
        !ctx->draw_after.ensure(n_gen * sizeof(uint64_t)) ||
        !ensure_draw_bitsets(ctx->draw_bits, req->nshots, n_gen, bitset_words, stream)) {
        return fail();
    }
    timer.report("ler upload");

    if (paulice_launch_ler_draw_parity(
            req->seed, req->n_generators, req->nshots, ctx->draw_cdf.as<double>(),
            ctx->draw_table_offset.as<uint32_t>(), ctx->draw_table_len.as<uint32_t>(),
            ctx->draw_table_klo.as<uint32_t>(), ctx->draw_table_of.as<uint32_t>(),
            req->probe_words, ctx->draw_words.as<uint64_t>(), ctx->draw_counts.as<uint32_t>(),
            ctx->draw_pos.as<uint32_t>(), req->pos_stride, ctx->draw_bits.as<uint64_t>(),
            static_cast<uint32_t>(bitset_words), ctx->draw_after.as<uint64_t>(),
            stream) != hipSuccess ||
        hipMemcpyAsync(req->out_counts, ctx->draw_counts.as<uint32_t>(), n_gen * sizeof(uint32_t),
                       hipMemcpyDeviceToHost, stream) != hipSuccess ||
        hipMemcpyAsync(req->out_pos, ctx->draw_pos.as<uint32_t>(), positions * sizeof(uint32_t),
                       hipMemcpyDeviceToHost, stream) != hipSuccess ||
        hipMemcpyAsync(req->out_after, ctx->draw_after.as<uint64_t>(), n_gen * sizeof(uint64_t),
                       hipMemcpyDeviceToHost, stream) != hipSuccess ||
        (words > 0 &&
         hipMemcpyAsync(req->out_words, ctx->draw_words.as<uint64_t>(), words * sizeof(uint64_t),
                        hipMemcpyDeviceToHost, stream) != hipSuccess) ||
        hipStreamSynchronize(stream) != hipSuccess) {
        return fail();
    }
    timer.report("ler draw");
    return 0;
}

int paulice_rocm_ler_transpose(PauliceEngineCtx *ctx, const PauliceLerTranspose *req) {
    if (ctx == nullptr || req == nullptr || req->n_generators == 0 ||
        req->n_generators > PAULICE_ROCM_LER_MAX_GENERATORS || req->n_shots == 0 ||
        req->pos_stride == 0) {
        return -1;
    }
    // The scatter writes every firing into `tr_indices`, which is sized from
    // the per-generator bound, so the counts must respect it.
    for (uint32_t g = 0; g < req->n_generators; ++g) {
        if (req->gen_counts[g] > req->pos_stride) {
            return -1;
        }
    }
    if (hipSetDevice(ctx->device_id) != hipSuccess) {
        return -1;
    }
    // Overwrites the buffers a resident sample lives in.
    ctx->sample_resident = false;
    const hipStream_t stream = ctx->stream;
    StageTimer timer(stream);
    const auto fail = [ctx]() {
        (void)hipStreamSynchronize(ctx->alt_stream);
        (void)hipStreamSynchronize(ctx->stream);
        return -1;
    };

    const size_t n_shots = req->n_shots;
    const size_t positions = static_cast<size_t>(req->pos_stride) * req->n_generators;
    if (!upload(ctx->draw_pos, req->pos, positions, stream) ||
        !upload(ctx->draw_counts, req->gen_counts, req->n_generators, stream) ||
        !ctx->tr_shot_counts.ensure(n_shots * sizeof(uint32_t)) ||
        !ctx->tr_chunk_sums.ensure(paulice_ler_transpose_chunk_sums(req->n_shots) *
                                   sizeof(uint32_t)) ||
        !ctx->tr_offsets.ensure((n_shots + 1) * sizeof(uint32_t)) ||
        !ctx->tr_cursor.ensure((n_shots + 1) * sizeof(uint32_t)) ||
        !ctx->tr_indices.ensure(std::max<size_t>(positions, 1) * sizeof(uint32_t)) ||
        !ctx->tr_nonempty.ensure(sizeof(unsigned long long)) ||
        hipMemsetAsync(ctx->tr_shot_counts.as<void>(), 0, n_shots * sizeof(uint32_t), stream) !=
            hipSuccess ||
        hipMemsetAsync(ctx->tr_nonempty.as<void>(), 0, sizeof(unsigned long long), stream) !=
            hipSuccess) {
        return fail();
    }
    timer.report("ler upload");

    unsigned long long nonempty = 0;
    if (paulice_launch_ler_transpose(
            ctx->draw_pos.as<uint32_t>(), req->pos_stride, ctx->draw_counts.as<uint32_t>(),
            req->n_generators, req->n_shots, ctx->tr_shot_counts.as<uint32_t>(),
            ctx->tr_chunk_sums.as<uint32_t>(), ctx->tr_offsets.as<uint32_t>(),
            ctx->tr_cursor.as<uint32_t>(), ctx->tr_indices.as<uint32_t>(),
            ctx->tr_nonempty.as<unsigned long long>(), stream) != hipSuccess ||
        hipMemcpyAsync(req->out_offsets, ctx->tr_offsets.as<uint32_t>(),
                       (n_shots + 1) * sizeof(uint32_t), hipMemcpyDeviceToHost,
                       stream) != hipSuccess ||
        hipMemcpyAsync(&nonempty, ctx->tr_nonempty.as<unsigned long long>(), sizeof(nonempty),
                       hipMemcpyDeviceToHost, stream) != hipSuccess ||
        hipStreamSynchronize(stream) != hipSuccess) {
        return fail();
    }
    const uint32_t total = req->out_offsets[n_shots];
    if (total > req->n_indices_capacity) {
        return -1;
    }
    if (total > 0 &&
        (hipMemcpyAsync(req->out_indices, ctx->tr_indices.as<uint32_t>(),
                        static_cast<size_t>(total) * sizeof(uint32_t), hipMemcpyDeviceToHost,
                        stream) != hipSuccess ||
         hipStreamSynchronize(stream) != hipSuccess)) {
        return fail();
    }
    timer.report("ler transpose");
    *req->out_nonempty = nonempty;
    return 0;
}

int paulice_rocm_ler_sample_score(PauliceEngineCtx *ctx, const PauliceLerSampleScore *req) {
    if (ctx == nullptr || req == nullptr || req->n_generators <= 0 ||
        req->n_generators > PAULICE_ROCM_LER_MAX_GENERATORS || req->n_shots <= 0 ||
        req->stride <= 0 || req->stride > PAULICE_ROCM_LER_MAX_WORDS || req->planes_end < 0 ||
        req->post_end < req->planes_end || req->post_end > req->stride || req->pos_stride == 0) {
        return -1;
    }
    const uint32_t n_gen32 = static_cast<uint32_t>(req->n_generators);
    const uint32_t n_shots32 = static_cast<uint32_t>(req->n_shots);
    const bool reuse = req->reuse_sample != 0;
    if (reuse && (!ctx->sample_resident || ctx->resident_shots != n_shots32 ||
                  ctx->resident_generators != n_gen32)) {
        return -1;
    }
    if (!reuse && (req->n_tables == 0 || req->n_cdf == 0)) {
        return -1;
    }
    if (hipSetDevice(ctx->device_id) != hipSuccess) {
        return -1;
    }
    const hipStream_t stream = ctx->stream;
    StageTimer timer(stream);
    // A draw that stops part way can leave bitsets dirty and the transpose
    // buffers half written, so neither is trusted after a failure.
    const auto fail = [ctx]() {
        (void)hipStreamSynchronize(ctx->alt_stream);
        (void)hipStreamSynchronize(ctx->stream);
        ctx->draw_bits.release();
        ctx->sample_resident = false;
        return -1;
    };

    const size_t n_gen = n_gen32;
    const size_t n_shots = n_shots32;
    const size_t n_words = n_gen * static_cast<size_t>(req->stride);
    const size_t positions = n_gen * static_cast<size_t>(req->pos_stride);
    // Accepted, logical, nonempty and overflow. A reusing dispatch clears only
    // the first two.
    if (!upload(ctx->ler_gen_words, req->gen_words, n_words, stream) ||
        !ctx->ler_counts.ensure(4 * sizeof(unsigned long long)) ||
        hipMemsetAsync(ctx->ler_counts.as<void>(), 0,
                       (reuse ? 2 : 4) * sizeof(unsigned long long), stream) != hipSuccess) {
        return fail();
    }
    unsigned long long *counters = ctx->ler_counts.as<unsigned long long>();

    if (!reuse) {
        ctx->sample_resident = false;
        if (!upload(ctx->draw_cdf, req->cdf_all, req->n_cdf, stream) ||
            !upload(ctx->draw_table_offset, req->table_offset, req->n_tables, stream) ||
            !upload(ctx->draw_table_len, req->table_len, req->n_tables, stream) ||
            !upload(ctx->draw_table_klo, req->table_klo, req->n_tables, stream) ||
            !upload(ctx->draw_table_of, req->table_of, n_gen, stream) ||
            !ctx->draw_pos.ensure(positions * sizeof(uint32_t)) ||
            !ctx->draw_counts.ensure(n_gen * sizeof(uint32_t)) ||
            !ctx->tr_shot_counts.ensure(n_shots * sizeof(uint32_t)) ||
            !ctx->tr_chunk_sums.ensure(paulice_ler_transpose_chunk_sums(n_shots32) *
                                       sizeof(uint32_t)) ||
            !ctx->tr_offsets.ensure((n_shots + 1) * sizeof(uint32_t)) ||
            !ctx->tr_cursor.ensure((n_shots + 1) * sizeof(uint32_t)) ||
            !ctx->tr_indices.ensure(positions * sizeof(uint32_t))) {
            return fail();
        }

        // The bitsets are the one allocation that scales with lanes times
        // shots, so the lane count is fitted to half of what the device has
        // free, counting the bitsets already held as free since `ensure`
        // replaces them. Too few lanes would serialise the draw, and then the
        // host is the better place for it.
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        if (hipMemGetInfo(&free_bytes, &total_bytes) != hipSuccess) {
            return fail();
        }
        constexpr size_t MIN_LANES = 4096;
        const size_t per_lane = ((n_shots + 63) / 64) * sizeof(uint64_t);
        const size_t budget = (free_bytes + ctx->draw_bits.capacity()) / 2;
        const size_t lanes = std::min(n_gen, budget / per_lane);
        if (lanes < std::min(n_gen, MIN_LANES)) {
            return fail();
        }
        size_t bitset_words = 0;
        if (!ensure_draw_bitsets(ctx->draw_bits, n_shots, lanes, bitset_words, stream) ||
            hipMemsetAsync(ctx->tr_shot_counts.as<void>(), 0, n_shots * sizeof(uint32_t),
                           stream) != hipSuccess) {
            return fail();
        }
        timer.report("ler upload");

        if (paulice_launch_ler_draw(req->seed, n_gen32, n_shots32, ctx->draw_cdf.as<double>(),
                                    ctx->draw_table_offset.as<uint32_t>(),
                                    ctx->draw_table_len.as<uint32_t>(),
                                    ctx->draw_table_klo.as<uint32_t>(),
                                    ctx->draw_table_of.as<uint32_t>(),
                                    ctx->draw_counts.as<uint32_t>(), ctx->draw_pos.as<uint32_t>(),
                                    req->pos_stride, ctx->draw_bits.as<uint64_t>(),
                                    static_cast<uint32_t>(bitset_words),
                                    static_cast<uint32_t>(lanes), counters + 3,
                                    stream) != hipSuccess) {
            return fail();
        }
        timer.report("ler draw");
        if (paulice_launch_ler_transpose(
                ctx->draw_pos.as<uint32_t>(), req->pos_stride, ctx->draw_counts.as<uint32_t>(),
                n_gen32, n_shots32, ctx->tr_shot_counts.as<uint32_t>(),
                ctx->tr_chunk_sums.as<uint32_t>(), ctx->tr_offsets.as<uint32_t>(),
                ctx->tr_cursor.as<uint32_t>(), ctx->tr_indices.as<uint32_t>(), counters + 2,
                stream) != hipSuccess) {
            return fail();
        }
        timer.report("ler transpose");
    } else {
        timer.report("ler upload");
    }

    unsigned long long host_counts[4] = {0, 0, 0, 0};
    if (paulice_launch_ler_score(ctx->ler_gen_words.as<uint64_t>(), req->stride, req->planes_end,
                                 req->post_end, ctx->tr_offsets.as<uint32_t>(),
                                 ctx->tr_indices.as<uint32_t>(), req->n_shots, counters,
                                 counters + 1, stream) != hipSuccess ||
        hipMemcpyAsync(host_counts, counters, sizeof(host_counts), hipMemcpyDeviceToHost,
                       stream) != hipSuccess ||
        hipStreamSynchronize(stream) != hipSuccess) {
        return fail();
    }
    timer.report("ler score");
    // `pos_stride` comes from the same tables the device drew from, so an
    // overflow means host and device disagree about a window; scoring the
    // short sample would be wrong.
    if (!reuse && host_counts[3] != 0) {
        return fail();
    }

    if (!reuse) {
        ctx->sample_resident = true;
        ctx->resident_shots = n_shots32;
        ctx->resident_generators = n_gen32;
        *req->out_nonempty = host_counts[2];
    }
    *req->out_accepted = host_counts[0];
    *req->out_logical = host_counts[1];
    return 0;
}
