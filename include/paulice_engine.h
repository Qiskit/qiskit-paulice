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

// The scoring-engine ABI.
//
// An engine is a separately installed package that scores candidate checks on
// other hardware. It hands qiskit-paulice one `PauliceEngine` table, wrapped in
// a PyCapsule named `PAULICE_ENGINE_CAPSULE`; qiskit-paulice calls through it
// from native code, without the GIL. Every call may decline by returning
// non-zero, and qiskit-paulice then scores on the host.
//
// The contract is bit-identical results. The structs below fix the layout, and
// the version also fixes the semantics an engine must reproduce: the gate
// encoding, the noise-generator enumeration, the summation order of gamma, the
// packed LER rows, the Philox4x64-10 streams and the Poisson tables. Any change
// to those bumps `PAULICE_ENGINE_ABI_VERSION`, and qiskit-paulice refuses an
// engine built for another version.
//
// A gamma batch is a set of independent candidate checks. The engine propagates
// each candidate's cumulants, counts the generators that contribute to gamma,
// and returns one score per candidate. The LER entry points score one
// candidate's Monte Carlo sample.
//
// `tests/common/mod.rs` is a CPU engine written against this header, which
// `cargo test` holds to the conformance checks. It is the plainest statement of
// what each entry computes.

#ifndef PAULICE_ENGINE_H
#define PAULICE_ENGINE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Gate opcodes, matching `rustiq_core::structures::CliffordGate`.
#define PAULICE_GATE_CNOT 0
#define PAULICE_GATE_CZ 1
#define PAULICE_GATE_H 2
#define PAULICE_GATE_S 3
#define PAULICE_GATE_SD 4
#define PAULICE_GATE_SQRTX 5
#define PAULICE_GATE_SQRTXD 6

/// Noise-model block kinds the device path understands. Every generator inside
/// one block carries the same rate, which is what lets the device reproduce the
/// host's summation order exactly -- see `PauliceEngine::gamma_batch`.
#define PAULICE_NOISE_DEPOLARIZING 0
#define PAULICE_NOISE_READOUT 1

typedef struct PauliceNoiseBlock {
    int32_t kind;
    double rate;
} PauliceNoiseBlock;

/// One batch of candidates. Every pointer is caller-owned host memory, read
/// only for the duration of the call.
///
/// Per-candidate arrays are concatenated and sliced by the `*_offset` arrays,
/// which hold `n_candidates + 1` entries.
typedef struct PauliceGammaBatch {
    int32_t n_candidates;
    int32_t nqubits;

    /// Gates of each candidate's checked circuit, packed by `paulice_pack_gate`.
    const uint32_t *gates;
    const int32_t *gate_offset;
    /// Number of 2-qubit gates in each candidate, i.e. of recorded wire pairs.
    const int32_t *n_two_qubit;

    /// Z-support bitmasks of the rows to propagate, `row_words` words per row.
    /// Post-selected rows are the checks; logical rows are the measured qubits.
    const uint64_t *post_seed;
    const int32_t *post_row_offset;
    const uint64_t *logical_seed;
    const int32_t *logical_row_offset;
    /// Words per row seed mask, i.e. ceil(nqubits / 64).
    int32_t row_words;

    /// Column of each qubit's readout wire, or -1 when that wire is not
    /// recorded (its qubit's last gate is single-qubit). `nqubits` per candidate.
    const int32_t *readout_column;

    /// Noise-model blocks in emission order, shared by every candidate.
    const PauliceNoiseBlock *blocks;
    int32_t n_blocks;

    /// Output: one accumulated rate per candidate. The caller applies `exp`,
    /// so that the transcendental comes from one library rather than two --
    /// double addition is pinned down by IEEE-754, `exp` is not.
    double *out_rates;
} PauliceGammaBatch;

/// Packs a gate into the representation `gates` expects.
static inline uint32_t paulice_pack_gate(uint32_t opcode, uint32_t q0, uint32_t q1) {
    return (opcode << 24) | (q0 << 12) | q1;
}

/// Opaque per-device context holding the engine's streams and scratch buffers.
typedef struct PauliceEngineCtx PauliceEngineCtx;

/// Widest LER generator row, in 64-bit words. One wavefront scores a shot with
/// the accumulator in registers, so this is a compile-time bound.
#define PAULICE_LER_MAX_WORDS 1024

/// A host-drawn LER sample and the packed generator rows to score it against.
typedef struct PauliceLerBatch {
    /// `n_generators * stride` words, generator-major. A row is
    /// `[X plane | Z plane | post-selected signature | logical signature]`,
    /// split at `planes_end` and `post_end`.
    const uint64_t *gen_words;
    int32_t n_generators;
    int32_t stride;
    int32_t planes_end;
    int32_t post_end;

    /// Shot `s` fired generators `indices[offsets[s]..offsets[s + 1]]`;
    /// `offsets` has `n_shots + 1` entries.
    const uint32_t *offsets;
    const uint32_t *indices;
    int32_t n_shots;
    int32_t n_indices;

    /// Output: shots that survive post-selection, and how many of those carry a
    /// logical error.
    uint64_t *out_accepted;
    uint64_t *out_logical;
} PauliceLerBatch;


/// Most generators the transpose handles; the generator is its grid's y index.
#define PAULICE_LER_MAX_GENERATORS 65535

/// Count reported for a generator whose draw would exceed `pos_stride`. The
/// draw refuses rather than truncate, since a truncated draw would consume a
/// different number of stream words than the host.
#define PAULICE_LER_DRAW_OVERFLOW 0xFFFFFFFFu

/// The device draw of `sample_shot_positions`, reported step by step so a
/// test can check each stage against the host.
typedef struct PauliceLerDrawParity {
    /// Generator `g` draws from the stream keyed on `(seed, g)`.
    uint64_t seed;
    uint32_t n_generators;
    uint32_t nshots;

    /// The Poisson CDF tables, concatenated. Table `t` is
    /// `cdf_all[table_offset[t] .. + table_len[t]]` and its first entry is the
    /// count `table_klo[t]`. Generator `g` uses table `table_of[g]`.
    const double *cdf_all;
    uint32_t n_cdf;
    const uint32_t *table_offset;
    const uint32_t *table_len;
    const uint32_t *table_klo;
    uint32_t n_tables;
    const uint32_t *table_of;

    /// Raw stream words to report per generator, from a fresh stream.
    uint32_t probe_words;

    /// Output: `probe_words` words per generator; one count per generator
    /// (`PAULICE_LER_DRAW_OVERFLOW` if it did not fit); `pos_stride` positions
    /// per generator of which the first `out_counts[g]` are live, in draw
    /// order; and the next stream word after each generator's draw.
    uint64_t *out_words;
    uint32_t *out_counts;
    uint32_t *out_pos;
    uint32_t pos_stride;
    uint64_t *out_after;
} PauliceLerDrawParity;


/// The per-generator to per-shot transpose of host-supplied positions.
typedef struct PauliceLerTranspose {
    /// Generator `g` fired in shots `pos[g * pos_stride .. + gen_counts[g]]`,
    /// in any order.
    const uint32_t *pos;
    uint32_t pos_stride;
    const uint32_t *gen_counts;
    uint32_t n_generators;
    uint32_t n_shots;

    /// Output: `n_shots + 1` offsets, `offsets[n_shots]` generator indices
    /// (at most `n_indices_capacity`), and the number of shots in which
    /// anything fired.
    uint32_t *out_offsets;
    uint32_t *out_indices;
    uint32_t n_indices_capacity;
    uint64_t *out_nonempty;
} PauliceLerTranspose;


/// One candidate's LER, with the sample drawn and transposed on the device so
/// that none of it crosses the bus.
typedef struct PauliceLerSampleScore {
    /// The packed generator table, as in `PauliceLerBatch`.
    const uint64_t *gen_words;
    int32_t n_generators;
    int32_t stride;
    int32_t planes_end;
    int32_t post_end;
    int32_t n_shots;

    /// The draw, as in `PauliceLerDrawParity`.
    uint64_t seed;
    const double *cdf_all;
    uint32_t n_cdf;
    const uint32_t *table_offset;
    const uint32_t *table_len;
    const uint32_t *table_klo;
    uint32_t n_tables;
    const uint32_t *table_of;
    /// Room per generator for its draw: the tables' largest count plus one. A
    /// generator that would exceed it fails the whole dispatch.
    uint32_t pos_stride;

    /// Score the sample left in `ctx` by the last successful draw instead of
    /// drawing again. Only the caller knows whether that sample came from the
    /// same seed, shot count and rates; the draw and table inputs are then
    /// ignored and `out_nonempty` is not written.
    int32_t reuse_sample;

    uint64_t *out_accepted;
    uint64_t *out_logical;
    /// Shots in which anything fired.
    uint64_t *out_nonempty;
} PauliceLerSampleScore;


#define PAULICE_ENGINE_ABI_VERSION 1
#define PAULICE_ENGINE_CAPSULE "qiskit_paulice.engine"

/// The table an engine exports. Every entry is required. Each scoring entry
/// returns 0 on success and non-zero to decline, in which case its outputs are
/// unspecified and qiskit-paulice scores that work on the host.
///
/// The table and `name` must stay valid for the rest of the process. Calls on
/// one context never overlap, but successive calls may come from different
/// threads. qiskit-paulice opens at most one context and may never destroy it.
typedef struct PauliceEngine {
    /// `PAULICE_ENGINE_ABI_VERSION` as the engine was built.
    uint32_t abi_version;
    /// `sizeof(PauliceEngine)` as the engine was built.
    uint32_t size;
    /// Shown to users, e.g. "rocm".
    const char *name;

    /// Number of devices, or a negative error code.
    int (*device_count)(void);
    /// A context on `device_id`, or NULL if the engine cannot use that device.
    PauliceEngineCtx *(*create)(int device_id);
    void (*destroy)(PauliceEngineCtx *ctx);

    /// Scores a gamma batch into `out_rates`.
    int (*gamma_batch)(PauliceEngineCtx *ctx, const PauliceGammaBatch *batch);
    /// Draws, transposes and scores one candidate's LER sample. On failure
    /// nothing may be left resident to reuse.
    int (*ler_sample_score)(PauliceEngineCtx *ctx, const PauliceLerSampleScore *req);

    /// The stages of `ler_sample_score` on their own, for the conformance
    /// suite. A transpose leaves no sample resident to reuse.
    int (*ler_score)(PauliceEngineCtx *ctx, const PauliceLerBatch *batch);
    int (*ler_draw_parity)(PauliceEngineCtx *ctx, const PauliceLerDrawParity *req);
    int (*ler_transpose)(PauliceEngineCtx *ctx, const PauliceLerTranspose *req);
} PauliceEngine;

#ifdef __cplusplus
}
#endif

#endif // PAULICE_ENGINE_H
