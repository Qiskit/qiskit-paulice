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

// The ROCm engine's entry points, which `engine_table.cpp` exports as a
// `PauliceEngine` table. The structs are qiskit-paulice's, from the vendored
// `paulice_engine.h`.

#ifndef PAULICE_ROCM_H
#define PAULICE_ROCM_H

#include "paulice_engine.h"

// This engine's own limits. qiskit-paulice does not know them: each entry
// declines work past them, and qiskit-paulice scores that work on the CPU.

/// Widest gamma batch, in qubits. The propagation kernel keeps a row's X and Z
/// words in registers and is specialised on 1 to 4 words per row.
#define PAULICE_ROCM_MAX_QUBITS 256

/// Widest LER generator row, in 64-bit words. One wavefront scores a shot with
/// the accumulator in registers, so this is a compile-time bound.
#define PAULICE_ROCM_LER_MAX_WORDS 1024

/// Most LER generators the transpose handles, since the generator is its
/// grid's y index.
#define PAULICE_ROCM_LER_MAX_GENERATORS 65535

#ifdef __cplusplus
extern "C" {
#endif

int paulice_rocm_device_count(void);
/// NULL unless `device_id` is a 64-wide-wavefront device.
PauliceEngineCtx *paulice_rocm_create(int device_id);
void paulice_rocm_destroy(PauliceEngineCtx *ctx);

int paulice_rocm_gamma_batch(PauliceEngineCtx *ctx, const PauliceGammaBatch *batch);
int paulice_rocm_ler_sample_score(PauliceEngineCtx *ctx, const PauliceLerSampleScore *req);
int paulice_rocm_ler_score(PauliceEngineCtx *ctx, const PauliceLerBatch *batch);
int paulice_rocm_ler_draw_parity(PauliceEngineCtx *ctx, const PauliceLerDrawParity *req);
int paulice_rocm_ler_transpose(PauliceEngineCtx *ctx, const PauliceLerTranspose *req);

/// The table handed to qiskit-paulice.
const PauliceEngine *paulice_rocm_engine(void);

#ifdef __cplusplus
}
#endif

#endif
