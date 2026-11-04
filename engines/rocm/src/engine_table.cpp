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

#include "paulice_rocm.h"

namespace {

const PauliceEngine kEngine = {
    PAULICE_ENGINE_ABI_VERSION,
    sizeof(PauliceEngine),
    "rocm",
    paulice_rocm_device_count,
    paulice_rocm_create,
    paulice_rocm_destroy,
    paulice_rocm_gamma_batch,
    paulice_rocm_ler_sample_score,
    paulice_rocm_ler_score,
    paulice_rocm_ler_draw_parity,
    paulice_rocm_ler_transpose,
};

} // namespace

const PauliceEngine *paulice_rocm_engine(void) { return &kEngine; }
