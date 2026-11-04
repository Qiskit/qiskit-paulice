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

// `qiskit_paulice_rocm._engine`: hands the engine table to Python as a
// capsule. Built against the limited API, so one wheel serves every CPython.

#ifndef Py_LIMITED_API
#define Py_LIMITED_API 0x030A0000
#endif
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "paulice_rocm.h"

static PyObject *engine_capsule(PyObject *self, PyObject *args) {
    (void)self;
    (void)args;
    return PyCapsule_New((void *)paulice_rocm_engine(), PAULICE_ENGINE_CAPSULE, NULL);
}

static PyMethodDef methods[] = {
    {"engine_capsule", engine_capsule, METH_NOARGS,
     "Return the PauliceEngine table as a capsule named \"" PAULICE_ENGINE_CAPSULE "\"."},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "_engine", NULL, -1, methods, NULL, NULL, NULL, NULL,
};

PyMODINIT_FUNC PyInit__engine(void) { return PyModule_Create(&module); }
