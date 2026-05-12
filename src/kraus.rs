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

/// Utils to manipulate sparse Kraus operators & invert them
use crate::sparse_pauli::SparsePauli;

pub struct Kraus {}

impl Kraus {
    /// Builds a (first order) approximation of a Krause operator from a list of generators (Pauli strings + coefficients)
    pub fn from_generators(generators: &Vec<(Vec<bool>, f64)>) -> Self {
        todo!()
    }
    pub fn inverse(&self) -> Self {
        todo!()
    }
}
