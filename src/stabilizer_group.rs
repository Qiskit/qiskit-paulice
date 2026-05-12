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

use super::utils::{nullspace, transpose};
use rustiq_core::routines::f2_linalg::row_echelon;

#[derive(Clone, Debug)]
/// Implementation of a stabilizer group specified by a set of generators
/// Phases are ignored
pub struct StabilizerGroup {
    pub generators: Vec<Vec<bool>>,
}

impl StabilizerGroup {
    pub fn new(generators: Vec<Vec<bool>>) -> Self {
        let mut group = Self { generators };
        group.normalize();
        group
    }

    /// Normalizes the group's generators
    pub fn normalize(&mut self) {
        let k = self.generators.len();
        if k > 0 {
            row_echelon(&mut self.generators, k);
        }
    }

    pub fn get_nullspace(&self) -> Vec<Vec<bool>> {
        transpose(&nullspace(&transpose(&self.generators)))
    }

    /// Checks if a Pauli is in the group
    pub fn contains(&self, pauli: &[bool]) -> bool {
        let mut generators = self.generators.clone();
        generators.push(pauli.to_vec());
        let k = generators.len();
        row_echelon(&mut generators, k - 1);
        let reduced_pauli = generators.pop().unwrap();
        reduced_pauli.iter().all(|b| !*b)
    }

    pub fn add_generator(&mut self, pauli: Vec<bool>) {
        self.generators.push(pauli);
        self.normalize();
    }

    pub fn add_ancilla(&mut self, qbit: usize, nqbits: usize) {
        let mut generator = vec![false; 2 * nqbits];
        generator[qbit + nqbits] = true;
        self.add_generator(generator);
    }

    pub fn is_trivial(&self) -> bool {
        self.generators.is_empty()
    }
}
