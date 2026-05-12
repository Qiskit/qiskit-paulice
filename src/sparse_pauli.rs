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

use super::wire::Wire;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct SparsePauli {
    pub paulis: HashMap<Wire, u8>,
}

fn _prod_pauli(p1: u8, p2: u8) -> u8 {
    if p1 == p2 {
        return 0;
    }
    for i in 1..4 {
        if p1 != i && p2 != i {
            return i;
        }
    }
    panic!("This should never happen :thinking:")
}

impl Default for SparsePauli {
    fn default() -> Self {
        Self::new()
    }
}

impl SparsePauli {
    pub fn new() -> Self {
        Self {
            paulis: HashMap::new(),
        }
    }
    pub fn from_slice(data: Vec<(Wire, u8)>) -> Self {
        Self {
            paulis: data.into_iter().collect(),
        }
    }
    pub fn update(&mut self, wire: Wire, pauli: u8) {
        if pauli != 0 {
            self.paulis
                .entry(wire.clone())
                .and_modify(|other_pauli| *other_pauli = _prod_pauli(*other_pauli, pauli))
                .or_insert(pauli);
            if *self.paulis.get(&wire).unwrap() == 0u8 {
                self.paulis.remove(&wire);
            }
        }
    }
    pub fn mult(&self, other: &SparsePauli) -> SparsePauli {
        let mut result = self.clone();
        for (wire, pauli) in other.paulis.iter() {
            result.update(wire.clone(), *pauli);
        }
        result
    }
    pub fn mult_inplace(&mut self, other: &SparsePauli) {
        for (wire, pauli) in other.paulis.iter() {
            self.update(wire.clone(), *pauli);
        }
    }
}
