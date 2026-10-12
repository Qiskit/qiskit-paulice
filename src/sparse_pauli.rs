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

    /// Stable 64-bit hash of the Pauli support and types.
    ///
    /// Hashes the full wire identity (`Input` qubit, or gate index **and**
    /// intra-gate qubit), not just `Wire::gate_index()`, so supports that differ
    /// only in which qubit of a two-qubit gate they touch hash apart in
    /// practice. Being a 64-bit hash, collisions are still possible in
    /// principle: this is a bucketing/identity hint, not a proof of equality, so
    /// any caller that needs strict correctness must confirm equality
    /// separately rather than trusting the fingerprint alone.
    pub fn fingerprint(&self) -> u64 {
        let mut wires: Vec<_> = self.paulis.iter().collect();
        wires.sort_by(|a, b| a.0.cmp(b.0));
        let mut h = wires.len() as u64;
        for (w, p) in wires {
            match w {
                Wire::Input(q) => {
                    h = h.wrapping_mul(31).wrapping_add(1);
                    h = h.wrapping_mul(31).wrapping_add(*q as u64 + 1);
                }
                Wire::GateWire(gi, qi) => {
                    h = h.wrapping_mul(31).wrapping_add(2);
                    h = h.wrapping_mul(31).wrapping_add(*gi as u64 + 1);
                    h = h.wrapping_mul(31).wrapping_add(*qi as u64 + 1);
                }
            }
            h = h.wrapping_mul(31).wrapping_add(*p as u64);
        }
        h
    }
}

#[cfg(test)]
mod fingerprint_tests {
    use super::super::wire::Wire;
    use super::*;

    #[test]
    fn fingerprint_distinguishes_input_qubits() {
        let mut a = SparsePauli::new();
        a.update(Wire::Input(0), 1);
        let mut b = SparsePauli::new();
        b.update(Wire::Input(1), 1);
        assert_ne!(a.fingerprint(), b.fingerprint());
    }

    #[test]
    fn fingerprint_distinguishes_gate_qubit_index() {
        let mut a = SparsePauli::new();
        a.update(Wire::GateWire(3, 0), 3);
        let mut b = SparsePauli::new();
        b.update(Wire::GateWire(3, 1), 3);
        assert_ne!(a.fingerprint(), b.fingerprint());
    }

    #[test]
    fn fingerprint_distinguishes_mixed_supports_that_share_gate_index() {
        let mut a = SparsePauli::new();
        a.update(Wire::Input(0), 1);
        a.update(Wire::GateWire(3, 0), 3);
        let mut b = SparsePauli::new();
        b.update(Wire::Input(1), 1);
        b.update(Wire::GateWire(3, 1), 3);
        assert_ne!(a.fingerprint(), b.fingerprint());
    }

    #[test]
    fn fingerprint_is_order_independent() {
        let mut a = SparsePauli::new();
        a.update(Wire::Input(0), 1);
        a.update(Wire::GateWire(2, 1), 3);
        let mut b = SparsePauli::new();
        b.update(Wire::GateWire(2, 1), 3);
        b.update(Wire::Input(0), 1);
        assert_eq!(a.fingerprint(), b.fingerprint());
    }
}
