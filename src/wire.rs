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

use rustiq_core::structures::{CliffordCircuit, CliffordGate};
use std::cmp::{Ord, Ordering};

/// Size of the dense wire-id range a circuit can produce. See `Wire::dense_id`.
pub fn wire_id_space(circuit: &CliffordCircuit) -> usize {
    circuit.nqbits + 2 * circuit.gates.len()
}

/// Struct representing a wire in a circuit
/// A wire is either an input wire (indexed by qbit) or a gate wire (indexed by gate index and qbit)
#[derive(PartialEq, Eq, Hash, Clone, Debug)]
pub enum Wire {
    Input(usize),
    GateWire(usize, usize),
}

impl PartialOrd for Wire {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Wire {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match (self, other) {
            (Wire::Input(_), Wire::GateWire(_, _)) => Ordering::Less,
            (Wire::GateWire(_, _), Wire::Input(_)) => Ordering::Greater,
            (Wire::Input(q1), Wire::Input(q2)) => q1.cmp(q2),
            (Wire::GateWire(g1, q1), Wire::GateWire(g2, q2)) => {
                if let Ordering::Equal = g1.cmp(g2) {
                    q1.cmp(q2)
                } else {
                    g1.cmp(g2)
                }
            }
        }
    }
}

impl Wire {
    /// Returns the gate index of the wire or -1 if the wire is an input wire
    pub fn gate_index(&self) -> i32 {
        match self {
            Self::Input(_) => -1,
            Self::GateWire(gi, _) => *gi as i32,
        }
    }
    /// Checks if the wire comes right after a 2-qubit gate
    pub fn is_two_qubit_gate(&self, circuit: &CliffordCircuit) -> bool {
        match self.gate_index() {
            -1 => false,
            _ => matches!(
                circuit.gates.get(self.gate_index() as usize).unwrap(),
                CliffordGate::CNOT(_, _) | CliffordGate::CZ(_, _)
            ),
        }
    }
    /// Dense index of this wire inside a circuit of `nqbits` qubits.
    ///
    /// Inputs take `[0, nqbits)`, and the (at most two) output wires of gate
    /// `g` take `nqbits + 2g` and `nqbits + 2g + 1`. Packing wires into a
    /// contiguous range lets the coverage tables index columns with an array
    /// lookup instead of hashing a `Wire`, which is what the inner loop of
    /// gamma scoring does for every generator term.
    pub fn dense_id(&self, nqbits: usize) -> usize {
        match self {
            Self::Input(q) => *q,
            Self::GateWire(gi, qi) => nqbits + 2 * gi + qi,
        }
    }

    /// Returns the global qbit index of the wire
    pub fn get_qbit(&self, circuit: &CliffordCircuit, offset: usize) -> usize {
        match self {
            Self::Input(q) => *q,
            Self::GateWire(gi, qi) => {
                let gate = circuit.gates[*gi + offset];
                match gate {
                    CliffordGate::CNOT(i, j) => {
                        if *qi == 0 {
                            i
                        } else {
                            j
                        }
                    }
                    CliffordGate::CZ(i, j) => {
                        if *qi == 0 {
                            i
                        } else {
                            j
                        }
                    }
                    CliffordGate::H(i) => i,
                    CliffordGate::S(i) => i,
                    CliffordGate::Sd(i) => i,
                    CliffordGate::SqrtX(i) => i,
                    CliffordGate::SqrtXd(i) => i,
                }
            }
        }
    }
}
