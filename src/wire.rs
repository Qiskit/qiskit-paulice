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
    2 * circuit.nqbits + 2 * circuit.gates.len()
}

/// Struct representing a wire in a circuit
/// A wire is an input wire (indexed by qbit), a gate wire (indexed by gate index and qbit), or
/// a readout wire (indexed by qbit).
///
/// A readout wire stands for the measurement of a qubit at the end of the circuit, in whatever
/// basis that is. An operator read out through a qubit holds a Z on its readout wire and the
/// readout error is an X there, so the error flips exactly the operators read out through it.
#[derive(PartialEq, Eq, Hash, Clone, Debug)]
pub enum Wire {
    Input(usize),
    GateWire(usize, usize),
    Readout(usize),
}

impl PartialOrd for Wire {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Wire {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Circuit order: inputs, then gates, then readouts.
        let key = |wire: &Wire| match wire {
            Wire::Input(q) => (0, *q, 0),
            Wire::GateWire(g, q) => (1, *g, *q),
            Wire::Readout(q) => (2, *q, 0),
        };
        key(self).cmp(&key(other))
    }
}

impl Wire {
    /// Returns the gate index of the wire or -1 if the wire is not a gate wire
    pub fn gate_index(&self) -> i32 {
        match self {
            Self::Input(_) | Self::Readout(_) => -1,
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
    /// Inputs take `[0, nqbits)`, readouts `[nqbits, 2 * nqbits)`, and the (at
    /// most two) output wires of gate `g` take `2 * nqbits + 2g` and
    /// `2 * nqbits + 2g + 1`. Packing wires into a
    /// contiguous range lets the coverage tables index columns with an array
    /// lookup instead of hashing a `Wire`, which is what the inner loop of
    /// gamma scoring does for every generator term.
    pub fn dense_id(&self, nqbits: usize) -> usize {
        match self {
            Self::Input(q) => *q,
            Self::Readout(q) => nqbits + q,
            Self::GateWire(gi, qi) => 2 * nqbits + 2 * gi + qi,
        }
    }

    /// Returns the global qbit index of the wire
    pub fn get_qbit(&self, circuit: &CliffordCircuit, offset: usize) -> usize {
        match self {
            Self::Input(q) | Self::Readout(q) => *q,
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Every wire of a circuit has its own dense id, inside `wire_id_space`.
    #[test]
    fn dense_ids_are_distinct_and_in_range() {
        let mut circuit = CliffordCircuit::new(3);
        circuit.gates.push(CliffordGate::H(0));
        circuit.gates.push(CliffordGate::CZ(0, 2));
        let nqbits = circuit.nqbits;
        let mut wires: Vec<Wire> = (0..nqbits).map(Wire::Input).collect();
        wires.extend((0..nqbits).map(Wire::Readout));
        for gate in 0..circuit.gates.len() {
            wires.extend([Wire::GateWire(gate, 0), Wire::GateWire(gate, 1)]);
        }
        let mut ids: Vec<usize> = wires.iter().map(|w| w.dense_id(nqbits)).collect();
        ids.sort_unstable();
        assert_eq!(ids, (0..wire_id_space(&circuit)).collect::<Vec<_>>());
    }
}
