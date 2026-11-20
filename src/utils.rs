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
use rustiq_core::routines::f2_linalg::rowop;
use rustiq_core::structures::{CliffordCircuit, CliffordGate};

pub fn get_qbits(gate: &CliffordGate) -> Vec<usize> {
    let (qbits, arity) = gate_qbits(gate);
    qbits[..arity].to_vec()
}

/// The qubits a gate acts on, and how many -- the allocation-free form of
/// `get_qbits`. Walks over a circuit's gates run this once per gate, and a
/// `Vec` each would cost more than the work being done.
pub fn gate_qbits(gate: &CliffordGate) -> ([usize; 2], usize) {
    match gate {
        CliffordGate::CNOT(i, j) => ([*i, *j], 2),
        CliffordGate::CZ(i, j) => ([*i, *j], 2),
        CliffordGate::H(i) => ([*i, 0], 1),
        CliffordGate::S(i) => ([*i, 0], 1),
        CliffordGate::Sd(i) => ([*i, 0], 1),
        CliffordGate::SqrtX(i) => ([*i, 0], 1),
        CliffordGate::SqrtXd(i) => ([*i, 0], 1),
    }
}

pub fn get_wires(gate: &CliffordGate, gate_index: usize) -> Vec<Wire> {
    (0..get_qbits(gate).len())
        .map(|qi| Wire::GateWire(gate_index, qi))
        .collect()
}

/// Where each qubit's readout error enters the coverage tables: one
/// `(wire, pauli)` per qubit, in qubit order.
///
/// A readout error is an X on the qubit just before it is measured. The
/// cumulant tables only record the wires of 2-qubit gates and the input
/// wires, so the error is pulled back through the single-qubit gates that end
/// the qubit's line, onto the last recorded wire: the qubit's last 2-qubit
/// gate, or its input if it has none. Each of those gates maps a Pauli to a
/// Pauli, so the result is the X, Y or Z (encoded 1, 2, 3) that the trailing
/// gates turn back into X. It commutes with a propagated operator on that wire
/// exactly when the X does at the measurement.
pub fn readout_terms(circuit: &CliffordCircuit) -> Vec<(Wire, u8)> {
    let mut out: Vec<(Wire, u8)> = (0..circuit.nqbits).map(|q| (Wire::Input(q), 1)).collect();
    let mut placed = vec![false; circuit.nqbits];
    for (gi, gate) in circuit.gates.iter().enumerate().rev() {
        let (qbits, arity) = gate_qbits(gate);
        if arity == 2 {
            for (qi, q) in qbits.iter().enumerate() {
                if !placed[*q] {
                    out[*q].0 = Wire::GateWire(gi, qi);
                    placed[*q] = true;
                }
            }
        } else if !placed[qbits[0]] {
            let pauli = &mut out[qbits[0]].1;
            *pauli = pull_back_through_1q(gate, *pauli);
        }
    }
    out
}

/// The Pauli (1, 2, 3 for X, Y, Z) that single-qubit `gate` maps to `pauli`,
/// up to sign. A gate and its inverse permute the Paulis the same way.
fn pull_back_through_1q(gate: &CliffordGate, pauli: u8) -> u8 {
    let fixed = match gate {
        CliffordGate::H(_) => 2,
        CliffordGate::S(_) | CliffordGate::Sd(_) => 3,
        CliffordGate::SqrtX(_) | CliffordGate::SqrtXd(_) => 1,
        CliffordGate::CNOT(_, _) | CliffordGate::CZ(_, _) => {
            unreachable!("only single-qubit gates are pulled through")
        }
    };
    // Each gate fixes one Pauli and swaps the other two; 1 + 2 + 3 = 6.
    if pauli == fixed {
        pauli
    } else {
        6 - fixed - pauli
    }
}
pub fn nullspace(matrix: &[Vec<bool>]) -> Vec<Vec<bool>> {
    let mut matrix = matrix.to_vec();
    let mut witness = vec![vec![false; matrix.len()]; matrix.len()];
    for (i, row) in witness.iter_mut().enumerate() {
        row[i] = true;
    }
    let mut rank = 0;
    for i in 0..matrix[0].len() {
        let mut pivot = None;
        for (j, row) in matrix.iter().enumerate().skip(rank) {
            if row[i] {
                pivot = Some(j);
                break;
            }
        }
        if let Some(pivot) = pivot {
            if pivot != rank {
                matrix.swap(pivot, rank);
                witness.swap(pivot, rank);
            }
            for j in rank + 1..matrix.len() {
                if matrix[j][i] && j != rank {
                    rowop(&mut matrix, rank, j);
                    rowop(&mut witness, rank, j);
                }
            }
            rank += 1;
        }
    }
    witness.drain(rank..).collect()
}

#[allow(clippy::ptr_arg)]
pub fn mult_f2(left: &Vec<Vec<bool>>, right: &Vec<Vec<bool>>) -> Vec<Vec<bool>> {
    let mut result = vec![vec![false; right[0].len()]; left.len()];
    for i in 0..left.len() {
        for j in 0..right[0].len() {
            for (k, right_row) in right.iter().enumerate() {
                result[i][j] ^= left[i][k] && right_row[j];
            }
        }
    }
    result
}
#[allow(clippy::ptr_arg)]
pub fn transpose(matrix: &Vec<Vec<bool>>) -> Vec<Vec<bool>> {
    let mut result = vec![vec![false; matrix.len()]; matrix[0].len()];
    for i in 0..matrix.len() {
        for (j, res_row) in result.iter_mut().enumerate() {
            res_row[i] = matrix[i][j];
        }
    }
    result
}

pub fn get_all_wires(circuit: &CliffordCircuit, qbit: usize) -> Vec<Wire> {
    let mut wires = vec![Wire::Input(qbit)];
    for (i, gate) in circuit.gates.iter().enumerate() {
        if gate.arity() < 2 {
            continue;
        }
        if get_qbits(gate).contains(&qbit) {
            let qindex = get_qbits(gate).iter().position(|&q| q == qbit).unwrap();
            wires.push(Wire::GateWire(i, qindex));
        }
    }
    wires
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The pulled-back Pauli depends on the order of the trailing gates, and a
    /// qubit with no 2-qubit gate is placed on its input wire.
    #[test]
    fn readout_terms_pull_x_back_through_trailing_gates() {
        let mut circuit = CliffordCircuit::new(4);
        circuit.gates.push(CliffordGate::H(1));
        circuit.gates.push(CliffordGate::CZ(0, 1));
        circuit.gates.push(CliffordGate::CNOT(1, 2));
        // Qubit 0 ends in H then S, qubit 1 in S then H, qubit 3 in SqrtX then H.
        circuit.gates.push(CliffordGate::H(0));
        circuit.gates.push(CliffordGate::S(0));
        circuit.gates.push(CliffordGate::S(1));
        circuit.gates.push(CliffordGate::H(1));
        circuit.gates.push(CliffordGate::SqrtX(3));
        circuit.gates.push(CliffordGate::H(3));
        assert_eq!(
            readout_terms(&circuit),
            vec![
                (Wire::GateWire(1, 0), 2),
                (Wire::GateWire(2, 0), 3),
                (Wire::GateWire(2, 1), 1),
                (Wire::Input(3), 2),
            ]
        );
    }
}
