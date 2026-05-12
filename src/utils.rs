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
    match gate {
        CliffordGate::CNOT(i, j) => vec![*i, *j],
        CliffordGate::CZ(i, j) => vec![*i, *j],
        CliffordGate::H(i) => vec![*i],
        CliffordGate::S(i) => vec![*i],
        CliffordGate::Sd(i) => vec![*i],
        CliffordGate::SqrtX(i) => vec![*i],
        CliffordGate::SqrtXd(i) => vec![*i],
    }
}

pub fn get_wires(gate: &CliffordGate, gate_index: usize) -> Vec<Wire> {
    (0..get_qbits(gate).len())
        .map(|qi| Wire::GateWire(gate_index, qi))
        .collect()
}

pub fn get_last_wire(circuit: &CliffordCircuit, qubit: usize) -> Wire {
    for (gi, gate) in circuit.gates.iter().enumerate().rev() {
        if get_qbits(gate).contains(&qubit) {
            return Wire::GateWire(
                gi,
                get_qbits(gate).iter().position(|&q| q == qubit).unwrap(),
            );
        }
    }
    Wire::Input(qubit)
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
