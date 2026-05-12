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

//! ALAP scheduling implementation in order to compute effective noise strength for each wire
use super::wire::Wire;
use rustiq_core::structures::{CliffordCircuit, CliffordGate};
use std::collections::HashMap;

const SINGLE_DELAY: f64 = 50.;
const DOUBLE_DELAY: f64 = 60.;

fn _get_qbits(gate: &CliffordGate) -> Vec<usize> {
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

pub fn _get_duration(gate: &CliffordGate) -> f64 {
    match gate {
        CliffordGate::CNOT(_, _) => DOUBLE_DELAY,
        CliffordGate::CZ(_, _) => DOUBLE_DELAY,
        CliffordGate::H(_) => SINGLE_DELAY,
        CliffordGate::SqrtX(_) => SINGLE_DELAY,
        CliffordGate::SqrtXd(_) => SINGLE_DELAY,
        _ => 0.,
    }
}

pub fn get_alap_delays(circuit: &CliffordCircuit) -> HashMap<Wire, f64> {
    let mut idle_before: Vec<_> = vec![0.; circuit.nqbits];
    let mut wire_noise = HashMap::new();
    for (index, gate) in circuit.gates.iter().enumerate().rev() {
        let qbits = _get_qbits(gate);
        let time_end = qbits
            .iter()
            .map(|q| idle_before[*q])
            .max_by(|s1: &f64, s2| (*s1).total_cmp(s2))
            .unwrap();
        for (qindex, qbit) in qbits.iter().enumerate() {
            let wire = Wire::GateWire(index, qindex);
            wire_noise.insert(wire, time_end - idle_before[*qbit]);
        }
        let t0 = qbits
            .iter()
            .map(|q| idle_before[*q])
            .max_by(|s1, s2| (*s1).total_cmp(s2))
            .unwrap();
        let t1 = t0 + _get_duration(gate);
        for qbit in qbits.iter() {
            idle_before[*qbit] = t1;
        }
    }
    wire_noise.retain(|_, d| *d > 0.);
    wire_noise
}
