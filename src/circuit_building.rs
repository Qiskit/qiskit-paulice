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

use super::sparse_pauli::SparsePauli;
use super::utils::get_qbits;
use super::wire::Wire;
use rustiq_core::structures::{CliffordCircuit, CliffordGate, PauliLike, PauliSet};
use std::collections::{HashMap, HashSet};

fn _cp_gate(control: usize, target: usize, pauli: u8) -> Vec<CliffordGate> {
    let mut piece = Vec::new();
    match pauli {
        1 => {
            piece.push(CliffordGate::H(target));
            piece.push(CliffordGate::CZ(control, target));
            piece.push(CliffordGate::H(target));
        }
        2 => {
            piece.push(CliffordGate::SqrtX(target));
            piece.push(CliffordGate::CZ(control, target));
            piece.push(CliffordGate::SqrtXd(target));
        }
        3 => piece.push(CliffordGate::CZ(control, target)),
        _ => panic!("Identity Pauli in check :thinking: pauli = {}", pauli),
    }
    piece
}

#[derive(Debug)]
enum WireToTrack {
    CheckElement(Wire, u8),
}

/// Returns the new circuit with injected check & initial position of the ancilla
/// (which might differ from the original one, due to routing)
pub fn add_check_no_allocate(
    mut circuit: CliffordCircuit,
    check: &SparsePauli,
    ancilla: usize,
) -> CliffordCircuit {
    let mut wires_to_track: Vec<_> = check
        .paulis
        .iter()
        .map(|(w, p)| WireToTrack::CheckElement(w.clone(), *p))
        .collect();
    wires_to_track.sort_by_key(|to_track| match to_track {
        WireToTrack::CheckElement(w, _) => 2 * (w.gate_index() + 1) + 1,
        // WireToTrack::Rotation(w, _) => 2 * (w.gate_index() + 1),
    });

    let mut offset = 1;
    assert!(
        circuit
            .gates
            .iter()
            .all(|g| !get_qbits(g).contains(&ancilla)),
        "Ancilla {} is already used in the circuit",
        ancilla
    );
    circuit.gates.insert(0, CliffordGate::H(ancilla));
    let mut corresp: HashMap<i32, usize> = HashMap::new();
    for to_track in wires_to_track.iter() {
        match to_track {
            WireToTrack::CheckElement(wire, pauli) => {
                let wire_gate_index = wire.gate_index() + 1;
                corresp.entry(wire_gate_index - 1).or_insert(offset);
                let qbit = wire.get_qbit(&circuit, corresp[&(wire_gate_index - 1)]);
                for gate in _cp_gate(ancilla, qbit, *pauli) {
                    circuit
                        .gates
                        .insert(offset + wire_gate_index as usize, gate);
                    offset += 1;
                }
            }
        }
    }
    circuit.gates.push(CliffordGate::H(ancilla));

    circuit
}

fn _remap_gate(gate: &CliffordGate, mapping: &[usize]) -> CliffordGate {
    match gate {
        CliffordGate::CNOT(i, j) => CliffordGate::CNOT(mapping[*i], mapping[*j]),
        CliffordGate::CZ(i, j) => CliffordGate::CZ(mapping[*i], mapping[*j]),
        CliffordGate::H(i) => CliffordGate::H(mapping[*i]),
        CliffordGate::S(i) => CliffordGate::S(mapping[*i]),
        CliffordGate::Sd(i) => CliffordGate::Sd(mapping[*i]),
        CliffordGate::SqrtX(i) => CliffordGate::SqrtX(mapping[*i]),
        CliffordGate::SqrtXd(i) => CliffordGate::SqrtXd(mapping[*i]),
    }
}

struct Layer<'a> {
    nqubits: usize,
    singles: Vec<CliffordGate>,
    czs: HashSet<(usize, usize)>,
    layer_types: &'a [HashSet<(usize, usize)>],
}

impl<'a> Layer<'a> {
    fn new(nqubits: usize, layer_types: &'a [HashSet<(usize, usize)>]) -> Self {
        Self {
            nqubits,
            singles: Vec::new(),
            czs: HashSet::new(),
            layer_types,
        }
    }

    fn add_gate(&mut self, gate: CliffordGate) {
        if gate.arity() == 2 {
            assert!(matches!(gate, CliffordGate::CZ(_, _)));
            let qbits = get_qbits(&gate);
            let q0 = std::cmp::min(qbits[0], qbits[1]);
            let q1 = std::cmp::max(qbits[0], qbits[1]);
            self.czs.insert((q0, q1));
        } else {
            self.singles.push(gate);
        }
    }

    fn commmutes(&self, gate: &CliffordGate) -> bool {
        if gate.arity() == 1 {
            let qbit = get_qbits(gate)[0];
            !self.czs.iter().any(|(a, b)| a == &qbit || b == &qbit)
        } else {
            let qbits = get_qbits(gate);
            let q0 = std::cmp::min(qbits[0], qbits[1]);
            let q1 = std::cmp::max(qbits[0], qbits[1]);
            !self.czs.iter().any(|(a, b)| a == &q0 || b == &q1)
        }
    }

    fn can_insert(&self, gate: &CliffordGate) -> bool {
        if gate.arity() == 1 {
            let qbit = get_qbits(gate)[0];
            self.czs.iter().any(|(a, b)| a == &qbit || b == &qbit)
        } else {
            let qbits = get_qbits(gate);
            let q0 = std::cmp::min(qbits[0], qbits[1]);
            let q1 = std::cmp::max(qbits[0], qbits[1]);
            if self.czs.iter().any(|(a, b)| a == &q0 || b == &q1) {
                return false;
            }
            let mut czs_copy = self.czs.clone();
            czs_copy.insert((q0, q1));
            self.layer_types.iter().any(|t| czs_copy.is_subset(t))
        }
    }

    fn to_circuit(&self) -> (CliffordCircuit, usize) {
        let li = self.layer_index();
        let mut circuit = CliffordCircuit::new(self.nqubits);
        for (a, b) in self.czs.iter() {
            circuit.gates.push(CliffordGate::CZ(*a, *b));
        }
        circuit.gates.extend(self.singles.iter());
        (circuit, li)
    }

    fn layer_index(&self) -> usize {
        for (i, t) in self.layer_types.iter().enumerate() {
            if self.czs == *t {
                return i;
            }
        }
        panic!("Layer {:?} does not match any known layer type", self.czs);
    }
}

pub fn get_layered_circuit(
    circuit: CliffordCircuit,
    layer_types: &[HashSet<(usize, usize)>],
) -> Vec<(CliffordCircuit, Option<usize>)> {
    assert!(
        circuit
            .gates
            .iter()
            .all(|g| !matches!(g, CliffordGate::CNOT(_, _))),
        "Cannot layer a circuit with CNOTs"
    );
    let mut first_single_layer = CliffordCircuit::new(circuit.nqbits);
    let mut layers: Vec<Layer> = Vec::new();

    for gate in circuit.gates.iter() {
        if gate.arity() == 1 {
            let mut inserted = false;
            for layer in layers.iter_mut() {
                if layer.commmutes(gate) {
                    continue;
                }
                assert!(layer.can_insert(gate));
                layer.add_gate(*gate);
                inserted = true;
                break;
            }
            if !inserted {
                first_single_layer.gates.push(*gate);
            }
            continue;
        }
        assert!(gate.arity() == 2);
        let mut max_i = None;
        for (index, layer) in layers.iter().enumerate() {
            if layer.commmutes(gate) {
                max_i = Some(index);
                continue;
            }
        }
        if let Some(mut max_i) = max_i {
            loop {
                if layers[max_i].can_insert(gate) {
                    layers[max_i].add_gate(*gate);
                    break;
                }
                if max_i == 0 {
                    let mut new_layer = Layer::new(circuit.nqbits, layer_types);
                    new_layer.add_gate(*gate);
                    layers.insert(0, new_layer);
                    break;
                }
                max_i -= 1;
            }
        } else {
            let mut new_layer = Layer::new(circuit.nqbits, layer_types);
            new_layer.add_gate(*gate);
            layers.insert(0, new_layer);
        }
    }

    let mut finished_layers: Vec<_> = layers
        .into_iter()
        .rev()
        .map(|l| {
            let (subc, li) = l.to_circuit();
            (subc, Some(li))
        })
        .collect();
    finished_layers.insert(0, (first_single_layer, None));
    finished_layers
}

pub fn fix_check_phase(circuit: &mut CliffordCircuit, ancilla: usize, virtual_zs: &[usize]) {
    let mut prop = PauliSet::new_empty(circuit.nqbits, 1);
    prop.set_entry(0, ancilla, false, true);
    for q in virtual_zs.iter() {
        prop.set_entry(0, *q, false, true);
    }
    prop.conjugate_with_circuit(&circuit.dagger());
    let h_index = circuit
        .gates
        .iter()
        .position(|g| *g == CliffordGate::H(ancilla))
        .expect("No Hadamard on ancilla found")
        + 1;
    let (phase, vec) = prop.get_as_vec_bool(0);
    match (phase, vec[ancilla]) {
        (false, false) => {}
        (false, true) => {
            circuit.gates.insert(h_index, CliffordGate::Sd(ancilla));
        }
        (true, false) => {
            circuit.gates.insert(h_index, CliffordGate::S(ancilla));
            circuit.gates.insert(h_index, CliffordGate::S(ancilla));
        }
        (true, true) => {
            circuit.gates.insert(h_index, CliffordGate::S(ancilla));
        }
    }
}

#[cfg(test)]
mod check_building_tests {
    use super::super::wire::Wire;
    use super::*;

    #[test]
    fn simple_check() {
        let mut circuit = CliffordCircuit::new(3);
        circuit.gates.push(CliffordGate::H(0));
        circuit.gates.push(CliffordGate::CNOT(0, 1));

        let ancilla = 2;
        let mut check = SparsePauli::new();
        check.update(Wire::Input(0), 1);
        check.update(Wire::GateWire(0, 0), 3);
        check.update(Wire::GateWire(1, 0), 3);

        let circuit = add_check_no_allocate(circuit, &check, ancilla);
        println!("{:?}", circuit);
    }

    #[test]
    fn test_layering_1() {
        let mut layer_types = Vec::new();
        layer_types.push(HashSet::from([(0, 1), (2, 3)]));
        for i in 0..4 {
            layer_types.push(HashSet::from([(i, i + 4)]));
        }
        let mut circuit = CliffordCircuit::new(8);
        for q in 0..8 {
            circuit.gates.push(CliffordGate::H(q));
        }
        for pair in [(0, 1), (4, 0), (5, 1), (0, 1), (2, 3), (6, 2)] {
            circuit.gates.push(CliffordGate::CZ(pair.0, pair.1));
            for q in 0..8 {
                circuit.gates.push(CliffordGate::H(q));
            }
        }
        let layers = get_layered_circuit(circuit, &layer_types);
        println!("{:?}", layers);
    }
}
