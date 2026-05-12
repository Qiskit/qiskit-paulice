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

use super::circuit_building::get_layered_circuit;
use super::scheduling::get_alap_delays;
use super::sparse_pauli::SparsePauli;
use super::utils::get_last_wire;
use super::wire::Wire;
use rustiq_core::structures::{CliffordCircuit, CliffordGate};
use std::collections::{HashMap, HashSet};
use std::fmt::Display;

use pyo3::prelude::*;

pub type NoiseGenerator = (SparsePauli, f64);
pub type LayerDescription = HashMap<Vec<(usize, usize)>, Vec<(String, f64)>>;
pub type GateDescription = HashMap<(usize, usize), Vec<((u8, u8), f64)>>;

pub trait NoiseModelLike {
    fn get_generators(&self, circuit: &CliffordCircuit) -> (Vec<NoiseGenerator>, CliffordCircuit);
}

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

/// A noise model that applies uniform depolarizing noise after each 2-qubit gate, independently on each qubit.
/// The depolarizing probability is specified by the parameter `depol_p`.
/// The corresponding error probability of each individual Pauli error (X, Y, or Z) is `depol_p / 3`.
#[derive(Clone, Debug)]
pub struct UniformDepolarizing {
    depol_p: f64,
}
impl UniformDepolarizing {
    pub fn new(depol_p: f64) -> Self {
        Self {
            depol_p: 5. * depol_p / 4.,
        }
    }
}

impl Default for UniformDepolarizing {
    fn default() -> Self {
        Self { depol_p: 8e-4 }
    }
}

impl NoiseModelLike for UniformDepolarizing {
    fn get_generators(&self, circuit: &CliffordCircuit) -> (Vec<NoiseGenerator>, CliffordCircuit) {
        let mut generators = Vec::new();
        let rate = -1. / 4. * (1. - 4. * self.depol_p / 15.).ln();

        for (index, gate) in circuit.gates.iter().enumerate() {
            if gate.arity() == 2 {
                for p1 in 0..=3u8 {
                    for p2 in 0..=3u8 {
                        if p1 != 0 || p2 != 0 {
                            let mut pauli = SparsePauli::new();
                            pauli.update(Wire::GateWire(index, 0), p1);
                            pauli.update(Wire::GateWire(index, 1), p2);
                            generators.push((pauli, rate));
                        }
                    }
                }
            }
        }
        (generators, circuit.clone())
    }
}
/// A noise model that applies a specified set of Pauli generators after each 2-qubit gate.
/// The generators are specified in a dictionary mapping (qbit1, qbit2) tuples to lists of (pauli_pair, rate) tuples.
/// Here, pauli_pair is a tuple (p1, p2) where p1 and p2 are in {0, 1, 2, 3} representing I, X, Y, Z respectively.
/// For example, to apply an X error on the first qubit and a Z error on the second qubit with rate 0.01 after a CZ gate between qubits 0 and 1, you would include the entry:
/// (0, 1): [((1, 3), 0.01)]
/// in the dictionary.
#[derive(Clone, Debug, Default)]
pub struct GateWiseNoiseModel {
    gate_models: GateDescription,
}
impl GateWiseNoiseModel {
    pub fn new(gate_models: GateDescription) -> Self {
        Self { gate_models }
    }
}

impl NoiseModelLike for GateWiseNoiseModel {
    fn get_generators(&self, circuit: &CliffordCircuit) -> (Vec<NoiseGenerator>, CliffordCircuit) {
        let mut generators = Vec::new();
        for (index, gate) in circuit.gates.iter().enumerate() {
            if gate.arity() == 2 {
                let qbits = _get_qbits(gate);
                let k = (qbits[0], qbits[1]);
                let loc_generators = self.gate_models.get(&k);
                if let Some(loc_generators) = loc_generators {
                    for (pauli_pair, rate) in loc_generators.iter() {
                        let mut pauli = SparsePauli::new();
                        let wire0 = Wire::GateWire(index, 0);
                        let wire1 = Wire::GateWire(index, 1);
                        if pauli_pair.0 != 0 {
                            pauli.update(wire0, pauli_pair.0);
                        }
                        if pauli_pair.1 != 0 {
                            pauli.update(wire1, pauli_pair.1);
                        }
                        generators.push((pauli, *rate));
                    }
                }
            }
        }
        (generators, circuit.clone())
    }
}

/// A noise model that applies a specified set of Pauli generators *before* each layer of gates acting on a specific set of pairs of qubits.
/// The generators are specified in a dictionary mapping lists of (qbit1, qbit2) tuples to lists of (pauli_list, rate) tuples.
/// Here, pauli_list is a list of length equal to the number of qubits in the system, where each entry is in {0, 1, 2, 3} representing I, X, Y, Z respectively.
/// For example, to apply an X error on qubit 0 and an independent Z error on qubit 1 with respective rates 0.01 and 0.02 before a layer of gates acting on qubits 0 and 1, you would include the entry:
/// [(0, 1)]: [([1, 0], 0.01), ([0, 3], 0.02)]
/// in the dictionary.
#[derive(Clone, Debug, Default)]
pub struct LayeredNoiseModel {
    layer_models: LayerDescription,
}
impl LayeredNoiseModel {
    pub fn new(layer_models: LayerDescription) -> Self {
        Self { layer_models }
    }
}

impl NoiseModelLike for LayeredNoiseModel {
    fn get_generators(&self, circuit: &CliffordCircuit) -> (Vec<NoiseGenerator>, CliffordCircuit) {
        let layer_types = self
            .layer_models
            .keys()
            .map(|d| HashSet::from_iter(d.iter().cloned()))
            .collect::<Vec<_>>();
        let layers = get_layered_circuit(circuit.clone(), &layer_types);
        let mut new_circuit = CliffordCircuit::new(circuit.nqbits);
        let mut last_wires: Vec<_> = (0..circuit.nqbits).map(Wire::Input).collect();
        let mut generators = Vec::new();
        for (layer, layer_index) in layers {
            if let Some(i) = layer_index {
                let mut layer_key: Vec<(usize, usize)> = layer_types[i].iter().cloned().collect();
                layer_key.sort();
                if let Some(loc_generators) = self.layer_models.get(&layer_key) {
                    for (generator, rate) in loc_generators.iter() {
                        let mut spauli = SparsePauli::new();
                        for (wire, pauli) in last_wires.iter().zip(generator.chars()) {
                            match pauli {
                                'X' => spauli.update(wire.clone(), 1),
                                'Y' => spauli.update(wire.clone(), 2),
                                'Z' => spauli.update(wire.clone(), 3),
                                _ => (),
                            }
                        }
                        generators.push((spauli, *rate));
                    }
                }
            }
            for gate in layer.gates.into_iter() {
                let qbits = _get_qbits(&gate);
                let new_wires: Vec<_> = qbits
                    .iter()
                    .map(|&q| {
                        Wire::GateWire(
                            new_circuit.gates.len(),
                            qbits.iter().position(|&x| x == q).unwrap(),
                        )
                    })
                    .collect();
                for (q, new_wire) in qbits.iter().zip(new_wires.iter()) {
                    last_wires[*q] = new_wire.clone();
                }
                new_circuit.gates.push(gate);
            }
        }
        (generators, new_circuit)
    }
}
impl Display for LayeredNoiseModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "LayeredNoiseModel(")?;
        for (layer, noise) in self.layer_models.iter() {
            writeln!(f, "\tlayer: {:?}", layer)?;
            writeln!(f, "\t  ->  {:?}\n", noise)?;
        }
        write!(f, ")")
    }
}

#[derive(Clone, Debug)]
pub struct Idling {
    decay_rate: f64,
}

impl Idling {
    pub fn new(decay_rate: f64) -> Self {
        Self { decay_rate }
    }
}

impl Default for Idling {
    fn default() -> Self {
        Self { decay_rate: 1e5 }
    }
}

impl NoiseModelLike for Idling {
    fn get_generators(&self, circuit: &CliffordCircuit) -> (Vec<NoiseGenerator>, CliffordCircuit) {
        let delays = get_alap_delays(circuit);
        let mut generators = Vec::new();

        for (index, gate) in circuit.gates.iter().enumerate() {
            if gate.arity() == 2 {
                for output_index in 0..2 {
                    let wire = Wire::GateWire(index, output_index);
                    let delay = delays.get(&wire).unwrap_or(&0.);
                    if *delay > 0. {
                        let wire_rate = -1. / 4.
                            * (1. - 4. * (1. - (-delay / self.decay_rate).exp()) / 3.).ln();
                        for p in 1u8..=3 {
                            let mut pauli = SparsePauli::new();
                            pauli.update(wire.clone(), p);
                            generators.push((pauli.clone(), wire_rate));
                        }
                    }
                }
            }
        }
        (generators, circuit.clone())
    }
}

#[derive(Clone, Debug)]
pub struct Readout {
    error_rate: f64,
}

impl Readout {
    pub fn new(error_rate: f64) -> Self {
        Self { error_rate }
    }
}

impl Default for Readout {
    fn default() -> Self {
        Self { error_rate: 1e-2 }
    }
}

impl NoiseModelLike for Readout {
    fn get_generators(&self, circuit: &CliffordCircuit) -> (Vec<NoiseGenerator>, CliffordCircuit) {
        let mut generators = Vec::new();
        let rate = -1. / 2. * (1. - 2. * self.error_rate).ln();
        for qbit in 0..circuit.nqbits {
            let wire = get_last_wire(circuit, qbit);
            let mut pauli = SparsePauli::new();
            pauli.update(wire.clone(), 1);
            generators.push((pauli.clone(), rate));
        }

        (generators, circuit.clone())
    }
}
/// A wrapper enum to simplify interfacing with Python.
#[derive(Clone, Debug)]
pub enum UNoiseModel {
    UniformDepolarizing(UniformDepolarizing),
    GateWise(GateWiseNoiseModel),
    Layered(LayeredNoiseModel),
    Readout(Readout),
    Idling(Idling),
}

impl NoiseModelLike for UNoiseModel {
    fn get_generators(&self, circuit: &CliffordCircuit) -> (Vec<NoiseGenerator>, CliffordCircuit) {
        match self {
            UNoiseModel::UniformDepolarizing(m) => m.get_generators(circuit),
            UNoiseModel::GateWise(m) => m.get_generators(circuit),
            UNoiseModel::Layered(m) => m.get_generators(circuit),
            UNoiseModel::Readout(m) => m.get_generators(circuit),
            UNoiseModel::Idling(m) => m.get_generators(circuit),
        }
    }
}

/// The actual python binded interface (can't pybind enums)
#[pyclass]
#[derive(Clone, Debug)]
pub struct NoiseModel {
    pub model: UNoiseModel,
}

#[pymethods]
impl NoiseModel {
    /// Adds a single-qubit depolarizing channel after each 2-qubit gate in the circuit.
    /// The depolarizing probability is specified by the parameter `proba`.
    /// The corresponding error probability of each individual Pauli error (X, Y, or Z) is `proba / 3`.
    #[staticmethod]
    pub fn uniform_depolarizing(proba: f64) -> Self {
        Self {
            model: UNoiseModel::UniformDepolarizing(UniformDepolarizing::new(proba)),
        }
    }
    /// Adds a custom channel after each 2-qubit gate in the circuit.
    /// Channels are specified by a mapping from (qbit1, qbit2) tuples to lists of (pauli_pair, rate) tuples.
    #[staticmethod]
    pub fn gate_wise(gate_models: GateDescription) -> Self {
        Self {
            model: UNoiseModel::GateWise(GateWiseNoiseModel::new(gate_models)),
        }
    }
    ///Adds a layer of noise generators in front of each entangling layer in the circuit.
    ///Layers are specified by a mapping from lists of (qbit1, qbit2) tuples to lists of (pauli_list, rate) tuples.
    #[staticmethod]
    pub fn layered(layer_models: LayerDescription) -> Self {
        Self {
            model: UNoiseModel::Layered(LayeredNoiseModel::new(layer_models)),
        }
    }
    /// Adds single X error generators on each output qubit of the circuit, with the specified error rate.
    #[staticmethod]
    pub fn readout(error_rate: f64) -> Self {
        Self {
            model: UNoiseModel::Readout(Readout::new(error_rate)),
        }
    }
    /// Adds a single qubit depolarizing channel on each internal wire of the circuit with
    /// a strength depending on the wire's duration in an ALAP scheduling.
    /// The total error probability is given by `1 - exp(-t / decay_rate)`, where t is the wire's duration.
    #[staticmethod]
    pub fn idling(decay_rate: f64) -> Self {
        Self {
            model: UNoiseModel::Idling(Idling::new(decay_rate)),
        }
    }
}
