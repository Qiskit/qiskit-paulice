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

use crate::sparse_pauli::SparsePauli;

use super::check_decoder::CheckDecoder;
use super::check_evaluator::CheckEvaluator;
use super::check_group::CheckGroup;
use super::circuit_building::{add_check_no_allocate, fix_check_phase};
use super::coverage::is_covered_single;
use super::metric::PyMetric as Metric;
use super::noise_model::NoiseModel;
use super::pauli::string_to_pauli;
use super::stabilizer_group::StabilizerGroup;
use super::utils::get_all_wires;
use super::wire::Wire;

use rustiq_core::structures::CliffordCircuit;

use pyo3::prelude::*;

type LogicalData = (Vec<usize>, StabilizerGroup);
type CheckData = (Vec<usize>, Vec<Vec<usize>>);

type PyWire = (i32, usize);

fn _to_rust_wire(py_wire: PyWire) -> Wire {
    if py_wire.0 == -1 {
        Wire::Input(py_wire.1)
    } else {
        Wire::GateWire(py_wire.0 as usize, py_wire.1)
    }
}

fn _to_py_wire(wire: Wire) -> PyWire {
    match wire {
        Wire::Input(q) => (-1, q),
        Wire::GateWire(gi, qi) => (gi as i32, qi),
    }
}

#[pyclass(skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct CheckPicker {
    /// Target circuit
    circuit: CliffordCircuit,
    /// Available qubit measurements (to extend the check via virtual zs)
    /// and available input stabilizer group
    logical_data: LogicalData,
    /// Already existing check qubits & their final measurements components (if any)
    check_data: CheckData,
    /// CheckEvaluator structure to evaluate check performances
    check_evaluator: Option<CheckEvaluator>,
    /// Possible CheckGroup data structure
    check_group: Option<CheckGroup>,
    /// Possible CheckDecoder data structure
    check_decoder: Option<CheckDecoder>,
}

#[pymethods]
/// Public interface
impl CheckPicker {
    #[new]
    pub fn new(
        circuit: Vec<(String, Vec<usize>)>,
        nqubits: Option<usize>,
        measured_qubits: Option<Vec<usize>>,
        stabilizer_group: Option<Vec<String>>,
        check_qubits: Option<Vec<usize>>,
        virtual_zs: Option<Vec<Vec<usize>>>,
    ) -> Self {
        let mut circuit = CliffordCircuit::from_vec(circuit);
        if let Some(nqubits) = nqubits {
            circuit.nqbits = nqubits;
        }
        let measured_qubits = measured_qubits.unwrap_or_default();
        let stabilizer_group = StabilizerGroup::new(
            stabilizer_group
                .unwrap_or_default()
                .into_iter()
                .map(|s| string_to_pauli(&s, circuit.nqbits))
                .collect(),
        );
        let check_qubits = check_qubits.unwrap_or_default();
        let virtual_zs = virtual_zs.unwrap_or_default();
        assert!(
            measured_qubits.is_empty() || stabilizer_group.is_trivial(),
            "Cannot have both non trivial stabilizer group and set of measurements"
        );
        Self {
            circuit,
            logical_data: (measured_qubits, stabilizer_group),
            check_data: (check_qubits, virtual_zs),
            check_evaluator: None,
            check_group: None,
            check_decoder: None,
        }
    }

    /// Returns the set of wires corresponding to a global qbit index
    pub fn get_wires(&self, qbit_index: usize) -> Vec<PyWire> {
        get_all_wires(&self.circuit, qbit_index)
            .into_iter()
            .map(_to_py_wire)
            .collect()
    }

    /// Sets all the data required to evaluate check's performances
    /// - Noise models: a list of noise models
    /// - Metric: the metric used to evaluate the performance of the check
    /// - Ancilla: the ancilla qubit that will implement the check
    pub fn set_evaluation_data(
        &mut self,
        noise_models: Vec<NoiseModel>,
        metric: Metric,
        ancilla: usize,
    ) {
        self.check_evaluator = Some(CheckEvaluator::new(
            self.circuit.clone(),
            metric._data.clone(),
            noise_models.into_iter().map(|n| n.model).collect(),
            self.logical_data.1.clone(),
            self.logical_data.0.clone(),
            self.check_data.0.clone(),
            self.check_data.1.clone(),
            ancilla,
        ));
    }

    /// Sets the target set of wires to use as support for the check.
    ///
    /// `seed` is forwarded to the underlying `CheckDecoder` to control the
    /// middle-wire choice in `find_checks`. Pass `None` for OS-seeded randomness
    /// (the default behavior); pass `Some(s)` for a reproducible decoder anchor.
    #[pyo3(signature = (wires, paulis, seed=None))]
    pub fn set_support(&mut self, wires: Vec<PyWire>, paulis: Vec<u8>, seed: Option<u64>) {
        let wires: Vec<_> = wires.into_iter().map(_to_rust_wire).collect();
        self.check_group = Some(CheckGroup::new(
            &self.circuit,
            &wires,
            &paulis,
            &self.logical_data.0,
            &self.logical_data.1,
        ));
        self.check_decoder = Some(CheckDecoder::new(
            &self.circuit,
            &wires,
            &paulis,
            &self.logical_data.0,
            &self.logical_data.1,
            seed,
        ));
    }
    /// Computes the dimension of the underlying check group
    pub fn get_dimension(&self) -> usize {
        assert!(
            self.check_group.is_some(),
            "Please first set the check's support"
        );
        self.check_group.as_ref().unwrap().get_dimension()
    }

    /// Evaluates some check via the F_2^n => G morphism
    pub fn evaluate(&self, vec: Vec<bool>) -> f64 {
        assert!(
            self.check_group.is_some(),
            "Please first set the check's support"
        );
        let (check, vzs) = self.check_group.as_ref().unwrap().get_check(&vec);
        self.check_evaluator
            .as_ref()
            .unwrap()
            .evaluate(&check, &vzs)
    }

    /// Commits a check specified by its coordinates
    pub fn commit_check_bv(&self, vec: Vec<bool>) -> Option<Self> {
        let (check, vzs) = self.check_group.as_ref().unwrap().get_check(&vec);
        Some(self.commit_check(check, vzs))
    }

    /// Generates a few checks and their costs and commmits the best one.
    pub fn find_good_checks(&self) -> Option<(Self, f64)> {
        assert!(
            self.check_decoder.is_some(),
            "Please first set the check's support"
        );
        let checks = self.check_decoder.as_ref().unwrap().find_checks();
        let mut best_check = None;
        let mut best_cost = f64::MAX;
        for check in checks {
            let vzs = if self.logical_data.0.is_empty() {
                Vec::new()
            } else {
                self.check_evaluator.as_ref().unwrap().compute_vzs(&check)
            };
            let cost = self
                .check_evaluator
                .as_ref()
                .unwrap()
                .evaluate(&check, &vzs);
            if cost < best_cost {
                best_cost = cost;
                best_check = Some((check, vzs));
            }
        }
        if let Some((check, vzs)) = best_check {
            Some((self.commit_check(check, vzs), best_cost))
        } else {
            None
        }
    }

    /// Returns a python compatible description of the current circuit
    pub fn get_circuit(&self) -> Vec<(String, Vec<usize>)> {
        self.circuit
            .gates
            .iter()
            .map(|gate| gate.to_vec())
            .collect()
    }

    /// Returns the virtual CZs stored for each of the currentl checks
    pub fn get_virtual_zs(&self) -> Vec<Vec<usize>> {
        self.check_data.1.clone()
    }

    /// Returns the current energy (as given by the chosen Metric)
    pub fn get_current_energy(&self) -> f64 {
        self.check_evaluator.as_ref().unwrap().get_current_energy()
    }

    /// Makes a copy of the CheckPicker
    pub fn copy(&self) -> Self {
        self.clone()
    }

    /// Returns all the uncovered single qubit Paulis in the circuit.
    pub fn get_uncovered_paulis(&self) -> Vec<(PyWire, u8)> {
        let propagator = super::pauli_propagator::PauliPropagator::new(&self.circuit);
        let cumulants =
            propagator.get_check_cumulants(&self.check_data.0, &Some(&self.check_data.1));
        let mut uncovered = Vec::new();
        for qbit in 0..self.circuit.nqbits {
            let wires = get_all_wires(&self.circuit, qbit);
            for wire in wires.iter() {
                for pauli in 1..=3 {
                    if !is_covered_single(&cumulants, pauli, wire) {
                        uncovered.push((_to_py_wire(wire.clone()), pauli));
                    }
                }
            }
        }
        uncovered
    }
}

impl CheckPicker {
    /// Commits a check
    fn commit_check(&self, check: SparsePauli, vzs: Vec<usize>) -> Self {
        let ancilla = self.check_evaluator.as_ref().unwrap().get_ancilla();
        let mut checked_circuit = add_check_no_allocate(self.circuit.clone(), &check, ancilla);
        fix_check_phase(&mut checked_circuit, ancilla, &vzs);
        let mut logical_data = self.logical_data.clone();
        if !logical_data.0.is_empty() {
            logical_data.0.push(ancilla);
        }
        if !logical_data.1.is_trivial() {
            logical_data.1.add_ancilla(ancilla, checked_circuit.nqbits);
        }
        let mut check_data = self.check_data.clone();
        check_data.0.push(ancilla);
        check_data.1.push(vzs);

        Self {
            circuit: checked_circuit,
            logical_data,
            check_data,
            check_evaluator: None,
            check_group: None,
            check_decoder: None,
        }
    }
}
