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

use super::circuit_building::add_check_no_allocate;
use super::coverage::Coverage;
use super::metric::Metric;
use super::noise_model::UNoiseModel;
use super::pauli::Pauli;
use super::sparse_pauli::SparsePauli;
use super::stabilizer_group::StabilizerGroup;
use rustiq_core::structures::{CliffordCircuit, PauliLike, PauliSet};

#[derive(Clone, Debug)]
pub struct CheckEvaluator {
    base_circuit: CliffordCircuit,
    metric: Metric,
    noise_models: Vec<UNoiseModel>,
    stabilizers: Vec<Pauli>,
    measured_qubits: Vec<usize>,
    current_checks: Vec<usize>,
    current_virtual_zs: Vec<Vec<usize>>,
    ancilla: usize,
}

impl CheckEvaluator {
    #![allow(clippy::too_many_arguments)]
    pub fn new(
        base_circuit: CliffordCircuit,
        metric: Metric,
        noise_models: Vec<UNoiseModel>,
        stabilizer_group: StabilizerGroup,
        measured_qubits: Vec<usize>,
        current_checks: Vec<usize>,
        current_virtual_zs: Vec<Vec<usize>>,
        ancilla: usize,
    ) -> Self {
        Self {
            base_circuit,
            metric,
            noise_models,
            stabilizers: stabilizer_group.generators,
            measured_qubits,
            current_checks,
            current_virtual_zs,
            ancilla,
        }
    }

    pub fn get_ancilla(&self) -> usize {
        self.ancilla
    }
    /// Evaluates some check (described by a SparsePauli object and a collection of virtual Zs at the end of the circuit)
    pub fn evaluate(&self, check: &SparsePauli, virtual_zs: &[usize]) -> f64 {
        let checked_circuit = add_check_no_allocate(self.base_circuit.clone(), check, self.ancilla);
        let mut all_check_qubits = self.current_checks.clone();
        all_check_qubits.push(self.ancilla);
        let mut all_virtual_zs = self.current_virtual_zs.clone();
        all_virtual_zs.push(virtual_zs.to_vec());
        let mut coverage = Coverage::new(&checked_circuit, &self.noise_models);
        coverage.set_check_cumulants(&all_check_qubits, &all_virtual_zs);
        coverage.set_logical_cumulants(&self.stabilizers, &self.measured_qubits);
        match self.metric {
            Metric::LogicalErrorRate(nshots) => coverage.approximate_psr_ler(nshots).1,
            Metric::Gamma => coverage.gamma_apx(),
            Metric::BalancedGamma => coverage.balanced_gamma_apx(),
        }
    }
    /// Utility method to infer the virtual Zs for a given check.
    /// Also checks that the check ends up diagonal when pulled to the end.
    pub fn compute_vzs(&self, check: &SparsePauli) -> Vec<usize> {
        let checked_circuit = add_check_no_allocate(self.base_circuit.clone(), check, self.ancilla);
        let mut pset = PauliSet::new_empty(checked_circuit.nqbits, 1);
        pset.set_entry(0, self.ancilla, false, true);
        pset.conjugate_with_circuit(&checked_circuit);
        let final_pauli = pset.get_as_vec_bool(0).1;
        if final_pauli
            .iter()
            .take(checked_circuit.nqbits)
            .enumerate()
            .filter(|(i, _)| *i != self.ancilla)
            .any(|(_, b)| *b)
        {
            panic!(
                "Unexpected X component in forward propagatated check in the presence of measured qubits."
            );
        }
        final_pauli
            .iter()
            .enumerate()
            .filter(|(i, b)| **b && *i != self.ancilla)
            .map(|(i, _)| i - checked_circuit.nqbits)
            .collect()
    }
    /// Returns the current energy of the base circuit
    pub fn get_current_energy(&self) -> f64 {
        let mut coverage = Coverage::new(&self.base_circuit, &self.noise_models);

        coverage.set_check_cumulants(&self.current_checks, &self.current_virtual_zs);
        coverage.set_logical_cumulants(&self.stabilizers, &self.measured_qubits);
        match self.metric {
            Metric::LogicalErrorRate(nshots) => coverage.approximate_psr_ler(nshots).1,
            Metric::Gamma => coverage.gamma_apx(),
            Metric::BalancedGamma => coverage.balanced_gamma_apx(),
        }
    }
}
