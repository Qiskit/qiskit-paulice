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

use super::pauli::Pauli;
use super::sparse_pauli::SparsePauli;
use super::utils::{get_qbits, get_wires};
use super::wire::Wire;
use rustiq_core::structures::{CliffordCircuit, PauliLike, PauliSet};
fn get_pauli_set_cumulants_forward(
    circuit: &CliffordCircuit,
    pset: &mut PauliSet,
    restrict_to_2q_gates: bool,
) -> Vec<SparsePauli> {
    let mut cumulants = vec![SparsePauli::new(); pset.len()];
    for (gate_index, gate) in circuit.gates.iter().enumerate() {
        pset.conjugate_with_gate(gate);
        if gate.arity() < 2 && restrict_to_2q_gates {
            continue;
        }
        for (i, cumulant) in cumulants.iter_mut().enumerate() {
            for (qbit, wire) in get_qbits(&circuit.gates[gate_index])
                .into_iter()
                .zip(get_wires(&circuit.gates[gate_index], gate_index).into_iter())
            {
                let x_part = pset.get_entry(qbit, i);
                let z_part = pset.get_entry(qbit + circuit.nqbits, i);
                let pauli = match (x_part, z_part) {
                    (false, false) => 0u8,
                    (true, false) => 1u8,
                    (true, true) => 2u8,
                    (false, true) => 3u8,
                };
                cumulant.update(wire, pauli);
            }
        }
    }
    cumulants
}

fn get_pauli_set_cumulants_backward(
    circuit: &CliffordCircuit,
    pset: &mut PauliSet,
    restrict_to_2q_gates: bool,
) -> Vec<SparsePauli> {
    let mut cumulants: Vec<_> = (0..pset.len()).map(|_| SparsePauli::new()).collect();

    let mut gate_index = (circuit.gates.len() - 1) as i32;
    while gate_index >= -1 {
        if gate_index != -1 {
            if circuit.gates[gate_index as usize].arity() < 2 && restrict_to_2q_gates {
                pset.conjugate_with_gate(&circuit.gates[gate_index as usize].dagger());
                gate_index -= 1;
                continue;
            }
            for (i, cumulant) in cumulants.iter_mut().enumerate() {
                for (qbit, wire) in get_qbits(&circuit.gates[gate_index as usize])
                    .into_iter()
                    .zip(
                        get_wires(&circuit.gates[gate_index as usize], gate_index as usize)
                            .into_iter(),
                    )
                {
                    let x_part = pset.get_entry(qbit, i);
                    let z_part = pset.get_entry(qbit + circuit.nqbits, i);
                    let pauli = match (x_part, z_part) {
                        (false, false) => 0u8,
                        (true, false) => 1u8,
                        (true, true) => 2u8,
                        (false, true) => 3u8,
                    };
                    cumulant.update(wire, pauli);
                }
            }
            pset.conjugate_with_gate(&circuit.gates[gate_index as usize].dagger());
            gate_index -= 1;
            continue;
        }
        for (i, cumulant) in cumulants.iter_mut().enumerate() {
            for qbit in 0..circuit.nqbits {
                let x_part = pset.get_entry(qbit, i);
                let z_part = pset.get_entry(qbit + circuit.nqbits, i);
                let pauli = match (x_part, z_part) {
                    (false, false) => 0u8,
                    (true, false) => 1u8,
                    (true, true) => 2u8,
                    (false, true) => 3u8,
                };
                cumulant.update(Wire::Input(qbit), pauli);
            }
        }
        break;
    }
    cumulants
}

fn propagate_paulis_backward(circuit: &CliffordCircuit, wires: &[Wire], paulis: &[u8]) -> PauliSet {
    let max_index = wires.iter().map(|wire| wire.gate_index()).max();
    let mut propagators = PauliSet::new_empty(circuit.nqbits, paulis.len() * wires.len());
    if let Some(max_index) = max_index {
        let mut gate_index = max_index;
        while gate_index >= -1 {
            for (w_index, wire) in wires.iter().enumerate() {
                for (p_index, pauli) in paulis.iter().enumerate() {
                    if wire.gate_index() == gate_index {
                        let qbit = wire.get_qbit(circuit, 0);
                        match *pauli {
                            1 => propagators.set_entry(
                                w_index * paulis.len() + p_index,
                                qbit,
                                true,
                                false,
                            ),
                            2 => propagators.set_entry(
                                w_index * paulis.len() + p_index,
                                qbit,
                                true,
                                true,
                            ),
                            3 => propagators.set_entry(
                                w_index * paulis.len() + p_index,
                                qbit,
                                false,
                                true,
                            ),
                            _ => (),
                        }
                    }
                }
            }
            if gate_index > -1 {
                propagators.conjugate_with_gate(&circuit.gates[gate_index as usize].dagger());
            }
            gate_index -= 1;
        }
    }
    propagators
}

fn propagate_paulis_forward(circuit: &CliffordCircuit, wires: &[Wire], paulis: &[u8]) -> PauliSet {
    let min_index = wires.iter().map(|wire| wire.gate_index()).min();
    let mut propagators = PauliSet::new_empty(circuit.nqbits, paulis.len() * wires.len());
    if let Some(min_index) = min_index {
        for gate_index in min_index..circuit.gates.len() as i32 {
            if gate_index > -1 {
                propagators.conjugate_with_gate(&circuit.gates[gate_index as usize]);
            }
            for (w_index, wire) in wires.iter().enumerate() {
                for (p_index, pauli) in paulis.iter().enumerate() {
                    if wire.gate_index() == gate_index {
                        let qbit = wire.get_qbit(circuit, 0);
                        match *pauli {
                            1 => propagators.set_entry(
                                w_index * paulis.len() + p_index,
                                qbit,
                                true,
                                false,
                            ),
                            2 => propagators.set_entry(
                                w_index * paulis.len() + p_index,
                                qbit,
                                true,
                                true,
                            ),
                            3 => propagators.set_entry(
                                w_index * paulis.len() + p_index,
                                qbit,
                                false,
                                true,
                            ),
                            _ => (),
                        }
                    }
                }
            }
        }
    }
    propagators
}
pub enum Direction {
    Forward,
    Backward,
}

#[derive(Clone, Debug)]
pub struct PauliPropagator<'a> {
    circuit: &'a CliffordCircuit,
}

impl<'a> PauliPropagator<'a> {
    /// Creates a new Pauli propagator object from a circuit's reference
    pub fn new(circuit: &'a CliffordCircuit) -> Self {
        Self { circuit }
    }

    /// Propagate (either backward or forward) single qubit Pauli operators positioned on a list of wires.
    /// Returns a list of n-qubit Pauli operators.
    pub fn propagate_paulis_from_wires(
        &self,
        wire_list: &[Wire],
        paulis: &[u8],
        direction: Direction,
    ) -> Vec<Pauli> {
        let prop_pset = match direction {
            Direction::Backward => propagate_paulis_backward(self.circuit, wire_list, paulis),
            Direction::Forward => propagate_paulis_forward(self.circuit, wire_list, paulis),
        };
        (0..prop_pset.len())
            .map(|i| prop_pset.get_as_vec_bool(i).1)
            .collect()
    }

    /// Compute the cumulants of a list of Pauli operators (either backward or forward)
    /// The cumulants can be restricted to 2-qubit gates if needed (saves space)
    pub fn get_cumulants_from_paulis(
        &self,
        paulis: &[Pauli],
        direction: Direction,
        restrict_to_2q_gates: bool,
    ) -> Vec<SparsePauli> {
        let mut pset = PauliSet::new_empty(self.circuit.nqbits, paulis.len());
        for (i, pauli) in paulis.iter().enumerate() {
            for qbit in 0..pauli.len() / 2 {
                pset.set_entry(i, qbit, pauli[qbit], pauli[qbit + self.circuit.nqbits]);
            }
        }
        match direction {
            Direction::Backward => {
                get_pauli_set_cumulants_backward(self.circuit, &mut pset, restrict_to_2q_gates)
            }
            Direction::Forward => {
                get_pauli_set_cumulants_forward(self.circuit, &mut pset, restrict_to_2q_gates)
            }
        }
    }

    pub fn get_check_cumulants(
        &self,
        check_qubits: &[usize],
        virtual_zs: &Option<&Vec<Vec<usize>>>,
    ) -> Vec<SparsePauli> {
        let mut paulis = Vec::new();
        if let Some(vzs) = virtual_zs {
            if vzs.len() == check_qubits.len() {
                for zs in vzs.iter() {
                    let mut pauli = vec![false; self.circuit.nqbits * 2];
                    for &qbit in zs.iter() {
                        pauli[qbit + self.circuit.nqbits] = true;
                    }
                    paulis.push(pauli);
                }
            }
        }
        if paulis.is_empty() {
            for &qbit in check_qubits.iter() {
                let mut pauli = vec![false; self.circuit.nqbits * 2];
                pauli[qbit + self.circuit.nqbits] = true;
                paulis.push(pauli);
            }
        }

        self.get_cumulants_from_paulis(&paulis, Direction::Backward, true)
    }
}
