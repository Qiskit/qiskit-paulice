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

use super::pauli_propagator::{Direction, PauliPropagator};
use super::sparse_pauli::SparsePauli;
use super::stabilizer_group::StabilizerGroup;
use super::utils::{mult_f2, nullspace};
use super::wire::Wire;
use rustiq_core::structures::{CliffordCircuit, PauliLike, PauliSet};
#[derive(Clone, Debug)]
/// A structure giving access to a the group of valid checks on a given subset of wires of a circuit
/// Elements of the group are accessible through a morphism F_2^n -> G
pub struct CheckGroup {
    pub generators: Vec<(SparsePauli, Vec<usize>)>,
}

impl CheckGroup {
    pub fn new(
        circuit: &CliffordCircuit,
        accessible_wires: &[Wire],
        paulis: &[u8],
        measured_qubits: &[usize],
        stabilizer_group: &StabilizerGroup,
    ) -> Self {
        let propagator = PauliPropagator::new(circuit);
        let mut props =
            propagator.propagate_paulis_from_wires(accessible_wires, paulis, Direction::Backward);
        let mut pset = PauliSet::new_empty(circuit.nqbits, measured_qubits.len());
        for (index, qbit) in measured_qubits.iter().enumerate() {
            pset.set_entry(index, *qbit, false, true);
        }
        pset.conjugate_with_circuit(&circuit.dagger());
        props.extend((0..measured_qubits.len()).map(|i| pset.get_as_vec_bool(i).1));
        let mut all_paulis = Vec::new();
        for wire in accessible_wires.iter() {
            for pauli in paulis {
                all_paulis.push((wire.clone(), *pauli));
            }
        }

        if !stabilizer_group.is_trivial() {
            let stab_nsp = stabilizer_group.get_nullspace();
            props = mult_f2(&props, &stab_nsp);
        };

        let nsp = nullspace(&props);
        let generators = nsp
            .into_iter()
            .map(|bv| {
                let mut pauli = SparsePauli::new();
                let mut virtual_zs = Vec::new();
                for (i, b) in bv.iter().enumerate() {
                    if *b {
                        if i < all_paulis.len() {
                            let (wire, p) = &all_paulis[i];
                            pauli.update(wire.clone(), *p);
                        } else {
                            virtual_zs.push(i - all_paulis.len());
                        }
                    }
                }
                (pauli, virtual_zs)
            })
            .collect();

        Self { generators }
    }

    pub fn get_check(&self, coordinates: &[bool]) -> (SparsePauli, Vec<usize>) {
        let mut check = SparsePauli::new();
        let mut virtual_zs = Vec::new();
        for (i, b) in coordinates.iter().enumerate() {
            if *b {
                check.mult_inplace(&self.generators[i].0);
                for q in self.generators[i].1.iter() {
                    if virtual_zs.contains(q) {
                        virtual_zs.remove(virtual_zs.iter().position(|x| *x == *q).unwrap());
                    } else {
                        virtual_zs.push(*q);
                    }
                }
            }
        }
        (check, virtual_zs)
    }

    pub fn get_dimension(&self) -> usize {
        self.generators.len()
    }
}

#[cfg(test)]
mod check_group_tests {
    use super::*;
    use rustiq_core::structures::CliffordGate;

    #[test]
    fn test_simple() {
        let mut circuit = CliffordCircuit::new(2);
        circuit.gates.push(CliffordGate::CNOT(0, 1));

        let wires = vec![Wire::Input(0), Wire::Input(1), Wire::GateWire(0, 1)];
        let paulis = vec![3u8];
        let check_group = CheckGroup::new(
            &circuit,
            &wires,
            &paulis,
            &[],
            &StabilizerGroup::new(Vec::new()),
        );
        assert_eq!(check_group.get_dimension(), 1);
    }
    #[test]
    fn test_measurement() {
        let mut circuit = CliffordCircuit::new(2);
        circuit.gates.push(CliffordGate::CNOT(0, 1));

        let wires = vec![Wire::Input(0), Wire::Input(1), Wire::GateWire(0, 1)];
        let paulis = vec![3u8];
        let check_group = CheckGroup::new(
            &circuit,
            &wires,
            &paulis,
            &[1],
            &StabilizerGroup::new(Vec::new()),
        );
        assert_eq!(check_group.get_dimension(), 2);
    }
    #[test]
    fn test_stabilizers() {
        let mut circuit = CliffordCircuit::new(2);
        circuit.gates.push(CliffordGate::CNOT(0, 1));

        let wires = vec![Wire::Input(0), Wire::Input(1), Wire::GateWire(0, 1)];
        let paulis = vec![3u8];
        let check_group = CheckGroup::new(
            &circuit,
            &wires,
            &paulis,
            &[],
            &StabilizerGroup::new(vec![vec![false, false, true, false]]),
        );
        assert_eq!(check_group.get_dimension(), 2);
    }
}
