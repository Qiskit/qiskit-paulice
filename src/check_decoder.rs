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
use super::utils::{mult_f2, nullspace, transpose};
use super::wire::Wire;
use super::decoding::information_set_decoding;
use itertools::Itertools;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::{StdRng, ThreadRng};
use rustiq_core::routines::f2_linalg::xor_vec;
use rustiq_core::structures::{CliffordCircuit, PauliLike, PauliSet};

fn _recombine_solution(solution: &[bool], wires: &[(Wire, u8)]) -> SparsePauli {
    let mut output = SparsePauli::new();
    for (b, (wire, pauli)) in solution.iter().zip(wires.iter()) {
        if *b {
            output.update(wire.clone(), *pauli);
        }
    }
    output
}
#[derive(Clone, Debug)]
pub struct CheckDecoder {
    b_matrix: Vec<Vec<bool>>,
    all_paulis: Vec<(Wire, u8)>,
    wires: Vec<Wire>,
    /// Optional seed for the middle-wire selection in `find_checks`. When `Some(s)`,
    /// `find_checks` uses a `StdRng` seeded with `s` so the picker is reproducible
    /// across process boundaries (subject to residual randomness inside
    /// `rustiq_core::information_set_decoding`, which uses an OS-seeded thread RNG).
    seed: Option<u64>,
}

impl CheckDecoder {
    pub fn new(
        circuit: &CliffordCircuit,
        accessible_wires: &[Wire],
        paulis: &[u8],
        measured_qubits: &[usize],
        stabilizer_group: &StabilizerGroup,
        seed: Option<u64>,
    ) -> Self {
        let propagator = PauliPropagator::new(circuit);
        let mut props =
            propagator.propagate_paulis_from_wires(accessible_wires, paulis, Direction::Backward);
        let mut all_paulis = Vec::new();
        for wire in accessible_wires.iter() {
            for pauli in paulis {
                all_paulis.push((wire.clone(), *pauli));
            }
        }

        if !stabilizer_group.is_trivial() {
            let stab_nsp = stabilizer_group.get_nullspace();
            props = mult_f2(&props, &stab_nsp);
        }
        if !measured_qubits.is_empty() {
            let mut pset = PauliSet::new_empty(circuit.nqbits, measured_qubits.len());
            for (index, qbit) in measured_qubits.iter().enumerate() {
                pset.set_entry(index, *qbit, false, true);
            }
            pset.conjugate_with_circuit(&circuit.dagger());
            let meas_props: Vec<_> = (0..measured_qubits.len())
                .map(|i| pset.get_as_vec_bool(i).1)
                .collect();
            let nsp = transpose(&nullspace(&transpose(&meas_props)));
            props = mult_f2(&props, &nsp);
        }
        Self {
            b_matrix: props,
            all_paulis,
            wires: accessible_wires.to_vec(),
            seed,
        }
    }

    pub fn find_checks(&self) -> Vec<SparsePauli> {
        // Build one RNG per `find_checks` call. With `Some(seed)` it's a deterministic
        // `StdRng` -- both the middle-wire pick AND the parity shuffles inside our
        // vendored `information_set_decoding` will draw from it, giving reproducible
        // output across processes. With `None` we keep the historical `ThreadRng`.
        let mut seeded_rng;
        let mut thread_rng_default;
        let rng: &mut dyn rand::RngCore = match self.seed {
            Some(s) => {
                seeded_rng = StdRng::seed_from_u64(s);
                &mut seeded_rng
            }
            None => {
                thread_rng_default = ThreadRng::default();
                &mut thread_rng_default
            }
        };
        let middle_idx = rng.random_range(0..self.wires.len() - 1);
        let middle_wire = &self.wires[middle_idx];
        let other_wire = &self.wires[self.wires.len() - 1];

        let actual_paulis: Vec<_> = self
            .all_paulis
            .iter()
            .filter(|(w, _)| w != middle_wire && w != other_wire)
            .cloned()
            .collect();
        let restricted_b_matrix: Vec<_> = self
            .b_matrix
            .iter()
            .zip(self.all_paulis.iter())
            .filter(|(_, a)| actual_paulis.contains(a))
            .map(|(vec, _)| vec.clone())
            .collect();

        let mut checks = Vec::new();
        for paulis in (1..=3).cartesian_product(1..=3) {
            let vec_size = self.b_matrix.first().unwrap().len();
            let mut target_vector = vec![false; vec_size];
            xor_vec(
                &mut target_vector,
                &self.b_matrix[self
                    .all_paulis
                    .iter()
                    .position(|(w, p)| *middle_wire == *w && paulis.0 == *p)
                    .unwrap()],
            );
            xor_vec(
                &mut target_vector,
                &self.b_matrix[self
                    .all_paulis
                    .iter()
                    .position(|(w, p)| *other_wire == *w && paulis.1 == *p)
                    .unwrap()],
            );
            let solution =
                information_set_decoding(&restricted_b_matrix, &target_vector, 1, true, rng);
            if let Some(solution) = solution {
                let mut solution = _recombine_solution(&solution, &actual_paulis);
                solution.update(middle_wire.clone(), paulis.0);
                solution.update(other_wire.clone(), paulis.1);
                checks.push(solution);
            }
        }

        checks
    }
}
