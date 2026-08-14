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

use super::cumulant_table::CumulantTable;
use super::noise_model::{NoiseGenerator, NoiseModelLike, UNoiseModel};
use super::pauli::Pauli;
use super::pauli_propagator::{Direction, PauliPropagator};
use super::sparse_pauli::SparsePauli;
use super::wire::Wire;

use rand::Rng;
use rand::distr::Distribution;
use rand::rngs::ThreadRng;
use rand_distr::Poisson;
use rayon::prelude::*;
use rustiq_core::structures::CliffordCircuit;

use std::collections::{BTreeSet, HashMap};
use std::sync::Mutex;

fn proba_from_rate(rate: f64) -> f64 {
    1. - (1. + (-2. * rate).exp()) / 2.
}

/// Legacy per-row representation of `CumulantTable::syndrome`'s single-row
/// question: does any wire of `error` anticommute with the row. Kept for the
/// `get_uncovered_paulis` path and as the reference in equivalence tests.
fn is_covered_single_cumulant(cumulant: &SparsePauli, error: &SparsePauli) -> bool {
    for (w, p) in error.paulis.iter() {
        if *cumulant.paulis.get(w).unwrap_or(p) != *p {
            return true;
        }
    }
    false
}

/// Legacy twin of `CumulantTable::syndrome` (reference implementation).
pub fn get_syndrome_legacy(cumulants: &[SparsePauli], error: &SparsePauli) -> Vec<bool> {
    cumulants
        .iter()
        .map(|c| is_covered_single_cumulant(c, error))
        .collect()
}

/// Legacy twin of `CumulantTable::covered_parity_any` (reference implementation).
pub fn is_covered_legacy(cumulants: &[SparsePauli], error: &SparsePauli) -> bool {
    for cumulant in cumulants.iter() {
        if error
            .paulis
            .iter()
            .filter(|(w, p)| *cumulant.paulis.get(w).unwrap_or(*p) != **p)
            .count()
            % 2
            == 1
        {
            return true;
        }
    }
    false
}

pub fn is_covered_single(cumulants: &[SparsePauli], pauli: u8, wire: &Wire) -> bool {
    cumulants
        .iter()
        .any(|c| *c.paulis.get(wire).unwrap_or(&pauli) != pauli)
}

pub struct Coverage<'a> {
    circuit: &'a CliffordCircuit,
    noise_models: &'a Vec<UNoiseModel>,
    logical_cumulants: CumulantTable,
    post_selected_cumulants: CumulantTable,
}

impl<'a> Coverage<'a> {
    pub fn new(circuit: &'a CliffordCircuit, noise_models: &'a Vec<UNoiseModel>) -> Self {
        Self {
            circuit,
            noise_models,
            logical_cumulants: CumulantTable::default(),
            post_selected_cumulants: CumulantTable::default(),
        }
    }

    fn get_generator_errors(&self) -> Vec<NoiseGenerator> {
        let mut noise_generators = Vec::new();
        let mut circuit = self.circuit.clone();
        for noise_model in self.noise_models {
            let (new_generators, new_circuit) = noise_model.get_generators(&circuit);
            noise_generators.extend(new_generators);
            circuit = new_circuit;
        }
        noise_generators
    }
    /// Computes & stores the backcumulants of a collection of checks specified
    /// by some measured qubits and some virtual CZs gates
    pub fn set_check_cumulants(&mut self, check_qubits: &[usize], virtual_zs: &[Vec<usize>]) {
        let propagator = PauliPropagator::new(self.circuit);
        let as_paulis: Vec<_> = check_qubits
            .iter()
            .zip(virtual_zs.iter())
            .map(|(q, vzs)| {
                let mut pauli = vec![false; 2 * self.circuit.nqbits];
                pauli[*q + self.circuit.nqbits] = true;
                for oq in vzs.iter() {
                    pauli[*oq + self.circuit.nqbits] = true;
                }
                pauli
            })
            .collect();
        self.post_selected_cumulants.extend(
            propagator.get_cumulant_table_from_paulis(&as_paulis, Direction::Backward, true),
        );
    }
    /// Computes & stores the cumulants of a collection of logical operators
    /// Those can either be initial stabilizers (that will be forward-propagated)
    /// or measured qubits (we will backpropagate Z operators for each of them)
    pub fn set_logical_cumulants(&mut self, stabilizers: &[Pauli], measured_qubits: &[usize]) {
        let propagator = PauliPropagator::new(self.circuit);
        self.logical_cumulants.extend(propagator.get_cumulant_table_from_paulis(
            stabilizers,
            Direction::Forward,
            true,
        ));
        let as_paulis: Vec<_> = measured_qubits
            .iter()
            .map(|q| {
                let mut pauli = vec![false; 2 * self.circuit.nqbits];
                pauli[*q + self.circuit.nqbits] = true;
                pauli
            })
            .collect();
        self.logical_cumulants.extend(propagator.get_cumulant_table_from_paulis(
            &as_paulis,
            Direction::Backward,
            true,
        ));
    }
    pub fn balanced_gamma_apx(&self) -> f64 {
        let error_generators = self.get_generator_errors();
        let mut gammas: HashMap<Vec<bool>, Vec<f64>> = HashMap::new();
        for (generator, rate) in error_generators.iter() {
            let syndrome = self.post_selected_cumulants.syndrome(generator);
            gammas.entry(syndrome).or_default().push(*rate);
        }
        let nclasses = gammas.len() as f64;
        // let gammas: Vec<f64> = gammas.into_values().map(|r| (4. * r).exp()).collect();
        return nclasses;
        // 1. / (gammas.into_iter().map(|g| 1. / g).sum::<f64>() / nclasses)
    }
    pub fn balanced_gamma_apx_old(&self, nshots: usize) -> f64 {
        let error_generators = self.get_generator_errors();

        let reverse_bins: Vec<_> = (0..nshots).map(|_| Mutex::new(Vec::new())).collect();
        error_generators
            .clone()
            .into_par_iter()
            .enumerate()
            .for_each(|(index, (_, e))| {
                let lambda = (nshots as f64) * proba_from_rate(e);
                let trng = &mut ThreadRng::default();
                let poisson = Poisson::new(lambda).unwrap();
                let num_successes = poisson.sample(trng) as usize;
                let mut pos = BTreeSet::new();
                while pos.len() < num_successes {
                    pos.insert(trng.random_range(0..nshots));
                }
                for elem in pos.into_iter() {
                    reverse_bins[elem].lock().unwrap().push(index);
                }
            });
        let reverse_bins_buckets: Vec<Mutex<Vec<Vec<usize>>>> = (0..rayon::current_num_threads())
            .map(|_| Mutex::new(Vec::new()))
            .collect();
        reverse_bins.into_par_iter().for_each(|b| {
            reverse_bins_buckets[rayon::current_thread_index().unwrap()]
                .lock()
                .unwrap()
                .push(b.lock().unwrap().clone());
        });
        let measured_cumulants = self.post_selected_cumulants.clone();

        let gammas_threads: Vec<_> = (0..rayon::current_num_threads())
            .map(|_| Mutex::new(HashMap::<Vec<bool>, usize>::new()))
            .collect();
        reverse_bins_buckets.into_par_iter().for_each(|bins| {
            for bin in bins.lock().unwrap().iter() {
                let mut error = SparsePauli::new();
                for index in bin {
                    error.mult_inplace(&error_generators[*index].0);
                }
                let syndrome = measured_cumulants.syndrome(&error);
                *gammas_threads[rayon::current_thread_index().unwrap_or_default()]
                    .lock()
                    .unwrap()
                    .entry(syndrome.clone())
                    .or_insert(0) += 1;
            }
        });
        let mut gammas = HashMap::<Vec<bool>, usize>::new();
        for gt in gammas_threads {
            let gt = gt.lock().unwrap();
            for (k, v) in gt.iter() {
                *gammas.entry(k.clone()).or_insert(0) += *v;
            }
        }

        // println!("Number of classes: {}", gammas.len());
        // println!("nshots: {}", nshots);
        let nclasses = gammas.len() as f64;
        gammas.into_values().sum::<usize>() as f64 / nclasses / nshots as f64
    }

    pub fn gamma_apx(&self) -> f64 {
        let t = std::time::Instant::now();
        let error_generators = self.get_generator_errors();
        crate::bench_timing::record("metric/generators", t);

        let t = std::time::Instant::now();
        // Coverage flags are computed in parallel but collected in generator
        // order (indexed collect preserves order), and the rates are then
        // summed sequentially in that order. This keeps the result independent
        // of thread count and scheduling — the legacy per-thread accumulators
        // summed in a scheduling-dependent order, which made gamma jitter by
        // ulps across runs and could flip ties in check selection.
        let uncovered: Vec<bool> = error_generators
            .par_iter()
            .map(|(generator, _)| {
                !self.post_selected_cumulants.covered_parity_any(generator)
                    && self.logical_cumulants.covered_parity_any(generator)
            })
            .collect();
        let mut acc = 0.;
        for ((_, rate), flag) in error_generators.iter().zip(uncovered.iter()) {
            if *flag {
                acc += rate;
            }
        }
        let out = (2. * acc).exp();
        crate::bench_timing::record("metric/score_generators", t);
        out
    }

    pub fn approximate_psr_ler(&self, nshots: usize) -> (f64, f64) {
        let t = std::time::Instant::now();
        let error_generators = self.get_generator_errors();
        crate::bench_timing::record("ler/generators", t);

        let t = std::time::Instant::now();
        let reverse_bins: Vec<_> = (0..nshots).map(|_| Mutex::new(Vec::new())).collect();
        error_generators
            .clone()
            .into_par_iter()
            .enumerate()
            .for_each(|(index, (_, e))| {
                let lambda = (nshots as f64) * proba_from_rate(e);
                let trng = &mut ThreadRng::default();
                let poisson = Poisson::new(lambda).unwrap();
                let num_successes = poisson.sample(trng) as usize;
                let mut pos = BTreeSet::new();
                while pos.len() < num_successes {
                    pos.insert(trng.random_range(0..nshots));
                }
                for elem in pos.into_iter() {
                    reverse_bins[elem].lock().unwrap().push(index);
                }
            });
        crate::bench_timing::record("ler/sample", t);
        let t = std::time::Instant::now();
        let reverse_bins_buckets: Vec<Mutex<Vec<Vec<usize>>>> = (0..rayon::current_num_threads())
            .map(|_| Mutex::new(Vec::new()))
            .collect();
        reverse_bins.into_par_iter().for_each(|b| {
            if !b.lock().unwrap().is_empty() {
                reverse_bins_buckets[rayon::current_thread_index().unwrap()]
                    .lock()
                    .unwrap()
                    .push(b.lock().unwrap().clone());
            }
        });
        crate::bench_timing::record("ler/redistribute", t);
        let accepted_errors: Vec<_> = (0..rayon::current_num_threads())
            .map(|_| Mutex::new(0))
            .collect();
        let accepted_logical_errors: Vec<_> = (0..rayon::current_num_threads())
            .map(|_| Mutex::new(0))
            .collect();

        let cumulants = self.logical_cumulants.clone();
        let measured_cumulants = self.post_selected_cumulants.clone();

        let t = std::time::Instant::now();
        let relevant_indices_len = reverse_bins_buckets
            .iter()
            .map(|b| b.lock().unwrap().len())
            .sum::<usize>();
        reverse_bins_buckets.into_par_iter().for_each(|bins| {
            let mut loc_a_e = 0;
            let mut loc_a_l_e = 0;
            for bin in bins.lock().unwrap().iter() {
                let mut error = SparsePauli::new();
                for index in bin {
                    error.mult_inplace(&error_generators[*index].0);
                }
                if error.paulis.is_empty() {
                    continue;
                }
                if measured_cumulants.covered_parity_any(&error) {
                    continue;
                }
                loc_a_e += 1;
                if !cumulants.covered_parity_any(&error) {
                    continue;
                }
                loc_a_l_e += 1;
            }
            *accepted_errors[rayon::current_thread_index().unwrap_or_default()]
                .lock()
                .unwrap() += loc_a_e;
            *accepted_logical_errors[rayon::current_thread_index().unwrap_or_default()]
                .lock()
                .unwrap() += loc_a_l_e;
        });
        crate::bench_timing::record("ler/score", t);
        let accepted_errors: i32 = accepted_errors
            .into_iter()
            .map(|m| *m.lock().unwrap())
            .sum::<i32>()
            + (nshots as i32 - relevant_indices_len as i32);
        let accepted_logical_errors: i32 = accepted_logical_errors
            .into_iter()
            .map(|m| *m.lock().unwrap())
            .sum();
        (
            accepted_errors as f64 / nshots as f64,
            accepted_logical_errors as f64 / accepted_errors as f64,
        )
    }
}
