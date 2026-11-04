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
use super::generator_table::GeneratorTable;
use super::noise_generators::build_noise_generators;
use super::noise_model::{NoiseGenerator, UNoiseModel};
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

pub(crate) fn proba_from_rate(rate: f64) -> f64 {
    1. - (1. + (-2. * rate).exp()) / 2.
}

pub fn is_covered_single(cumulants: &[SparsePauli], pauli: u8, wire: &Wire) -> bool {
    cumulants
        .iter()
        .any(|c| *c.paulis.get(wire).unwrap_or(&pauli) != pauli)
}

pub struct Coverage<'a> {
    circuit: &'a CliffordCircuit,
    noise_models: &'a [UNoiseModel],
    logical_cumulants: CumulantTable,
    post_selected_cumulants: CumulantTable,
}

impl<'a> Coverage<'a> {
    pub fn new(circuit: &'a CliffordCircuit, noise_models: &'a [UNoiseModel]) -> Self {
        Self {
            circuit,
            noise_models,
            logical_cumulants: CumulantTable::default(),
            post_selected_cumulants: CumulantTable::default(),
        }
    }

    pub fn post_selected_cumulants(&self) -> &CumulantTable {
        &self.post_selected_cumulants
    }

    pub fn logical_cumulants(&self) -> &CumulantTable {
        &self.logical_cumulants
    }

    pub(crate) fn get_generator_errors(&self) -> Vec<NoiseGenerator> {
        build_noise_generators(self.circuit, self.noise_models)
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
        self.post_selected_cumulants
            .extend(propagator.get_cumulant_table_from_paulis(
                &as_paulis,
                Direction::Backward,
                true,
            ));
    }
    /// Computes & stores the cumulants of a collection of logical operators
    /// Those can either be initial stabilizers (that will be forward-propagated)
    /// or measured qubits (we will backpropagate Z operators for each of them)
    pub fn set_logical_cumulants(&mut self, stabilizers: &[Pauli], measured_qubits: &[usize]) {
        let propagator = PauliPropagator::new(self.circuit);
        self.logical_cumulants
            .extend(propagator.get_cumulant_table_from_paulis(
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
        self.logical_cumulants
            .extend(propagator.get_cumulant_table_from_paulis(
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
        GeneratorTable::build(self.circuit, self.noise_models)
            .gamma_score(&self.post_selected_cumulants, &self.logical_cumulants)
    }

    /// Monte Carlo estimate of `(post-selection acceptance, logical error rate)`.
    ///
    /// With `seed = Some(s)` the estimate is reproducible: identical across
    /// repeated calls and across thread counts. The sample is keyed per
    /// generator index (see `ler::sample_shots`), so it does not depend on
    /// scheduling.
    ///
    /// With `seed = None` a fresh base seed is drawn from an OS-seeded
    /// generator on every call, so the result varies from call to call.
    pub fn approximate_psr_ler(&self, nshots: usize, seed: Option<u64>) -> (f64, f64) {
        let job = self.prepare_psr_ler(nshots, seed);
        let (counts, nonempty) = crate::ler::score_batch(std::slice::from_ref(&job))
            .pop()
            .expect("score_batch returns one result per job");
        job.finish(counts, nonempty)
    }

    /// Everything `approximate_psr_ler` does except scoring, so that a round
    /// of candidates can be prepared in parallel and scored together with
    /// `ler::score_batch`.
    pub fn prepare_psr_ler(&self, nshots: usize, seed: Option<u64>) -> crate::ler::LerJob {
        let error_generators = self.get_generator_errors();
        crate::ler::LerJob::prepare(
            self.pack_ler(&error_generators),
            &error_generators,
            nshots,
            seed,
        )
    }

    /// `prepare_psr_ler`, leaving the draw to the device when it would take
    /// the job.
    pub fn prepare_psr_ler_deferred(&self, nshots: usize, seed: Option<u64>) -> crate::ler::LerJob {
        let error_generators = self.get_generator_errors();
        crate::ler::LerJob::prepare_deferred(
            self.pack_ler(&error_generators),
            &error_generators,
            nshots,
            seed,
        )
    }

    fn pack_ler(&self, error_generators: &[NoiseGenerator]) -> crate::ler::LerTables {
        crate::ler::LerTables::pack(
            error_generators,
            &self.post_selected_cumulants,
            &self.logical_cumulants,
            self.circuit.nqbits,
        )
    }

    /// The hash-map scorer, kept as the reference the packed scorer's parity
    /// tests check against.
    #[cfg(test)]
    pub fn approximate_psr_ler_reference(&self, nshots: usize, seed: Option<u64>) -> (f64, f64) {
        let error_generators = self.get_generator_errors();
        let sample = crate::ler::sample_shots(&error_generators, nshots, seed);

        let mut accepted_errors: i64 = 0;
        let mut accepted_logical_errors: i64 = 0;
        for shot in 0..nshots {
            let lo = sample.offsets[shot] as usize;
            let hi = sample.offsets[shot + 1] as usize;
            if lo == hi {
                continue;
            }
            let mut error = SparsePauli::new();
            for index in &sample.indices[lo..hi] {
                error.mult_inplace(&error_generators[*index as usize].0);
            }
            if error.paulis.is_empty() {
                continue;
            }
            if self.post_selected_cumulants.covered_parity_any(&error) {
                continue;
            }
            accepted_errors += 1;
            if !self.logical_cumulants.covered_parity_any(&error) {
                continue;
            }
            accepted_logical_errors += 1;
        }
        let accepted_errors = accepted_errors + (nshots - sample.nonempty) as i64;
        (
            accepted_errors as f64 / nshots as f64,
            accepted_logical_errors as f64 / accepted_errors as f64,
        )
    }
}
