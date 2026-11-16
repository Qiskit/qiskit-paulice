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
use rayon::prelude::*;

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

type LogicalData = (Vec<usize>, StabilizerGroup);
type CheckData = (Vec<usize>, Vec<Vec<usize>>);

type PyWire = (i32, usize);

/// Memoized candidate evaluations, keyed by the canonical (sorted) form of the
/// candidate check. Valid exactly as long as the evaluator state is unchanged:
/// the map is replaced with a fresh one in `set_evaluation_data` and
/// `commit_check`. Until then it dedups a candidate that appears more than
/// once in a batch, and across scoring calls on the same picker state. It is
/// shared (via `Arc`) by `copy`/`clone`, so copies made by `station.copy()`
/// share it too. Gamma evaluation is a pure function of (evaluator state,
/// check), so a hit returns exactly what recomputation would; an LER estimate
/// is reused rather than resampled.
type EvalMemo = Arc<Mutex<HashMap<Vec<(PyWire, u8)>, (f64, Vec<usize>)>>>;

/// A candidate awaiting evaluation: its index in the candidate list, its
/// virtual Zs, and its memo key.
type PendingEval = (usize, Vec<usize>, Vec<(PyWire, u8)>);

fn _memo_key(check: &SparsePauli) -> Vec<(PyWire, u8)> {
    let mut key: Vec<_> = check
        .paulis
        .iter()
        .map(|(w, p)| (_to_py_wire(w.clone()), *p))
        .collect();
    key.sort_unstable();
    key
}

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
    /// Input stabilizers whose forward images define the logical errors counted by the
    /// metric. `None` means the input stabilizer group in `logical_data` plays that role too.
    logical_stabilizers: Option<StabilizerGroup>,
    /// Already existing check qubits & their final measurements components (if any)
    check_data: CheckData,
    /// CheckEvaluator structure to evaluate check performances
    check_evaluator: Option<CheckEvaluator>,
    /// Possible CheckGroup data structure
    check_group: Option<CheckGroup>,
    /// Possible CheckDecoder data structure
    check_decoder: Option<CheckDecoder>,
    /// Shared memo of candidate evaluations for the current evaluator state
    eval_memo: EvalMemo,
}

#[pymethods]
/// Public interface
impl CheckPicker {
    /// Builds a picker for `circuit`.
    ///
    /// Pauli labels are internal labels over `nqubits` qubits: character `i` acts on qubit
    /// `i`, the reverse of a Qiskit label (shorter labels are padded with identities).
    ///
    /// - `stabilizer_group` decides validity: Paulis stabilizing the input state, and a check
    ///   is valid iff its back-propagated product lies in that group, possibly times Z on
    ///   `measured_qubits` at the output. By default it also decides the cost.
    /// - `logical_stabilizers` decides cost only, never validity: input stabilizers whose
    ///   forward images define the logical errors the metric counts, in place of
    ///   `stabilizer_group`.
    #[new]
    #[pyo3(signature = (
        circuit,
        nqubits=None,
        measured_qubits=None,
        stabilizer_group=None,
        check_qubits=None,
        virtual_zs=None,
        logical_stabilizers=None,
    ))]
    pub fn new(
        circuit: Vec<(String, Vec<usize>)>,
        nqubits: Option<usize>,
        measured_qubits: Option<Vec<usize>>,
        stabilizer_group: Option<Vec<String>>,
        check_qubits: Option<Vec<usize>>,
        virtual_zs: Option<Vec<Vec<usize>>>,
        logical_stabilizers: Option<Vec<String>>,
    ) -> Self {
        let mut circuit = CliffordCircuit::from_vec(circuit);
        if let Some(nqubits) = nqubits {
            circuit.nqbits = nqubits;
        }
        let measured_qubits = measured_qubits.unwrap_or_default();
        let to_group = |labels: Vec<String>| {
            StabilizerGroup::new(
                labels
                    .into_iter()
                    .map(|s| string_to_pauli(&s, circuit.nqbits))
                    .collect(),
            )
        };
        let stabilizer_group = to_group(stabilizer_group.unwrap_or_default());
        let logical_stabilizers = logical_stabilizers.map(to_group);
        let check_qubits = check_qubits.unwrap_or_default();
        let virtual_zs = virtual_zs.unwrap_or_default();
        assert!(
            measured_qubits.is_empty() || stabilizer_group.is_trivial(),
            "Cannot have both non trivial stabilizer group and set of measurements"
        );
        Self {
            circuit,
            logical_data: (measured_qubits, stabilizer_group),
            logical_stabilizers,
            check_data: (check_qubits, virtual_zs),
            check_evaluator: None,
            check_group: None,
            check_decoder: None,
            eval_memo: EvalMemo::default(),
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
            self.logical_stabilizers
                .clone()
                .unwrap_or_else(|| self.logical_data.1.clone()),
            self.logical_data.0.clone(),
            self.check_data.0.clone(),
            self.check_data.1.clone(),
            ancilla,
        ));
        // New evaluator state -> previously memoized evaluations no longer apply.
        self.eval_memo = EvalMemo::default();
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

    /// Commits a check specified by its coordinates; with measured qubits, the stored virtual Zs
    /// include the ancilla, as in `find_good_checks`
    pub fn commit_check_bv(&self, vec: Vec<bool>) -> Option<Self> {
        let (check, mut vzs) = self.check_group.as_ref().unwrap().get_check(&vec);
        if !self.logical_data.0.is_empty() {
            vzs.push(self.check_evaluator.as_ref().unwrap().get_ancilla());
        }
        Some(self.commit_check(check, vzs))
    }

    /// Generates a few checks and their costs and commits the best one.
    pub fn find_good_checks(&self) -> Option<(Self, f64)> {
        assert!(
            self.check_decoder.is_some(),
            "Please first set the check's support"
        );
        let checks = self.check_decoder.as_ref().unwrap().find_checks();
        self.evaluate_and_commit_best(checks)
    }

    /// Batched form of `find_good_checks`: decodes the candidates of every
    /// window in one call, scores them all together, and commits the single
    /// cheapest one.
    ///
    /// Equivalent to running `set_support` + `find_good_checks` once per window
    /// on independent copies and keeping the lowest-cost result: every window
    /// is decoded and scored against the same uncommitted state either way, and
    /// candidates stay in window order (and, within a window, in `find_checks`
    /// order) so ties resolve to the same candidate. Batching lets all
    /// windows' candidates be decoded and scored in one parallel pass instead
    /// of one window at a time.
    ///
    /// `seeds` supplies one decoder seed per window, `None` meaning OS-seeded as
    /// in `set_support`; the caller owns the draw order so a seeded search
    /// stays reproducible.
    pub fn find_good_checks_windowed(
        &self,
        windows: Vec<Vec<PyWire>>,
        paulis: Vec<u8>,
        seeds: Vec<Option<u64>>,
    ) -> Option<(Self, f64)> {
        assert!(
            self.check_evaluator.is_some(),
            "Please first set the evaluation data"
        );
        assert_eq!(
            windows.len(),
            seeds.len(),
            "Expected one decoder seed per window"
        );
        let decoded: Vec<Vec<SparsePauli>> = windows
            .into_par_iter()
            .zip(seeds)
            .map(|(window, seed)| {
                let wires: Vec<_> = window.into_iter().map(_to_rust_wire).collect();
                let decoder = CheckDecoder::new(
                    &self.circuit,
                    &wires,
                    &paulis,
                    &self.logical_data.0,
                    &self.logical_data.1,
                    seed,
                );
                decoder.find_checks()
            })
            .collect();
        self.evaluate_and_commit_best(decoded.into_iter().flatten().collect())
    }

    /// Returns a python compatible description of the current circuit
    pub fn get_circuit(&self) -> Vec<(String, Vec<usize>)> {
        self.circuit
            .gates
            .iter()
            .map(|gate| gate.to_vec())
            .collect()
    }

    /// Returns the virtual CZs stored for each of the current checks
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
    /// Scores every candidate in `checks` against the current evaluator state.
    /// Returns one `(virtual zs, cost)` per input, in input order.
    ///
    /// Candidates already in `eval_memo` are served from it; the rest go
    /// through a single `evaluate_batch` call, so the fresh candidates are
    /// scored in one parallel pass, each distinct one only once.
    /// Because the costs are collected by index rather than compared as they
    /// arrive, the outcome does not depend on which candidates happened to be
    /// memoized.
    ///
    /// The virtual Zs of the fresh candidates are computed in parallel:
    /// `compute_vzs` only reads the evaluator, so the candidates are
    /// independent, and the memo is consulted beforehand so no lock is held
    /// while they run.
    fn score_checks(&self, checks: &[SparsePauli]) -> Vec<(Vec<usize>, f64)> {
        let mut scored: Vec<Option<(Vec<usize>, f64)>> = vec![None; checks.len()];
        let mut misses: Vec<(usize, Vec<(PyWire, u8)>)> = Vec::new();
        let mut pending_keys: HashSet<Vec<(PyWire, u8)>> = HashSet::new();
        let mut repeats: Vec<(usize, Vec<(PyWire, u8)>)> = Vec::new();
        {
            let memo = self.eval_memo.lock().unwrap();
            for (index, check) in checks.iter().enumerate() {
                let key = _memo_key(check);
                if let Some((cost, vzs)) = memo.get(&key) {
                    scored[index] = Some((vzs.clone(), *cost));
                } else if pending_keys.contains(&key) {
                    repeats.push((index, key));
                } else {
                    pending_keys.insert(key.clone());
                    misses.push((index, key));
                }
            }
        }
        let pending: Vec<PendingEval> = if self.logical_data.0.is_empty() {
            misses
                .into_iter()
                .map(|(index, key)| (index, Vec::new(), key))
                .collect()
        } else {
            let evaluator = self.check_evaluator.as_ref().unwrap();
            misses
                .into_par_iter()
                .map(|(index, key)| (index, evaluator.compute_vzs(&checks[index]), key))
                .collect()
        };
        if !pending.is_empty() {
            let evaluator = self.check_evaluator.as_ref().unwrap();
            let items: Vec<(SparsePauli, Vec<usize>)> = pending
                .iter()
                .map(|(index, vzs, _)| (checks[*index].clone(), vzs.clone()))
                .collect();
            let costs = evaluator.evaluate_batch(&items);
            let mut memo = self.eval_memo.lock().unwrap();
            for ((index, vzs, key), cost) in pending.into_iter().zip(costs) {
                memo.insert(key, (cost, vzs.clone()));
                scored[index] = Some((vzs, cost));
            }
            for (index, key) in repeats {
                let (cost, vzs) = memo[&key].clone();
                scored[index] = Some((vzs, cost));
            }
        }
        scored
            .into_iter()
            .map(|entry| entry.expect("every candidate is either memoized or evaluated"))
            .collect()
    }

    /// Scores `checks` and commits the cheapest one. Ties go to the earliest
    /// candidate, so the winner only depends on the order of `checks`. Returns
    /// `None` when no candidate costs less than `f64::MAX`.
    fn evaluate_and_commit_best(&self, mut checks: Vec<SparsePauli>) -> Option<(Self, f64)> {
        if checks.is_empty() {
            return None;
        }
        let mut scored = self.score_checks(&checks);
        let mut best: Option<usize> = None;
        let mut best_cost = f64::MAX;
        for (index, (_, cost)) in scored.iter().enumerate() {
            if *cost < best_cost {
                best_cost = *cost;
                best = Some(index);
            }
        }
        let best = best?;
        let (vzs, _) = scored.swap_remove(best);
        let check = checks.swap_remove(best);
        Some((self.commit_check(check, vzs), best_cost))
    }

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
            logical_stabilizers: self.logical_stabilizers.clone(),
            check_data,
            check_evaluator: None,
            check_group: None,
            check_decoder: None,
            // Committing changes the circuit and check data; start a fresh memo.
            eval_memo: EvalMemo::default(),
        }
    }
}

#[cfg(test)]
mod check_picker_tests {
    use super::*;

    /// A picker over a small measured circuit, so candidates carry virtual Zs
    /// and `score_checks` takes the parallel `compute_vzs` path.
    fn measured_picker(metric: Metric, noise: f64) -> CheckPicker {
        let gates = [
            ("H", vec![0]),
            ("CX", vec![0, 1]),
            ("S", vec![1]),
            ("CX", vec![1, 2]),
            ("H", vec![2]),
            ("CX", vec![2, 0]),
            ("CX", vec![0, 1]),
            ("H", vec![1]),
        ];
        let circuit = gates
            .into_iter()
            .map(|(name, qubits)| (name.to_string(), qubits))
            .collect();
        let mut picker = CheckPicker::new(
            circuit,
            Some(4),
            Some(vec![0, 1, 2]),
            None,
            None,
            None,
            None,
        );
        picker.set_evaluation_data(vec![NoiseModel::uniform_depolarizing(noise)], metric, 3);
        picker
    }

    /// Candidates decoded from several windows of every qubit, listed twice so
    /// each one is repeated; also returns how many distinct candidates there are.
    fn repeated_candidates(picker: &mut CheckPicker) -> (Vec<SparsePauli>, usize) {
        let mut checks = Vec::new();
        for qubit in 0..3 {
            for seed in 0..3 {
                picker.set_support(picker.get_wires(qubit), vec![1, 2, 3], Some(seed));
                checks.extend(picker.check_decoder.as_ref().unwrap().find_checks());
            }
        }
        let distinct: HashSet<_> = checks.iter().map(_memo_key).collect();
        assert!(!distinct.is_empty());
        let repeated = checks.iter().chain(checks.iter()).cloned().collect();
        (repeated, distinct.len())
    }

    #[test]
    fn score_checks_matches_direct_evaluation() {
        let mut picker = measured_picker(Metric::gamma(), 0.01);
        let (repeated, distinct) = repeated_candidates(&mut picker);

        let scored = picker.score_checks(&repeated);

        let evaluator = picker.check_evaluator.as_ref().unwrap();
        assert_eq!(scored.len(), repeated.len());
        for (check, (vzs, cost)) in repeated.iter().zip(&scored) {
            let expected_vzs = evaluator.compute_vzs(check);
            let expected_cost = evaluator.evaluate(check, &expected_vzs);
            assert_eq!(*vzs, expected_vzs);
            assert_eq!(cost.to_bits(), expected_cost.to_bits());
        }
        assert!(scored.iter().any(|(vzs, _)| !vzs.is_empty()));
        assert_eq!(picker.eval_memo.lock().unwrap().len(), distinct);
    }

    /// LER is a Monte Carlo estimate, so a repeat only gets the bit-identical
    /// cost of its first occurrence if it was served rather than resampled.
    #[test]
    fn score_checks_scores_each_distinct_candidate_once() {
        let mut picker = measured_picker(Metric::logical_error_rate(200, None), 0.2);
        let (repeated, distinct) = repeated_candidates(&mut picker);

        let scored = picker.score_checks(&repeated);

        let mut first = HashMap::new();
        for (check, (vzs, cost)) in repeated.iter().zip(&scored) {
            let seen = first
                .entry(_memo_key(check))
                .or_insert_with(|| (vzs.clone(), cost.to_bits()));
            assert_eq!(*seen, (vzs.clone(), cost.to_bits()));
        }
        assert_eq!(picker.eval_memo.lock().unwrap().len(), distinct);
    }
}
