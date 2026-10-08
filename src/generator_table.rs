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

//! Flat storage for the noise generators used in coverage scoring.
//!
//! A generator is a Pauli on a handful of wires plus a rate. The legacy form
//! is one `SparsePauli` (a `HashMap`) per generator, which for a realistic
//! circuit means tens of thousands of small hash maps rebuilt for every
//! candidate check. This table holds the same data as four flat vectors in
//! CSR form -- rates, term offsets, dense wire ids, Paulis -- so building it
//! costs a few pushes per generator and scoring costs an array index per term.

use super::cumulant_table::{CumulantTable, pauli_xz_bits};
use super::noise_generators::{GeneratorSink, emit_generators};
use super::noise_model::{NoiseGenerator, UNoiseModel};
use super::wire::Wire;
use rustiq_core::structures::CliffordCircuit;

#[derive(Clone, Debug, Default)]
pub struct GeneratorTable {
    nqbits: usize,
    rates: Vec<f64>,
    /// `rates.len() + 1` entries; generator `g` owns terms `offsets[g]..offsets[g + 1]`.
    offsets: Vec<u32>,
    /// Per term: the dense wire id (`Wire::dense_id`) and the Pauli on it.
    wire_ids: Vec<u32>,
    paulis: Vec<u8>,
}

/// Collects generators straight into a `GeneratorTable`.
struct TableSink {
    table: GeneratorTable,
}

impl GeneratorSink for TableSink {
    fn reserve(&mut self, generators: usize, terms: usize) {
        self.table.rates.reserve(generators);
        self.table.offsets.reserve(generators + 1);
        self.table.wire_ids.reserve(terms);
        self.table.paulis.reserve(terms);
    }

    fn push(&mut self, terms: &[(Wire, u8)], rate: f64) {
        let nqbits = self.table.nqbits;
        for (wire, pauli) in terms {
            self.table.wire_ids.push(wire.dense_id(nqbits) as u32);
            self.table.paulis.push(*pauli);
        }
        self.table.rates.push(rate);
        self.table.offsets.push(self.table.wire_ids.len() as u32);
    }
}

impl GeneratorTable {
    fn empty(nqbits: usize) -> Self {
        Self {
            nqbits,
            rates: Vec::new(),
            offsets: vec![0],
            wire_ids: Vec::new(),
            paulis: Vec::new(),
        }
    }

    /// Builds the table directly from the circuit and its noise models, without
    /// materialising a `SparsePauli` per generator. Produces exactly the same
    /// generators, in the same order, as `from_sparse(&build_noise_generators(..))`
    /// -- the order matters, because the gamma sum is accumulated in it.
    pub fn build(circuit: &CliffordCircuit, noise_models: &[UNoiseModel]) -> Self {
        let mut sink = TableSink {
            table: Self::empty(circuit.nqbits),
        };
        emit_generators(circuit, noise_models, &mut sink);
        sink.table
    }

    pub fn from_sparse(nqbits: usize, generators: &[NoiseGenerator]) -> Self {
        let mut sink = TableSink {
            table: Self::empty(nqbits),
        };
        for (pauli, rate) in generators {
            sink.push_sparse(pauli, *rate);
        }
        sink.table
    }

    pub fn len(&self) -> usize {
        self.rates.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rates.is_empty()
    }

    pub fn nqbits(&self) -> usize {
        self.nqbits
    }

    pub fn rates(&self) -> &[f64] {
        &self.rates
    }

    pub fn offsets(&self) -> &[u32] {
        &self.offsets
    }

    pub fn wire_ids(&self) -> &[u32] {
        &self.wire_ids
    }

    pub fn paulis(&self) -> &[u8] {
        &self.paulis
    }

    fn generator_covered(&self, gen_index: usize, cumulants: &CumulantTable) -> bool {
        if cumulants.nrows() == 0 {
            return false;
        }
        let terms = self.offsets[gen_index] as usize..self.offsets[gen_index + 1] as usize;
        let nwords = cumulants.nwords();
        if nwords == 1 {
            let mut acc = 0u64;
            for t in terms {
                let (xe, ze) = pauli_xz_bits(self.paulis[t]);
                let column = cumulants.column_of_id(self.wire_ids[t] as usize);
                acc ^= cumulants.parity_word_at(column, xe, ze, 0);
            }
            return acc != 0;
        }
        let mut acc = vec![0u64; nwords];
        for t in terms {
            let (xe, ze) = pauli_xz_bits(self.paulis[t]);
            let column = cumulants.column_of_id(self.wire_ids[t] as usize);
            for (k, slot) in acc.iter_mut().enumerate() {
                *slot ^= cumulants.parity_word_at(column, xe, ze, k);
            }
        }
        acc.iter().any(|w| *w != 0)
    }

    /// Whether generator `g` counts toward gamma: invisible to the checks but
    /// visible to the logical operators.
    #[inline]
    pub fn counts_toward_gamma(
        &self,
        g: usize,
        post_selected: &CumulantTable,
        logical: &CumulantTable,
    ) -> bool {
        !self.generator_covered(g, post_selected) && self.generator_covered(g, logical)
    }

    /// Same semantics as `Coverage::gamma_apx` over `sparse` generators.
    ///
    /// Sums the rates sequentially, in generator order, so the result is
    /// bit-reproducible from run to run regardless of thread count.
    pub fn gamma_score(&self, post_selected: &CumulantTable, logical: &CumulantTable) -> f64 {
        let mut acc = 0.;
        for (g, rate) in self.rates.iter().enumerate() {
            if self.counts_toward_gamma(g, post_selected, logical) {
                acc += rate;
            }
        }
        (2. * acc).exp()
    }
}

#[cfg(test)]
mod generator_table_tests {
    use super::super::coverage::Coverage;
    use super::super::noise_generators::build_noise_generators;
    use super::super::noise_model::{
        GateDescription, GateWiseNoiseModel, Idling, LayerDescription, LayeredNoiseModel, Readout,
        UNoiseModel, UniformDepolarizing,
    };
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use rustiq_core::structures::CliffordGate;

    fn toy_circuit() -> CliffordCircuit {
        let mut c = CliffordCircuit::new(3);
        c.gates.push(CliffordGate::CZ(0, 1));
        c.gates.push(CliffordGate::CZ(1, 2));
        c
    }

    fn random_circuit(rng: &mut StdRng, nqbits: usize, ngates: usize) -> CliffordCircuit {
        let mut circuit = CliffordCircuit::new(nqbits);
        for _ in 0..ngates {
            let q = rng.random_range(0..nqbits);
            let gate = match rng.random_range(0..5) {
                0 => CliffordGate::H(q),
                1 => CliffordGate::S(q),
                2 => CliffordGate::SqrtX(q),
                _ => {
                    let mut q2 = rng.random_range(0..nqbits);
                    while q2 == q {
                        q2 = rng.random_range(0..nqbits);
                    }
                    CliffordGate::CZ(q, q2)
                }
            };
            circuit.gates.push(gate);
        }
        circuit
    }

    fn gatewise(nqbits: usize) -> UNoiseModel {
        let mut models = GateDescription::new();
        for a in 0..nqbits - 1 {
            models.insert(
                (a, a + 1),
                vec![((1, 1), 0.001 + 0.0001 * a as f64), ((3, 0), 0.002)],
            );
            models.insert((a + 1, a), vec![((2, 3), 0.0015), ((0, 1), 0.0005)]);
        }
        UNoiseModel::GateWise(GateWiseNoiseModel::new(models))
    }

    fn layered(nqbits: usize) -> UNoiseModel {
        let mut layers = LayerDescription::new();
        let even: Vec<(usize, usize)> = (0..nqbits - 1).step_by(2).map(|i| (i, i + 1)).collect();
        let gens: Vec<(String, f64)> = even
            .iter()
            .map(|(a, b)| {
                let mut label = vec!['I'; nqbits];
                label[*a] = 'X';
                label[*b] = 'Z';
                (label.into_iter().collect::<String>(), 0.0015)
            })
            .collect();
        layers.insert(even, gens);
        UNoiseModel::Layered(LayeredNoiseModel::new(layers))
    }

    /// The structural builder must agree with the `SparsePauli` route exactly:
    /// same generators, same order, same rate bits. The gamma sum is
    /// accumulated in generator order, so a permutation would change results.
    #[test]
    fn build_matches_from_sparse() {
        let mut rng = StdRng::seed_from_u64(20260828);
        for trial in 0..40 {
            let nqbits = 3 + trial % 5;
            let circuit = random_circuit(&mut rng, nqbits, 6 + 3 * (trial % 8));
            let model_sets: Vec<Vec<UNoiseModel>> = vec![
                vec![UNoiseModel::UniformDepolarizing(UniformDepolarizing::new(
                    0.003,
                ))],
                vec![
                    UNoiseModel::UniformDepolarizing(UniformDepolarizing::new(0.003)),
                    UNoiseModel::Readout(Readout::new(0.01)),
                ],
                vec![UNoiseModel::Readout(Readout::new(0.02))],
                vec![
                    UNoiseModel::Idling(Idling::new(1e5)),
                    UNoiseModel::Readout(Readout::new(0.01)),
                ],
                vec![gatewise(nqbits), UNoiseModel::Readout(Readout::new(0.01))],
                vec![layered(nqbits), UNoiseModel::Readout(Readout::new(0.01))],
            ];
            for models in model_sets {
                let expected =
                    GeneratorTable::from_sparse(nqbits, &build_noise_generators(&circuit, &models));
                let got = GeneratorTable::build(&circuit, &models);
                assert_eq!(got.len(), expected.len(), "generator count (trial {trial})");
                assert_eq!(got.offsets(), expected.offsets(), "offsets (trial {trial})");
                assert_eq!(got.wire_ids(), expected.wire_ids(), "wires (trial {trial})");
                assert_eq!(got.paulis(), expected.paulis(), "paulis (trial {trial})");
                let got_bits: Vec<u64> = got.rates().iter().map(|r| r.to_bits()).collect();
                let want_bits: Vec<u64> = expected.rates().iter().map(|r| r.to_bits()).collect();
                assert_eq!(got_bits, want_bits, "rates (trial {trial})");
            }
        }
    }

    #[test]
    fn gamma_score_matches_coverage() {
        let circuit = toy_circuit();
        let models = vec![
            UNoiseModel::UniformDepolarizing(UniformDepolarizing::new(0.01)),
            UNoiseModel::Readout(Readout::new(0.02)),
        ];
        let table = GeneratorTable::build(&circuit, &models);
        let mut coverage = Coverage::new(&circuit, &models);
        coverage.set_check_cumulants(&[0], &[vec![]]);
        coverage.set_logical_cumulants(&[], &[1, 2]);
        let expected = coverage.gamma_apx();
        let got = table.gamma_score(
            coverage.post_selected_cumulants(),
            coverage.logical_cumulants(),
        );
        assert!((expected - got).abs() < 1e-12, "{expected} vs {got}");
    }
}
