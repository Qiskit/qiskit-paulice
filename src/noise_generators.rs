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

//! Build noise-generator lists for a fixed circuit.

use super::noise_model::{NoiseGenerator, NoiseModelLike, UNoiseModel};
use super::sparse_pauli::SparsePauli;
use super::utils::last_wires;
use super::wire::Wire;
use rustiq_core::structures::CliffordCircuit;

/// Receives noise generators in emission order.
///
/// The order matters: the gamma sum is accumulated over generators in the
/// order they are produced, so any sink must see exactly the sequence
/// `build_noise_generators` would return.
pub trait GeneratorSink {
    /// `terms` holds `(wire, pauli)` pairs in ascending wire order, with no
    /// identity Paulis and no repeated wires -- the normalised form
    /// `SparsePauli` would have produced.
    fn push(&mut self, terms: &[(Wire, u8)], rate: f64);

    /// Hint that `generators` generators carrying `terms` terms in total are
    /// about to arrive, so sinks backed by flat buffers can allocate once
    /// instead of growing: a candidate check emits tens of thousands of them.
    fn reserve(&mut self, _generators: usize, _terms: usize) {}

    /// Pushes a generator that is already a `SparsePauli`, normalising its
    /// term order first.
    fn push_sparse(&mut self, pauli: &SparsePauli, rate: f64) {
        let mut terms: Vec<(Wire, u8)> =
            pauli.paulis.iter().map(|(w, p)| (w.clone(), *p)).collect();
        terms.sort_by(|a, b| a.0.cmp(&b.0));
        self.push(&terms, rate);
    }
}

/// Collects generators into the legacy `Vec<(SparsePauli, rate)>` form.
struct SparseSink {
    generators: Vec<NoiseGenerator>,
}

impl GeneratorSink for SparseSink {
    fn reserve(&mut self, generators: usize, _terms: usize) {
        self.generators.reserve(generators);
    }

    fn push(&mut self, terms: &[(Wire, u8)], rate: f64) {
        let mut pauli = SparsePauli::new();
        for (wire, p) in terms {
            pauli.update(wire.clone(), *p);
        }
        self.generators.push((pauli, rate));
    }

    fn push_sparse(&mut self, pauli: &SparsePauli, rate: f64) {
        self.generators.push((pauli.clone(), rate));
    }
}

/// Runs every noise model over the circuit, feeding `sink` in emission order.
///
/// Each model may rewrite the circuit (`LayeredNoiseModel` relayers it), and
/// later models see the rewritten version -- which is why `Readout` recomputes
/// the last wires here rather than trusting the original circuit.
pub fn emit_generators(
    circuit: &CliffordCircuit,
    noise_models: &[UNoiseModel],
    sink: &mut dyn GeneratorSink,
) {
    let (generators, terms) = noise_models
        .iter()
        .map(|m| m.generator_hint(circuit))
        .fold((0, 0), |(g, t), (dg, dt)| (g + dg, t + dt));
    sink.reserve(generators, terms);

    // Only materialise a rewritten circuit when a model actually rewrites one.
    let mut rewritten: Option<CliffordCircuit> = None;
    for noise_model in noise_models {
        let next = {
            let current = rewritten.as_ref().unwrap_or(circuit);
            match noise_model {
                UNoiseModel::Readout(m) => {
                    // Recompute on the *current* circuit: a prior model (e.g.
                    // LayeredNoiseModel) may have rewritten gate indices / last wires.
                    m.emit_for_last_wires(&last_wires(current), sink);
                    None
                }
                _ => noise_model.emit_generators(current, sink),
            }
        };
        if next.is_some() {
            rewritten = next;
        }
    }
}

pub fn build_noise_generators(
    circuit: &CliffordCircuit,
    noise_models: &[UNoiseModel],
) -> Vec<NoiseGenerator> {
    let mut sink = SparseSink {
        generators: Vec::new(),
    };
    emit_generators(circuit, noise_models, &mut sink);
    sink.generators
}

#[cfg(test)]
mod generator_build_tests {
    use super::super::noise_model::{LayeredNoiseModel, Readout, UNoiseModel};
    use super::*;
    use rustiq_core::structures::CliffordGate;
    use std::collections::HashMap;

    fn sequential_generators(
        circuit: &CliffordCircuit,
        models: &[UNoiseModel],
    ) -> Vec<NoiseGenerator> {
        let mut gens = Vec::new();
        let mut circuit = circuit.clone();
        for model in models {
            let (new_gens, new_circuit) = model.get_generators(&circuit);
            gens.extend(new_gens);
            circuit = new_circuit;
        }
        gens
    }

    fn gen_sig(g: &NoiseGenerator) -> (u64, u64) {
        (g.0.fingerprint(), g.1.to_bits())
    }

    #[test]
    fn readout_follows_circuit_mutated_by_prior_model() {
        let mut circuit = CliffordCircuit::new(2);
        circuit.gates.push(CliffordGate::H(0));
        circuit.gates.push(CliffordGate::H(1));
        circuit.gates.push(CliffordGate::CZ(0, 1));

        let mut layer_models = HashMap::new();
        layer_models.insert(vec![(0, 1)], vec![("XI".to_string(), 0.01)]);
        let models = vec![
            UNoiseModel::Layered(LayeredNoiseModel::new(layer_models)),
            UNoiseModel::Readout(Readout::new(0.02)),
        ];

        let got = build_noise_generators(&circuit, &models);
        let expected = sequential_generators(&circuit, &models);
        let got_sigs: Vec<_> = got.iter().map(gen_sig).collect();
        let expected_sigs: Vec<_> = expected.iter().map(gen_sig).collect();
        assert_eq!(got_sigs, expected_sigs);
    }
}
