# This code is a Qiskit project.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for ``CheckedCircuit.get_syndrome_flips``."""

from __future__ import annotations

import unittest
import warnings

import numpy as np
import pytest
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import PauliLindbladMap
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel as AerNoiseModel
from qiskit_aer.noise import depolarizing_error
from qiskit_paulice import add_pauli_checks
from qiskit_paulice.noise_models import NoiseModel

pytest.importorskip("samplomatic")

import samplomatic

NUM_QUBITS, DEPTH, SHOTS, NUM_RANDOMIZATIONS = 4, 6, 200, 8


def _bare_circuit():
    qc = QuantumCircuit(NUM_QUBITS)
    qc.h(range(NUM_QUBITS))
    for d in range(DEPTH):
        for i in range(d % 2, NUM_QUBITS - 1, 2):
            qc.cz(i, i + 1)
        for q in range(NUM_QUBITS):
            qc.sx(q)
    qc.measure_all()
    return qc


class TestSyndromeFlips(unittest.TestCase):
    """The twirl's effect on the syndrome is known exactly and must be undone."""

    @classmethod
    def setUpClass(cls):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cls.checked = add_pauli_checks(
                _bare_circuit(),
                list(range(NUM_QUBITS)),
                NoiseModel(gate_noise=1e-3, readout_noise=1e-2),
                seed=1,
            )[-1]
            cls.boxed = cls.checked.boxed_circuit
            cls.template, cls.samplex = samplomatic.build(cls.boxed)

            # ref names come from a counter shared across the process, so discover them
            def _refs(prefix):
                return [
                    spec.name.split(".", 1)[1] for spec in cls.samplex.inputs().get_specs(prefix)
                ]

            # identity maps: no injected noise, so this isolates the twirl
            inputs = cls.samplex.inputs().bind(
                pauli_lindblad_maps={
                    ref: PauliLindbladMap.identity(cls.boxed.num_qubits)
                    for ref in _refs("pauli_lindblad_maps")
                },
                basis_changes={ref: [0] * cls.boxed.num_qubits for ref in _refs("basis_changes")},
            )
            cls.outputs = cls.samplex.sample(inputs, num_randomizations=NUM_RANDOMIZATIONS, rng=99)

    def test_shape_and_values(self):
        """One 0/1 bit per check, per randomization."""
        flips = self.checked.get_syndrome_flips(self.outputs)
        expected_leading = np.asarray(self.outputs["measurement_flips.checks_c"]).shape[:-1]
        self.assertEqual(flips.shape, (*expected_leading, len(self.checked.check_support)))
        self.assertTrue(set(np.unique(flips)) <= {0, 1})
        # the twirl really does flip syndromes -- otherwise this test proves nothing
        self.assertTrue(flips.any())

    def test_matches_correcting_the_bits(self):
        """Correcting the syndrome equals correcting the bits, as F2-linearity requires."""
        flips = self.checked.get_syndrome_flips(self.outputs)
        postselect = self.checked.get_postselection_method()
        rng = np.random.default_rng(0)
        for r in range(NUM_RANDOMIZATIONS):
            bit_flips = np.zeros(self.checked.circuit.num_qubits, dtype=np.byte)
            for creg in self.checked.circuit.cregs:
                block = np.asarray(self.outputs[f"measurement_flips.{creg.name}"])[r].ravel()
                for i, clbit in enumerate(creg):
                    clbit_index = self.checked.circuit.find_bit(clbit).index
                    bit_flips[self.checked._cb_to_q[clbit_index]] = block[i]
            for _ in range(5):
                bits = rng.integers(0, 2, self.checked.circuit.num_qubits).astype(np.byte)
                np.testing.assert_array_equal(
                    postselect(bits ^ bit_flips),
                    postselect(bits) ^ flips[r].ravel(),
                )

    def test_recovers_untwirled_acceptance(self):
        """End to end: raw bits post-select the wrong shots, corrected bits do not."""
        aer_noise = AerNoiseModel()
        aer_noise.add_all_qubit_quantum_error(depolarizing_error(0.02, 2), ["cz"])
        sim = AerSimulator(noise_model=aer_noise, seed_simulator=7)
        postselect = self.checked.get_postselection_method()

        # reference: the same circuit, untwirled
        reference = sim.run(
            transpile(self.checked.circuit, sim), shots=SHOTS * NUM_RANDOMIZATIONS
        ).result()
        counts = reference.get_counts()
        untwirled = sum(n for bs, n in counts.items() if not postselect(bs).any()) / (
            SHOTS * NUM_RANDOMIZATIONS
        )

        flips = self.checked.get_syndrome_flips(self.outputs)
        raw_kept = corrected_kept = total = 0
        for r in range(NUM_RANDOMIZATIONS):
            bound = self.template.assign_parameters(self.outputs["parameter_values"][r])
            memory = sim.run(transpile(bound, sim), shots=SHOTS, memory=True).result().get_memory()
            for shot in memory:
                syndrome = postselect(shot)
                total += 1
                raw_kept += not syndrome.any()
                corrected_kept += not (syndrome ^ flips[r].ravel()).any()

        corrected = corrected_kept / total
        raw = raw_kept / total
        # the corrected acceptance reproduces the untwirled one
        self.assertAlmostEqual(corrected, untwirled, delta=0.05)
        # ... and the uncorrected one is badly wrong, which is the whole point
        self.assertLess(raw, untwirled - 0.2)

    def test_rejects_outputs_without_flips(self):
        """Outputs that carry no measurement flips are an error, not a silent no-op."""
        with self.assertRaises(ValueError):
            self.checked.get_syndrome_flips({"parameter_values": np.zeros((2, 2))})


if __name__ == "__main__":
    unittest.main()
