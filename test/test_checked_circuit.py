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

"""Test checked_circuit module."""

from __future__ import annotations

import unittest

import numpy as np
from qiskit import QuantumCircuit
from qiskit_paulice import CheckedCircuit, UncoveredPauli


def _bell_with_measure() -> QuantumCircuit:
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure(0, 0)
    qc.measure(1, 1)
    return qc


class TestCheckedCircuit(unittest.TestCase):
    """Tests covering :class:`CheckedCircuit`."""

    def test_post_init_coerces_sequences(self):
        """List inputs to tuple-typed fields are coerced (and nested lists too)."""
        cc = CheckedCircuit(
            circuit=_bell_with_measure(),
            target_qubits=[0, 1],
            check_qubits=[],
            check_support=[[0, 1]],
        )
        self.assertIsInstance(cc.target_qubits, tuple)
        self.assertIsInstance(cc.check_qubits, tuple)
        self.assertEqual(cc.check_support, ((0, 1),))

    def test_uncovered_paulis_shape_and_types(self):
        """``uncovered_paulis`` returns ``UncoveredPauli`` triples with sane fields."""
        cc = CheckedCircuit(circuit=_bell_with_measure())
        ups = cc.uncovered_paulis
        self.assertIsInstance(ups, tuple)
        # An unchecked Clifford circuit has many uncovered single-qubit errors.
        self.assertGreater(len(ups), 0)
        n_inst = len(cc.circuit.data)
        for up in ups:
            self.assertIsInstance(up, UncoveredPauli)
            self.assertIn(up.pauli, ("X", "Y", "Z"))
            self.assertIn(up.qubit, range(cc.circuit.num_qubits))
            self.assertTrue(
                up.after_instruction is None or 0 <= up.after_instruction < n_inst,
                msg=f"after_instruction out of range: {up.after_instruction}",
            )
        # Input-wire errors (after_instruction is None) exist for every qubit and Pauli.
        input_wire = {(up.qubit, up.pauli) for up in ups if up.after_instruction is None}
        for q in range(cc.circuit.num_qubits):
            for p in ("X", "Y", "Z"):
                self.assertIn((q, p), input_wire)

    def test_uncovered_paulis_is_cached(self):
        """Repeated access returns the same tuple object (cached_property)."""
        cc = CheckedCircuit(circuit=_bell_with_measure())
        self.assertIs(cc.uncovered_paulis, cc.uncovered_paulis)

    def test_postselection_bitstring_with_measurements(self):
        """Bitstring path uses ``measure`` instructions to map clbits to qubits."""
        cc = CheckedCircuit(
            circuit=_bell_with_measure(),
            check_support=[[0, 1]],
        )
        f = cc.get_postselection_method()
        # check_support = {0, 1}: syndrome bit = m[0] XOR m[1]
        np.testing.assert_array_equal(f("00"), np.array([0]))
        np.testing.assert_array_equal(f("11"), np.array([0]))
        np.testing.assert_array_equal(f("10"), np.array([1]))
        np.testing.assert_array_equal(f("01"), np.array([1]))

    def test_postselection_bitstring_strips_whitespace(self):
        """Spaces inside the bitstring (e.g. register separators) are ignored."""
        cc = CheckedCircuit(circuit=_bell_with_measure(), check_support=[[0, 1]])
        f = cc.get_postselection_method()
        np.testing.assert_array_equal(f("1 0"), f("10"))

    def test_postselection_multiple_checks(self):
        """Multiple rows of the support matrix produce independent syndrome bits."""
        cc = CheckedCircuit(
            circuit=_bell_with_measure(),
            check_support=[[0, 1], [1]],
        )
        f = cc.get_postselection_method()
        # Bitstring path. "10" → m[1]=1, m[0]=0 → x=[0,1]; rows [1,1] and [0,1].
        np.testing.assert_array_equal(f("10"), np.array([1, 1]))
        # "01" → m[1]=0, m[0]=1 → x=[1,0]; rows [1,1] and [0,1].
        np.testing.assert_array_equal(f("01"), np.array([1, 0]))
        # Array path: input is qubit-indexed.
        np.testing.assert_array_equal(
            f(np.array([1, 0], dtype=np.byte)),
            np.array([1, 0]),
        )

    def test_postselection_rejects_wrong_length_with_measurements(self):
        """Bitstrings whose length doesn't match num_clbits raise ValueError."""
        cc = CheckedCircuit(circuit=_bell_with_measure(), check_support=[[0, 1]])
        f = cc.get_postselection_method()
        with self.assertRaisesRegex(ValueError, "expected 2"):
            f("1")
        with self.assertRaisesRegex(ValueError, "expected 2"):
            f("101")

    def test_postselection_rejects_wrong_length_without_measurements(self):
        """The qubit-indexed fallback also enforces an exact length match."""
        qc = QuantumCircuit(3)
        qc.h(0)
        cc = CheckedCircuit(circuit=qc, check_support=[[0, 1]])
        f = cc.get_postselection_method()
        with self.assertRaisesRegex(ValueError, "expected 3"):
            f("10")

    def test_postselection_no_measurements_uses_qubit_indexing(self):
        """Without measure ops, bitstrings are interpreted as qubit-indexed."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        cc = CheckedCircuit(circuit=qc, check_support=[[0, 1]])
        f = cc.get_postselection_method()
        np.testing.assert_array_equal(f("10"), np.array([1]))
        np.testing.assert_array_equal(f("11"), np.array([0]))
