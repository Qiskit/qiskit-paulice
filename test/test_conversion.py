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

"""Test conversion of Qiskit circuits to rustiq gate lists."""

from __future__ import annotations

import unittest

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import RZGate, U1Gate
from qiskit.quantum_info import Clifford
from qiskit_paulice._internal.conversion import convert_to_rustiq_circuit
from qiskit_paulice.checked_circuit import _RUSTIQ_GATES

_ANGLES = (0, np.pi / 2, np.pi, 3 * np.pi / 2, -np.pi / 2, 2 * np.pi, 5 * np.pi / 2, -1e-12)


def _rebuild(gates: list[tuple[str, list[int]]], num_qubits: int) -> QuantumCircuit:
    """The Qiskit circuit applying a rustiq gate list."""
    circuit = QuantumCircuit(num_qubits)
    for name, qubits in gates:
        circuit.append(_RUSTIQ_GATES[name], qubits)
    return circuit


def _rz(param) -> QuantumCircuit:
    """A one-qubit circuit holding a single ``rz`` whose parameter is stored as given."""
    gate = RZGate(0.0)
    gate.params[0] = param  # bypasses Qiskit's conversion of complex values to float
    circuit = QuantumCircuit(1)
    circuit.append(gate, [0])
    return circuit


class TestConvertToRustiqCircuit(unittest.TestCase):
    """Tests for :func:`convert_to_rustiq_circuit`."""

    def test_equivalent_clifford(self):
        """Every supported gate converts to the same Clifford, with a correct index map."""
        circuit = QuantumCircuit(2, 2)
        circuit.h(0)
        circuit.x(0)
        circuit.z(1)
        circuit.id(0)
        circuit.barrier()
        circuit.sx(1)
        circuit.sxdg(0)
        circuit.sdg(1)
        circuit.append(U1Gate(np.pi / 2), [0])
        circuit.cx(0, 1)
        circuit.cz(1, 0)
        circuit.s(0)
        circuit.measure([0, 1], [0, 1])
        gates, indices = convert_to_rustiq_circuit(circuit)
        self.assertEqual(
            Clifford(_rebuild(gates, 2)), Clifford(circuit.remove_final_measurements(False))
        )
        self.assertEqual(indices, [0, 1, 1, 2, 2, 5, 6, 7, 8, 9, 10, 11])

    def test_clifford_rz_angles(self):
        """``rz`` by a multiple of pi/2 converts to zero, one or two phase gates."""
        for angle in _ANGLES:
            with self.subTest(angle=angle):
                circuit = QuantumCircuit(1)
                circuit.h(0)
                circuit.rz(angle, 0)
                gates, indices = convert_to_rustiq_circuit(circuit)
                self.assertEqual(Clifford(_rebuild(gates, 1)), Clifford(circuit))
                self.assertEqual(len(gates), 1 + [0, 1, 2, 1][round(angle / (np.pi / 2)) % 4])
                self.assertEqual(indices, [0] + [1] * (len(gates) - 1))

    def test_bound_and_complex_params(self):
        """Bound parameter expressions and real-valued complex parameters are accepted."""
        theta = Parameter("theta")
        circuit = QuantumCircuit(1)
        circuit.h(0)
        circuit.rz(2 * theta, 0)
        bound = circuit.assign_parameters({theta: 3 * np.pi / 4})
        self.assertEqual(
            Clifford(_rebuild(convert_to_rustiq_circuit(bound)[0], 1)), Clifford(bound)
        )
        for param in (complex(np.pi, 0), complex(-np.pi / 2, 1e-12)):
            with self.subTest(param=param):
                reference = QuantumCircuit(1)
                reference.rz(param.real, 0)
                gates, _ = convert_to_rustiq_circuit(_rz(param))
                self.assertEqual(Clifford(_rebuild(gates, 1)), Clifford(reference))

    def test_unbound_param_raises(self):
        """An ``rz`` with a free parameter cannot be classified and raises."""
        circuit = QuantumCircuit(1)
        circuit.rz(Parameter("theta"), 0)
        with self.assertRaisesRegex(ValueError, "unbound parameter"):
            convert_to_rustiq_circuit(circuit)

    def test_non_clifford_rz_raises(self):
        """Any angle off a real multiple of pi/2 raises, symmetrically in sign."""
        for param in (np.pi / 4, 0.3, 5e-5, -5e-5, 2 * np.pi - 5e-5, complex(np.pi / 2, 1.0)):
            with self.subTest(param=param), self.assertRaisesRegex(ValueError, "non-Clifford"):
                convert_to_rustiq_circuit(_rz(param))

    def test_unsupported_gate_raises(self):
        """A gate outside the supported set raises ``ValueError``."""
        circuit = QuantumCircuit(1)
        circuit.t(0)
        with self.assertRaisesRegex(ValueError, "Unsupported gate"):
            convert_to_rustiq_circuit(circuit)


if __name__ == "__main__":
    unittest.main()
