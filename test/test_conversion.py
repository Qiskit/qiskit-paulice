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

"""Test conversions between Qiskit objects and the Rust picker's conventions."""

from __future__ import annotations

import unittest

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import RZGate, U1Gate
from qiskit.quantum_info import Clifford, Pauli
from qiskit_paulice._internal.conversion import (
    RUSTIQ_GATES as _RUSTIQ_GATES,
)
from qiskit_paulice._internal.conversion import (
    clifford_of,
    convert_stabilizers,
    convert_to_rustiq_circuit,
    normalize_measured_qubits,
    normalize_stabilizers,
)

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


def _payload() -> QuantumCircuit:
    """A 3-qubit Clifford circuit whose ``rz`` gates are Clifford angles."""
    circuit = QuantumCircuit(3)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.rz(np.pi / 2, 1)
    circuit.cx(1, 2)
    circuit.sx(2)
    return circuit


def _output_stabilizer(circuit: QuantumCircuit, qubits) -> Pauli:
    """The stabilizer of the prepared state that is the image of ``Z`` on ``qubits``."""
    z = Pauli("I" * circuit.num_qubits)
    z.z[list(qubits)] = True
    return z.evolve(clifford_of(circuit), frame="s")


class TestCliffordOf(unittest.TestCase):
    """Tests for :func:`clifford_of`."""

    def test_matches_qiskit_and_ignores_non_unitaries(self):
        """Equals Qiskit's Clifford of the same circuit with ``rz(pi/2)`` written as ``s``;
        barriers and measurements are ignored."""
        with_s = QuantumCircuit(3)
        with_s.h(0)
        with_s.cx(0, 1)
        with_s.s(1)
        with_s.cx(1, 2)
        with_s.sx(2)
        circuit = _payload()
        circuit.barrier()
        circuit.measure_all()
        self.assertEqual(clifford_of(circuit), Clifford(with_s))

    def test_non_clifford_raises(self):
        circuit = QuantumCircuit(1)
        circuit.rz(0.3, 0)
        with self.assertRaises(ValueError):
            clifford_of(circuit)


class TestNormalize(unittest.TestCase):
    """Tests for :func:`normalize_stabilizers` and :func:`normalize_measured_qubits`."""

    def test_stabilizers(self):
        self.assertEqual(normalize_stabilizers(None, 3), [])
        self.assertEqual(normalize_stabilizers("all", 3), ["ZII", "IZI", "IIZ"])
        # A Qiskit label reads right to left; the internal label reads left to right.
        self.assertEqual(normalize_stabilizers([Pauli("-XZ"), Pauli("iYI")], 2), ["ZX", "IY"])
        self.assertEqual(normalize_stabilizers(["ZX"], 2), ["ZX"])
        with self.assertRaises(ValueError):
            normalize_stabilizers("some", 2)

    def test_measured_qubits(self):
        self.assertEqual(normalize_measured_qubits(None, 3), [])
        self.assertEqual(normalize_measured_qubits("all", 3), [0, 1, 2])
        self.assertEqual(normalize_measured_qubits((2, 0), 3), [2, 0])
        with self.assertRaises(ValueError):
            normalize_measured_qubits("some", 3)


class TestConvertStabilizers(unittest.TestCase):
    """Tests for :func:`convert_stabilizers`."""

    def setUp(self):
        self.circuit = _payload()

    def test_images_of_z_map_back_to_z(self):
        """The forward image of ``Z`` on qubit ``q`` converts to the internal label with ``Z``
        at character ``q``, for Paulis and for Qiskit labels, whatever the sign."""
        for q in range(3):
            stabilizer = _output_stabilizer(self.circuit, [q])
            expected = "I" * q + "Z" + "I" * (2 - q)
            with self.subTest(qubit=q):
                self.assertEqual(
                    convert_stabilizers(
                        [stabilizer, -stabilizer, stabilizer.to_label()], self.circuit
                    ),
                    [expected] * 3,
                )
        self.assertEqual(
            convert_stabilizers([_output_stabilizer(self.circuit, [0, 2])], self.circuit), ["ZIZ"]
        )

    def test_payload_qubits_embed_the_circuit_in_a_wider_register(self):
        """With ``payload_qubits``, register qubit ``payload_qubits[v]`` is circuit qubit ``v``."""
        payload = [4, 1, 2]
        stabilizer = _output_stabilizer(self.circuit, [0, 1])
        register = Pauli("I" * 6)
        register.x[payload] = stabilizer.x
        register.z[payload] = stabilizer.z
        self.assertEqual(
            convert_stabilizers([register], self.circuit, payload, num_qubits=6), ["ZZI"]
        )
        off_payload = register.copy()
        off_payload.z[0] = True
        with self.assertRaisesRegex(ValueError, "outside the payload"):
            convert_stabilizers([off_payload], self.circuit, payload, num_qubits=6)
        with self.assertRaisesRegex(ValueError, "acts on 3 qubits; expected 6"):
            convert_stabilizers([stabilizer], self.circuit, payload, num_qubits=6)
        with self.assertRaisesRegex(ValueError, "num_qubits is required"):
            convert_stabilizers([register], self.circuit, payload)

    def test_rejects_wrong_width_and_non_stabilizers(self):
        with self.assertRaisesRegex(ValueError, "acts on 2 qubits; expected 3"):
            convert_stabilizers(["ZZ"], self.circuit)
        generators = [_output_stabilizer(self.circuit, [q]) for q in range(3)]
        not_stabilizer = next(
            label
            for label in ("XII", "IXI", "IIX", "ZII")
            if any(not g.commutes(Pauli(label)) for g in generators)
        )
        with self.assertRaisesRegex(ValueError, "not a stabilizer"):
            convert_stabilizers([not_stabilizer], self.circuit)
