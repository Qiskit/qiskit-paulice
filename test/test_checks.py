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

"""Test checks module (target-qubit indexing in ISA / non-ISA modes)."""

from __future__ import annotations

import unittest

from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap
from qiskit_paulice.checks import add_pauli_checks
from qiskit_paulice.noise_models import NoiseModel

# Physical qubits the payload occupies after transpilation. Chosen adjacent on
# a line so no routing swaps are inserted at optimization_level=0.
_PAYLOAD_PHYS = [5, 6, 7]
_ANCILLA_PHYS = [4, 8]


def _deep_clifford(nq: int = 3, layers: int = 4) -> QuantumCircuit:
    """A Clifford circuit deep enough that each qubit has several wires."""
    qc = QuantumCircuit(nq)
    for _ in range(layers):
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.s(0)
        qc.s(2)
    qc.measure_all()
    return qc


def _isa_circuit() -> QuantumCircuit:
    """Transpile the Clifford circuit onto a line with an explicit layout."""
    qc = _deep_clifford()
    isa = transpile(
        qc,
        coupling_map=CouplingMap.from_line(10),
        basis_gates=["h", "s", "cx", "measure"],
        initial_layout=_PAYLOAD_PHYS,
        optimization_level=0,
    )
    assert isa.layout is not None, "expected transpile to record a layout (ISA mode)"
    return isa


class TestAddPauliChecksTargetIndexing(unittest.TestCase):
    """Covers the ISA physical-target contract and non-ISA range checks."""

    def setUp(self):
        self.noise_model = NoiseModel(gate_noise=1e-3, readout_noise=1e-2)

    def test_isa_accepts_physical_targets(self):
        """In ISA mode, target_qubits are physical indices the payload occupies."""
        circuit_isa = _isa_circuit()
        targets = [5, 7]  # physical, in the payload
        result = add_pauli_checks(
            circuit_isa,
            targets,
            self.noise_model,
            ancilla_qubits=_ANCILLA_PHYS,
            cost="gamma",
            method="windowed",
            seed=123,
        )

        # One variant per added check, plus the bare circuit.
        self.assertEqual(len(result), len(targets) + 1)

        # Bare circuit has no checks.
        self.assertEqual(result[0].target_qubits, ())
        self.assertEqual(result[0].check_qubits, ())

        # Fully-checked variant reports the *physical* targets we passed.
        self.assertEqual(result[-1].target_qubits, (5, 7))

        # Ancillas come from the physical ancilla pool we supplied.
        self.assertEqual(len(result[-1].check_qubits), 2)
        self.assertLessEqual(set(result[-1].check_qubits), set(_ANCILLA_PHYS))

        # ISA output reuses the transpiled physical width (no new qreg).
        self.assertEqual(result[-1].circuit.num_qubits, circuit_isa.num_qubits)

    def test_isa_rejects_non_payload_targets(self):
        """Old-style virtual indices (not payload physical qubits) error loudly."""
        circuit_isa = _isa_circuit()
        # [0, 1] are valid *virtual* indices but not payload physical qubits
        # (payload is physical [5, 6, 7]); this must raise, not silently work.
        with self.assertRaisesRegex(ValueError, "not payload qubits"):
            add_pauli_checks(
                circuit_isa,
                [0, 1],
                self.noise_model,
                ancilla_qubits=_ANCILLA_PHYS,
                cost="gamma",
                method="windowed",
                seed=123,
            )

    def test_non_isa_out_of_range_targets(self):
        """Non-ISA: targets index the circuit you passed; out-of-range errors."""
        qc = _deep_clifford()
        with self.assertRaisesRegex(ValueError, "out of range"):
            add_pauli_checks(
                qc,
                [99],
                self.noise_model,
                cost="gamma",
                method="windowed",
                seed=123,
            )

    def test_non_isa_happy_path_unchanged(self):
        """Non-ISA path still accepts circuit-native target indices."""
        qc = _deep_clifford()
        result = add_pauli_checks(
            qc,
            [0, 1],
            self.noise_model,
            cost="gamma",
            method="windowed",
            seed=123,
        )
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].target_qubits, ())
        self.assertEqual(result[-1].target_qubits, (0, 1))
