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

"""Tests for the ``checks`` module."""

from __future__ import annotations

import unittest

from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap
from qiskit_paulice import CheckedCircuit
from qiskit_paulice.checks import add_pauli_checks
from qiskit_paulice.noise_models import NoiseModel

_DEFAULT_NOISE = NoiseModel(gate_noise=1e-3, readout_noise=1e-2)
_PAYLOAD_PHYS = [5, 6, 7]
_ANCILLA_PHYS = [4, 8]


def _clifford(nq: int = 3, layers: int = 2) -> QuantumCircuit:
    """A Clifford circuit with terminal measurements, parameterized by depth."""
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
    """Transpile a deep Clifford onto a line with an explicit layout."""
    isa = transpile(
        _clifford(layers=4),
        coupling_map=CouplingMap.from_line(10),
        basis_gates=["h", "s", "cx", "measure"],
        initial_layout=_PAYLOAD_PHYS,
        optimization_level=0,
    )
    assert isa.layout is not None, "expected transpile to record a layout (ISA mode)"
    return isa


def _assert_variant_progression(test, result, *, expected_targets):
    """Validate the shape contract of an ``add_pauli_checks`` result.

    * Output is a list of ``len(expected_targets) + 1`` ``CheckedCircuit``\\ s.
    * Variant ``k`` has exactly ``k`` checks committed; its ``target_qubits``,
      ``check_qubits``, and ``check_support`` are length-``k`` prefixes of the
      fully-checked variant's values.
    * The first variant is bare; the last carries all targets in order.
    """
    n = len(expected_targets)
    test.assertEqual(len(result), n + 1)
    final = result[-1]
    test.assertEqual(final.target_qubits, tuple(expected_targets))
    for k, variant in enumerate(result):
        test.assertIsInstance(variant, CheckedCircuit)
        test.assertEqual(len(variant.target_qubits), k)
        test.assertEqual(len(variant.check_qubits), k)
        test.assertEqual(len(variant.check_support), k)
        test.assertEqual(variant.target_qubits, final.target_qubits[:k])
        test.assertEqual(variant.check_qubits, final.check_qubits[:k])
        test.assertEqual(variant.check_support, final.check_support[:k])


class TestAddPauliChecksBasic(unittest.TestCase):
    """Test basic usage."""

    def test_minimal_non_isa_call(self):
        """Basic virtual circuit."""
        qc = _clifford()
        result = add_pauli_checks(qc, [0], _DEFAULT_NOISE, seed=0)

        _assert_variant_progression(self, result, expected_targets=[0])

        bare, checked = result
        # Bare variant reuses the input width; the checked variant adds one
        # ancilla qubit in its own qreg.
        self.assertEqual(bare.circuit.num_qubits, qc.num_qubits)
        self.assertEqual(checked.circuit.num_qubits, qc.num_qubits + 1)

        # The committed check qubit lives in the appended ancilla register
        # (i.e. outside the original payload range).
        (check_q,) = checked.check_qubits
        self.assertGreaterEqual(check_q, qc.num_qubits)
        self.assertLess(check_q, checked.circuit.num_qubits)

        # Cost metadata is populated for every variant (the bare variant carries
        # the baseline cost the picker is trying to improve on).
        self.assertIsInstance(bare.cost, float)
        self.assertIsInstance(checked.cost, float)
        self.assertEqual(bare.cost_metric, "gamma")
        self.assertEqual(checked.cost_metric, "gamma")

        # Every check has non-empty support (must touch at least the target).
        for support in checked.check_support:
            self.assertGreater(len(support), 0)


class TestAddPauliChecksTargetIndexing(unittest.TestCase):
    """Covers the ISA physical-target contract and non-ISA range checks."""

    def test_isa_accepts_physical_targets(self):
        """In ISA mode, target_qubits are physical indices the payload occupies."""
        circuit_isa = _isa_circuit()
        targets = [5, 7]  # physical, in the payload
        result = add_pauli_checks(
            circuit_isa,
            targets,
            _DEFAULT_NOISE,
            ancilla_qubits=_ANCILLA_PHYS,
            cost="gamma",
            method="windowed",
            seed=123,
        )

        _assert_variant_progression(self, result, expected_targets=targets)

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
                _DEFAULT_NOISE,
                ancilla_qubits=_ANCILLA_PHYS,
                cost="gamma",
                method="windowed",
                seed=123,
            )

    def test_non_isa_out_of_range_targets(self):
        """Non-ISA: targets index the circuit you passed; out-of-range errors."""
        qc = _clifford(layers=4)
        with self.assertRaisesRegex(ValueError, "out of range"):
            add_pauli_checks(
                qc,
                [99],
                _DEFAULT_NOISE,
                cost="gamma",
                method="windowed",
                seed=123,
            )


# All 15 non-identity 2-qubit Pauli labels, useful for building per-edge generators.
_PAULI_2Q = [
    "IX", "IY", "IZ",
    "XI", "XX", "XY", "XZ",
    "YI", "YX", "YY", "YZ",
    "ZI", "ZX", "ZY", "ZZ",
]


class TestAddPauliChecksPermutations(unittest.TestCase):
    """Permutations of the input args to :func:`add_pauli_checks`.

    Each test exercises one dimension of the input space (cost metric, method,
    noise-model shape, register naming, seed determinism, ...) and asserts the
    call runs end-to-end and returns a shape-valid result. Corner cases (errors,
    invalid combinations) are covered separately.
    """

    def test_cost_ler(self):
        """``cost="LER"`` produces a probability cost in [0, 1] for each variant."""
        result = add_pauli_checks(
            _clifford(), [0], _DEFAULT_NOISE,
            cost="LER", cost_nshots=200, seed=0,
        )
        _assert_variant_progression(self, result, expected_targets=[0])
        for variant in result:
            self.assertIsInstance(variant.cost, float)
            self.assertEqual(variant.cost_metric, "LER")
            self.assertGreaterEqual(variant.cost, 0.0)
            self.assertLessEqual(variant.cost, 1.0)

    def test_method_genetic(self):
        result = add_pauli_checks(
            _clifford(), [0], _DEFAULT_NOISE, method="genetic", seed=0,
        )
        _assert_variant_progression(self, result, expected_targets=[0])
        self.assertIsInstance(result[-1].cost, float)

    def test_method_windowed_genetic(self):
        result = add_pauli_checks(
            _clifford(), [0], _DEFAULT_NOISE, method="windowed_genetic", seed=0,
        )
        _assert_variant_progression(self, result, expected_targets=[0])
        self.assertIsInstance(result[-1].cost, float)

    def test_gate_wise_noise(self):
        """Per-edge dict noise keyed by ``(a, b)`` int pairs."""
        gate_noise = {
            (0, 1): [(p, 1e-4) for p in _PAULI_2Q],
            (1, 2): [(p, 1e-4) for p in _PAULI_2Q],
        }
        noise = NoiseModel(gate_noise=gate_noise, readout_noise=1e-2)
        result = add_pauli_checks(_clifford(), [0], noise, seed=0)
        _assert_variant_progression(self, result, expected_targets=[0])
        self.assertIsInstance(result[-1].cost, float)

    def test_readout_only_noise(self):
        """Picker accepts a noise model with no gate noise."""
        noise = NoiseModel(readout_noise=1e-2)
        result = add_pauli_checks(_clifford(), [0], noise, seed=0)
        _assert_variant_progression(self, result, expected_targets=[0])
        self.assertIsInstance(result[-1].cost, float)

    def test_custom_register_names(self):
        """Custom check_qreg_name / check_creg_name propagate to the output circuit."""
        result = add_pauli_checks(
            _clifford(), [0], _DEFAULT_NOISE,
            check_creg_name="my_creg", check_qreg_name="my_qreg",
            seed=0,
        )
        final = result[-1].circuit
        self.assertIn("my_qreg", {qr.name for qr in final.qregs})
        self.assertIn("my_creg", {cr.name for cr in final.cregs})

    def test_seed_reproducibility_for_gamma_windowed(self):
        """``cost="gamma"`` + ``method="windowed"`` is fully deterministic with a seed."""
        qc = _clifford()
        kwargs = dict(cost="gamma", method="windowed", seed=42)
        r1 = add_pauli_checks(qc, [0, 1], _DEFAULT_NOISE, **kwargs)
        r2 = add_pauli_checks(qc, [0, 1], _DEFAULT_NOISE, **kwargs)
        self.assertEqual([v.check_qubits for v in r1], [v.check_qubits for v in r2])
        self.assertEqual([v.check_support for v in r1], [v.check_support for v in r2])
        self.assertEqual([v.cost for v in r1], [v.cost for v in r2])

    def test_non_isa_silently_ignores_ancilla_qubits(self):
        """In non-ISA mode ``ancilla_qubits`` has no effect (documented behavior)."""
        qc = _clifford()
        kwargs = dict(cost="gamma", method="windowed", seed=0)
        baseline = add_pauli_checks(qc, [0], _DEFAULT_NOISE, **kwargs)
        # A value that would error in ISA mode (out-of-range index) is silently
        # ignored here.
        with_bogus = add_pauli_checks(
            qc, [0], _DEFAULT_NOISE, ancilla_qubits=[99], **kwargs
        )
        self.assertEqual(baseline[-1].check_qubits, with_bogus[-1].check_qubits)
        self.assertEqual(baseline[-1].cost, with_bogus[-1].cost)
