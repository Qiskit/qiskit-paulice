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

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit.quantum_info import PauliLindbladMap
from qiskit.transpiler import CouplingMap
from qiskit_paulice import CheckedCircuit
from qiskit_paulice.checks import (
    _lift_to_isa_circuit,
    _remove_inactive_qubits,
    add_pauli_checks,
)
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


# All 15 non-identity 2-qubit Pauli strings, useful for building per-edge generators.
# Each string is paired left-to-right with the edge tuple ("XY" on (a, b) = X on a, Y on b),
# mirroring the sparse form of :class:`~qiskit.quantum_info.PauliLindbladMap`.
_PAULI_2Q = [p_a + p_b for p_a in "IXYZ" for p_b in "IXYZ" if (p_a, p_b) != ("I", "I")]


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
            _clifford(),
            [0],
            _DEFAULT_NOISE,
            cost="LER",
            cost_nshots=200,
            seed=0,
        )
        _assert_variant_progression(self, result, expected_targets=[0])
        for variant in result:
            self.assertIsInstance(variant.cost, float)
            self.assertEqual(variant.cost_metric, "LER")
            self.assertGreaterEqual(variant.cost, 0.0)
            self.assertLessEqual(variant.cost, 1.0)

    def test_method_genetic(self):
        result = add_pauli_checks(
            _clifford(),
            [0],
            _DEFAULT_NOISE,
            method="genetic",
            seed=0,
        )
        _assert_variant_progression(self, result, expected_targets=[0])
        self.assertIsInstance(result[-1].cost, float)

    def test_method_windowed_genetic(self):
        result = add_pauli_checks(
            _clifford(),
            [0],
            _DEFAULT_NOISE,
            method="windowed_genetic",
            seed=0,
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
            _clifford(),
            [0],
            _DEFAULT_NOISE,
            check_creg_name="my_creg",
            check_qreg_name="my_qreg",
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
        with_bogus = add_pauli_checks(qc, [0], _DEFAULT_NOISE, ancilla_qubits=[99], **kwargs)
        self.assertEqual(baseline[-1].check_qubits, with_bogus[-1].check_qubits)
        self.assertEqual(baseline[-1].cost, with_bogus[-1].cost)


class TestAddPauliChecksAncillaEdgeNoiseInference(unittest.TestCase):
    """Noise models that don't cover the ancilla/target connections inserted by check
    picking still run end-to-end: gate-wise edges and layered layers are inferred."""

    def test_gate_wise_without_ancilla_edges(self):
        """Gate-wise noise on payload edges only — ancilla/target edges are inferred."""
        # Payload edges from `_clifford(layers=2)`: (0, 1) and (1, 2). No ancilla edges.
        gate_noise = {
            (0, 1): [(p, 1e-4) for p in _PAULI_2Q],
            (1, 0): [(p, 1e-4) for p in _PAULI_2Q],
            (1, 2): [(p, 1e-4) for p in _PAULI_2Q],
            (2, 1): [(p, 1e-4) for p in _PAULI_2Q],
        }
        noise = NoiseModel(gate_noise=gate_noise, readout_noise=1e-2)
        result = add_pauli_checks(_clifford(), [0], noise, seed=0)
        _assert_variant_progression(self, result, expected_targets=[0])
        self.assertIsInstance(result[-1].cost, float)

    def test_layered_without_ancilla_layers(self):
        """Layered noise covering only payload layers used to panic on the ancilla
        gates added by check insertion; it now infers them."""
        # Layered noise requires a CZ-only circuit (the layering pass rejects CNOTs).
        qc = QuantumCircuit(3)
        for _ in range(2):
            qc.h(0)
            qc.cz(0, 1)
            qc.cz(1, 2)
            qc.s(0)
            qc.s(2)
        qc.measure_all()
        # Two payload layers (position i in the Pauli string maps to qubit i).
        layered_noise: dict[tuple[tuple[int, int], ...], list[tuple[str, float]]] = {
            # Pauli strings in Qiskit convention: rightmost = qubit 0.
            ((0, 1),): [("IYX", 1e-4)],
            ((1, 2),): [("YXI", 1e-4)],
        }
        noise = NoiseModel(gate_noise=layered_noise, readout_noise=1e-2)
        result = add_pauli_checks(qc, [0], noise, seed=0)
        _assert_variant_progression(self, result, expected_targets=[0])
        self.assertIsInstance(result[-1].cost, float)

    def test_layered_factory_with_asymmetric_generator_end_to_end(self):
        """An asymmetric Pauli generator built via from_pauli_lindblad_maps flows through
        add_pauli_checks end-to-end (covers the full factory → boundary → picker path)."""
        qc = QuantumCircuit(3)
        for _ in range(2):
            qc.h(0)
            qc.cz(0, 1)
            qc.cz(1, 2)
            qc.s(0)
            qc.s(2)
        qc.measure_all()
        # Asymmetric XZ generator at indices [0, 1] in a 3-qubit system: X on q0, Z on q1.
        plm = PauliLindbladMap([("XZ", [0, 1], 1e-3)], num_qubits=3)
        noise = NoiseModel.from_pauli_lindblad_maps([plm])
        noise.readout_noise = 1e-2
        result = add_pauli_checks(qc, [0], noise, seed=0)
        _assert_variant_progression(self, result, expected_targets=[0])
        self.assertIsInstance(result[-1].cost, float)

    def test_inferred_ancilla_noise_shifts_picker_cost(self):
        """Inferred gate-wise noise on the ancilla/target edges is actually applied — the
        picker's cost is strictly worse than a baseline where the ancilla edges are
        explicitly given empty (zero-rate) noise so the inference fallback doesn't fire."""
        payload_rate = 1e-3
        payload_gens = [(p, payload_rate) for p in _PAULI_2Q]
        payload = {
            (0, 1): payload_gens,
            (1, 0): payload_gens,
            (1, 2): payload_gens,
            (2, 1): payload_gens,
        }
        # Inferred-ancilla model: ancilla edges are absent → fallback fires with median rate.
        inferred = NoiseModel(gate_noise=dict(payload), readout_noise=1e-2)
        # Explicit-empty model: ancilla edges present with empty generator lists, so the
        # Rust applicator finds them and skips the inference fallback. The non-ISA ancilla
        # for a 3-qubit input is qubit 3.
        ancilla_edges = {(3, 0): [], (0, 3): [], (3, 1): [], (1, 3): [], (3, 2): [], (2, 3): []}
        explicit_empty = NoiseModel(
            gate_noise={**payload, **ancilla_edges},
            readout_noise=1e-2,
        )

        r_inferred = add_pauli_checks(_clifford(), [0], inferred, cost="gamma", seed=0)
        r_empty = add_pauli_checks(_clifford(), [0], explicit_empty, cost="gamma", seed=0)
        # Inferred ancilla noise adds uncaught logical errors; the gamma cost must be
        # strictly higher than the zero-ancilla-noise baseline.
        self.assertGreater(r_inferred[-1].cost, r_empty[-1].cost)


class TestAddPauliChecksErrorPaths(unittest.TestCase):
    """Argument-validation error paths for :func:`add_pauli_checks`."""

    def test_invalid_cost_value_raises(self):
        with self.assertRaisesRegex(ValueError, "Invalid cost value"):
            add_pauli_checks(_clifford(), [0], _DEFAULT_NOISE, cost="not_a_metric", seed=0)

    def test_isa_without_ancilla_qubits_raises(self):
        with self.assertRaisesRegex(ValueError, "ancilla_qubits"):
            add_pauli_checks(_isa_circuit(), [5], _DEFAULT_NOISE, seed=0)

    def test_isa_wrong_length_ancilla_qubits_raises(self):
        with self.assertRaisesRegex(ValueError, "one entry per target"):
            add_pauli_checks(_isa_circuit(), [5, 7], _DEFAULT_NOISE, ancilla_qubits=[4], seed=0)

    def test_isa_out_of_range_ancilla_qubits_raises(self):
        with self.assertRaisesRegex(ValueError, "physical qubit indices"):
            add_pauli_checks(_isa_circuit(), [5], _DEFAULT_NOISE, ancilla_qubits=[999], seed=0)

    def test_isa_duplicate_ancilla_qubits_raises(self):
        with self.assertRaisesRegex(ValueError, "duplicates"):
            add_pauli_checks(_isa_circuit(), [5, 7], _DEFAULT_NOISE, ancilla_qubits=[4, 4], seed=0)

    def test_isa_ancilla_overlap_with_payload_raises(self):
        # _PAYLOAD_PHYS = [5, 6, 7] — passing one of those as ancilla overlaps the payload.
        with self.assertRaisesRegex(ValueError, "overlap with payload"):
            add_pauli_checks(_isa_circuit(), [5], _DEFAULT_NOISE, ancilla_qubits=[6], seed=0)

    def test_isa_check_creg_name_collision_raises(self):
        # Build an ISA circuit that already has a creg named "checks_c".
        circuit_isa = _isa_circuit()
        circuit_isa.add_register(ClassicalRegister(1, "checks_c"))
        with self.assertRaisesRegex(ValueError, "classical register named"):
            add_pauli_checks(circuit_isa, [5], _DEFAULT_NOISE, ancilla_qubits=[4], seed=0)

    def test_qreg_creg_name_collision_raises(self):
        with self.assertRaisesRegex(ValueError, "must differ"):
            add_pauli_checks(
                _clifford(),
                [0],
                _DEFAULT_NOISE,
                check_qreg_name="same",
                check_creg_name="same",
                seed=0,
            )

    def test_existing_qreg_name_collision_raises(self):
        qc = QuantumCircuit(QuantumRegister(3, "checks_q"))
        qc.measure_all()
        with self.assertRaisesRegex(ValueError, "quantum register named"):
            add_pauli_checks(qc, [0], _DEFAULT_NOISE, seed=0)

    def test_existing_creg_name_collision_raises(self):
        qc = _clifford()
        qc.add_register(ClassicalRegister(1, "checks_c"))
        with self.assertRaisesRegex(ValueError, "classical register named"):
            add_pauli_checks(qc, [0], _DEFAULT_NOISE, seed=0)

    def test_empty_noise_model_raises(self):
        with self.assertRaisesRegex(ValueError, "may not be empty"):
            add_pauli_checks(_clifford(), [0], NoiseModel(), seed=0)

    def test_empty_gate_noise_dict_is_ignored(self):
        # Empty dict trips the early-return guards in _is_layered_gate_noise /
        # _is_gate_wise_noise; with non-None readout, the call still succeeds.
        noise = NoiseModel(gate_noise={}, readout_noise=1e-2)
        result = add_pauli_checks(_clifford(), [0], noise, seed=0)
        _assert_variant_progression(self, result, expected_targets=[0])

    def test_seed_none_smoke(self):
        # The ``seed is None`` branch isn't covered by tests that always pass a seed.
        result = add_pauli_checks(_clifford(), [0], _DEFAULT_NOISE)
        _assert_variant_progression(self, result, expected_targets=[0])


class TestInternalHelpers(unittest.TestCase):
    """Direct exercises of internal helpers for paths the public API can't reach."""

    def test_remove_inactive_qubits_falls_back_to_anonymous_qreg(self):
        # qregs=None forces the fallback to a single anonymous "q" register. Also
        # include a measure (measure-skip branch) and a gate on an out-of-range qubit
        # (the "all q < n_keep" guard).
        circ = QuantumCircuit(5, 1)
        circ.cx(0, 1)
        circ.cx(0, 4)  # qubit 4 is beyond n_keep=3 -> dropped by the helper guard
        circ.measure(0, 0)

        out = _remove_inactive_qubits(circ, num_original_qubits=3, num_active_checks=0)
        self.assertEqual(out.num_qubits, 3)
        self.assertEqual([qr.name for qr in out.qregs], ["q"])
        # The measure and the out-of-range cx must both be stripped.
        op_names = [inst.operation.name for inst in out.data]
        self.assertEqual(op_names, ["cx"])

    def test_strip_measurements_uses_correct_creg_when_multiple_present(self):
        # _strip_measurements_cregs_barriers iterates ``circuit.cregs`` to find which
        # creg owns a measure's clbit. With multiple cregs and a measure into the
        # second one, the inner loop must skip past the first creg before matching.
        from qiskit_paulice.checks import _strip_measurements_cregs_barriers

        cr1 = ClassicalRegister(1, "c1")
        cr2 = ClassicalRegister(1, "c2")
        qc = QuantumCircuit(2)
        qc.add_register(cr1)
        qc.add_register(cr2)
        qc.h(0)
        qc.measure(0, cr2[0])

        _, measurement_info, _, _ = _strip_measurements_cregs_barriers(qc)
        # One measurement recorded, attributed to ``c2`` (not ``c1``).
        self.assertEqual(measurement_info, [(0, 0, "c2")])

    def test_loose_clbit_measurement_raises(self):
        # Measurements into a Clbit that isn't part of any ClassicalRegister can't be
        # round-tripped through the picker (the strip step records measurements as
        # (qubit, position-in-creg, creg-name) and has nowhere to put a loose clbit).
        # We raise explicitly rather than silently dropping the measurement.
        from qiskit.circuit import Clbit

        loose = Clbit()
        qc = QuantumCircuit(2)
        qc.add_bits([loose])
        qc.h(0)
        qc.measure(0, loose)
        with self.assertRaisesRegex(ValueError, "loose Clbits"):
            add_pauli_checks(qc, [0], _DEFAULT_NOISE, seed=0)

    def test_isa_circuit_drops_gates_outside_payload(self):
        # The ISA rebuild filters gates whose qubits aren't all in the payload-physical
        # mapping. Add such a gate post-transpile and verify the call still succeeds.
        circuit_isa = _isa_circuit()
        # Gate on physical qubits (2, 3): neither is in payload [5, 6, 7], so the
        # virtual reconstruction must drop it.
        circuit_isa.cx(2, 3)
        result = add_pauli_checks(circuit_isa, [5], _DEFAULT_NOISE, ancilla_qubits=[4], seed=0)
        _assert_variant_progression(self, result, expected_targets=[5])

    def test_lift_to_isa_circuit_drops_inactive_ancilla_and_measure(self):
        # A synthetic variant with (a) a gate that survives, (b) a gate on an
        # "inactive" ancilla index (beyond num_payload_virtual + num_active_checks),
        # and (c) a measure. The lift helper must keep (a) and drop (b) and (c).
        variant = QuantumCircuit(5, 1)
        variant.cx(0, 1)  # payload-only gate, should survive
        variant.cx(0, 4)  # touches inactive ancilla index 4 -> dropped
        variant.measure(0, 0)  # measure in variant -> dropped

        out = _lift_to_isa_circuit(
            variant=variant,
            qregs=[QuantumRegister(10, "q")],
            cregs=[],
            measurement_info=[],
            num_payload_virtual=2,
            num_active_checks=1,
            payload_phys=[5, 6],
            ancilla_qubits=[8],
        )
        # Only the surviving payload cx should remain (no measure, no inactive-ancilla gate).
        kept = [inst for inst in out.data if inst.operation.name not in ("measure",)]
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].operation.name, "cx")
