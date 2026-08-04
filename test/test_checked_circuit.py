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
import warnings
from collections import Counter

import numpy as np
import samplomatic
from qiskit import QuantumCircuit
from qiskit.quantum_info import Clifford
from qiskit.transpiler.passes import RemoveBarriers
from qiskit_paulice import CheckedCircuit, UncoveredPauli, add_pauli_checks
from qiskit_paulice.checked_circuit import BOXING_DEFAULTS
from qiskit_paulice.noise_models import NoiseModel
from samplomatic.annotations import InjectNoise
from samplomatic.transpiler import generate_boxing_pass_manager
from samplomatic.utils import get_annotation


def _bell_with_measure() -> QuantumCircuit:
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure(0, 0)
    qc.measure(1, 1)
    return qc


def _bare_circuit(nq=4, depth=4, barriers=False):
    """A brickwork Clifford payload, optionally with its layer boundaries marked."""
    qc = QuantumCircuit(nq)
    qc.h(range(nq))
    for d in range(depth):
        for i in range(d % 2, nq - 1, 2):
            qc.cz(i, i + 1)
        for q in range(nq):
            qc.sx(q)
        if barriers:
            qc.barrier()
    qc.measure_all()
    return qc


def _brickwork_layers(nq):
    """The two unique entangling layers of the brickwork payload."""
    return [{(i, i + 1) for i in range(0, nq - 1, 2)}, {(i, i + 1) for i in range(1, nq - 1, 2)}]


def _gate_counts(circuit):
    """Gate tallies, ignoring the barriers that mark layer boundaries."""
    return Counter(inst.operation.name for inst in circuit.data if inst.operation.name != "barrier")


def _box_edges(instruction, boxed):
    """The entangled qubit pairs inside one box."""
    body = instruction.operation.blocks[0]
    qmap = [boxed.find_bit(q).index for q in instruction.qubits]
    return [
        tuple(sorted(qmap[body.find_bit(q).index] for q in sub.qubits))
        for sub in body.data
        if len(sub.qubits) == 2
    ]


def _box_edge_sets(boxed):
    """The set of entangled qubit pairs inside each box, in circuit order."""
    out = []
    for instruction in boxed.data:
        if instruction.operation.name != "box":
            continue
        edges = frozenset(_box_edges(instruction, boxed))
        if edges:
            out.append(edges)
    return out


def _checked_example(nq=4, depth=4, seed=1):
    """A ``CheckedCircuit`` and its boxed form."""
    qc = _bare_circuit(nq, depth)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        checked = add_pauli_checks(
            qc, list(range(nq)), NoiseModel(gate_noise=1e-3, readout_noise=1e-2), seed=seed
        )[-1]
        boxed = checked.box()
    return checked, boxed


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


class TestBox(unittest.TestCase):
    """Tests for ``CheckedCircuit.box``."""

    def test_boxes_the_executed_circuit(self):
        """Every entangling gate of the checked circuit -- check gates included -- lands in a box."""
        checked, boxed = _checked_example()
        ancillas = set(checked.check_qubits)
        circuit_edges = [
            tuple(sorted(checked.circuit.find_bit(q).index for q in inst.qubits))
            for inst in checked.circuit.data
            if len(inst.qubits) == 2
        ]
        boxed_edges = [
            edge
            for instruction in boxed.data
            if instruction.operation.name == "box"
            for edge in _box_edges(instruction, boxed)
        ]
        self.assertEqual(sorted(boxed_edges), sorted(circuit_edges))
        # the check gates are really in there
        self.assertTrue(any(set(e) & ancillas for e in boxed_edges))

    def test_rejects_non_gate_instructions(self):
        """Anything but unitary gates, measures, and barriers is rejected.

        Resets alter the check propagation, and control flow can be reordered by
        stratification, which tracks qubit wires but not clbit dataflow.
        """
        checked, _ = _checked_example()
        checked.circuit.reset(0)
        with self.assertRaisesRegex(ValueError, "reset"):
            checked.box()


class TestIsolatedCheckLayers(unittest.TestCase):
    """Tests for the isolated check layers ``CheckedCircuit.box`` produces."""

    def setUp(self):
        self.checked, self.isolated = _checked_example(nq=6, depth=8, seed=4)

    def test_same_circuit(self):
        """Isolating check gates only regroups them: same gates, same unitary."""
        stripped = self.checked._stratify(None)
        self.assertEqual(_gate_counts(self.checked.circuit), _gate_counts(stripped))
        original = RemoveBarriers()(self.checked.circuit.remove_final_measurements(inplace=False))
        restratified = RemoveBarriers()(stripped.remove_final_measurements(inplace=False))
        self.assertEqual(Clifford(original), Clifford(restratified))

    def test_each_check_gate_boxed_alone(self):
        """A check box is exactly its one gate: a single edge on a two-qubit box."""
        ancillas = set(self.checked.check_qubits)
        saw_check_box = False
        for instruction in self.isolated.data:
            if instruction.operation.name != "box":
                continue
            edges = _box_edges(instruction, self.isolated)
            if any(set(e) & ancillas for e in edges):
                self.assertEqual(len(edges), 1)
                self.assertEqual(len(instruction.qubits), 2)
                saw_check_box = True
        self.assertTrue(saw_check_box)

    def test_unique_layers_is_payload_plus_one_per_check(self):
        """The whole point: two brickwork payload layers plus one unique layer per check."""
        ancillas = set(self.checked.check_qubits)
        payload, check = set(), set()
        for edges in _box_edge_sets(self.isolated):
            (check if any(set(e) & ancillas for e in edges) else payload).add(edges)
        self.assertEqual(len(payload), 2)
        self.assertEqual(len(check), len(self.checked.check_support))
        # ... and every layer recurs rather than proliferating
        self.assertLess(len(check) + len(payload), sum(1 for _ in _box_edge_sets(self.isolated)))

    def test_payload_layers_split_what_packing_would_merge(self):
        """The palette is authoritative: gates that packing would share a stratum get split."""
        qc = QuantumCircuit(4)
        qc.cz(0, 1)
        qc.cz(2, 3)
        qc.measure_all()
        checked = CheckedCircuit(circuit=qc)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            packed = checked.box()
            # the reversed edge also checks pair normalization
            split = checked.box(payload_layers=[{(1, 0)}, {(2, 3)}])
        self.assertEqual(len(set(_box_edge_sets(packed))), 1)
        self.assertEqual(len(set(_box_edge_sets(split))), 2)

    def test_rejects_foreign_payload_layers(self):
        """Layers that do not cover the payload's edges are an error, not a mislabelling."""
        with self.assertRaises(ValueError):
            self.checked.box(payload_layers=_brickwork_layers(4))

    def test_rejects_ambiguous_payload_layers(self):
        """An edge sitting in two unique layers has no well-defined boxing."""
        layers = _brickwork_layers(6)
        layers[1].add((0, 1))
        with self.assertRaises(ValueError):
            self.checked.box(payload_layers=layers)

    def test_builds_a_samplex(self):
        """The isolated boxing is a working samplomatic circuit."""
        samplomatic.build(self.isolated)


class TestBareModelReuse(unittest.TestCase):
    """A model learned once on the bare circuit binds to the checked circuit's payload boxes."""

    def setUp(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.bare = _bare_circuit(nq=6, depth=8, barriers=True)
            self.checked = add_pauli_checks(
                RemoveBarriers()(self.bare),
                list(range(6)),
                NoiseModel(gate_noise=1e-3, readout_noise=1e-2),
                seed=4,
            )[-1]
            self.isolated = self.checked.box(payload_layers=_brickwork_layers(6))
            # the bare circuit boxed exactly as the checked circuit's payload boxes are
            self.boxed_bare = generate_boxing_pass_manager(**BOXING_DEFAULTS).run(self.bare)

    def test_payload_refs_come_from_the_bare_circuit(self):
        """Every payload box carries a ref the bare circuit's boxing also carries."""
        bare_refs = {
            inject.ref
            for instruction in self.boxed_bare.data
            if instruction.operation.name == "box"
            and (inject := get_annotation(instruction.operation, InjectNoise)) is not None
            and inject.ref
        }
        ancillas = set(self.checked.check_qubits)
        payload_refs, check_refs = set(), set()
        for instruction in self.isolated.data:
            if instruction.operation.name != "box":
                continue
            inject = get_annotation(instruction.operation, InjectNoise)
            edges = _box_edges(instruction, self.isolated)
            if inject is None or not edges:
                continue
            target = check_refs if any(set(e) & ancillas for e in edges) else payload_refs
            target.add(inject.ref)
        self.assertTrue(payload_refs)
        self.assertLessEqual(payload_refs, bare_refs)
        # the check layers are the only thing the bare model does not cover
        self.assertFalse(check_refs & bare_refs)
        # a brickwork payload keeps its two layers, and each check contributes one small layer
        self.assertEqual(len(payload_refs), 2)
        self.assertEqual(len(check_refs), len(self.checked.check_support))
