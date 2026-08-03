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

"""Tests for ``CheckedCircuit.box``."""

from __future__ import annotations

import unittest
import warnings
from collections import Counter

import samplomatic
from qiskit import QuantumCircuit
from qiskit.quantum_info import Clifford
from qiskit.transpiler.passes import RemoveBarriers
from qiskit_paulice import add_pauli_checks
from qiskit_paulice.checked_circuit import (
    _BOXING_DEFAULTS,
    _ISOLATED_BOXING_DEFAULTS,
)
from qiskit_paulice.noise_models import NoiseModel
from samplomatic.annotations import InjectNoise
from samplomatic.transpiler import generate_boxing_pass_manager
from samplomatic.utils import get_annotation


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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            merged = checked.box(check_layers="merged")
        for layout, circ in [("isolated", boxed), ("merged", merged)]:
            with self.subTest(layout):
                boxed_edges = [
                    edge
                    for instruction in circ.data
                    if instruction.operation.name == "box"
                    for edge in _box_edges(instruction, circ)
                ]
                self.assertEqual(sorted(boxed_edges), sorted(circuit_edges))
                # the check gates are really in there
                self.assertTrue(any(set(e) & ancillas for e in boxed_edges))

    def test_rejects_resets(self):
        """Resets alter the Heisenberg evolution and are rejected until supported."""
        checked, _ = _checked_example()
        checked.circuit.reset(0)
        with self.assertRaises(ValueError):
            checked.box()

    def test_rejects_unknown_check_layers(self):
        """A typo'd strategy is an error, not a silent fallback."""
        checked, _ = _checked_example()
        with self.assertRaises(ValueError):
            checked.box(check_layers="seperate")


class TestIsolatedCheckLayers(unittest.TestCase):
    """Tests for ``CheckedCircuit.box(check_layers="isolated")``."""

    def setUp(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.checked, _ = _checked_example(nq=6, depth=8, seed=4)
            self.isolated = self.checked.box(check_layers="isolated")

    def test_same_circuit(self):
        """Isolating check gates only regroups them: same gates, same unitary."""
        stripped = self.checked._stratify(None)
        self.assertEqual(_gate_counts(self.checked.circuit), _gate_counts(stripped))
        original = RemoveBarriers()(self.checked.circuit.remove_final_measurements(inplace=False))
        restratified = RemoveBarriers()(stripped.remove_final_measurements(inplace=False))
        self.assertEqual(Clifford(original), Clifford(restratified))

    def test_each_check_gate_boxed_alone(self):
        """No box mixes a check gate with anything else."""
        ancillas = set(self.checked.check_qubits)
        saw_check_box = False
        for edges in _box_edge_sets(self.isolated):
            if any(set(e) & ancillas for e in edges):
                self.assertEqual(len(edges), 1)
                saw_check_box = True
        self.assertTrue(saw_check_box)

    def test_unique_layers_is_payload_plus_one_per_check(self):
        """The whole point: check layers contribute exactly one unique layer each."""
        ancillas = set(self.checked.check_qubits)
        payload, check = set(), set()
        for edges in _box_edge_sets(self.isolated):
            (check if any(set(e) & ancillas for e in edges) else payload).add(edges)
        self.assertEqual(len(check), len(self.checked.check_support))
        # ... and every check's single-gate layer recurs rather than proliferating
        self.assertLess(len(check) + len(payload), sum(1 for _ in _box_edge_sets(self.isolated)))

    def test_preserves_authored_payload_layers(self):
        """Given the bare circuit's layer boundaries, the payload's own layers come back exactly."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bare = _bare_circuit(nq=6, depth=8, barriers=True)
            checked = add_pauli_checks(
                RemoveBarriers()(bare),
                list(range(6)),
                NoiseModel(gate_noise=1e-3, readout_noise=1e-2),
                seed=4,
            )[-1]
            isolated = checked.box(check_layers="isolated", payload_layers=bare)
        ancillas = set(checked.check_qubits)
        payload = {
            edges for edges in _box_edge_sets(isolated) if not any(set(e) & ancillas for e in edges)
        }
        # the brickwork payload has exactly two distinct entangling layers
        self.assertEqual(len(payload), 2)

    def test_rejects_foreign_payload_layers(self):
        """Layer boundaries from a different circuit are an error, not a mislabelling."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            other = _bare_circuit(nq=6, depth=2, barriers=True)
        with self.assertRaises(ValueError):
            self.checked.box(check_layers="isolated", payload_layers=other)

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
            self.isolated = self.checked.box(check_layers="isolated", payload_layers=self.bare)
            # the bare circuit boxed exactly as the checked circuit's payload boxes are
            self.boxed_bare = generate_boxing_pass_manager(
                **{**_BOXING_DEFAULTS, **_ISOLATED_BOXING_DEFAULTS}
            ).run(self.bare)

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

    def test_check_layers_are_two_qubit_models(self):
        """Isolated check layers are cheap to characterize, not full-width."""
        ancillas = set(self.checked.check_qubits)
        for instruction in self.isolated.data:
            if instruction.operation.name != "box":
                continue
            edges = _box_edges(instruction, self.isolated)
            if edges and any(set(e) & ancillas for e in edges):
                self.assertEqual(len(instruction.qubits), 2)


if __name__ == "__main__":
    unittest.main()
