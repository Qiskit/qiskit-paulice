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

"""Tests for ``CheckedCircuit.box`` and ``CheckedCircuit.compute_local_scales``."""

from __future__ import annotations

import unittest
import warnings
from collections import Counter

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Clifford, Pauli, PauliLindbladMap
from qiskit.transpiler.passes import RemoveBarriers
from qiskit_paulice import add_pauli_checks
from qiskit_paulice.checked_circuit import (
    _BOXING_DEFAULTS,
    _ISOLATED_BOXING_DEFAULTS,
)
from qiskit_paulice.noise_models import NoiseModel

pytest.importorskip("samplomatic")
pytest.importorskip("qiskit_addon_slc.utils")

import samplomatic
from qiskit_addon_slc.utils import (
    generate_noise_model_paulis,
    map_modifier_ref_to_ref,
)
from samplomatic.annotations import InjectionSite, InjectNoise
from samplomatic.transpiler import generate_boxing_pass_manager
from samplomatic.utils import find_unique_box_instructions, get_annotation


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
        body = instruction.operation.blocks[0]
        qmap = [boxed.find_bit(q).index for q in instruction.qubits]
        edges = frozenset(
            tuple(sorted(qmap[body.find_bit(q).index] for q in sub.qubits))
            for sub in body.data
            if len(sub.qubits) == 2
        )
        if edges:
            out.append(edges)
    return out


def _fake_learned_rates(boxed, num_qubits, seed=1):
    """Stand-in for a noise learner: random rates on the generators of each unique layer."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        unique = find_unique_box_instructions(boxed, normalize_annotations=None, undress_boxes=True)
        paulis = generate_noise_model_paulis(unique, None, boxed)
    rng = np.random.default_rng(seed)
    return {
        ref: PauliLindbladMap.from_sparse_list(
            [(p, idx, float(rng.random() * 0.01 + 1e-4)) for p, idx in qspl.to_sparse_list()],
            num_qubits=num_qubits,
        )
        for ref, qspl in paulis.items()
    }


def _checked_example(nq=4, depth=4, seed=1):
    """A ``CheckedCircuit``, its boxed form, and per-``ref`` learned rates for that boxed form."""
    qc = _bare_circuit(nq, depth)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        checked = add_pauli_checks(
            qc, list(range(nq)), NoiseModel(gate_noise=1e-3, readout_noise=1e-2), seed=seed
        )[-1]
        boxed = checked.boxed_circuit
    return checked, boxed, _fake_learned_rates(boxed, checked.circuit.num_qubits, seed)


class TestBox(unittest.TestCase):
    """Tests for ``CheckedCircuit.box``."""

    def test_boxes_the_executed_circuit(self):
        """Every entangling gate of the checked circuit -- check gates included -- lands in a box."""
        checked, boxed, _ = _checked_example()
        ancillas = set(checked.check_qubits)
        boxed_edges = []
        for instruction in boxed.data:
            if instruction.operation.name != "box":
                continue
            body = instruction.operation.blocks[0]
            qmap = [boxed.find_bit(q).index for q in instruction.qubits]
            for sub in body.data:
                if len(sub.qubits) == 2:
                    a, b = (qmap[body.find_bit(q).index] for q in sub.qubits)
                    boxed_edges.append((min(a, b), max(a, b)))
        circuit_edges = [
            tuple(sorted(checked.circuit.find_bit(q).index for q in inst.qubits))
            for inst in checked.circuit.data
            if len(inst.qubits) == 2
        ]
        self.assertEqual(sorted(boxed_edges), sorted(circuit_edges))
        # the check gates are really in there
        self.assertTrue(any(set(e) & ancillas for e in boxed_edges))

    def test_cached(self):
        """``boxed_circuit`` is computed once."""
        checked, boxed, _ = _checked_example()
        self.assertIs(boxed, checked.boxed_circuit)

    def test_rejects_unknown_check_layers(self):
        """A typo'd strategy is an error, not a silent fallback."""
        checked, _, _ = _checked_example()
        with self.assertRaises(ValueError):
            checked.box(check_layers="seperate")


class TestIsolatedCheckLayers(unittest.TestCase):
    """Tests for ``CheckedCircuit.box(check_layers="isolated")``."""

    def setUp(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.checked, self.merged, _ = _checked_example(nq=6, depth=8, seed=4)
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

    def test_builds_and_shades(self):
        """The isolated boxing is a working samplomatic circuit that can be shaded."""
        rates = _fake_learned_rates(self.isolated, self.checked.circuit.num_qubits)
        scales = self.checked.compute_local_scales(rates, self.isolated)
        self.assertEqual(set(scales), set(map_modifier_ref_to_ref(self.isolated)))
        stacked = np.concatenate(list(scales.values()))
        self.assertTrue(0.0 in stacked and 1.0 in stacked)
        _, samplex = samplomatic.build(self.isolated)
        samplex.inputs().bind(**{f"noise_scales.{m}": -1.0 for m in scales}, local_scales=scales)

    def test_boxing_costs_reports_both_strategies(self):
        """``boxing_costs`` describes each layout, and isolation trades boxes for layers."""
        costs = self.checked.boxing_costs()
        self.assertEqual(set(costs), {"merged", "isolated"})
        self.assertEqual(
            costs["isolated"].unique_layers - len(self.checked.check_support),
            len(
                {
                    edges
                    for edges in _box_edge_sets(self.isolated)
                    if not any(set(e) & set(self.checked.check_qubits) for e in edges)
                }
            ),
        )
        # isolation always costs more boxes; that is the trade it makes
        self.assertGreater(costs["isolated"].boxes, costs["merged"].boxes)


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
        self.bare_rates = _fake_learned_rates(self.boxed_bare, self.checked.circuit.num_qubits)

    def test_payload_refs_come_from_the_bare_circuit(self):
        """Every payload box is covered by the bare circuit's model, and the generators match."""
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
        self.assertLessEqual(payload_refs, set(self.bare_rates))
        # the check layers are the only thing the bare model does not cover
        self.assertFalse(check_refs & set(self.bare_rates))
        # a brickwork payload keeps its two layers, and each check contributes one small layer
        self.assertEqual(len(payload_refs), 2)
        self.assertEqual(len(check_refs), len(self.checked.check_support))

    def test_shades_and_binds_with_the_reused_model(self):
        """The bare model plus per-check models shades and binds to the checked circuit."""
        rates = dict(self.bare_rates)
        rates.update(_fake_learned_rates(self.isolated, self.checked.circuit.num_qubits, seed=9))
        scales = self.checked.compute_local_scales(rates, self.isolated)
        self.assertEqual(set(scales), set(map_modifier_ref_to_ref(self.isolated)))
        _, samplex = samplomatic.build(self.isolated)
        samplex.inputs().bind(**{f"noise_scales.{m}": -1.0 for m in scales}, local_scales=scales)

    def test_check_layers_are_two_qubit_models(self):
        """Isolated check layers are cheap to characterize, not full-width."""
        ancillas = set(self.checked.check_qubits)
        for instruction in self.isolated.data:
            if instruction.operation.name != "box":
                continue
            edges = _box_edges(instruction, self.isolated)
            if edges and any(set(e) & ancillas for e in edges):
                self.assertEqual(len(instruction.qubits), 2)

    def test_uncovered_ref_names_the_cause(self):
        """A model that does not cover a box explains itself instead of raising KeyError."""
        with self.assertRaises(ValueError) as caught:
            self.checked.compute_local_scales(self.bare_rates, self.isolated)
        self.assertIn("no entry for", str(caught.exception))
        self.assertIn("twirling_strategy", str(caught.exception))


class TestComputeLocalScales(unittest.TestCase):
    """Tests for the detection-shaded ``local_scales``."""

    def setUp(self):
        self.checked, self.boxed, self.noise_rates = _checked_example()
        self.scales = self.checked.compute_local_scales(self.noise_rates)
        self.id_map = map_modifier_ref_to_ref(self.boxed)

    def test_structure_and_samplex_compatible(self):
        """Keyed by modifier_ref, 0/1 float arrays aligned to the noise model, and it binds."""
        self.assertEqual(set(self.scales), set(self.id_map))
        for mod, mask in self.scales.items():
            self.assertEqual(mask.dtype, float)
            self.assertTrue(set(np.unique(mask)) <= {0.0, 1.0})
            self.assertEqual(len(mask), self.noise_rates[self.id_map[mod]].num_terms)
        # some noise is detected and some survives -- the checks actually do something
        stacked = np.concatenate(list(self.scales.values()))
        self.assertTrue(0.0 in stacked and 1.0 in stacked)
        # binds into a samplex built from the very circuit that was shaded
        _, samplex = samplomatic.build(self.boxed)
        samplex.inputs().bind(
            **{f"noise_scales.{m}": -1.0 for m in self.scales}, local_scales=self.scales
        )

    def test_matches_independent_backpropagation(self):
        """The single-pass mask agrees with a per-generator back-propagation of the checks."""
        n = self.checked.circuit.num_qubits
        checks = [
            Pauli("".join("Z" if q in set(s) else "I" for q in range(n))[::-1])
            for s in self.checked.check_support
        ]
        boxes = list(self.boxed.data)

        def tail_after(box_idx):
            """Everything the circuit does after box ``box_idx``, flattened."""
            sub = QuantumCircuit(n)
            for instruction in boxes[box_idx + 1 :]:
                if instruction.operation.name == "box":
                    body = instruction.operation.blocks[0]
                    qmap = [self.boxed.find_bit(q).index for q in instruction.qubits]
                    for s in body.data:
                        if s.operation.name in ("measure", "barrier", "delay", "reset"):
                            continue
                        sub.append(s.operation, [qmap[body.find_bit(q).index] for q in s.qubits])
                elif instruction.operation.name not in ("measure", "barrier", "delay", "reset"):
                    sub.append(
                        instruction.operation,
                        [self.boxed.find_bit(q).index for q in instruction.qubits],
                    )
            return sub

        checked_any = False
        for i, instruction in enumerate(boxes):
            if instruction.operation.name != "box":
                continue
            ann = get_annotation(instruction.operation, InjectNoise)
            if ann is None or not ann.ref:
                continue
            checked_any = True
            # `CheckedCircuit.box` pins site="after", so the channel sits at the box exit
            self.assertEqual(ann.site, InjectionSite.AFTER)
            here = [c.evolve(tail_after(i), frame="h") for c in checks]
            qmap = sorted(self.boxed.find_bit(q).index for q in instruction.qubits)
            for j, (pauli, idx, _r) in enumerate(self.noise_rates[ann.ref].to_sparse_list()):
                chars = ["I"] * n
                for c, q in zip(pauli, idx, strict=True):
                    chars[n - 1 - qmap[q]] = c
                generator = Pauli("".join(chars))
                expected = 0.0 if any(not generator.commutes(c) for c in here) else 1.0
                self.assertEqual(self.scales[ann.modifier_ref][j], expected)
        self.assertTrue(checked_any)

    def test_respects_injection_site(self):
        """Shading follows ``InjectNoise.site``; the two sites give genuinely different masks."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            before = self.checked.box(inject_noise_site="before")
        rates = _fake_learned_rates(before, self.checked.circuit.num_qubits)
        # same layers, so the same generators; only the injection point moved
        self.assertEqual(
            sorted(r.num_terms for r in rates.values()),
            sorted(r.num_terms for r in self.noise_rates.values()),
        )
        scales_before = self.checked.compute_local_scales(rates, before)
        self.assertEqual(set(scales_before), set(self.scales))
        stacked_before = np.concatenate([scales_before[m] for m in sorted(scales_before)])
        stacked_after = np.concatenate([self.scales[m] for m in sorted(self.scales)])
        self.assertFalse(np.array_equal(stacked_before, stacked_after))

    def test_rejects_boxed_bare_circuit(self):
        """The boxed *bare* circuit -- the classic mistake -- is rejected."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            boxed_bare = generate_boxing_pass_manager(
                twirling_strategy="active_circuit",
                inject_noise_strategy="individual_modification",
                inject_noise_targets="gates",
                measure_annotations="all",
            ).run(_bare_circuit())
        # it even has colliding modifier_refs, which is exactly why this must be caught
        self.assertTrue(set(map_modifier_ref_to_ref(boxed_bare)) & set(self.scales))
        with self.assertRaises(ValueError):
            self.checked.compute_local_scales(self.noise_rates, boxed_bare)

    def test_rejects_unrelated_circuit(self):
        """A boxed checked circuit from a different payload is rejected."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            other, _, _ = _checked_example(nq=6, depth=4, seed=2)
        with self.assertRaises(ValueError):
            self.checked.compute_local_scales(self.noise_rates, other.boxed_circuit)

    def test_gamma_reduction_is_detected_share(self):
        """log(gamma_full / gamma_reduced) equals 4 * sum of the detected generators' rates."""
        detected_rate_sum = 0.0
        log_full = log_reduced = 0.0
        for mod, mask in self.scales.items():
            rates = np.asarray(self.noise_rates[self.id_map[mod]].rates)
            log_full += 4 * rates.sum()
            log_reduced += 4 * (rates * mask).sum()
            detected_rate_sum += rates[mask == 0.0].sum()
        self.assertAlmostEqual(log_full - log_reduced, 4 * detected_rate_sum)
        self.assertLess(np.exp(log_reduced), np.exp(log_full))  # gamma genuinely drops


if __name__ == "__main__":
    unittest.main()
