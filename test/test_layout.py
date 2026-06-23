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

"""Test layout module."""

from __future__ import annotations

import unittest

from qiskit.transpiler import CouplingMap
from qiskit_paulice.layout import get_check_qubits, get_low_overhead_ancillas


class TestGetLowOverheadAncillas(unittest.TestCase):
    """Tests covering :func:`get_low_overhead_ancillas`."""

    def test_line_middle_layout(self):
        """Layout in the interior of a line picks up both endpoint ancillas."""
        result = get_low_overhead_ancillas(CouplingMap.from_line(5), [1, 2, 3])
        self.assertEqual(result, {0: [1], 4: [3]})

    def test_endpoint_layout(self):
        """Layout at a line endpoint has exactly one adjacent ancilla."""
        self.assertEqual(get_low_overhead_ancillas(CouplingMap.from_line(5), [0]), {1: [0]})

    def test_single_interior_qubit(self):
        """A single interior layout qubit has two adjacent ancillas."""
        self.assertEqual(
            get_low_overhead_ancillas(CouplingMap.from_line(5), [1]),
            {0: [1], 2: [1]},
        )

    def test_full_layout_yields_no_ancillas(self):
        """If the layout covers every physical qubit, no ancillas remain."""
        self.assertEqual(
            get_low_overhead_ancillas(CouplingMap.from_line(5), list(range(5))),
            {},
        )

    def test_empty_layout(self):
        """An empty layout produces an empty ancilla map."""
        self.assertEqual(get_low_overhead_ancillas(CouplingMap.from_line(5), []), {})

    def test_ancilla_shared_by_multiple_layout_qubits(self):
        """An ancilla adjacent to several layout qubits maps to all of them."""
        # Star topology centered on qubit 1.
        cm = CouplingMap([(0, 1), (1, 0), (1, 2), (2, 1), (1, 3), (3, 1)])
        result = get_low_overhead_ancillas(cm, [0, 2])
        self.assertEqual(list(result.keys()), [1])
        # Order tracks the layout iteration order.
        self.assertEqual(result[1], [0, 2])

    def test_key_order_is_deterministic(self):
        """Keys are sorted, so `neighbors` iteration instability can't leak out."""
        # Edges added so a single layout qubit's neighbors are not in index order.
        cm = CouplingMap([(5, 2), (5, 8), (5, 1), (1, 5), (2, 5), (8, 5)])
        self.assertEqual(list(get_low_overhead_ancillas(cm, [5])), [1, 2, 8])


class _FakeBackend:
    """Minimal stand-in exposing the ``coupling_map`` attribute the wrapper reads."""

    def __init__(self, coupling_map: CouplingMap):
        self.coupling_map = coupling_map


class TestGetCheckQubits(unittest.TestCase):
    """Tests covering :func:`get_check_qubits`."""

    def test_line_pairs_each_target_with_its_ancilla(self):
        # Line 0-1-2-3-4, payload on the interior: 0 checks 1, 4 checks 3.
        targets, ancillas = get_check_qubits(CouplingMap.from_line(5), [1, 2, 3])
        self.assertEqual(targets, [1, 3])
        self.assertEqual(ancillas, [0, 4])

    def test_accepts_backend_like_object(self):
        # Anything exposing `.coupling_map` works the same as passing the map.
        cm = CouplingMap.from_line(5)
        self.assertEqual(
            get_check_qubits(_FakeBackend(cm), [1, 2, 3]),
            get_check_qubits(cm, [1, 2, 3]),
        )

    def test_each_ancilla_takes_a_distinct_target(self):
        # Ancilla 0 neighbors only target 1; ancilla 3 neighbors targets 1 and 2.
        # Walking ancillas in order, 0 claims 1 and 3 falls through to 2, so both
        # targets get a check rather than competing for target 1.
        cm = CouplingMap([(0, 1), (1, 0), (3, 1), (1, 3), (3, 2), (2, 3)])
        targets, ancillas = get_check_qubits(cm, [1, 2])
        self.assertEqual(targets, [1, 2])
        self.assertEqual(ancillas, [0, 3])

    def test_pairs_are_valid_and_unique(self):
        # Star centered on 2 plus a tail: every pair must use a distinct ancilla
        # outside the layout that genuinely neighbors its target.
        cm = CouplingMap.from_line(6)
        layout = [1, 2, 3, 4]
        targets, ancillas = get_check_qubits(cm, layout)
        self.assertEqual(len(targets), len(ancillas))
        self.assertEqual(len(set(targets)), len(targets))
        self.assertEqual(len(set(ancillas)), len(ancillas))
        for t, a in zip(targets, ancillas, strict=True):
            self.assertIn(t, layout)
            self.assertNotIn(a, layout)
            self.assertIn(a, list(cm.neighbors(t)))

    def test_ancilla_with_no_free_target_is_dropped(self):
        # Ancillas 0 and 2 both border only target 1; the first claims it and the
        # second is left unmatched, so only one pair comes back.
        cm = CouplingMap([(0, 1), (1, 0), (2, 1), (1, 2)])
        self.assertEqual(get_check_qubits(cm, [1]), ([1], [0]))

    def test_empty_layout_returns_empty_lists(self):
        self.assertEqual(get_check_qubits(CouplingMap.from_line(5), []), ([], []))

    def test_full_layout_returns_empty_lists(self):
        self.assertEqual(get_check_qubits(CouplingMap.from_line(5), list(range(5))), ([], []))
