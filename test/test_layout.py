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
from qiskit_paulice.layout import get_low_overhead_ancillas


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
