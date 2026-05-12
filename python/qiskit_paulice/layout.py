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

"""Functionality for finding layouts for checks dressed with spacetime Pauli checks."""

from __future__ import annotations

from collections.abc import Sequence

from qiskit.transpiler import CouplingMap


def get_low_overhead_ancillas(
    coupling_map: CouplingMap, layout: Sequence[int]
) -> dict[int, list[int]]:
    """Find ancilla qubits adjacent to the layout qubits in the coupling graph.

    This function identifies physical qubits that are not in the layout but are
    connected to one or more layout qubits via the coupling map.

    Args:
        coupling_map: A qubit connectivity graph
        layout: Physical qubit indices for which to find adjacent ancillas

    Returns:
        A dictionary mapping ancilla qubit indices to lists of layout qubit indices
        to which it is adjacent.
    """
    layout_set = set(layout)
    ancilla_targets: dict[int, list[int]] = {}

    for qubit in layout:
        for neighbor in coupling_map.neighbors(qubit):
            if neighbor not in layout_set:
                if neighbor not in ancilla_targets:
                    ancilla_targets[neighbor] = []
                ancilla_targets[neighbor].append(qubit)

    return ancilla_targets
