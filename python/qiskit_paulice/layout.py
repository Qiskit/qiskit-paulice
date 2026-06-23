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

from qiskit.providers import BackendV2
from qiskit.transpiler import CouplingMap


def get_check_qubits(
    backend: BackendV2 | CouplingMap, layout: Sequence[int]
) -> tuple[list[int], list[int]]:
    """Pair qubits in ``layout`` with neighboring ancillas.

    Generate equal-length lists of target and ancilla qubits, such that
    target qubit ``i`` is adjacent to ancilla qubit ``i``in the coupling map.
    Target and ancilla qubits may appear at most one time in their
    respective lists.

    Args:
        backend: The target backend, or a :class:`~qiskit.transpiler.CouplingMap`
            describing its connectivity.
        layout: A list of physical qubit indices.

    Returns:
        A length-2 tuple of lists, ``(target_qubits, ancilla_qubits)``. ``target_qubits[i]``
        pairs with ``ancilla_qubits[i]``.
    """
    coupling_map = getattr(backend, "coupling_map", backend)
    ancilla_to_payload = get_low_overhead_ancillas(coupling_map, layout)

    # Give each ancilla its first not-yet-claimed neighbor, so every target and
    # ancilla is used at most once. Sorting makes the choice deterministic.
    matched: dict[int, int] = {}  # target qubit -> ancilla
    for ancilla in sorted(ancilla_to_payload):
        for target in sorted(ancilla_to_payload[ancilla]):
            if target not in matched:
                matched[target] = ancilla
                break

    targets = sorted(matched)
    return [int(t) for t in targets], [int(matched[t]) for t in targets]


def get_low_overhead_ancillas(
    coupling_map: CouplingMap, layout: Sequence[int]
) -> dict[int, list[int]]:
    """Create a mapping from ancillas to ``layout`` qubits to which they are adjacent.

    Args:
        coupling_map: A qubit connectivity graph.
        layout: Physical qubit indices occupied by the payload circuit.

    Returns:
        A mapping from ancilla indices to the list of ``layout`` qubits to which it is
        adjacent. An ancilla bordering several layout qubits maps to all of them, and
        a layout qubit bordering several ancillas appears in each of their lists.
    """
    layout_set = set(layout)
    ancilla_targets: dict[int, list[int]] = {}

    for qubit in layout:
        for neighbor in sorted(coupling_map.neighbors(qubit)):
            if neighbor not in layout_set:
                if neighbor not in ancilla_targets:
                    ancilla_targets[neighbor] = []
                ancilla_targets[neighbor].append(qubit)

    return ancilla_targets
