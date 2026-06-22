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
    """Pair payload qubits with neighboring ancillas for low-overhead checks.

    Wraps :func:`get_low_overhead_ancillas` and resolves the ancilla map into two
    parallel lists ready to pass to :func:`qiskit_paulice.add_pauli_checks` as
    ``target_qubits`` and ``ancilla_qubits``: ``ancilla_qubits[i]`` implements a
    check with ``target_qubits[i]``.

    Each check consumes one ancilla and one distinct target qubit, so the pairing
    is chosen as a maximum bipartite matching between ancillas and payload qubits.
    This maximizes the number of target qubits that can be checked (a simpler
    first-available assignment can strand a target whose only free ancilla was
    given to a neighbor). Ancillas and their candidate targets are visited in
    sorted order, so the result is deterministic.

    Args:
        backend: The target backend, or a :class:`~qiskit.transpiler.CouplingMap`
            describing its connectivity.
        layout: Physical qubit indices occupied by the payload circuit.

    Returns:
        A ``(target_qubits, ancilla_qubits)`` tuple of equal-length physical qubit
        index lists, ordered by target qubit index.
    """
    coupling_map = getattr(backend, "coupling_map", backend)
    ancilla_to_payload = get_low_overhead_ancillas(coupling_map, layout)

    # Maximum bipartite matching (Kuhn's algorithm): match each ancilla to a
    # distinct target qubit it neighbors. `matched` maps target qubit -> ancilla.
    matched: dict[int, int] = {}

    def _augment(ancilla: int, visited: set[int]) -> bool:
        for target in sorted(ancilla_to_payload[ancilla]):
            if target in visited:
                continue
            visited.add(target)
            if target not in matched or _augment(matched[target], visited):
                matched[target] = ancilla
                return True
        return False

    for ancilla in sorted(ancilla_to_payload):
        _augment(ancilla, set())

    target_qubits = sorted(matched)
    ancilla_qubits = [int(matched[t]) for t in target_qubits]
    return [int(t) for t in target_qubits], ancilla_qubits


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
