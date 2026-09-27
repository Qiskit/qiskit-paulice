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


"""Wires: the stretches of a qubit's timeline between consecutive gates."""

from __future__ import annotations

from typing import NamedTuple


class Wire(NamedTuple):
    """A stretch of one qubit's timeline between two consecutive gates on that qubit.

    Barriers and measurements are not gates, so they do not start or end a wire.

    Attributes:
        qubit: Index of the qubit.
        after_instruction: Index into ``QuantumCircuit.data`` of the gate on ``qubit`` that the
            wire follows. ``None`` denotes the qubit's input wire, before its first gate.
    """

    qubit: int
    after_instruction: int | None
