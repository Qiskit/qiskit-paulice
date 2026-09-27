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


"""Checked circuits doped with non-Clifford rotations."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
from qiskit import QuantumCircuit

from .wire import Wire

if TYPE_CHECKING:
    from .checked_circuit import CheckedCircuit


@dataclass(frozen=True, eq=False)
class DopedCircuit:
    """A checked circuit with doping rotations inserted by :meth:`.CheckedCircuit.dope`.

    Doping preserves the checks, so post-selection and boxing carry over unchanged. Analyses
    of the Clifford skeleton (:attr:`.CheckedCircuit.uncovered_paulis`,
    :meth:`.CheckedCircuit.estimate_fault_rates`) belong to the undoped :attr:`checked`.

    Attributes:
        circuit: The doped circuit; parametrized if it was doped with ``angle=None``.
        doped_wires: The wires holding the rotations, sorted by circuit position, with
            instruction indices into :attr:`checked`'s circuit.
        checked: The undoped :class:`.CheckedCircuit` this was made from.
    """

    circuit: QuantumCircuit
    doped_wires: tuple[Wire, ...]
    checked: CheckedCircuit

    def get_postselection_method(self) -> Callable[[str | np.ndarray], np.ndarray]:
        """See :meth:`.CheckedCircuit.get_postselection_method`; the checks are unchanged."""
        return self.checked.get_postselection_method()

    def box(
        self,
        payload_layers: Iterable[Iterable[tuple[int, int]]] | None = None,
        **kwargs,
    ) -> QuantumCircuit:
        """See :meth:`.CheckedCircuit.box`, applied to the doped circuit."""
        return replace(self.checked, circuit=self.circuit).box(payload_layers, **kwargs)
