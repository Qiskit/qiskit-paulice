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

"""A class for specifying a circuit containing coherent spacetime Pauli checks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from typing import Literal, NamedTuple

import numpy as np
from qiskit import QuantumCircuit

from ._internal import Metric as _Metric
from ._internal.conversion import convert_to_rustiq_circuit as _convert_to_rustiq_circuit
from ._internal.utils import build_check_picker as _build_check_picker


class UncoveredPauli(NamedTuple):
    """A spacetime location at which a single qubit Pauli error is undetectable by a set of checks.

    Attributes:
        qubit: Index of the qubit where the undetected error sits
        after_instruction: Index (into ``circuit.data``) of the instruction the error occurs after;
            ``None`` means the error sits on the qubit's input wire.
        pauli: The undetected Pauli error (``"X"``, ``"Y"``, or ``"Z"``)
    """

    qubit: int
    after_instruction: int | None
    pauli: Literal["X", "Y", "Z"]


@dataclass(frozen=True, eq=False)
class CheckedCircuit:
    """A quantum circuit and information about spacetime Pauli checks it contains.

    Attributes:
        circuit: A quantum circuit containing ``0`` or more spacetime Pauli checks.
        target_qubits: Qubit indices of ``circuit`` which were used to entangle the check
            qubits to the payload. ``None`` if ``circuit`` contains no checks.
        check_qubits: Qubit indices of the ancilla qubits in ``circuit``. The ``i``th
            check uses ``check_qubits[i]`` to detect errors on ``target_qubits[i]`` and other
            qubits in ``check_support[i]``.
        check_support: For each check, the qubit indices whose measurement outcomes XOR
            together to give that check's syndrome bit.
        cost: The value of the cost function with respect to the checks in ``circuit``
        cost_metric: The metric used to evaluate check quality (``gamma`` or ``LER``)
    """

    circuit: QuantumCircuit
    target_qubits: tuple[int, ...] = ()
    check_qubits: tuple[int, ...] = ()
    check_support: tuple[tuple[int, ...], ...] = ()
    cost: float | None = None
    cost_metric: str | None = None

    def __post_init__(self) -> None:
        """Coerce mutable sequence inputs to tuples."""
        object.__setattr__(self, "target_qubits", tuple(self.target_qubits))
        object.__setattr__(self, "check_qubits", tuple(self.check_qubits))
        object.__setattr__(
            self,
            "check_support",
            tuple(tuple(s) for s in self.check_support),
        )

    @cached_property
    def uncovered_paulis(self) -> tuple[UncoveredPauli, ...]:
        """Locations where a single qubit Pauli error is undetectable by some checks.

        Each entry is an ``UncoveredPauli(qubit, after_instruction, pauli)`` triple,
        where ``qubit`` is the qubit of the single-qubit error, ``after_instruction``
        is the ``circuit.data`` index of the instruction which immediately precedes
        the error, and ``pauli`` is the type of error (``"X"``, ``"Y"``, or ``"Z"``).

        Only locations on input wires and immediately after 2-qubit gates are
        enumerated; errors after single qubit gates are folded into the next
        2-qubit-gate wire.
        """
        check_picker = _build_check_picker(
            self.circuit,
            _Metric.gamma(),
            [],
            None,
            None,
            list(self.check_qubits),
            [list(s) for s in self.check_support],
        )
        # The picker stores a rustiq-converted form of `self.circuit`; build
        # the same conversion's qiskit-instruction-index map so we can name
        # each rustiq wire in qiskit terms.
        _, qiskit_inst_indices = _convert_to_rustiq_circuit(self.circuit)
        out = []
        for (gate_idx, slot), p in check_picker.get_uncovered_paulis():
            pauli: Literal["X", "Y", "Z"] = "IXYZ"[p]  # type: ignore[assignment]
            if gate_idx == -1:
                # Input wire: the rustiq slot field is just the qubit index.
                out.append(UncoveredPauli(qubit=int(slot), after_instruction=None, pauli=pauli))
            else:
                qiskit_inst_idx = qiskit_inst_indices[gate_idx]
                qiskit_gate = self.circuit.data[qiskit_inst_idx]
                qubit = self.circuit.find_bit(qiskit_gate.qubits[slot]).index
                out.append(
                    UncoveredPauli(qubit=qubit, after_instruction=qiskit_inst_idx, pauli=pauli)
                )
        return tuple(out)

    @cached_property
    def _cb_to_q(self) -> dict[int, int]:
        cb_to_q: dict[int, int] = {}
        for inst in self.circuit.data:
            if inst.operation.name == "measure":
                q = self.circuit.find_bit(inst.qubits[0]).index
                cb = self.circuit.find_bit(inst.clbits[0]).index
                cb_to_q[cb] = q
        return cb_to_q

    @cached_property
    def _sub_array(self) -> np.ndarray:
        n_qubits_full = self.circuit.num_qubits
        sub_array = np.zeros((len(self.check_support), n_qubits_full), dtype=np.byte)
        for i, vzs in enumerate(self.check_support):
            for q in vzs:
                sub_array[i, q] = 1
        return sub_array

    def get_postselection_method(self) -> Callable[[str | np.ndarray], np.ndarray]:
        """Return a function that maps a single shot's outcome to a syndrome vector.

        No errors were detected iff every entry of the returned vector is zero. The
        returned function accepts either bitstrings or bit arrays.
        """
        n_qubits_full = self.circuit.num_qubits
        cb_to_q = self._cb_to_q
        sub_array = self._sub_array

        def _aux(bitstring_or_array: str | np.ndarray) -> np.ndarray:
            if isinstance(bitstring_or_array, str):
                s = bitstring_or_array.replace(" ", "")
                x = np.zeros(n_qubits_full, dtype=np.byte)
                if cb_to_q:
                    for cb, q in cb_to_q.items():
                        if cb < len(s) and q < n_qubits_full:
                            x[q] = 1 if s[-(cb + 1)] == "1" else 0
                else:
                    # Fallback: bitstring is qubit-indexed (e.g. circuit was
                    # output of `pick_checks` with a user-applied measure_all).
                    for q in range(min(n_qubits_full, len(s))):
                        x[q] = 1 if s[-(q + 1)] == "1" else 0
            else:
                x = bitstring_or_array
            return (sub_array @ x) % 2

        return _aux
