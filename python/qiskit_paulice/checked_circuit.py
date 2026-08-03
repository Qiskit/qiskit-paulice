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

from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import cached_property
from itertools import groupby
from typing import Any, Literal, NamedTuple

import numpy as np
from qiskit import QuantumCircuit
from samplomatic.transpiler import generate_boxing_pass_manager

from ._internal import Metric as _Metric
from ._internal.conversion import convert_to_rustiq_circuit as _convert_to_rustiq_circuit
from ._internal.utils import build_check_picker as _build_check_picker

_NOT_PROPAGATED = frozenset({"measure", "barrier", "delay"})
_BOXING_DEFAULTS: dict[str, Any] = {
    "twirling_strategy": "active",
    "inject_noise_strategy": "individual_modification",
    "inject_noise_targets": "gates",
    "inject_noise_site": "after",
    "measure_annotations": "all",
    "remove_barriers": "after_stratification",
}


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
        n_clbits_full = self.circuit.num_clbits
        cb_to_q = self._cb_to_q
        sub_array = self._sub_array

        def _aux(bitstring_or_array: str | np.ndarray) -> np.ndarray:
            if isinstance(bitstring_or_array, str):
                s = bitstring_or_array.replace(" ", "")
                x = np.zeros(n_qubits_full, dtype=np.byte)
                if cb_to_q:
                    if len(s) != n_clbits_full:
                        raise ValueError(
                            f"Bitstring has length {len(s)}; expected "
                            f"{n_clbits_full} (one bit per clbit)."
                        )
                    for cb, q in cb_to_q.items():
                        x[q] = 1 if s[-(cb + 1)] == "1" else 0
                else:
                    # Fallback: bitstring is qubit-indexed (e.g. circuit was
                    # output of `pick_checks` with a user-applied measure_all).
                    if len(s) != n_qubits_full:
                        raise ValueError(
                            f"Bitstring has length {len(s)}; expected "
                            f"{n_qubits_full} (one bit per qubit)."
                        )
                    for q in range(n_qubits_full):
                        x[q] = 1 if s[-(q + 1)] == "1" else 0
            else:
                x = bitstring_or_array
            return (sub_array @ x) % 2

        return _aux

    def box(
        self,
        payload_layers: Iterable[Iterable[tuple[int, int]]] | None = None,
        **kwargs,
    ) -> QuantumCircuit:
        """Box :attr:`circuit` for samplomatic, with every check gate isolated in its own layer.

        Runs :func:`~samplomatic.transpiler.generate_boxing_pass_manager` on :attr:`circuit` --
        the circuit that is actually executed, ancillas and check gates included -- after
        regrouping it so each check gate sits alone in a two-qubit layer scoped to its own
        qubits. Repeats of a box with the same entangling gates then share a ``ref``: the unique
        layers a noise learner must characterize are the payload's own plus one small layer per
        check, a count that does not grow with circuit depth, and the payload boxes carry the
        same ``ref`` values as a boxing of the bare circuit, so a noise model learned once on the
        bare circuit binds to them.

        Box the executed circuit, never the bare one: the bare circuit's boxes carry different
        ``ref`` values while their ``modifier_ref`` values collide with this circuit's. For a
        depth-preserving boxing without check isolation, run samplomatic's boxing pass on
        :attr:`circuit` directly.

        Args:
            payload_layers: the payload's unique entangling layers, each a collection of
                qubit-index pairs. Supply them when the payload's layer structure is known; every
                payload stratum is then an instance of exactly one of these layers. An edge may
                appear in only one layer. When ``None``, layers are derived by packing payload
                gates as early as possible, which may not recover an intended structure.
            **kwargs: overrides for :func:`~samplomatic.transpiler.generate_boxing_pass_manager`.

        Returns:
            :attr:`circuit`, boxed and annotated.

        Raises:
            ValueError: ``payload_layers`` does not describe this circuit's payload gates, or
                :attr:`circuit` contains a reset, which is not yet supported.
        """
        if any(instruction.operation.name == "reset" for instruction in self.circuit.data):
            raise ValueError(
                "circuits containing resets are not supported yet: propagating check "
                "operators through a reset is not implemented."
            )
        options = {**_BOXING_DEFAULTS, **kwargs}
        return generate_boxing_pass_manager(**options).run(self._stratify(payload_layers))

    def _stratify(
        self, payload_layers: Iterable[Iterable[tuple[int, int]]] | None
    ) -> QuantumCircuit:
        """Re-emit :attr:`circuit` with each check gate alone in its own barrier-delimited layer.

        Every instruction gets a stratum key and the circuit is stable-sorted by it. The keys are
        monotone along every qubit's wire, so no two instructions sharing a qubit are ever
        reordered and the result implements the same unitary.
        """
        circuit = self.circuit
        ancillas = set(self.check_qubits)
        data = [inst for inst in circuit.data if inst.operation.name != "barrier"]
        indices = [[circuit.find_bit(q).index for q in inst.qubits] for inst in data]

        edge_to_layer = _edge_to_layer(payload_layers) if payload_layers is not None else None
        stratum_ids: list[int] = []
        after_layer: dict[int, int] = defaultdict(lambda: -1)
        checks_in_gap: dict[int, int] = defaultdict(int)
        free_from: dict[int, int] = defaultdict(int)

        # Payload gates pack as early as their qubits allow; each check gate gets a stratum of
        # its own, in the gap after the last payload layer to touch its target.
        keys: dict[int, tuple[int, int, int]] = {}
        for i, inst in enumerate(data):
            if len(inst.qubits) != 2 or inst.operation.name in _NOT_PROPAGATED:
                continue
            a, b = sorted(indices[i])
            if {a, b} & ancillas:
                target = b if a in ancillas else a
                gap = after_layer[target]
                checks_in_gap[gap] += 1
                keys[i] = (gap, 1, checks_in_gap[gap])
                continue
            layer = max(free_from[a], free_from[b])
            if edge_to_layer is not None:
                if (a, b) not in edge_to_layer:
                    raise ValueError(
                        "payload_layers does not describe this circuit's payload gates: edge "
                        f"{(a, b)} is in no layer."
                    )
                # Never mix gates from different unique layers in one stratum.
                unique = edge_to_layer[(a, b)]
                while layer < len(stratum_ids) and stratum_ids[layer] != unique:
                    layer += 1
                if layer == len(stratum_ids):
                    stratum_ids.append(unique)
            keys[i] = (layer, 0, 0)
            free_from[a] = free_from[b] = layer + 1
            after_layer[a] = after_layer[b] = layer

        # Every other instruction rides with the next entangler on any of its qubits, or the end.
        end = (len(data), 0, 0)
        next_key: dict[int, tuple[int, int, int]] = defaultdict(lambda: end)
        for i in reversed(range(len(data))):
            if i in keys:
                for q in indices[i]:
                    next_key[q] = keys[i]
            else:
                keys[i] = min((next_key[q] for q in indices[i]), default=end)

        out = circuit.copy_empty_like()
        order = sorted(range(len(data)), key=keys.__getitem__)
        for key, stratum in groupby(order, key=keys.__getitem__):
            for i in stratum:
                out.append(data[i])
            if key != end:
                out.barrier()
        return out


def _edge_to_layer(
    payload_layers: Iterable[Iterable[tuple[int, int]]],
) -> dict[tuple[int, int], int]:
    """Map each entangling edge to the index of the unique payload layer containing it."""
    edge_to_layer: dict[tuple[int, int], int] = {}
    for index, layer in enumerate(payload_layers):
        for a, b in layer:
            edge = (min(a, b), max(a, b))
            if edge_to_layer.setdefault(edge, index) != index:
                raise ValueError(f"edge {edge} appears in more than one payload layer.")
    return edge_to_layer
