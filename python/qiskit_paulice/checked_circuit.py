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

from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Literal, NamedTuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, PauliLindbladMap, PauliList
from samplomatic.annotations import InjectionSite, InjectNoise
from samplomatic.transpiler import generate_boxing_pass_manager
from samplomatic.utils import get_annotation

from ._internal import Metric as _Metric
from ._internal.conversion import convert_to_rustiq_circuit as _convert_to_rustiq_circuit
from ._internal.utils import build_check_picker as _build_check_picker

# Instructions which do not act on the Heisenberg-propagated syndrome operators.
_NOT_PROPAGATED = frozenset({"measure", "barrier", "delay", "reset"})

# Boxing options used by :meth:`CheckedCircuit.box`. ``individual_modification`` is what gives
# every box its own ``modifier_ref``, which is what makes per-box ``local_scales`` possible.
# ``inject_noise_site`` is pinned rather than left to samplomatic's default, which is currently
# ``"before"`` but is slated to become ``"after"``: a layer's learned noise describes what that
# layer does, so it belongs after the layer, and shading depends on which side it sits on.
_BOXING_DEFAULTS: dict[str, Any] = {
    "twirling_strategy": "active_circuit",
    "inject_noise_strategy": "individual_modification",
    "inject_noise_targets": "gates",
    "inject_noise_site": "after",
    "measure_annotations": "all",
}

# Extra options for ``check_layers="isolated"``. ``active`` scopes each box to the qubits it acts
# on, which is what lets the payload boxes carry the same ``ref`` and generators as the bare
# circuit's -- under ``active_circuit`` they would also span the ancillas and diverge. The barriers
# `_stratify` emits are what the boxing pass stratifies on, so they must survive until it runs.
_ISOLATED_BOXING_DEFAULTS: dict[str, Any] = {
    "twirling_strategy": "active",
    "remove_barriers": "after_stratification",
}


class BoxingCost(NamedTuple):
    """What a :meth:`CheckedCircuit.box` strategy costs.

    Attributes:
        unique_layers: Distinct layers a noise learner must characterize -- one experiment each.
        boxes: Entangling layers executed, which sets the circuit's depth and the number of points
            noise is injected at.
    """

    unique_layers: int
    boxes: int


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

    def get_syndrome_flips(self, samplex_outputs) -> np.ndarray:
        r"""The syndrome flips a samplomatic twirl induces, one bit per check per randomization.

        Twirling inserts random Paulis, which propagate to the measurements and flip some of the
        bits a check takes the parity of. The flip is *deterministic within a randomization*, not
        random per shot, so it does not average away: for a randomization whose flip on check
        ``i`` is ``1``, post-selecting that check's raw parity on ``0`` keeps precisely the shots
        in which an error *was* detected and discards the clean ones. Nothing about the result
        looks wrong -- the acceptance rate is plausible and the expectation values are quiet --
        so this must be corrected explicitly:

        .. code-block:: python

            flips = checked.get_syndrome_flips(samplex_outputs)
            postselect = checked.get_postselection_method()
            accepted = not (postselect(bits) ^ flips[randomization]).any()

        Correcting the syndrome rather than the bits is equivalent and much cheaper, because the
        syndrome is linear over :math:`\mathbb{F}_2`: ``syndrome(b ^ f) == syndrome(b) ^
        syndrome(f)``. So the whole correction is one bit per check per randomization, applied to
        a syndrome you already compute, and it costs no shots and adds no variance.

        Args:
            samplex_outputs: the outputs of :meth:`samplomatic.samplex.Samplex.sample` for a
                samplex built from a boxing of :attr:`circuit`, which carries one
                ``measurement_flips.<register>`` entry per measured classical register.

        Returns:
            A ``0``/``1`` array of shape ``samplex_outputs["measurement_flips.<register>"].shape[:-1]
            + (len(check_support),)`` -- typically ``(num_randomizations, 1, num_checks)`` -- giving
            each check's syndrome flip. XOR it into the syndrome from
            :meth:`get_postselection_method`.

        Raises:
            ValueError: this ``CheckedCircuit`` has no checks, or ``samplex_outputs`` carries no
                measurement flips for this circuit's classical registers.
        """
        if not self.check_support:
            raise ValueError("this CheckedCircuit has no checks; there is no syndrome to flip.")

        blocks = {
            creg: np.asarray(samplex_outputs[key])
            for creg in self.circuit.cregs
            if (key := f"measurement_flips.{creg.name}") in samplex_outputs
        }
        if not blocks:
            raise ValueError(
                "samplex_outputs carries no measurement flips for this circuit's classical "
                f"registers ({[creg.name for creg in self.circuit.cregs]}). Its samplex must be "
                "built from a boxing of this CheckedCircuit's `circuit` whose measurement boxes "
                "are twirled -- see `CheckedCircuit.box`, whose `measure_annotations='all'` does "
                "this. Without the flips the syndrome cannot be corrected, and post-selecting on "
                "raw bits silently keeps the wrong shots."
            )

        leading = next(iter(blocks.values())).shape[:-1]
        clbit_flips = np.zeros((*leading, self.circuit.num_clbits), dtype=np.byte)
        for creg, block in blocks.items():
            if block.shape[:-1] != leading:
                raise ValueError(
                    f"measurement flips for register {creg.name!r} have shape {block.shape}, which "
                    f"does not agree with {leading} from the other registers."
                )
            for i, clbit in enumerate(creg):
                clbit_flips[..., self.circuit.find_bit(clbit).index] = block[..., i]

        # The syndrome is indexed by qubit, the flips by clbit; go through the same map the
        # bitstring path uses.
        qubit_flips = np.zeros((*leading, self.circuit.num_qubits), dtype=np.byte)
        if self._cb_to_q:
            for clbit_index, qubit in self._cb_to_q.items():
                qubit_flips[..., qubit] = clbit_flips[..., clbit_index]
        else:
            shared = min(self.circuit.num_clbits, self.circuit.num_qubits)
            qubit_flips[..., :shared] = clbit_flips[..., :shared]
        return (qubit_flips @ self._sub_array.T) % 2

    def get_postselection_method(self) -> Callable[[str | np.ndarray], np.ndarray]:
        """Return a function that maps a single shot's outcome to a syndrome vector.

        No errors were detected iff every entry of the returned vector is zero. The
        returned function accepts either bitstrings or bit arrays.

        When the circuit is twirled, the raw syndrome this returns is not the physical one --
        correct it with :meth:`get_syndrome_flips` before post-selecting.
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
        *,
        check_layers: Literal["merged", "isolated"] = "merged",
        payload_layers: QuantumCircuit | None = None,
        **kwargs,
    ) -> QuantumCircuit:
        """Group this checked circuit into annotated boxes, ready for noise learning and sampling.

        Runs :func:`~samplomatic.transpiler.generate_boxing_pass_manager` on :attr:`circuit` --
        that is, on the circuit that is actually executed, ancillas and check gates included. The
        boxes it produces are the layers whose noise a learner characterizes, the layers a
        :class:`~samplomatic.samplex.Samplex` injects noise into, and the layers
        :meth:`compute_local_scales` shades, so all four stages agree by construction.

        Boxing the *bare* circuit instead is a silent error: check insertion changes which gates
        share a layer, so the bare circuit's layers carry different ``ref`` values (a noise model
        learned for them cannot be bound) while its ``modifier_ref`` values collide with this
        circuit's (so ``local_scales`` derived from it may bind to the wrong layers).

        ``check_layers`` trades noise-learning cost against circuit depth; see
        :meth:`boxing_costs`, which reports both for both settings.

        Args:
            check_layers: how to lay out the check gates. Each value is a coherent configuration,
                not an independent knob -- ``"isolated"`` also scopes each box to the qubits it
                acts on (``twirling_strategy="active"``), because that is what makes the payload
                boxes match the bare circuit's. Override either via ``**kwargs`` if you need to.

                - ``"merged"``: let the boxing pass pack check gates into whichever payload layer
                  has room, and twirl every active qubit in every box. Leaves the circuit's depth
                  alone and models idling noise everywhere, but each layer ends up holding a
                  different subset of the ancilla edges, so almost every layer is unique and the
                  number of layers to learn grows with circuit depth.
                - ``"isolated"``: give every check gate a layer of its own, and scope each box to
                  its own qubits. A one-gate box has the same content every time that check fires,
                  so the layers to learn become the payload's own plus one small two-qubit layer
                  per check -- a count that does not grow with depth. Crucially the payload boxes
                  now carry the same ``ref`` and the same generators as the *bare* circuit's, so a
                  noise model learned once on the bare circuit binds to them, and keeps binding as
                  the checks change. The costs are real: roughly two to three times the boxes, so a
                  deeper circuit carrying more idling noise, and idling on qubits a box does not
                  touch is neither twirled nor modelled.

            payload_layers: a circuit whose barriers (or boxes) mark the payload's intended layer
                boundaries, used only when ``check_layers="isolated"``. Supply the bare circuit
                the checks were added to when its layer structure is known; its payload layers are
                then reproduced exactly. When ``None``, layers are derived by packing payload gates
                as early as possible, which may not recover an intended structure -- a layering
                that leaves gaps gets repacked into more distinct layers than it started with.
            **kwargs: overrides for :func:`~samplomatic.transpiler.generate_boxing_pass_manager`.
                Defaults are ``twirling_strategy="active_circuit"``,
                ``inject_noise_strategy="individual_modification"``,
                ``inject_noise_targets="gates"``, ``inject_noise_site="after"`` and
                ``measure_annotations="all"``.

        Returns:
            :attr:`circuit`, boxed and annotated.

        Raises:
            ValueError: ``check_layers`` is not ``"merged"`` or ``"isolated"``, or
                ``payload_layers`` does not describe this circuit's payload gates.
        """
        if check_layers == "merged":
            return generate_boxing_pass_manager(**{**_BOXING_DEFAULTS, **kwargs}).run(self.circuit)
        if check_layers != "isolated":
            raise ValueError(f"check_layers must be 'merged' or 'isolated', got {check_layers!r}.")
        options = {
            **_BOXING_DEFAULTS,
            **_ISOLATED_BOXING_DEFAULTS,
            **kwargs,
        }
        return generate_boxing_pass_manager(**options).run(self._stratify(payload_layers))

    @cached_property
    def boxed_circuit(self) -> QuantumCircuit:
        """This checked circuit boxed with :meth:`box`'s default options."""
        return self.box()

    def boxing_costs(self, payload_layers: QuantumCircuit | None = None) -> dict[str, BoxingCost]:
        """What each :meth:`box` ``check_layers`` setting would cost, for this circuit.

        ``unique_layers`` is the number of distinct layers a noise learner must characterize -- one
        experiment each -- and ``boxes`` is the number of entangling layers executed, which sets the
        circuit's depth and the number of points noise is injected at.

        Neither setting dominates: ``"merged"`` grows with circuit depth and is flat in the number
        of checks, while ``"isolated"`` is flat in depth and grows with the number of checks. The
        counts alone do not decide it -- see ``check_layers`` in :meth:`box` for what each gives up,
        and note that ``"isolated"``'s check layers are two-qubit models, far cheaper to
        characterize than a full-width payload layer of the same count.

        Args:
            payload_layers: as in :meth:`box`.

        Returns:
            A dict with keys ``"merged"`` and ``"isolated"``, each a :class:`BoxingCost`.
        """
        strategies: dict[str, dict[str, Any]] = {
            "merged": {"check_layers": "merged"},
            "isolated": {"check_layers": "isolated"},
        }
        costs = {}
        for name, options in strategies.items():
            boxed = self.box(payload_layers=payload_layers, **options)
            refs, boxes = set(), 0
            for instruction in boxed.data:
                if instruction.operation.name != "box":
                    continue
                inject = get_annotation(instruction.operation, InjectNoise)
                if inject is not None and inject.ref:
                    refs.add(inject.ref)
                    boxes += 1
            costs[name] = BoxingCost(unique_layers=len(refs), boxes=boxes)
        return costs

    def _stratify(self, payload_layers: QuantumCircuit | None) -> QuantumCircuit:
        """Re-emit :attr:`circuit` with each check gate alone in its own barrier-delimited layer.

        Gates are only regrouped, never added, removed, or reordered relative to any qubit they
        share, so the result implements the same unitary.
        """
        circuit = self.circuit
        ancillas = set(self.check_qubits)
        data = list(circuit.data)
        indices = {i: [circuit.find_bit(q).index for q in data[i].qubits] for i in range(len(data))}
        entanglers = [
            i
            for i in range(len(data))
            if len(data[i].qubits) == 2 and data[i].operation.name not in _NOT_PROPAGATED
        ]

        layer_map = _payload_layer_map(payload_layers) if payload_layers is not None else None
        payload_layer: dict[int, int] = {}
        gaps: dict[int, list[int]] = defaultdict(list)
        after_layer: dict[int, int] = defaultdict(lambda: -1)
        occurrence: Counter = Counter()
        free_from: dict[int, int] = defaultdict(int)

        for i in entanglers:
            a, b = sorted(indices[i])
            if {a, b} & ancillas:
                # A check gate sits in the gap after the last payload layer to touch its target.
                target = b if a in ancillas else a
                gaps[after_layer[target]].append(i)
                continue
            if layer_map is None:
                # No authored layering to honour: pack as early as the qubits allow.
                layer = max(free_from[a], free_from[b])
            else:
                key = ((a, b), occurrence[(a, b)])
                if key not in layer_map:
                    raise ValueError(
                        "payload_layers does not describe this circuit's payload gates: no layer "
                        f"for occurrence {key[1]} of edge {key[0]}. It must contain the same "
                        "entangling gates as the bare circuit these checks were added to."
                    )
                layer = layer_map[key]
                if layer < max(free_from[a], free_from[b]):
                    raise ValueError(
                        f"payload_layers orders edge {(a, b)} inconsistently with this circuit: "
                        f"layer {layer} follows a gate already placed at or after it."
                    )
            occurrence[(a, b)] += 1
            payload_layer[i] = layer
            free_from[a] = free_from[b] = layer + 1
            after_layer[a] = after_layer[b] = layer

        by_layer: dict[int, list[int]] = defaultdict(list)
        for i, layer in payload_layer.items():
            by_layer[layer].append(i)
        schedule: list[list[int]] = [[i] for i in sorted(gaps[-1])]
        for layer in range(max(by_layer, default=-1) + 1):
            if by_layer[layer]:
                schedule.append(sorted(by_layer[layer]))
            schedule.extend([i] for i in sorted(gaps[layer]))

        return _emit_layers(circuit, data, indices, set(entanglers), schedule)

    def compute_local_scales(
        self,
        noise_rates: dict[str, PauliLindbladMap],
        boxed_circuit: QuantumCircuit | None = None,
    ) -> dict[str, np.ndarray]:
        r"""Per-box generator masks that zero out the noise these checks detect (samplomatic ``local_scales``).

        Companion to :func:`qiskit_addon_slc.compute_local_scales`, but shading by *detectability*
        instead of observable effect: a generator the checks detect is removed by post-selection, so
        it need not be cancelled by PEC and is given scale ``0``; a surviving generator gets ``1``.
        This lowers the PEC sampling cost (:math:`\gamma`), often dramatically, since a good set of
        checks detects the large majority of the model's generators.

        The truncation is exact to first order in the noise rates, not exact outright:
        post-selection discards a shot when its *total* accumulated error anticommutes with a check,
        which is a joint condition on all the generators that fired. Two individually-detected
        generators can fire together and multiply into the checks' centralizer, surviving
        post-selection with neither one cancelled by PEC. The residual bias this leaves is
        :math:`O(\lambda^2)` in the generator rates.

        Detectability is evaluated on ``boxed_circuit`` itself, in one Heisenberg backward pass:
        each check operator is propagated back from the end of the circuit, and a generator is
        detected iff it anticommutes with some check at the point the noise is injected -- the side
        of the box named by its :class:`~samplomatic.InjectNoise` ``site``. Generators supported on
        the check ancillas are handled on the same footing as payload generators.

        Args:
            noise_rates: the learned noise, mapping each ``InjectNoise.ref`` in ``boxed_circuit``
                to a :class:`~qiskit.quantum_info.PauliLindbladMap` (e.g. from ``NoiseLearnerV3``).
            boxed_circuit: this checked circuit, boxed. Defaults to :attr:`boxed_circuit`; pass one
                explicitly only if it was boxed with non-default options. It must be a boxing of
                :attr:`circuit` -- *not* of the bare circuit the checks were added to -- or a
                :class:`ValueError` is raised.

        Returns:
            A dict mapping each box's ``modifier_ref`` to a ``0``/``1`` array aligned with that
            box's ``noise_rates`` generators -- ready to pass as ``local_scales`` to a
            :class:`~samplomatic.samplex.Samplex` built from ``boxed_circuit``.

        Raises:
            ValueError: this ``CheckedCircuit`` has no checks, or ``boxed_circuit`` is not a boxing
                of :attr:`circuit`.
        """
        if not self.check_support:
            raise ValueError("this CheckedCircuit has no checks; nothing to detect against.")
        boxed = self.boxed_circuit if boxed_circuit is None else boxed_circuit
        if boxed.num_qubits != self.circuit.num_qubits or _entangling_edges(
            boxed
        ) != _entangling_edges(self.circuit):
            raise ValueError(
                "boxed_circuit is not a boxing of this CheckedCircuit's `circuit`: its entangling "
                "gates differ. Note that the circuit to box is the checked circuit that will be "
                "executed (see `CheckedCircuit.box`), not the bare circuit the checks were added to."
            )

        n = self.circuit.num_qubits
        # Each check's syndrome is the parity of the measured bits in its support, i.e. the Z
        # string on that support at the end of the circuit.
        detectors = PauliList(
            [
                Pauli("".join("Z" if q in set(support) else "I" for q in range(n))[::-1])
                for support in self.check_support
            ]
        )

        scales: dict[str, np.ndarray] = {}
        for instruction in reversed(boxed.data):
            if instruction.operation.name != "box":
                if instruction.operation.name not in _NOT_PROPAGATED:
                    qargs = [boxed.find_bit(q).index for q in instruction.qubits]
                    detectors = detectors.evolve(instruction.operation, qargs=qargs, frame="h")
                continue

            inject = get_annotation(instruction.operation, InjectNoise)
            shade = inject is not None and inject.ref
            if shade and inject.ref not in noise_rates:
                raise ValueError(
                    f"noise_rates has no entry for {inject.ref!r}, a layer of boxed_circuit. Its "
                    f"keys are {sorted(noise_rates)}. A model learned for one boxing does not "
                    "carry over to another: a box's ref covers its gates *and* the qubits it "
                    "spans, so the same layer boxed with a different `twirling_strategy` -- or "
                    "boxed from the bare circuit while this one has ancillas in span -- gets a "
                    "different ref. `CheckedCircuit.box(check_layers='isolated')` is the layout "
                    "whose payload boxes match the bare circuit's."
                )
            # The box's `InjectNoise.site` says whether the channel acts before or after the box's
            # gates. Shade at exactly that point: the two differ by conjugation through the box, so
            # getting it wrong silently mislabels which generators the checks see.
            after = shade and inject.site == InjectionSite.AFTER

            if after:
                scales[inject.modifier_ref] = _shade(
                    instruction, boxed, noise_rates[inject.ref], detectors
                )
            detectors = detectors.evolve(_box_body_as_circuit(instruction, boxed), frame="h")
            if shade and not after:
                scales[inject.modifier_ref] = _shade(
                    instruction, boxed, noise_rates[inject.ref], detectors
                )

        return scales


def _payload_layer_map(circuit: QuantumCircuit) -> dict[tuple[tuple[int, int], int], int]:
    """``{(edge, occurrence): layer}`` from a circuit whose layers barriers or boxes delimit.

    Keyed by occurrence rather than by position because check insertion may reorder entangling
    gates that act on disjoint qubits; per-edge order is what survives and is what we need.
    """
    layers: dict[tuple[tuple[int, int], int], int] = {}
    occurrence: Counter = Counter()
    layer = 0
    for instruction in circuit.data:
        name = instruction.operation.name
        if name == "barrier":
            layer += 1
            continue
        edges = _instruction_edges(instruction, circuit)
        for edge in edges:
            layers[(edge, occurrence[edge])] = layer
            occurrence[edge] += 1
        if name == "box" and edges:
            layer += 1
    ordered = sorted(set(layers.values()))
    return {key: ordered.index(value) for key, value in layers.items()}


def _instruction_edges(instruction, circuit: QuantumCircuit) -> list[tuple[int, int]]:
    """The qubit pairs an instruction entangles, looking inside it if it is a box."""
    if instruction.operation.name == "box":
        body = instruction.operation.blocks[0]
        qmap = [circuit.find_bit(q).index for q in instruction.qubits]
        return [
            (min(pair), max(pair))
            for sub in body.data
            if len(sub.qubits) == 2
            for pair in [[qmap[body.find_bit(q).index] for q in sub.qubits]]
        ]
    if len(instruction.qubits) == 2 and instruction.operation.name not in _NOT_PROPAGATED:
        pair = [circuit.find_bit(q).index for q in instruction.qubits]
        return [(min(pair), max(pair))]
    return []


def _emit_layers(
    circuit: QuantumCircuit,
    data: list,
    indices: dict[int, list[int]],
    entanglers: set[int],
    schedule: list[list[int]],
) -> QuantumCircuit:
    """Re-emit ``circuit`` with its entanglers grouped per ``schedule``, barriers between layers.

    Every qubit's instructions keep their original relative order, so the circuit is unchanged.
    """
    on_qubit: dict[int, list[int]] = defaultdict(list)
    for i in range(len(data)):
        if data[i].operation.name != "barrier":
            for q in indices[i]:
                on_qubit[q].append(i)

    cursor: Counter = Counter()
    emitted: set[int] = set()
    out = circuit.copy_empty_like()

    def flush_until(qubit: int, target: int) -> None:
        """Emit everything still pending on ``qubit`` ahead of instruction ``target``."""
        while on_qubit[qubit][cursor[qubit]] != target:
            i = on_qubit[qubit][cursor[qubit]]
            if i not in emitted and i not in entanglers:
                out.append(data[i])
                emitted.add(i)
            cursor[qubit] += 1

    for layer in schedule:
        for i in layer:
            for q in indices[i]:
                flush_until(q, i)
            out.append(data[i])
            emitted.add(i)
            for q in indices[i]:
                cursor[q] += 1
        out.barrier()
    for i in range(len(data)):
        if i not in emitted and data[i].operation.name != "barrier":
            out.append(data[i])
    return out


def _entangling_edges(circuit: QuantumCircuit) -> Counter:
    """The multiset of qubit pairs entangled by ``circuit``, looking inside any boxes."""
    edges: Counter = Counter()
    for instruction in circuit.data:
        edges.update(_instruction_edges(instruction, circuit))
    return edges


def _box_body_as_circuit(instruction, circuit: QuantumCircuit) -> QuantumCircuit:
    """A box's body, widened onto ``circuit``'s qubits, with non-unitary instructions dropped."""
    body = instruction.operation.blocks[0]
    qmap = [circuit.find_bit(q).index for q in instruction.qubits]
    out = QuantumCircuit(circuit.num_qubits)
    for sub in body.data:
        if sub.operation.name in _NOT_PROPAGATED:
            continue
        out.append(sub.operation, [qmap[body.find_bit(q).index] for q in sub.qubits])
    return out


def _shade(
    instruction, circuit: QuantumCircuit, rates: PauliLindbladMap, detectors: PauliList
) -> np.ndarray:
    """A ``0``/``1`` mask over ``rates``' generators: ``0`` where ``detectors`` sees the generator."""
    # A box's Pauli-Lindblad map indexes its qubits in outer-circuit order (see `InjectNoise`).
    qmap = sorted(circuit.find_bit(q).index for q in instruction.qubits)
    n = circuit.num_qubits
    mask = np.ones(rates.num_terms)
    for j, (pauli, indices, _rate) in enumerate(rates.to_sparse_list()):
        chars = ["I"] * n
        for char, q in zip(pauli, indices, strict=True):
            chars[n - 1 - qmap[q]] = char
        generator = Pauli("".join(chars))
        mask[j] = 0.0 if any(not generator.commutes(d) for d in detectors) else 1.0
    return mask
