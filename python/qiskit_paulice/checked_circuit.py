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
from itertools import groupby, product
from typing import Any, Literal, NamedTuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import Clifford, PauliList
from samplomatic.transpiler import generate_boxing_pass_manager

from ._internal import Metric as _Metric
from ._internal.conversion import convert_to_rustiq_circuit as _convert_to_rustiq_circuit
from ._internal.utils import build_check_picker as _build_check_picker
from .noise_models import NoiseModel

# Non-unitary instructions :meth:`CheckedCircuit.box` accepts; all else is rejected.
_NON_GATES = frozenset({"measure", "barrier"})

BOXING_DEFAULTS: dict[str, Any] = {
    "twirling_strategy": "active",
    "inject_noise_strategy": "individual_modification",
    "inject_noise_targets": "gates",
    "inject_noise_site": "after",
    "measure_annotations": "all",
    "remove_barriers": "after_stratification",
}
"""Options :meth:`CheckedCircuit.box` passes to
:func:`~samplomatic.transpiler.generate_boxing_pass_manager`, before ``**kwargs`` overrides."""


class UncoveredPauli(NamedTuple):
    """A spacetime location at which a single qubit Pauli error is undetectable by the set of checks.

    Attributes:
        qubit: Index of the qubit where the undetected error sits
        after_instruction: Index (into ``circuit.data``) of the instruction the error occurs after;
            ``None`` means the error sits on the qubit's input wire.
        pauli: The undetected Pauli error (``"X"``, ``"Y"``, or ``"Z"``)
    """

    qubit: int
    after_instruction: int | None
    pauli: Literal["X", "Y", "Z"]


class FaultRates(NamedTuple):
    r"""Monte Carlo estimates of a checked circuit's fault rates under a noise model.

    All rates are estimated from one common set of sampled fault configurations.

    Attributes:
        harmless_rate: Probability that a nonidentity fault configuration is harmless --
            back-propagates to a diagonal Pauli on the circuit input, acting as a global
            phase on :math:`|0^n\rangle` -- given that it is accepted (zero syndrome on
            every check).
        harmless_stderr: Standard error of ``harmless_rate``.
        logical_error_rate: Probability that some payload measurement outcome is flipped,
            given that the fault configuration is accepted -- the residual error rate
            surviving post-selection.
        logical_error_stderr: Standard error of ``logical_error_rate``.
        acceptance_rate: Probability that a fault configuration produces a zero syndrome on
            every check.
        acceptance_stderr: Standard error of ``acceptance_rate``.
        check_trigger_rates: For each check, the probability that its syndrome bit reads 1.
        check_trigger_stderrs: Standard errors of ``check_trigger_rates``.
        shots: Number of fault configurations sampled.
    """

    harmless_rate: float
    harmless_stderr: float
    logical_error_rate: float
    logical_error_stderr: float
    acceptance_rate: float
    acceptance_stderr: float
    check_trigger_rates: tuple[float, ...]
    check_trigger_stderrs: tuple[float, ...]
    shots: int


@dataclass(frozen=True, eq=False)
class CheckedCircuit:
    """A quantum circuit and information about spacetime Pauli checks it contains.

    Attributes:
        circuit: A quantum circuit containing ``0`` or more spacetime Pauli checks.
        target_qubits: Qubit indices of ``circuit`` which were used to entangle the check
            qubits to the payload. Empty if ``circuit`` contains no checks.
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

    def estimate_fault_rates(
        self,
        noise_model: NoiseModel,
        shots: int = 100_000,
        seed: int | np.random.Generator | None = None,
    ) -> FaultRates:
        r"""Estimate acceptance, harmless-fault, logical-error, and check trigger rates.

        One Monte Carlo simulation of Pauli fault configurations drawn from ``noise_model``
        yields four families of rates. A configuration is *accepted* if it flips no check
        syndrome (the ``acceptance_rate``, for shot budgeting), and *harmless* if it is
        nonidentity yet back-propagates to a diagonal Pauli on the circuit input -- it then
        acts as a global phase on :math:`|0^n\rangle`, leaving the Clifford output state
        untouched. The conditional rate :math:`\Pr(\text{harmless} \mid \text{accepted})` is
        the correction term in the doped fidelity bound of
        `arXiv:2607.25941 <https://arxiv.org/abs/2607.25941>`_ (Sec. S2): doping converts at
        worst every harmless fault into a harmful one, so

        .. math:: F_{\text{doped}} \;\geq\; F_{\text{Clifford}} - \Pr(H \mid A),

        where :math:`F_{\text{Clifford}}` is the post-selected fidelity of this (undoped)
        checked circuit, e.g. measured by direct fidelity estimation.

        The remaining rates support that certificate operationally. The
        ``logical_error_rate`` is the probability that an accepted configuration flips a
        payload measurement outcome -- the residual error surviving post-selection,
        evaluable for any noise model after the checks are fixed (unlike the ``cost``
        recorded at pick time). The ``check_trigger_rates`` predict how often each syndrome
        bit reads 1; comparing them against measured syndrome data validates the noise model
        itself, and thereby the harmless-rate correction computed from it.

        Faults are sampled from independent Pauli-Lindblad generators following each
        entangling gate (a rate-:math:`\lambda` generator fires with probability
        :math:`(1 - e^{-2\lambda})/2`), matching the noise conventions of
        :func:`~qiskit_paulice.checks.add_pauli_checks`; edges absent from a gate-wise model
        fall back to the median rate per Pauli pair. Readout errors enter the syndrome (hence
        acceptance) and the payload outcomes (hence the logical error rate) but, being
        classical, never the state. Layered and idling noise are not supported.

        Args:
            noise_model: The noise model to sample fault configurations from. Supports
                uniform (:class:`float`) and :data:`.GateWiseNoise` gate noise plus readout
                noise.
            shots: Number of fault configurations to sample.
            seed: A seed or generator for the fault sampling.

        Returns:
            The estimated rates with their standard errors.

        Raises:
            ValueError: The noise model is empty, or uses layered or idling noise.
            ValueError: :attr:`circuit` contains a non-Clifford instruction (call this on the
                undoped checked circuit -- the bound's reference point -- not the doped one).
            ValueError: No sampled configuration was accepted.
        """
        gate_noise = noise_model.gate_noise
        if noise_model.idling_noise is not None:
            raise ValueError("Idling noise is not supported by estimate_harmless_rate.")
        if (
            isinstance(gate_noise, dict)
            and gate_noise
            and isinstance(next(iter(gate_noise))[0], tuple)
        ):
            raise ValueError(
                "Layered gate noise is not supported by estimate_harmless_rate; "
                "use uniform or gate-wise noise."
            )
        if gate_noise is None and noise_model.readout_noise is None:
            raise ValueError("The noise model may not be empty.")
        if noise_model.readout_noise is not None and not 0 <= noise_model.readout_noise < 0.5:
            raise ValueError("readout_noise must lie in [0, 0.5).")
        rates, x_img, z_img, full_clifford = _fault_channels(self.circuit, gate_noise)

        masks = self._sub_array.astype(np.uint8)
        signatures = x_img.astype(np.uint8) @ masks.T % 2
        back = PauliList.from_symplectic(z_img, x_img).evolve(full_clifford, frame="h")
        measured = np.array(sorted(self._cb_to_q.values()), dtype=int)
        payload = np.array([q for q in measured if q not in set(self.check_qubits)], dtype=int)
        # A fault flips payload outcome q iff its output image anticommutes with Z_q.
        flip_rows = x_img[:, payload].astype(np.uint8)

        # One row of XOR accumulators per shot; each generator firing XORs in its syndrome
        # signature, payload outcome flips, and back-propagated symplectic rows.
        # Poisson(shots * rate) firings spread uniformly over shots give i.i.d.
        # Poisson(rate) counts per shot, whose odd-count (flip) probability is exactly the
        # Pauli-Lindblad (1 - exp(-2 rate))/2.
        rng = np.random.default_rng(seed)
        syndrome = np.zeros((shots, len(masks)), dtype=np.uint8)
        outcome = np.zeros((shots, len(payload)), dtype=np.uint8)
        back_x = np.zeros((shots, self.circuit.num_qubits), dtype=np.uint8)
        back_z = np.zeros_like(back_x)
        channel = np.repeat(np.arange(len(rates)), rng.poisson(shots * rates))
        shot = rng.integers(0, shots, len(channel))
        np.bitwise_xor.at(syndrome, shot, signatures[channel])
        np.bitwise_xor.at(outcome, shot, flip_rows[channel])
        np.bitwise_xor.at(back_x, shot, back.x[channel].astype(np.uint8))
        np.bitwise_xor.at(back_z, shot, back.z[channel].astype(np.uint8))
        if noise_model.readout_noise is not None:
            # A readout flip on measured qubit q toggles every check with q in its support,
            # and the outcome bit itself when q is a payload qubit.
            outcome_rows = (payload[None, :] == measured[:, None]).astype(np.uint8)
            readout_rate = -np.log(1 - 2 * noise_model.readout_noise) / 2
            index = np.repeat(
                np.arange(len(measured)), rng.poisson(shots * readout_rate, len(measured))
            )
            shot = rng.integers(0, shots, len(index))
            np.bitwise_xor.at(syndrome, shot, masks.T[measured[index]])
            np.bitwise_xor.at(outcome, shot, outcome_rows[index])

        accepted = ~syndrome.any(axis=1)
        num_accepted = int(accepted.sum())
        if num_accepted == 0:
            raise ValueError(
                f"None of the {shots} sampled fault configurations was accepted; increase "
                "shots or reduce the noise strength."
            )
        nonidentity = (back_x | back_z).any(axis=1)
        harmless = (accepted & nonidentity & ~back_x.any(axis=1)).sum() / num_accepted
        logical = (accepted & outcome.any(axis=1)).sum() / num_accepted
        acceptance = num_accepted / shots
        triggers = syndrome.mean(axis=0)

        def _stderr(probability: float, count: int) -> float:
            return float(np.sqrt(probability * (1 - probability) / count))

        return FaultRates(
            harmless_rate=float(harmless),
            harmless_stderr=_stderr(harmless, num_accepted),
            logical_error_rate=float(logical),
            logical_error_stderr=_stderr(logical, num_accepted),
            acceptance_rate=float(acceptance),
            acceptance_stderr=_stderr(acceptance, shots),
            check_trigger_rates=tuple(float(p) for p in triggers),
            check_trigger_stderrs=tuple(_stderr(float(p), shots) for p in triggers),
            shots=shots,
        )

    def box(
        self,
        payload_layers: Iterable[Iterable[tuple[int, int]]] | None = None,
        **kwargs,
    ) -> QuantumCircuit:
        """Box :attr:`circuit` while maintaining concurrent scheduling of payload layers.

        This method stratifies the entangling layers of the checked circuit into boxes such
        that the number of unique entangling layers is minimized. This is done by scheduling
        the entangling gates from the Pauli checks into boxes of their own, resulting in one
        unique layer per Pauli check in addition to the ``payload_layers``.

        Scheduling check gates into boxes of their own is beneficial in that the number of
        unique entangling layers is minimized, but it comes at a cost of sub-optimal
        gate scheduling.

        Args:
            payload_layers: The unique entangling layers of the bare payload circuit. Each inner
                list contains the edges for one unique layer. Edges should not be repeated
                within the same layer, but may appear in multiple layers; each stratum of the
                boxed circuit is then consistent with (a subset of) one of these layers.
            **kwargs: Overrides for :func:`~samplomatic.transpiler.generate_boxing_pass_manager`.
                Defaults to the key-value pairs in
                :data:`~qiskit_paulice.checked_circuit.BOXING_DEFAULTS`.

        Returns:
            :attr:`circuit`, boxed and annotated.

        Raises:
            ValueError: ``payload_layers`` does not describe this circuit's payload gates.
            ValueError: :attr:`circuit` contains an instruction other than one- and two-qubit
                unitary gates, measurements, and barriers.
        """
        for instruction in self.circuit.data:
            operation = instruction.operation
            if operation.name in _NON_GATES:
                continue
            if not isinstance(operation, Gate) or len(instruction.qubits) > 2:
                raise ValueError(
                    f"'{operation.name}' is not supported: a checked circuit may contain only "
                    "one- and two-qubit unitary gates, measurements, and barriers."
                )
        options = {**BOXING_DEFAULTS, **kwargs}
        return generate_boxing_pass_manager(**options).run(self._stratify(payload_layers))

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

    def _stratify(
        self, payload_layers: Iterable[Iterable[tuple[int, int]]] | None
    ) -> QuantumCircuit:
        """Return a copy of the checked circuit that is separated into layers.

        This method isolates entangling gates that are part of a Pauli check into
        their own stratum and maintains the payload layer scheduling. This has the
        downside of additional idling time on all qubits and the upside of having
        fewer unique entangling layers for which to learn noise.
        """
        # Unpack circuit into lists of instructions and qubit indices
        circuit = self.circuit
        ancillas = set(self.check_qubits)
        data = [inst for inst in circuit.data if inst.operation.name != "barrier"]
        indices = [[circuit.find_bit(q).index for q in inst.qubits] for inst in data]

        # Get a mapping from edges to their associated layer IDs
        edge_to_layers = _edge_to_layers(payload_layers) if payload_layers is not None else None

        # For a given layer of entangling gates (stratum), store which unique layers it is
        # still consistent with; joining gates narrow the set.
        viable_layers: list[set[int]] = []
        # Mapping from a gap between two payload layers to the number of checks it contains
        checks_in_gap: dict[int, int] = defaultdict(int)
        # Mapping from qubit ID to the earliest stratum ID where it is free
        free_from: dict[int, int] = defaultdict(int)

        # Mapping from entangling gates to the gap/stratum in which they belong
        keys: dict[int, tuple[int, int]] = {}
        for i, inst in enumerate(data):
            if len(inst.qubits) != 2:
                continue
            a, b = sorted(indices[i])
            # Instruction is a check gate
            if {a, b} & ancillas:
                target = b if a in ancillas else a
                # Check qubit should be sandwiched between payload strata where its target qubit is used
                gap = free_from[target] - 1
                checks_in_gap[gap] += 1
                keys[i] = (gap, checks_in_gap[gap])
                continue
            # Earliest stratum where both qubits are free: ASAP packing.
            layer = max(free_from[a], free_from[b])
            if edge_to_layers is not None:
                if (a, b) not in edge_to_layers:
                    raise ValueError(
                        "payload_layers does not describe this circuit's payload gates: edge "
                        f"{(a, b)} is in no layer."
                    )
                candidates = edge_to_layers[(a, b)]
                # Skip past strata consistent with none of the layers containing this edge.
                while layer < len(viable_layers) and not viable_layers[layer] & candidates:
                    layer += 1
                # Start a new stratum if necessary, else narrow the joined stratum's layers
                if layer == len(viable_layers):
                    viable_layers.append(set(candidates))
                else:
                    viable_layers[layer] &= candidates
            # Specify the stratum the payload instruction is associated with and hard-code 0 to indicate this is a payload stratum.
            keys[i] = (layer, 0)
            # Both qubits are now occupied through this stratum.
            free_from[a] = layer + 1
            free_from[b] = layer + 1

        # Associate single qubit gates with the entangling stratum to their right
        end = (len(data), 0)
        next_key: dict[int, tuple[int, int]] = defaultdict(lambda: end)
        for i in reversed(range(len(data))):
            if i in keys:
                for q in indices[i]:
                    next_key[q] = keys[i]
            else:
                keys[i] = min((next_key[q] for q in indices[i]), default=end)

        out = circuit.copy_empty_like()
        # Reorder the instructions by which stratum they're in. Original order maintained in ties.
        # Place instructions in new order such that original unique layers are maintained and
        # Pauli check gates have their own time-slice. Use barriers to delimit the strata.
        order = sorted(range(len(data)), key=keys.__getitem__)
        for stratum_key, members in groupby(order, key=keys.__getitem__):
            for i in members:
                out.append(data[i])
            if stratum_key != end:
                out.barrier()
        return out


def _edge_to_layers(
    payload_layers: Iterable[Iterable[tuple[int, int]]],
) -> dict[tuple[int, int], set[int]]:
    """Map each entangling edge to the indices of the unique payload layers containing it."""
    edge_to_layers: dict[tuple[int, int], set[int]] = defaultdict(set)
    for index, layer in enumerate(payload_layers):
        for a, b in layer:
            edge_to_layers[(min(a, b), max(a, b))].add(index)
    return dict(edge_to_layers)


_PAULI_PAIRS = ["".join(pair) for pair in product("IXYZ", repeat=2)][1:]


def _edge_generators(
    gate_noise: float | dict,
) -> Callable[[tuple[int, int]], list[tuple[str, float]]]:
    """Per-edge elementary ``(pauli_pair, lindblad_rate)`` generators of a gate noise spec.

    A uniform depolarizing probability ``p`` becomes the 15 non-identity pairs at rate
    ``-ln(1 - 4p/15)/4``; a gate-wise model is canonicalized to ``(min, max)`` edges, with
    the median rate per Pauli pair as the fallback for absent edges -- both matching the
    conventions of the Rust check evaluator.
    """
    if not isinstance(gate_noise, dict):
        rate = -np.log(1 - 4 * gate_noise / 15) / 4
        pairs = [(pair, rate) for pair in _PAULI_PAIRS]
        return lambda edge: pairs
    table = {}
    for (a, b), generators in gate_noise.items():
        table[min(a, b), max(a, b)] = [
            (pair if a <= b else pair[::-1], rate) for pair, rate in generators
        ]
    by_pair = defaultdict(list)
    for generators in table.values():
        for pair, rate in generators:
            by_pair[pair].append(rate)
    fallback = [(pair, float(np.median(rates))) for pair, rates in by_pair.items()]
    return lambda edge: table.get(edge, fallback)


def _fault_channels(
    circuit: QuantumCircuit, gate_noise: float | dict | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Clifford]:
    """Firing probabilities and output-frame images of every elementary fault channel.

    Sweeps the circuit backward maintaining the Clifford of the instruction suffix, which
    conjugates a Pauli fault following each entangling gate to the circuit output:
    destabilizer row ``q`` of the tableau is the image of ``X_q`` and stabilizer row ``q``
    that of ``Z_q``, so a generator's image is their XOR over its symplectic components.

    Returns:
        ``(rates, x, z, full_clifford)``: per channel the Lindblad rate (flip probability
        ``(1 - exp(-2 rate))/2``) and the symplectic rows of its image at the output, plus
        the Clifford of the entire circuit.

    Raises:
        ValueError: on a non-Clifford instruction or a non-terminal measurement.
    """
    generators = _edge_generators(gate_noise) if gate_noise is not None else lambda edge: []
    num_qubits = circuit.num_qubits
    suffix = Clifford.from_label("I" * num_qubits)
    touched: set[int] = set()
    rates: list[float] = []
    x_rows: list[np.ndarray] = []
    z_rows: list[np.ndarray] = []
    for inst in reversed(circuit.data):
        name = inst.operation.name
        if name == "barrier":
            continue
        qargs = [circuit.find_bit(qubit).index for qubit in inst.qubits]
        if name == "measure":
            if qargs[0] in touched:
                raise ValueError(
                    f"Qubit {qargs[0]} is used after its measurement; only terminal "
                    "measurements are supported."
                )
            continue
        touched.update(qargs)
        if len(qargs) == 2:
            for pair, rate in generators((min(qargs), max(qargs))):
                x = np.zeros(num_qubits, dtype=bool)
                z = np.zeros(num_qubits, dtype=bool)
                for qubit, char in zip(sorted(qargs), pair, strict=True):
                    if char in "XY":
                        x ^= suffix.destab_x[qubit]
                        z ^= suffix.destab_z[qubit]
                    if char in "ZY":
                        x ^= suffix.stab_x[qubit]
                        z ^= suffix.stab_z[qubit]
                rates.append(rate)
                x_rows.append(x)
                z_rows.append(z)
        try:
            suffix = suffix.dot(inst.operation, qargs=qargs)
        except QiskitError as exc:
            raise ValueError(f"Non-Clifford instruction in circuit: {name!r}") from exc
    x = np.asarray(x_rows, dtype=bool).reshape(len(x_rows), num_qubits)
    z = np.asarray(z_rows, dtype=bool).reshape(len(z_rows), num_qubits)
    return np.asarray(rates), x, z, suffix
