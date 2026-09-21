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
from qiskit.circuit import Gate
from qiskit.circuit.library import CXGate, CZGate, HGate, SdgGate, SGate, SXdgGate, SXGate
from qiskit.quantum_info import Clifford, PauliList
from samplomatic.transpiler import generate_boxing_pass_manager

from ._internal import Metric as _Metric
from ._internal import NoiseModel as _RustNoiseModel
from ._internal.conversion import convert_noise_model as _convert_noise_model
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
    r"""Monte Carlo fault-rate estimates for a checked circuit, from one common sample set.

    Attributes:
        harmless_rate: Fraction of accepted shots whose error is non-identity yet
            backpropagates to a diagonal Pauli on the circuit input, applying a global phase
            to :math:`|0^n\rangle`.
        harmless_stderr: Standard error of ``harmless_rate``.
        logical_error_rate: Fraction of accepted shots whose error flips one or more
            payload measurement outcomes.
        logical_error_stderr: Standard error of ``logical_error_rate``.
        acceptance_rate: Probability of a zero syndrome on every check.
        acceptance_stderr: Standard error of ``acceptance_rate``.
        check_trigger_rates: Per check, the probability that its syndrome bit reads 1.
        check_trigger_stderrs: Standard errors of ``check_trigger_rates``.
        shots: Number of noisy shots used to generate the instance's fields.
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
        """Locations where a single qubit Pauli error is undetectable by the checks.

        Each entry is an ``UncoveredPauli(qubit, after_instruction, pauli)`` triple. Only
        input wires and wires immediately after 2-qubit gates are enumerated; errors after
        single qubit gates are folded into the next 2-qubit-gate wire.
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

        One noisy Monte Carlo sampling under ``noise_model`` yields all rates. A shot is
        *accepted* if all check syndromes are :math:`0`. An error is *harmless* if it is
        non-identity yet backpropagates to a diagonal Pauli on the input, acting as a
        global phase on :math:`|0^n\rangle`. The *harmless rate* and *logical error rate*
        are the fractions of accepted shots whose error is harmless, or flips a payload
        measurement outcome; the *check trigger rate* is the fraction of all shots a given
        check flags with a non-zero syndrome.

        Args:
            noise_model: Noise to apply during Monte Carlo sampling.
            shots: Number of fault configurations to sample.
            seed: Seed or generator for the fault sampling.

        Returns:
            The estimated fault rates with their standard errors.

        Raises:
            ValueError: The noise model is empty or unsupported, :attr:`circuit` contains a
                non-Clifford instruction, or no sampled configuration was accepted.
        """
        model = _convert_noise_model(noise_model, self.circuit)
        rates, x_img, z_img, full_clifford = _fault_channels(self.circuit, model)

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


_RUSTIQ_GATES = {
    "CX": CXGate(),
    "CZ": CZGate(),
    "H": HGate(),
    "S": SGate(),
    "Sd": SdgGate(),
    "SqrtX": SXGate(),
    "SqrtXd": SXdgGate(),
}


def _fault_channels(
    circuit: QuantumCircuit, model: _RustNoiseModel | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Clifford]:
    """Specify every noise generator in the model with its end-of-circuit Pauli in symplectic form.

    Returns:
        ``(rates, x, z, full_clifford)``: for each generator its rate (it fires with
        probability ``(1 - exp(-2 rate))/2``) and the x and z bits of its image, plus the
        whole circuit's Clifford for pushing images back to the input.

    Raises:
        ValueError: on a non-Clifford instruction, a non-terminal measurement, or a rate
            that is negative or not finite.
    """
    touched: set[int] = set()
    for inst in reversed(circuit.data):
        qargs = [circuit.find_bit(qubit).index for qubit in inst.qubits]
        if inst.operation.name == "measure":
            if qargs[0] in touched:
                raise ValueError(
                    f"Qubit {qargs[0]} is used after its measurement; only terminal "
                    "measurements are supported."
                )
        elif inst.operation.name != "barrier":
            touched.update(qargs)
    try:
        gates, _ = _convert_to_rustiq_circuit(circuit)
    except (ValueError, AssertionError) as exc:
        raise ValueError(f"Non-Clifford instruction in circuit: {exc}") from exc

    num_qubits = circuit.num_qubits
    rates: list[float] = []
    x_rows: list[np.ndarray] = []
    z_rows: list[np.ndarray] = []
    channels: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    if model is not None:
        generators, gates = model.resolve_generators(gates, num_qubits)
        for components, rate in generators:
            rates.append(rate)
            x_rows.append(np.zeros(num_qubits, dtype=bool))
            z_rows.append(np.zeros(num_qubits, dtype=bool))
            for (gate_index, slot), pauli in components:
                channels[gate_index].append((len(rates) - 1, slot, pauli))
    suffix = Clifford.from_label("I" * num_qubits)
    for gate_index in range(len(gates) - 1, -1, -1):
        name, qubits = gates[gate_index]
        for row, slot, pauli in channels.get(gate_index, ()):
            _xor_image(x_rows[row], z_rows[row], suffix, qubits[slot], pauli)
        suffix = suffix.dot(_RUSTIQ_GATES[name], qargs=qubits)
    for row, qubit, pauli in channels.get(-1, ()):
        _xor_image(x_rows[row], z_rows[row], suffix, qubit, pauli)

    if not np.isfinite(rates).all() or any(rate < 0 for rate in rates):
        raise ValueError("The noise model produced non-finite or negative Lindblad rates.")
    x = np.asarray(x_rows, dtype=bool).reshape(len(x_rows), num_qubits)
    z = np.asarray(z_rows, dtype=bool).reshape(len(z_rows), num_qubits)
    return np.asarray(rates), x, z, suffix


def _xor_image(
    x_row: np.ndarray, z_row: np.ndarray, suffix: Clifford, qubit: int, pauli: int
) -> None:
    """XOR the suffix image of Pauli 1=X/2=Y/3=Z on ``qubit`` into a generator's rows."""
    if pauli != 3:
        x_row ^= suffix.destab_x[qubit]
        z_row ^= suffix.destab_z[qubit]
    if pauli != 1:
        x_row ^= suffix.stab_x[qubit]
        z_row ^= suffix.stab_z[qubit]
