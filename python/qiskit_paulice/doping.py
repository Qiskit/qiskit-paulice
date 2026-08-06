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

"""T-gate doping of Clifford circuits."""

from __future__ import annotations

from dataclasses import replace
from typing import NamedTuple

import numpy as np
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import Clifford, PauliList

from .checked_circuit import CheckedCircuit


class DopingSite(NamedTuple):
    """A circuit wire at which a magic-injecting ``T`` gate is inserted.

    Attributes:
        qubit: Index of the qubit whose wire is doped.
        after_instruction: Index (into ``circuit.data``) of the instruction the ``T`` gate
            is inserted after; ``None`` means the gate sits on the qubit's input wire.
    """

    qubit: int
    after_instruction: int | None


def dope_clifford_circuit(
    circuit: QuantumCircuit | CheckedCircuit,
    num_t_gates: int | None = None,
    *,
    after_entangling_only: bool = False,
    parametric: bool = False,
    seed: int | np.random.Generator | None = None,
) -> tuple[QuantumCircuit, list[DopingSite]] | tuple[CheckedCircuit, list[DopingSite]]:
    r"""Insert magic-injecting ``T`` gates into a Clifford circuit.

    Each candidate wire is classified by conjugating the ``Z`` generator of a ``T`` placed
    there through the surrounding Clifford gates
    (`arXiv:2607.25941 <https://arxiv.org/abs/2607.25941>`_, Sec. S1.3). Every wire segment
    is a candidate, and the reference's three pruning rewrites are iterated to a fixed
    point: remove a rotation that commutes with all previous rotations and back-propagates
    to a diagonal at the input (it acts trivially on :math:`|0^n\rangle`), or commutes with
    all following rotations and forward-propagates to a diagonal at the output (it is
    invisible to sampling); merge equal-generator pairs that commute with every rotation
    between them, keeping the earliest (the reference removes one at random). Each returned
    site therefore contributes an irreducible non-Clifford rotation.

    A :class:`.CheckedCircuit` input restricts sites to payload wires where a ``Z`` error is
    undetected by every check (cf. :attr:`.CheckedCircuit.uncovered_paulis`), so all
    syndromes remain deterministic and post-selection is unaffected. A new
    :class:`.CheckedCircuit` with the same check metadata is returned.

    Args:
        circuit: The Clifford circuit to dope, or a :class:`.CheckedCircuit` whose spacetime
            code the doping must preserve. Barriers and terminal measurements are ignored;
            sites past a qubit's measurement are excluded.
        num_t_gates: Number of ``T`` gates to insert, drawn uniformly from the valid sites;
            ``None`` uses every valid site. A drawn subset may itself be further reducible;
            pruned-away draws are redrawn until ``num_t_gates`` irreducible sites remain.
        after_entangling_only: Restrict sites to wires directly following an entangling
            gate, as in the reference: a ``Z`` rotation there merges into the following
            single-qubit layer as a zero-cost virtual ``RZ``, so every doping configuration
            shares one pulse schedule.
        parametric: Insert ``rz(dope[i])`` rotations (``dope[i]`` at ``sites[i]``) instead
            of ``T`` gates: one template covers every doping configuration
            (:math:`\pi/4` = ``T``, :math:`\pi/2` = ``S``, :math:`\pi` = ``Z``, ``0`` =
            identity), and every assignment preserves the code of a
            :class:`.CheckedCircuit`.
        seed: Seed or generator for the random site selection.

    Returns:
            * **QuantumCircuit | CheckedCircuit** -- A copy of ``circuit`` with the rotations
              inserted
            * **list[DopingSite]** -- The doped sites, sorted by circuit position

    Raises:
        ValueError: ``circuit`` contains a non-Clifford instruction or a non-terminal
            measurement, ``num_t_gates`` is out of range, or no irreducible subset of that
            size could be drawn.
    """
    checked = None
    if isinstance(circuit, CheckedCircuit):
        checked = circuit
        circuit = circuit.circuit
    site_qubits = set(range(circuit.num_qubits)) - set(checked.check_qubits if checked else ())
    positions, qubits, gens, full_clifford = _sweep_wire_segments(
        circuit, site_qubits, after_entangling_only
    )

    # The back-propagation of a generator P to the input is C^dag P C for the whole-circuit
    # Clifford C; diagonal images are stabilizers of |0^n>, so such rotations inject no magic.
    input_diagonal = ~gens.evolve(full_clifford, frame="h").x.any(axis=1)
    output_diagonal = np.asarray(~gens.x.any(axis=1))

    # A wire preserves a check iff Z there commutes with the check's back-cumulant. By
    # conjugation through the suffix, that is the commutation of the forward-propagated
    # generator with the check's syndrome operator, a Z-product on its support.
    active = np.ones(len(gens), dtype=bool)
    for support in checked.check_support if checked else ():
        active &= gens.x[:, list(support)].sum(axis=1) % 2 == 0
    _prune_to_fixpoint(gens, input_diagonal, output_diagonal, active)
    valid = [int(i) for i in np.flatnonzero(active)]

    if num_t_gates is None:
        chosen_idx = valid
    elif not 0 <= num_t_gates <= len(valid):
        raise ValueError(
            f"num_t_gates ({num_t_gates}) must be between 0 and the number of valid doping "
            f"sites ({len(valid)})."
        )
    else:
        # A subset of a fixed point need not be one: re-prune each draw and top up.
        rng = np.random.default_rng(seed)
        pool = [valid[i] for i in rng.permutation(len(valid))]
        selected = np.zeros(len(gens), dtype=bool)
        while need := num_t_gates - int(selected.sum()):
            if not pool:
                raise ValueError(
                    f"Could only draw {int(selected.sum())} irreducible doping sites of "
                    f"the requested {num_t_gates}; request fewer T gates or pass "
                    "num_t_gates=None."
                )
            selected[pool[:need]] = True
            del pool[:need]
            _prune_to_fixpoint(gens, input_diagonal, output_diagonal, selected)
        chosen_idx = [int(i) for i in np.flatnonzero(selected)]

    chosen = sorted((int(positions[i]), int(qubits[i])) for i in chosen_idx)
    inserts: dict[int, list[int]] = {}
    for position, qubit in chosen:
        inserts.setdefault(position, []).append(qubit)
    angles = iter(ParameterVector("dope", len(chosen))) if parametric else None
    doped = circuit.copy_empty_like()
    for position in range(len(circuit.data) + 1):
        for qubit in inserts.get(position, ()):
            if angles is None:
                doped.t(qubit)
            else:
                doped.rz(next(angles), qubit)
        if position < len(circuit.data):
            doped.append(circuit.data[position])

    sites = [DopingSite(qubit, position - 1 if position else None) for position, qubit in chosen]
    if checked is not None:
        return replace(checked, circuit=doped), sites
    return doped, sites


def _sweep_wire_segments(
    circuit: QuantumCircuit, site_qubits: set[int], entangling_only: bool
) -> tuple[np.ndarray, np.ndarray, PauliList, Clifford]:
    """Enumerate candidate sites, one per wire segment, with output-propagated generators.

    Sweeps the wire boundaries backward, maintaining the suffix Clifford ``S``; stabilizer
    row ``q`` of its tableau is the forward propagation ``S Z_q S^dag``. All boundaries of a
    wire segment (a maximal gate-free run on one qubit) share one generator with no rotation
    between them, so its earliest boundary represents it exhaustively. Segments past a
    terminal measurement are skipped; ``entangling_only`` keeps only wires directly
    following a multi-qubit gate.

    Returns:
        Time-sorted ``(positions, qubits, gens, full_clifford)``: a ``T`` at candidate ``i``
        precedes ``circuit.data[positions[i]]`` on ``qubits[i]`` and has generator
        ``gens[i]`` at the circuit output; ``full_clifford`` is the whole circuit's Clifford.

    Raises:
        ValueError: on a non-Clifford instruction or a non-terminal measurement.
    """
    data = circuit.data
    suffix = Clifford.from_label("I" * circuit.num_qubits)
    touched: set[int] = set()
    positions: list[int] = []
    qubits: list[int] = []
    x_rows: list[np.ndarray] = []
    z_rows: list[np.ndarray] = []

    def _record(position: int, qubit: int) -> None:
        positions.append(position)
        qubits.append(qubit)
        x_rows.append(suffix.stab_x[qubit].copy())
        z_rows.append(suffix.stab_z[qubit].copy())

    for position in range(len(data) - 1, -1, -1):
        inst = data[position]
        name = inst.operation.name
        if name == "barrier":
            continue
        qargs = [circuit.find_bit(qubit).index for qubit in inst.qubits]
        if name == "measure":
            # Sweeping backward, a gate already seen on this qubit lies after the measurement.
            if qargs[0] in touched:
                raise ValueError(
                    f"Qubit {qargs[0]} is used after its measurement; only terminal "
                    "measurements are supported."
                )
            continue
        touched.update(qargs)
        if len(qargs) > 1 or not entangling_only:
            for qubit in qargs:
                if qubit in site_qubits:
                    _record(position + 1, qubit)
        try:
            suffix = suffix.dot(inst.operation, qargs=qargs)
        except QiskitError as exc:
            raise ValueError(f"Non-Clifford instruction in circuit: {name!r}") from exc
    if not entangling_only:
        for qubit in sorted(site_qubits):
            _record(0, qubit)

    num_qubits = circuit.num_qubits
    gens = PauliList.from_symplectic(
        np.asarray(z_rows[::-1], dtype=bool).reshape(len(z_rows), num_qubits),
        np.asarray(x_rows[::-1], dtype=bool).reshape(len(x_rows), num_qubits),
    )
    return np.asarray(positions[::-1], dtype=int), np.asarray(qubits[::-1], dtype=int), gens, suffix


def _prune_to_fixpoint(
    gens: PauliList,
    input_diagonal: np.ndarray,
    output_diagonal: np.ndarray,
    active: np.ndarray,
) -> None:
    """Iterate the pruning rewrites of arXiv:2607.25941, Sec. S1.3 on ``active`` in place.

    Candidates must be time-sorted. A rotation is removed if it commutes with all previous
    active rotations and back-propagates to a diagonal on the input, or commutes with all
    following active rotations and forward-propagates to a diagonal on the output; of two
    rotations with equal propagated generators commuting with every active rotation between
    them, the later is removed. Repeats until no rewrite applies.
    """
    changed = True
    while changed:
        changed = False
        for i in np.flatnonzero(active & (input_diagonal | output_diagonal)):
            row = ~gens.commutes(gens[i])
            if (input_diagonal[i] and not (row[:i] & active[:i]).any()) or (
                output_diagonal[i] and not (row[i + 1 :] & active[i + 1 :]).any()
            ):
                active[i] = False
                changed = True
        groups: dict[bytes, list[int]] = {}
        for i in np.flatnonzero(active):
            groups.setdefault(gens.x[i].tobytes() + gens.z[i].tobytes(), []).append(int(i))
        for members in groups.values():
            # Equal generators share anticommutation rows, so one row serves the whole group.
            row = ~gens.commutes(gens[members[0]])
            anchor = members[0]
            for member in members[1:]:
                if (row[anchor + 1 : member] & active[anchor + 1 : member]).any():
                    anchor = member
                else:
                    active[member] = False
                    changed = True
