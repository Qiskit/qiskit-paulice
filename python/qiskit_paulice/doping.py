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
    preserve_distribution: bool = True,
    after_entangling_only: bool = False,
    parametric: bool = False,
    seed: int | np.random.Generator | None = None,
) -> tuple[QuantumCircuit, list[DopingSite]] | tuple[CheckedCircuit, list[DopingSite]]:
    r"""Insert magic-injecting ``T`` gates into a Clifford circuit.

    A ``T`` gate on a wire is the rotation :math:`e^{-i \pi Z / 8}` (up to global phase);
    conjugating its ``Z`` generator through the Clifford gates on either side of the wire
    classifies the site (`arXiv:2607.25941 <https://arxiv.org/abs/2607.25941>`_, Sec. S1.3).
    Two doping modes are supported:

    * **Distribution preserving** (``preserve_distribution=True``): a site is kept only if its
      generator propagated forward to the circuit output is diagonal (``I``/``Z``). Every
      inserted rotation is then a diagonal unitary commuting with ``Z`` measurements, so the
      doped circuit samples *exactly* the same computational-basis distribution as ``circuit``
      while its output state gains magic. Sites whose generator back-propagates to a diagonal
      on the input are discarded (the rotation acts as a global phase on :math:`|0^n\rangle`
      and injects no magic), and sites with coinciding propagated generators are deduplicated,
      keeping the earliest (two ``T`` gates on one propagated generator merge into a Clifford
      ``S``). Because all kept generators are diagonal at the output they mutually commute, so
      the pruning rewrites of the reference are provably already at their fixed point and each
      kept ``T`` is an irreducible non-Clifford rotation.

    * **XEB mode** (``preserve_distribution=False``): the selection of the reference is
      implemented instead, producing rotations that *change* the sampled distribution (the
      regime used for cross-entropy benchmarking). Every wire segment is a candidate, and the
      reference's three pruning rewrites are iterated to a fixed point: a rotation is removed
      if it commutes with all previous rotations and back-propagates to a diagonal on the
      input (it acts trivially), or commutes with all following rotations and forward-propagates
      to a diagonal on the output (it commutes into the final measurements); and of two
      rotations with equal propagated generators commuting with every rotation between them,
      only the earliest is kept (the pair would merge into a Clifford ``S``). Each returned
      site therefore contributes an irreducible non-Clifford rotation. Unlike the reference,
      which removes one member of a mergeable pair at random, the earliest is kept
      deterministically; the reference's restriction of candidates to wires directly
      following entangling gates is available via ``after_entangling_only``.

    When ``circuit`` is a :class:`.CheckedCircuit`, doping preserves its spacetime code: a
    wire is a valid site only if a ``Z`` on it commutes with the back-cumulant of every check,
    so all check syndromes remain deterministic in the doped circuit and post-selection is
    unaffected. Check ancilla wires are never doped. In distribution-preserving mode the
    constraint is automatically satisfied; in XEB mode it filters the candidate wires. The
    first return value is then a new :class:`.CheckedCircuit` wrapping the doped circuit with
    the same check metadata.

    Args:
        circuit: The Clifford circuit to dope, or a :class:`.CheckedCircuit` whose spacetime
            code the doping must preserve. Any instruction convertible to a
            `Clifford <https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.quantum_info.Clifford>`_
            is supported; barriers and terminal measurements are ignored (sites after a
            qubit's measurement are excluded).
        num_t_gates: The number of ``T`` gates to insert, drawn uniformly without replacement
            from the valid sites. ``None`` inserts a ``T`` at every valid site. In XEB mode a
            randomly drawn subset may itself be further reducible; sites are then redrawn
            until ``num_t_gates`` irreducible sites are reached.
        preserve_distribution: Whether to keep the sampled distribution exactly unchanged
            (``True``) or to select distribution-changing rotations as in the reference
            (``False``).
        after_entangling_only: Restrict sites to wires directly following an entangling
            (multi-qubit) gate, as in the reference. On hardware these wires sit inside the
            single-qubit layer that follows the entangling layer, so an inserted ``Z``
            rotation merges into it as a zero-cost virtual-``RZ`` phase and every doping
            configuration of one circuit shares a single pulse schedule.
        parametric: Insert ``rz(dope[i])`` rotations from a ``dope``
            `ParameterVector <https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.ParameterVector>`_
            (``dope[i]`` at ``sites[i]``) instead of ``T`` gates, yielding one template
            circuit for every doping configuration: bind :math:`\pi/4` for ``T``,
            :math:`\pi/2` for ``S``, :math:`\pi` for ``Z``, or ``0`` for identity. In
            distribution-preserving mode every assignment preserves the sampled distribution
            (all rotations remain diagonal), and on a :class:`.CheckedCircuit` every
            assignment preserves the code; the XEB-mode irreducibility guarantee applies to
            non-Clifford values.
        seed: A seed or generator for the random site selection. Unused when ``num_t_gates``
            is ``None``.

    Returns:
            * **QuantumCircuit | CheckedCircuit** -- A copy of ``circuit`` with ``T`` gates
              (or parametric ``RZ`` rotations) inserted (a :class:`.CheckedCircuit` when one
              was given)
            * **list[DopingSite]** -- The doped sites, sorted by circuit position

    Raises:
        ValueError: ``circuit`` contains a non-Clifford instruction or a non-terminal
            measurement.
        ValueError: ``num_t_gates`` is negative, exceeds the number of valid sites, or (in
            XEB mode) no subset of ``num_t_gates`` irreducible sites could be drawn.
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

    if preserve_distribution:
        # Kept generators are diagonal at the output, hence mutually commute: pruning is at
        # its fixed point and reduces to keeping the earliest site per generator class.
        classes: dict[bytes, int] = {}
        for i in np.flatnonzero(output_diagonal & ~input_diagonal):
            classes.setdefault(gens.z[i].tobytes(), int(i))
        valid = sorted(classes.values())
    else:
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
        rng = np.random.default_rng(seed)
        if preserve_distribution:
            chosen_idx = [valid[i] for i in rng.choice(len(valid), num_t_gates, replace=False)]
        else:
            # A subset of a fixed point need not be one: re-prune each draw and top up.
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
    """Enumerate candidate doping sites, one per wire segment, with propagated generators.

    Sweeps the wire boundaries backward, maintaining the Clifford implemented by the
    instruction suffix; stabilizer row ``q`` of its tableau is the forward propagation
    ``S Z_q S^dag``. A wire segment is a maximal run of boundaries on one qubit with no gate
    on that qubit in between: all its boundaries carry the same propagated generator with no
    rotation between them, so its earliest boundary represents it exhaustively. Segments past
    a qubit's terminal measurement are never emitted; with ``entangling_only``, only wires
    directly following a multi-qubit gate are emitted (the reference's site restriction).

    Returns:
        ``(positions, qubits, gens, full_clifford)``, time-sorted: per candidate the
        insertion position (a ``T`` there precedes ``circuit.data[position]``), its qubit,
        and (as a ``PauliList``) its generator propagated to the circuit output; plus the
        Clifford of the entire circuit.

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
