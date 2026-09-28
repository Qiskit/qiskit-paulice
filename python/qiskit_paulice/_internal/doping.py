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

"""Implementation of :meth:`.CheckedCircuit.dope`."""

from __future__ import annotations

import numpy as np
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import Clifford, PauliList

from ..wire import Wire


def dope_circuit(
    circuit: QuantumCircuit,
    check_qubits: tuple[int, ...],
    check_support: tuple[tuple[int, ...], ...],
    num_sites: int | None,
    wires: str,
    angle: float | None,
    seed: int | np.random.Generator | None,
) -> tuple[QuantumCircuit, list[Wire]]:
    """Insert ``RZ`` rotations that commute with all Pauli check stabilizers into a Clifford circuit.

    Candidate doping locations are the wires selected by ``wires`` on non-check qubits. A candidate is
    kept only if its rotation leaves every check's syndrome unchanged. Redundant rotations
    are then removed: those that only add a global phase or do not change Z-basis
    measurement outcomes, and those that merge into an earlier rotation about the same
    Pauli. ``num_sites`` of the rest are drawn at random. Each rotation is placed directly after the gate its wire
    follows, or at the start of the circuit for an input wire.

    Args:
        circuit: Clifford circuit, with barriers and terminal measurements allowed.
        check_qubits: Ancilla qubits of the checks.
        check_support: For each check, the qubits that comprise its syndrome.
        num_sites: Number of rotations to insert, or ``None`` for every valid wire.
        wires: Candidate wires: ``"all"``, ``"after_entangling"``, or ``"before_entangling"``.
        angle: Angle of every rotation, or ``None`` for a free parameter ``dope[i]`` per
            rotation.
        seed: Seed or generator for drawing ``num_sites`` wires.

    Returns:
        The doped circuit, and the wires holding its rotations sorted by circuit position.
        Wire indices refer to ``circuit``, and rotation ``i`` has parameter ``dope[i]`` when
        ``angle`` is ``None``.

    Raises:
        ValueError: ``wires`` is not an allowed value, ``circuit`` has a non-Clifford
            instruction, ``num_sites`` is out of range, or the random draw cannot reach
            ``num_sites`` irredundant rotations.
    """
    if wires not in ("all", "after_entangling", "before_entangling"):
        raise ValueError(
            f"wires must be 'all', 'after_entangling', or 'before_entangling', not {wires!r}."
        )
    site_qubits = set(range(circuit.num_qubits)) - set(check_qubits)
    candidates, output_paulis, full_clifford = _sweep_wire_segments(circuit, site_qubits, wires)

    # The back-propagation of a Pauli P to the input is C^dag P C for the whole-circuit
    # Clifford C; a diagonal image stabilizes |0^n>, so the rotation only applies a phase.
    input_diagonal = ~output_paulis.evolve(full_clifford, frame="h").x.any(axis=1)
    output_diagonal = ~output_paulis.x.any(axis=1)

    # A check's syndrome is the product of Z over its support at the output, so a rotation
    # preserves it iff the rotation's output Pauli has an even number of X/Y on the support.
    active = np.ones(len(candidates), dtype=bool)
    for support in check_support:
        active &= output_paulis.x[:, list(support)].sum(axis=1) % 2 == 0
    _prune(output_paulis, input_diagonal, output_diagonal, active)
    valid = [int(i) for i in np.flatnonzero(active)]

    if num_sites is None:
        chosen_idx = valid
    elif not 0 <= num_sites <= len(valid):
        raise ValueError(
            f"num_sites ({num_sites}) must be between 0 and the number of valid doping "
            f"sites ({len(valid)})."
        )
    else:
        # A subset of an irreducible set need not be irreducible: re-prune each draw and top up.
        rng = np.random.default_rng(seed)
        pool = [valid[i] for i in rng.permutation(len(valid))]
        selected = np.zeros(len(candidates), dtype=bool)
        while need := num_sites - int(selected.sum()):
            if not pool:
                raise ValueError(
                    f"Could only draw {int(selected.sum())} irreducible doping sites of "
                    f"the requested {num_sites}; request fewer sites or pass "
                    "num_sites=None."
                )
            selected[pool[:need]] = True
            del pool[:need]
            _prune(output_paulis, input_diagonal, output_diagonal, selected)
        chosen_idx = [int(i) for i in np.flatnonzero(selected)]

    chosen = [candidates[i] for i in chosen_idx]
    rotations: dict[int | None, list[int]] = {}
    for wire in chosen:
        rotations.setdefault(wire.after_instruction, []).append(wire.qubit)
    angles = iter(ParameterVector("dope", len(chosen))) if angle is None else None
    doped = circuit.copy_empty_like()

    def insert_rotations(after_instruction: int | None) -> None:
        for qubit in rotations.get(after_instruction, ()):
            doped.rz(angle if angles is None else next(angles), qubit)

    insert_rotations(None)
    for index, instruction in enumerate(circuit.data):
        doped.append(instruction)
        insert_rotations(index)
    return doped, chosen


def _sweep_wire_segments(
    circuit: QuantumCircuit, site_qubits: set[int], wires: str
) -> tuple[list[Wire], PauliList, Clifford]:
    """List the wires an ``RZ`` rotation could be placed on, and the Pauli each one rotates about.

    Only qubits in ``site_qubits`` get candidates, and ``wires`` selects which:

    * ``"all"``: every wire, including each qubit's input wire. A wire that ends in a
      measurement is included, and nothing after a measurement is.
    * ``"after_entangling"``: the wire directly after each multi-qubit gate, on each of its
      qubits.
    * ``"before_entangling"``: the wire directly before each multi-qubit gate, on each of its
      qubits.

    An ``RZ`` rotation on qubit ``q`` anywhere along a wire has the same effect as a rotation
    by the same angle about ``U Z_q U^dag`` at the end of the circuit, where ``U`` is the
    Clifford made of every gate after the wire starts. That Pauli, without its sign, is the
    candidate's output Pauli.

    Args:
        circuit: Clifford circuit; measurements must be terminal and barriers are ignored.
        site_qubits: Qubits that may receive a rotation.
        wires: ``"all"``, ``"after_entangling"``, or ``"before_entangling"``.

    Returns:
        ``(candidates, output_paulis, full_clifford)``: the candidate wires sorted by circuit
        position (input wires first, then by ``after_instruction``, then by qubit), the output
        Pauli of each, and the Clifford of the whole circuit.

    Raises:
        ValueError: ``circuit`` has a non-Clifford instruction.
    """
    data = circuit.data
    qargs = [[circuit.find_bit(qubit).index for qubit in inst.qubits] for inst in data]
    # For each instruction, the last gate before it on each of its qubits.
    previous_gate: list[dict[int, int | None]] = []
    last_gate: dict[int, int] = {}
    for index, inst in enumerate(data):
        previous_gate.append({qubit: last_gate.get(qubit) for qubit in qargs[index]})
        if inst.operation.name not in ("barrier", "measure"):
            last_gate.update(dict.fromkeys(qargs[index], index))

    suffix = Clifford.from_label("I" * circuit.num_qubits)
    candidates: list[Wire] = []
    x_rows: list[np.ndarray] = []
    z_rows: list[np.ndarray] = []

    def _record(wire: Wire) -> None:
        # Row q of the suffix tableau's stabilizers is the image of Z_q through the suffix.
        candidates.append(wire)
        x_rows.append(suffix.stab_x[wire.qubit].copy())
        z_rows.append(suffix.stab_z[wire.qubit].copy())

    # Sweep backward so that `suffix` is always the Clifford of every gate after `index`.
    for index in range(len(data) - 1, -1, -1):
        name = data[index].operation.name
        if name in ("barrier", "measure"):
            continue
        entangling = len(qargs[index]) > 1
        if wires == "all" or (wires == "after_entangling" and entangling):
            for qubit in qargs[index]:
                if qubit in site_qubits:
                    _record(Wire(qubit, index))
        try:
            suffix = suffix.dot(data[index].operation, qargs=qargs[index])
        except QiskitError as exc:
            raise ValueError(f"Non-Clifford instruction in circuit: {name!r}") from exc
        if wires == "before_entangling" and entangling:
            # The suffix now includes this gate, as seen from the wire leading into it.
            for qubit in qargs[index]:
                if qubit in site_qubits:
                    _record(Wire(qubit, previous_gate[index][qubit]))
    if wires == "all":
        for qubit in sorted(site_qubits):
            _record(Wire(qubit, None))

    order = sorted(
        range(len(candidates)),
        key=lambda i: (
            -1 if candidates[i].after_instruction is None else candidates[i].after_instruction,
            candidates[i].qubit,
        ),
    )
    num_qubits = circuit.num_qubits
    output_paulis = PauliList.from_symplectic(
        np.asarray(z_rows, dtype=bool).reshape(len(z_rows), num_qubits)[order],
        np.asarray(x_rows, dtype=bool).reshape(len(x_rows), num_qubits)[order],
    )
    return [candidates[i] for i in order], output_paulis, suffix


def _prune(
    output_paulis: PauliList,
    input_diagonal: np.ndarray,
    output_diagonal: np.ndarray,
    active: np.ndarray,
) -> None:
    """Deactivate doping candidates until none of the remaining rotations is redundant.

    Candidate ``i`` is an ``RZ`` rotation which, commuted to the circuit output, rotates about
    the Pauli ``output_paulis[i]``. Only active candidates are considered present. An active
    candidate is deactivated if either of these holds:

    1. It commutes with every earlier active candidate and is diagonal at the input, so it
       moves to the start and acts on ``|0...0>`` as a global phase; or it commutes with
       every later active candidate and is diagonal at the output, so it moves to the end
       and leaves the Z-basis measurement distribution unchanged.
    2. It has the same Pauli as an earlier active candidate, and no active candidate
       between the two anticommutes with it, so it merges into that earlier rotation.

    One pass over rule 1 in circuit order, then rule 2, leaves no rule applicable. Rule 2
    and the later input checks of rule 1 see every rule 1 deactivation. No other
    deactivation unblocks a candidate:

    * A later candidate that blocks an output-diagonal one anticommutes with it. So it is
      not output-diagonal, and it fails the input condition while the blocked one is active.
    * A candidate merged by rule 2 leaves an equal active candidate earlier, with no
      anticommuting active candidate between them, so it blocked nothing that the earlier
      one does not.

    Args:
        output_paulis: The Pauli of each candidate at the circuit output, sorted by circuit
            position.
        input_diagonal: Whether each Pauli, conjugated back to the circuit input, has no X or
            Y component.
        output_diagonal: Whether each Pauli has no X or Y component.
        active: Mask of candidates in play; updated in place.
    """
    for i in np.flatnonzero(active & (input_diagonal | output_diagonal)):
        row = ~output_paulis.commutes(output_paulis[i])
        if (input_diagonal[i] and not (row[:i] & active[:i]).any()) or (
            output_diagonal[i] and not (row[i + 1 :] & active[i + 1 :]).any()
        ):
            active[i] = False
    groups: dict[bytes, list[int]] = {}
    for i in np.flatnonzero(active):
        key = output_paulis.x[i].tobytes() + output_paulis.z[i].tobytes()
        groups.setdefault(key, []).append(int(i))
    for members in groups.values():
        # Equal Paulis share anticommutation rows, so one row serves the whole group.
        row = ~output_paulis.commutes(output_paulis[members[0]])
        anchor = members[0]
        for member in members[1:]:
            if (row[anchor + 1 : member] & active[anchor + 1 : member]).any():
                anchor = member
            else:
                active[member] = False
