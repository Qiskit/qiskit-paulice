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

from collections.abc import Sequence
from numbers import Real

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import CXGate, CZGate, HGate, SdgGate, SGate, SXdgGate, SXGate
from qiskit.quantum_info import Clifford, Pauli

from ._internal_r import NoiseModel as RustNoiseModel

RUSTIQ_GATES = {
    "CX": CXGate(),
    "CZ": CZGate(),
    "H": HGate(),
    "S": SGate(),
    "Sd": SdgGate(),
    "SqrtX": SXGate(),
    "SqrtXd": SXdgGate(),
}
"""Qiskit gate for each rustiq gate name emitted by :func:`convert_to_rustiq_circuit`."""

_NAMES_CONVERSION = {
    "cx": "CX",
    "cz": "CZ",
    "h": "H",
    "s": "S",
    "sdg": "Sd",
    "sxdg": "SqrtXd",
    "sx": "SqrtX",
    "x": "X",
    "z": "Z",
    "rz": "RZ",
    "u1": "RZ",
    "id": "I",
}


def convert_to_rustiq_circuit(circuit):
    """Convert a Clifford Qiskit circuit to rustiq's gate list, plus a Qiskit-index map.

    Measurements and barriers are skipped. ``x`` and ``z`` become two ``SqrtX`` or two ``S``
    gates, and ``rz``/``u1`` rotations by a multiple of pi/2 become zero, one or two ``S``/``Sd``
    gates (equal up to a global phase); ``id`` and rotations by a multiple of 2 pi emit nothing.

    Args:
        circuit: The circuit, built from ``cx``, ``cz``, ``h``, ``s``, ``sdg``, ``sx``,
            ``sxdg``, ``x``, ``z``, ``rz``, ``u1``, ``id``, ``measure`` and ``barrier``.

    Returns:
        ``(rustiq_circuit, qiskit_inst_indices)``: the list of ``(gate_name, qubit_indices)``
        pairs, and a parallel list whose ``i``-th entry is the index into ``circuit.data`` of
        the instruction that emitted the ``i``-th gate, for translating rustiq-side wire
        references back to positions in ``circuit``.

    Raises:
        ValueError: ``circuit`` contains any other instruction, an ``rz``/``u1`` with an
            unbound parameter, or one whose angle is not a real multiple of pi/2.
    """
    rustiq_circuit = []
    qiskit_inst_indices = []

    def emit(rustiq_gate, qiskit_inst_idx):
        rustiq_circuit.append(rustiq_gate)
        qiskit_inst_indices.append(qiskit_inst_idx)

    for inst_idx, gate in enumerate(circuit.data):
        # Skip measurements and barriers
        if gate.operation.name in ("measure", "barrier"):
            continue

        qbits = [circuit.find_bit(q).index for q in gate.qubits]
        if gate.operation.name not in _NAMES_CONVERSION:
            raise ValueError(f"Unsupported gate {gate}")
        name = _NAMES_CONVERSION[gate.operation.name]
        if name == "RZ":
            try:
                param = complex(gate.operation.params[0])
            except TypeError as exc:
                raise ValueError(
                    f"Unsupported gate {gate}: unbound parameter; bind it to a multiple of pi/2"
                ) from exc
            # Round to the nearest quarter turn with one absolute tolerance, so that
            # e.g. rz(-eps) and rz(eps) are classified alike.
            quarter_turns = param.real / (np.pi / 2)
            nearest = round(quarter_turns)
            if abs(param.imag) > 1e-8 or abs(quarter_turns - nearest) > 1e-8:
                raise ValueError(
                    f"Unsupported gate {gate}: non-Clifford rz angle {param:.4g} (not a real "
                    "multiple of pi/2)"
                )
            for rustiq_name in ((), ("S",), ("S", "S"), ("Sd",))[nearest % 4]:
                emit((rustiq_name, qbits), inst_idx)
            continue
        if name == "I":
            continue
        if name == "X":
            emit(("SqrtX", qbits), inst_idx)
            emit(("SqrtX", qbits), inst_idx)
        elif name == "Z":
            emit(("S", qbits), inst_idx)
            emit(("S", qbits), inst_idx)
        else:
            emit((name, qbits), inst_idx)
    return rustiq_circuit, qiskit_inst_indices


def clifford_of(circuit: QuantumCircuit) -> Clifford:
    """The Clifford ``circuit`` implements, ignoring measurements and barriers.

    Goes through :func:`convert_to_rustiq_circuit`, so every gate it accepts is supported,
    including ``rz`` by multiples of pi/2.

    Raises:
        ValueError: ``circuit`` contains a non-Clifford instruction.
    """
    gates, _ = convert_to_rustiq_circuit(circuit)
    clifford_circuit = QuantumCircuit(circuit.num_qubits)
    for name, qubits in gates:
        clifford_circuit.append(RUSTIQ_GATES[name], qubits)
    return Clifford(clifford_circuit)


def convert_stabilizers(
    stabilizers: Sequence[Pauli | str],
    circuit: QuantumCircuit,
    payload_qubits: Sequence[int] | None = None,
    num_qubits: int | None = None,
) -> list[str]:
    """Map stabilizers of the state ``circuit`` prepares to internal labels on its input.

    Args:
        stabilizers: Paulis or labels in Qiskit convention (the rightmost character of a label
            acts on qubit ``0``), each on ``num_qubits`` qubits. Without ``payload_qubits``
            they act on the qubits of ``circuit``; with it, on a register in which
            ``payload_qubits[v]`` holds qubit ``v`` of ``circuit`` and every other qubit must
            carry the identity.
        circuit: The measurement-free Clifford circuit, ``U``.
        payload_qubits: The register position of each qubit of ``circuit``, or ``None``.
        num_qubits: The width every stabilizer must have: the register width with
            ``payload_qubits``, where it is required, else ``circuit.num_qubits`` by default.

    Returns:
        For each stabilizer ``G``, the internal label of ``U^dagger G U``: character ``i`` acts
        on qubit ``i`` of ``circuit``, the reverse of a Qiskit label, and the sign is dropped.

    Raises:
        ValueError: A Pauli has the wrong width, acts outside ``payload_qubits``, or is not a
            stabilizer of the prepared state, i.e. its back-propagated image has an ``X`` or
            ``Y`` component; ``payload_qubits`` is given without ``num_qubits``.
    """
    clifford = clifford_of(circuit)
    if num_qubits is None:
        if payload_qubits is not None:
            raise ValueError("num_qubits is required when payload_qubits is given.")
        num_qubits = circuit.num_qubits
    labels = []
    for stabilizer in stabilizers:
        pauli = stabilizer if isinstance(stabilizer, Pauli) else Pauli(stabilizer)
        if pauli.num_qubits != num_qubits:
            raise ValueError(
                f"Stabilizer {stabilizer} acts on {pauli.num_qubits} qubits; expected "
                f"{num_qubits}."
            )
        if payload_qubits is not None:
            payload = list(payload_qubits)
            off_payload = np.ones(num_qubits, dtype=bool)
            off_payload[payload] = False
            if (pauli.x[off_payload] | pauli.z[off_payload]).any():
                raise ValueError(
                    f"Stabilizer {stabilizer} acts on qubits outside the payload "
                    f"{sorted(payload)}; stabilizers must be the identity on every other qubit."
                )
            pauli = Pauli((pauli.z[payload], pauli.x[payload]))
        # Heisenberg frame: the image of G on the input state |0...0>.
        image = pauli.evolve(clifford, frame="h")
        if image.x.any():
            raise ValueError(f"{stabilizer} is not a stabilizer of the state the circuit prepares.")
        labels.append("".join("Z" if z else "I" for z in image.z))
    return labels


def normalize_stabilizers(
    stabilizers: None | list[str] | list[Pauli] | str, num_qubits: int
) -> list[str]:
    """Turn a stabilizer specification into internal Pauli labels for the Rust picker.

    An internal label has character ``i`` acting on qubit ``i``: the reverse of a Qiskit label,
    whose rightmost character acts on qubit ``0``.

    Args:
        stabilizers: ``None`` (no stabilizers), ``"all"`` (Z on each of the ``num_qubits``
            qubits, the stabilizer group of the all-zeros state), a list of internal labels
            passed through unchanged, or a list of :class:`~qiskit.quantum_info.Pauli`, which
            are converted. A Pauli's phase is dropped, since neither group membership nor
            commutation depends on it. The Rust side pads labels shorter than its register
            with identities.
        num_qubits: Number of qubits ``"all"`` expands over.

    Raises:
        ValueError: ``stabilizers`` is a string other than ``"all"``.
    """
    if stabilizers is None:
        return []
    if isinstance(stabilizers, str):
        if stabilizers != "all":
            raise ValueError(f"Unexpected stabilizers value {stabilizers!r}; expected 'all'")
        return ["I" * q + "Z" + "I" * (num_qubits - q - 1) for q in range(num_qubits)]
    return [
        Pauli((s.z, s.x)).to_label()[::-1] if isinstance(s, Pauli) else s for s in stabilizers
    ]


def normalize_measured_qubits(measured_qubits: None | list[int] | str, num_qubits: int) -> list[int]:
    """Expand ``"all"`` to every qubit index and ``None`` to no qubits."""
    if measured_qubits is None:
        return []
    if isinstance(measured_qubits, str):
        if measured_qubits != "all":
            raise ValueError(f"Unexpected measured_qubits value {measured_qubits!r}; expected 'all'")
        return list(range(num_qubits))
    return list(measured_qubits)


def convert_to_qiskit_circuit(circuit, nqbits):
    """Turns a rustiq circuit into a qiskit circuit"""
    qs_circuit = QuantumCircuit(nqbits)
    for gate, qbits in circuit:
        if gate == "H":
            qs_circuit.h(*qbits)
        elif gate == "CNOT":
            qs_circuit.cx(*qbits)
        elif gate == "CZ":
            qs_circuit.cz(*qbits)
        elif gate == "S":
            qs_circuit.s(*qbits)
        elif gate == "SqrtX":
            qs_circuit.sx(*qbits)
        elif gate == "Sd":
            qs_circuit.sdg(*qbits)
        elif gate == "SqrtXd":
            qs_circuit.sxdg(*qbits)
        else:
            raise ValueError(f"Unknown rustiq gate {gate}")
    return qs_circuit


def convert_noise_model(noise_model, circuit: QuantumCircuit) -> RustNoiseModel | None:
    """Validate a :class:`~qiskit_paulice.NoiseModel` and convert its gate noise to Rust.

    Args:
        noise_model: The noise model to validate and convert.
        circuit: The circuit the noise model will be applied to; layered gate noise requires
            it to contain no ``cx`` gates.

    Returns:
        The Rust noise model for the gate noise, or ``None`` when there is no gate noise (an
        empty gate-noise dict counts as none). Readout noise is validated but not converted.

    Raises:
        ValueError: Idling noise is set; the model has neither gate nor readout noise;
            ``readout_noise`` lies outside ``[0, 0.5)``; uniform gate noise lies outside
            ``[0, 3)``; the gate noise is not a number, a layered or a gate-wise
            specification; or layered gate noise is paired with a circuit containing ``cx``
            gates.
    """
    if noise_model.idling_noise is not None:
        raise ValueError("Idling noise is not supported.")
    readout = noise_model.readout_noise
    if readout is not None and not 0 <= readout < 0.5:
        raise ValueError("readout_noise must lie in [0, 0.5).")
    gate_noise = noise_model.gate_noise
    first_key = next(iter(gate_noise), None) if isinstance(gate_noise, dict) else None
    if gate_noise is None or gate_noise == {}:
        model = None
    elif isinstance(gate_noise, Real) and not isinstance(gate_noise, bool):
        # The Rust rate -ln(1 - p/3)/4 is finite and non-negative only for p in [0, 3).
        if not 0 <= gate_noise < 3:
            raise ValueError(f"Uniform gate_noise must lie in [0, 3), not {gate_noise!r}.")
        model = RustNoiseModel.uniform_depolarizing(float(gate_noise))
    elif isinstance(first_key, tuple) and first_key and isinstance(first_key[0], tuple):
        if any(inst.operation.name == "cx" for inst in circuit.data):
            raise ValueError(
                "Layered gate noise requires a CZ-based circuit (the Rust layering pass "
                "does not support CX gates); transpile CX to CZ first."
            )
        model = RustNoiseModel.layered(convert_layered_noise(gate_noise))
    elif isinstance(first_key, tuple) and len(first_key) == 2 and isinstance(first_key[0], int):
        model = RustNoiseModel.gate_wise(convert_gate_wise_noise(gate_noise))
    else:
        raise ValueError(f"Unrecognized gate noise specification: {gate_noise!r}")
    if model is None and readout is None:
        raise ValueError("The noise model may not be empty.")
    return model


def convert_layered_noise(noise):
    """Convert layered gate noise to the form the Rust layered noise model expects.

    Args:
        noise: A :data:`~qiskit_paulice.noise_models.LayeredGateNoise` mapping each layer (a
            tuple of qubit-pair edges) to ``(pauli, rate)`` pairs, where ``pauli`` is a
            :class:`~qiskit.quantum_info.Pauli` or a label with qubit 0 rightmost.

    Returns:
        The same mapping with each edge written ``(min, max)``, each layer's edges sorted,
        and each Pauli given as a label with qubit 0 leftmost.

    Raises:
        ValueError: The edges of a layer share a qubit.
    """
    new_noise = {}
    for layer in noise:
        # The Rust layering pass always uses canonical ``(min, max)`` edge tuples internally,
        # so non-canonical user layer keys (e.g. ``((1, 0),)``) would otherwise silently
        # fail to match. Canonicalize each edge and re-sort the layer's edges here.
        canonical_layer = tuple(sorted((min(e), max(e)) for e in layer))
        # A layer's edges fire simultaneously, so they must be pairwise disjoint (a
        # matching); overlapping edges (including duplicates like ``((a, b), (b, a))``)
        # cannot be layered and would panic the Rust layering pass.
        qubits = [q for edge in canonical_layer for q in edge]
        if len(set(qubits)) != len(qubits):
            raise ValueError(
                f"Layer {layer!r} is not a matching: its edges must be pairwise disjoint."
            )
        converted_noise = []
        for p, r in noise[layer]:
            p_str = p.to_label() if isinstance(p, Pauli) else p
            # User-facing strings follow Qiskit convention (rightmost char = qubit 0);
            # the Rust consumer indexes left-to-right (leftmost char = qubit 0).
            converted_noise.append((p_str[::-1], r))
        new_noise[canonical_layer] = converted_noise
    return new_noise


def convert_gate_wise_noise(noise):
    """Convert gate-wise noise to the integer form the Rust gate-wise noise model expects.

    Args:
        noise: A :data:`~qiskit_paulice.noise_models.GateWiseNoise` mapping each edge
            ``(a, b)`` to ``(pauli, rate)`` pairs, where ``pauli`` is a 2-character label whose
            first character acts on ``a`` and second on ``b``.

    Returns:
        The same mapping with each label replaced by a pair of integers, 0=I, 1=X, 2=Y, 3=Z.

    Raises:
        ValueError: A Pauli is not a 2-character string of ``I``, ``X``, ``Y`` and ``Z``.
    """
    pauli_map = {"I": 0, "X": 1, "Y": 2, "Z": 3}
    new_noise = {}
    for edge in noise:
        converted_noise = []
        for p_str, r in noise[edge]:
            if not isinstance(p_str, str) or len(p_str) != 2 or not set(p_str) <= set(pauli_map):
                raise ValueError(
                    "Each gate-wise generator must be a 2-character Pauli string paired "
                    "left-to-right with the edge tuple (e.g. 'XZ' on edge (a, b) = X on a, "
                    "Z on b)."
                )
            # ``p_str[0]`` on edge[0], ``p_str[1]`` on edge[1] -- same convention as
            # PauliLindbladMap's sparse ``(pauli_str, indices)`` form.
            p_tuple = (pauli_map[p_str[0]], pauli_map[p_str[1]])
            converted_noise.append((p_tuple, r))
        new_noise[edge] = converted_noise
    return new_noise
