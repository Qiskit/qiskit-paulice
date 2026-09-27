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

from numbers import Real

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli

from ._internal_r import NoiseModel as RustNoiseModel

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
    """Convert a qiskit circuit to rustiq's gate list, plus a qiskit-index map.

    The second returned list ``qiskit_inst_indices`` runs parallel to
    ``rustiq_circuit``: ``qiskit_inst_indices[i]`` is the index into
    ``circuit.data`` of the qiskit ``CircuitInstruction`` that emitted the
    i-th rustiq gate. Some qiskit instructions emit zero rustiq gates (e.g.
    ``measure``, ``barrier``, ``id``, ``rz(0)``); some emit two (``x``, ``z``,
    ``rz(pi)``). This lets callers translate rustiq-side wire references back
    to positions in the original qiskit circuit.

    Measurements and barriers are ignored as they are not part of the Clifford
    circuit logic.
    """
    # Filter out measurements and barriers when checking gate set
    gate_names = set(
        q.operation.name for q in circuit if q.operation.name not in ("measure", "barrier")
    )
    assert gate_names <= set(
        ("cx", "h", "s", "x", "z", "sx", "sxdg", "sdg", "cz", "rz", "u1", "id")
    ), f"Gate set is: {gate_names}"

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
            param = gate.operation.params[0]
            if isinstance(param, (np.complex128, np.complex64, complex)):
                param = float(np.real(param))
            param = param % (2 * np.pi)
            if np.isclose(param, 0.0) or np.isclose(param, 2 * np.pi):
                continue
            if np.isclose(param, np.pi / 2):
                emit(("S", qbits), inst_idx)
                continue
            if np.isclose(param, np.pi):
                emit(("S", qbits), inst_idx)
                emit(("S", qbits), inst_idx)
                continue
            if np.isclose(param, 3 * np.pi / 2):
                emit(("Sd", qbits), inst_idx)
                continue
            emit(("RZ", qbits, str(param)), inst_idx)
        elif name == "I":
            continue
        elif name == "X":
            emit(("SqrtX", qbits), inst_idx)
            emit(("SqrtX", qbits), inst_idx)
        elif name == "Z":
            emit(("S", qbits), inst_idx)
            emit(("S", qbits), inst_idx)
        else:
            emit((name, qbits), inst_idx)
    return rustiq_circuit, qiskit_inst_indices


def convert_to_qiskit_circuit(circuit, nqbits):
    """Turns a rustiq circuit into a qiskit circuit
    """
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
