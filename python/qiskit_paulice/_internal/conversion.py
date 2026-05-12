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

import numpy as np
from qiskit import QuantumCircuit

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
