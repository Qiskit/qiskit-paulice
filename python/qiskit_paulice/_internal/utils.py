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

"""Some utility functions for qiskit_paulice_r.
"""


import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli

from ._internal_r import CheckPicker, NoiseModel
from ._internal_r import PyMetric as Metric
from .conversion import convert_to_rustiq_circuit


def random_square_circuit(num_qubits: int):
    """Generates a random squaure LNN circuit"""
    circuit = QuantumCircuit(num_qubits)
    circuit.h(range(num_qubits))
    for d in range(num_qubits):
        for i in range(d % 2, num_qubits - 1, 2):
            circuit.cz(i, i + 1)
        for q in range(num_qubits):
            if np.random.randint(0, 2):
                circuit.sx(q)
            if np.random.randint(0, 2):
                circuit.s(q)
            if np.random.randint(0, 2):
                circuit.sx(q)
    return circuit


def build_check_picker(
    circuit: QuantumCircuit,
    metric: Metric,
    noise_models: None | list[NoiseModel] = None,
    stabilizers: None | list[str] | list[Pauli] | str = None,
    measured_qubits: None | list[int] | str = None,
    check_qubits=None,
    virtual_zs=None,
):
    """Builds a rust CheckPicker object for a gicen qiskit circuit & some parameters
    """
    measured_qubits = measured_qubits or []
    stabilizers = stabilizers or []
    noise_models = noise_models or []
    if isinstance(measured_qubits, str):
        if measured_qubits == "all":
            measured_qubits = list(set(range(circuit.num_qubits)))
        else:
            raise ValueError("Unexpected measured_qubits type")
    assert isinstance(metric, Metric), "metric should be a Metric instance"
    if isinstance(stabilizers, str):
        if stabilizers == "all":
            stabilizers = [
                "".join("Z" if q == i else "I" for i in range(circuit.num_qubits))
                for q in set(range(circuit.num_qubits))
            ]
        else:
            raise ValueError("Unexpected stabilizers type")
    if stabilizers and isinstance(stabilizers[0], Pauli):
        if any(s.phase for s in stabilizers):
            raise ValueError("Pauli with phase not supported")
        stabilizers = [s.to_label()[::-1] for s in stabilizers]
    if check_qubits is None:
        check_qubits = []
    if virtual_zs is None:
        virtual_zs = []
    if len(virtual_zs) < len(check_qubits):
        virtual_zs = virtual_zs + [[]] * (len(check_qubits) - len(virtual_zs))
    rustiq_circuit, _ = convert_to_rustiq_circuit(circuit)
    picker = CheckPicker(
        rustiq_circuit,
        circuit.num_qubits,
        measured_qubits,
        stabilizers,
        check_qubits,
        virtual_zs,
    )
    picker.set_evaluation_data(noise_models, metric, circuit.num_qubits)
    return picker
