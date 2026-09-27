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


from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli

from ._internal_r import CheckPicker, NoiseModel
from ._internal_r import PyMetric as Metric
from .conversion import convert_to_rustiq_circuit


def validate_terminal_measurements(circuit: QuantumCircuit) -> None:
    """Check that every measurement in ``circuit`` is the last instruction on its qubit.

    Barriers may follow a measurement. Any other instruction on a measured qubit after its
    measurement, including a second measurement, is rejected.

    Args:
        circuit: The circuit to check.

    Raises:
        ValueError: A qubit has an instruction other than a barrier after its measurement.
    """
    measured: set[int] = set()
    for inst in circuit.data:
        if inst.operation.name == "barrier":
            continue
        for qubit in (circuit.find_bit(q).index for q in inst.qubits):
            if qubit in measured:
                raise ValueError(
                    f"Qubit {qubit} has an instruction after its measurement; only one "
                    "terminal measurement per qubit is supported."
                )
            if inst.operation.name == "measure":
                measured.add(qubit)


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
