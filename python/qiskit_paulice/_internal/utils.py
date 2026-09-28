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
from .conversion import (
    convert_to_rustiq_circuit,
    normalize_measured_qubits,
    normalize_stabilizers,
)


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
    logical_stabilizers: None | list[str] | list[Pauli] = None,
):
    """Builds a rust CheckPicker object for a given qiskit circuit & some parameters.

    ``stabilizers`` decides validity: a check may back-propagate to any product of them.
    ``logical_stabilizers`` decides cost only: the metric protects their forward images instead
    of those of ``stabilizers`` (see :class:`.CheckPickerStation`).
    """
    noise_models = noise_models or []
    assert isinstance(metric, Metric), "metric should be a Metric instance"
    measured_qubits = normalize_measured_qubits(measured_qubits, circuit.num_qubits)
    stabilizers = normalize_stabilizers(stabilizers, circuit.num_qubits)
    logical = normalize_stabilizers(logical_stabilizers, circuit.num_qubits) or None
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
        logical,
    )
    picker.set_evaluation_data(noise_models, metric, circuit.num_qubits)
    return picker
