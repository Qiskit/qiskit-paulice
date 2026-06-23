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

"""Python wrapper for simulation functionalities.
"""


from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli

from ._internal_r import NoiseModel
from ._internal_r import PyMetric as Metric
from .utils import build_check_picker


def get_gamma(
    circuit: QuantumCircuit,
    noise_models: None | list[NoiseModel] = None,
    stabilizers: None | list[str] | list[Pauli] | str = None,
    measured_qubits: None | list[int] | str = None,
    check_qubits: None | list[int] = None,
    virtual_zs: None | list[list[int]] = None,
):
    """Use Monte-Carlo simulation to estimate the post-selection rate and logical error rate
    """
    check_picker = build_check_picker(
        circuit,
        Metric.gamma(),
        noise_models,
        stabilizers,
        measured_qubits,
        check_qubits,
        virtual_zs,
    )
    return check_picker.get_current_energy()
