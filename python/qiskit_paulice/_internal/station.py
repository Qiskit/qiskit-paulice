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

import copy
import typing

from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli

from ._internal_r import CheckPicker, NoiseModel
from ._internal_r import PyMetric as Metric
from .conversion import convert_to_qiskit_circuit, convert_to_rustiq_circuit


class CheckPickerStation:
    """Docstring for CheckPickerStation
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        n_checks_to_add: int,
        metric: Metric = Metric.gamma(),
        noise_models: None | list[NoiseModel] = None,
        stabilizers: None | list[str] | list[Pauli] | str = None,
        measured_qubits: None | list[int] | str = None,
    ):
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
                    "".join(
                        "Z" if q == i else "I" for i in range(circuit.num_qubits + n_checks_to_add)
                    )
                    for q in set(range(circuit.num_qubits))
                ]
            else:
                raise ValueError("Unexpected stabilizers type")
        if stabilizers and isinstance(stabilizers[0], Pauli):
            if any(s.phase for s in stabilizers):
                raise ValueError("Pauli with phase not supported")
            stabilizers = [s.to_label()[::-1] for s in stabilizers]
        rustiq_circuit, _ = convert_to_rustiq_circuit(circuit)
        self.check_picker = CheckPicker(
            rustiq_circuit,
            circuit.num_qubits + n_checks_to_add,
            measured_qubits,
            stabilizers,
            None,
            None,
        )
        self.noise_models = noise_models
        self.metric = metric
        self.set_evaluation_data(noise_models, metric, circuit.num_qubits)
        self.ancilla = circuit.num_qubits
        self.tot_nqbits = circuit.num_qubits + n_checks_to_add

    def get_wires(self, qbit_index: int):
        """Returns the set of wires corresponding to a global qbit index

        Arguments:
            qbit_index: the qbit index
        """
        return self.check_picker.get_wires(qbit_index)

    def set_support(
        self,
        support: list[tuple[int, int]],
        paulis: None | list[int] = None,
        seed: None | int = None,
    ):
        """Sets the support of the check to pick. ``seed`` is forwarded to the
        Rust-side decoder so the middle-wire choice is reproducible across runs.
        """
        self.check_picker.set_support(support, paulis or [1, 2, 3], seed)

    def get_circuit(self) -> QuantumCircuit:
        """Returns the current circuit
        """
        rs_circuit = self.check_picker.get_circuit()
        return convert_to_qiskit_circuit(rs_circuit, self.tot_nqbits)

    def commit_check(self, check: list[bool]):
        """Commits a check specified by its coordinates
        """
        new_self = copy.copy(self)
        new_check_picker = self.check_picker.commit_check_bv(check)
        new_self.check_picker = new_check_picker
        new_self.set_evaluation_data(self.noise_models, self.metric, self.ancilla + 1)
        new_self.ancilla = self.ancilla + 1
        return new_self

    def get_check_data(self):
        """Returns the check qubit indices & their virtual CZz positions
        """
        czs = self.check_picker.get_virtual_zs()
        return list(range(self.tot_nqbits - len(czs), self.tot_nqbits)), czs

    def __getattribute__(self, name: str) -> typing.Any:
        try:
            return super().__getattribute__(name)
        except AttributeError:
            rust_check_picker = super().__getattribute__("check_picker")
            return getattr(rust_check_picker, name)

    def copy(self):
        """Makes a copy of the check picker"""
        new_self = copy.copy(self)
        new_self.check_picker = self.check_picker.copy()
        return new_self

    def find_good_check(self):
        """Explores a few different checks & commits the best one
        Might return None if the check decoding algorithm failed
        """
        new_check_picker = self.check_picker.find_good_checks()
        if new_check_picker is None:
            return new_check_picker
        new_check_picker, score = new_check_picker

        self.check_picker = new_check_picker
        self.set_evaluation_data(self.noise_models, self.metric, self.ancilla + 1)
        self.ancilla += 1
        return score
