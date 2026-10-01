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
from .conversion import (
    convert_to_qiskit_circuit,
    convert_to_rustiq_circuit,
    normalize_measured_qubits,
    normalize_stabilizers,
)


class CheckPickerStation:
    """Python-side driver of the Rust ``CheckPicker`` for one check-picking run.

    Exactly one of ``stabilizers`` and ``measured_qubits`` should be given; they define what
    a valid check is and what the metric protects.

    Args:
        circuit: Payload circuit, without measurements.
        n_checks_to_add: Number of ancilla qubits to reserve after the payload qubits.
        metric: Cost metric to minimize.
        noise_models: Rust noise models used by the metric.
        stabilizers: **Validity** (and, by default, cost). Paulis stabilizing the circuit's
            input state, as ``"all"`` (Z on every qubit, the stabilizer group of the all-zeros
            input), a list of :class:`~qiskit.quantum_info.Pauli` (phase ignored), or a list
            of internal labels, in which character ``i`` acts on qubit ``i``, the reverse of
            a Qiskit label. A check is valid iff its back-propagated product lies in the
            group they generate; its syndrome is the ancilla bit alone. Unless
            ``logical_stabilizers`` is given, the metric also counts an error as logical iff
            it anticommutes with the forward image of one of them.
        measured_qubits: **Validity and cost**, for measurement-anchored checks. Qubits
            measured in the Z basis at the end of the circuit, or ``"all"``. A check is valid
            iff its product, times Z on some measured qubits, propagates to the identity;
            those qubits join the syndrome. The metric counts an error as logical iff it flips
            a measured bit.
        logical_stabilizers: **Cost only**; never affects which checks are valid. Input
            stabilizers whose forward images the metric protects, in place of ``stabilizers``.
            Same formats as ``stabilizers``; only meaningful with ``stabilizers`` and not
            ``measured_qubits``.
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        n_checks_to_add: int,
        metric: Metric = Metric.gamma(),
        noise_models: None | list[NoiseModel] = None,
        stabilizers: None | list[str] | list[Pauli] | str = None,
        measured_qubits: None | list[int] | str = None,
        logical_stabilizers: None | list[str] | list[Pauli] = None,
    ):
        noise_models = noise_models or []
        assert isinstance(metric, Metric), "metric should be a Metric instance"
        measured_qubits = normalize_measured_qubits(measured_qubits, circuit.num_qubits)
        stabilizers = normalize_stabilizers(stabilizers, circuit.num_qubits)
        logical = normalize_stabilizers(logical_stabilizers, circuit.num_qubits) or None
        rustiq_circuit, _ = convert_to_rustiq_circuit(circuit)
        self.check_picker = CheckPicker(
            rustiq_circuit,
            circuit.num_qubits + n_checks_to_add,
            measured_qubits,
            stabilizers,
            None,
            None,
            logical,
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
