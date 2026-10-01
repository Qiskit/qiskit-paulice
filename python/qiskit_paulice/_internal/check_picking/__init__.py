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

"""Main interface for check picking."""

from qiskit import QuantumCircuit

from .._internal_r import NoiseModel
from .._internal_r import PyMetric as Metric
from ..station import CheckPickerStation
from .genetic_search import genetic_algorithm, windowed_genetic_algorithm
from .windowed_search import windowed_check_picker

_METHODS = {
    "windowed": windowed_check_picker,
    "genetic": genetic_algorithm,
    "windowed_genetic": windowed_genetic_algorithm,
}


class CheckedCircuits:
    """Internal data container holding every variant produced by a single
    ``pick_checks`` run -- the bare circuit followed by one circuit per
    committed check.

    Attributes:
        circuits: List of ``QuantumCircuit``s; ``circuits[k]`` has the first
            ``k`` checks committed (``circuits[0]`` is the bare payload).
        check_qubits: Qubit indices of the ancillas (one per committed check).
        virtual_zs: For each committed check, the list of qubit indices whose
            Z-basis outcomes XOR together to give that check's syndrome bit.
        costs: Picker-metric values, one per variant; ``costs[k]`` is the
            metric after committing the first ``k`` checks. Length is
            ``len(check_qubits) + 1``.
        committed_targets: The target qubits that actually received a check, in
            commit order. May be a subsequence of the requested targets when
            some have no valid check (e.g. too-shallow wires).

    This class is internal; the public-facing ``CheckedCircuit`` (singular)
    in ``qiskit_paulice.checked_circuit`` is what users see.
    """

    def __init__(self, circuits, check_qubits, virtual_zs, costs, committed_targets):
        self.circuits = circuits
        self.check_qubits = check_qubits
        self.virtual_zs = virtual_zs
        self.costs: list[float] = list(costs)
        self.committed_targets: list[int] = list(committed_targets)


def pick_checks(
    circuit: QuantumCircuit,
    targets: list[int],
    noise_models: list[NoiseModel],
    metric=Metric.gamma(),
    stabilizers=None,
    measured_qubits=None,
    method="windowed",
    verbose=False,
    seed=None,
    logical_stabilizers=None,
    **kwargs,
):
    """All-in-one wrapper.
    Feed it a circuit, some target data qubits, some metric to optimize and a noise model.
    It will spit out a circuit with the best checks that it could find.

    Additional arguments are passed to the check picking method.

    Arguments:
        circuit (QuantumCircuit): the base circuit, without measurements
        targets (List[int]): list of target data to attach the checks to
        noise_models (List[NoiseModel]): list of noise models to consider during check picking
        metric (Metric): the metric to optimize during check picking
        stabilizers (List[str] or List[Pauli] or None or "all"): validity (and, by default,
          cost). Paulis stabilizing the input state, as `Pauli` (phase ignored) or internal
          labels, in which character i acts on qubit i, the reverse of a Qiskit label; "all"
          is Z on every qubit. A valid check back-propagates into the
          group they generate and its syndrome is the ancilla bit alone (`virtual_zs` are
          empty). Mutually exclusive with `measured_qubits`. See `CheckPickerStation`.
        measured_qubits (List[int] or None or "all"): qubits measured in Z at the end of the
          circuit, or "all". A valid check may absorb Z on measured qubits, which then join
          its `virtual_zs`. Mutually exclusive with `stabilizers`.
        method (str): the check picking method to use. One of "windowed",
          "genetic, "windowed_genetic"
        verbose (bool): whether to print progress information
        seed (int or None): random seed for reproducible check selection
        logical_stabilizers (List[str] or List[Pauli] or None): cost only, never validity.
          Input stabilizers whose forward images the metric protects, instead of
          `stabilizers`. Only with `stabilizers`.
        **kwargs: additional arguments passed to the check picking method

    Returns:
        CheckedCircuits: data container with circuits, check_qubits, virtual_zs, costs.
    """
    assert method in _METHODS, f"Unknown method {method} (should be one of {_METHODS})"
    assert stabilizers or measured_qubits, "Either stabilizer or measured qubits must be specified"
    assert not stabilizers or not measured_qubits, (
        "Only one of stabilizer or measured qubits can be specified"
    )
    assert noise_models, "No noise models specified"

    method = _METHODS[method]
    check_picker = CheckPickerStation(
        circuit,
        len(targets),
        metric,
        noise_models,
        stabilizers,
        measured_qubits,
        logical_stabilizers=logical_stabilizers,
    )
    if verbose:
        print("[CHECK PICKING] Initial metric value:", check_picker.get_current_energy())
    circuits, check_qubits, virtual_zs, costs, committed_targets = method(
        check_picker, targets, verbose=verbose, seed=seed, **kwargs
    )
    return CheckedCircuits(circuits, check_qubits, virtual_zs, costs, committed_targets)
