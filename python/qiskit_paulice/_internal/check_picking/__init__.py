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
            ``len(check_qubits) + 1``. ``None`` if the picker method did not
            report it.

    This class is internal; the public-facing ``CheckedCircuit`` (singular)
    in ``qiskit_paulice.checked_circuit`` is what users see.
    """

    def __init__(self, circuits, check_qubits, virtual_zs, costs=None):
        self.circuits = circuits
        self.check_qubits = check_qubits
        self.virtual_zs = virtual_zs
        self.costs: list[float] | None = list(costs) if costs is not None else None


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
    **kwargs,
):
    """All-in-one wrapper.
    Feed it a circuit, some target data qubits, some metric to optimize and a noise model.
    It will spit out a circuit with the best checks that it could find.

    Additional arguments are passed to the check picking method.

    Arguments:
        circuit (QuantumCircuit): the base circuit
        targets (List[int]): list of target data to attach the checks to
        noise_models (List[NoiseModel]): list of noise models to consider during check picking
        metric (Metric): the metric to optimize during check picking
        stabilizers (List[str] or None or "all"): list of stabilizers to use as checks.
          Can be passed as a list of `Pauli`, a list of strings, or "all"
          (a shorthand to all Z stabilizers)
        measured_qubits (List[int] or None or "all"): list of measured qubits to use as checks.
          Can be passed as a list of integers, or "all" (a shorthand to all qubits)
        method (str): the check picking method to use. One of "windowed",
          "genetic, "windowed_genetic"
        verbose (bool): whether to print progress information
        seed (int or None): random seed for reproducible check selection
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
        circuit, len(targets), metric, noise_models, stabilizers, measured_qubits
    )
    if verbose:
        print("[CHECK PICKING] Initial metric value:", check_picker.get_current_energy())
    out = method(check_picker, targets, verbose=verbose, seed=seed, **kwargs)
    # Methods now return `(circuits, check_qubits, virtual_zs, costs)`; tolerate the
    # older 3-tuple form for backward compatibility.
    if len(out) == 4:
        circuits, check_qubits, virtual_zs, costs = out
    else:
        circuits, check_qubits, virtual_zs = out
        costs = None
    return CheckedCircuits(circuits, check_qubits, virtual_zs, costs)
