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

"""The two ways to anchor a check, for tests of invariants that must hold either way.

A check is anchored at the circuit's terminal measurements or at the input stabilizers of the
state it prepares. Tests of options that never touch this seam use the default, measurements.
"""

from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit_paulice import add_pauli_checks

MODES = ("measurements", "stabilizers")


def add_pauli_checks_in(mode: str, circuit: QuantumCircuit, targets, noise, **kwargs):
    """Run ``add_pauli_checks`` in ``mode`` on a circuit that has terminal measurements.

    In ``"stabilizers"`` mode the measurements are stripped and ``stabilizers="all"`` is passed.
    """
    if mode == "stabilizers":
        circuit = circuit.remove_final_measurements(inplace=False)
        kwargs["stabilizers"] = "all"
    return add_pauli_checks(circuit, targets, noise, **kwargs)
