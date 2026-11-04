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

"""Entry-point target for ``coheriq.engines.qiskit_paulice:rocm``."""

from __future__ import annotations

# Registers and materializes the domain before the engine looks it up.
import qiskit_paulice.acceleration  # noqa: F401  pylint: disable=unused-import
from coheriq import AccelerationEngine

from ._engine import engine_capsule

_engine = AccelerationEngine("qiskit_paulice", "rocm")


@_engine.override
def scoring_engine():
    """Return the ROCm engine's ``PauliceEngine`` table."""
    return engine_capsule()


_engine.materialize()
