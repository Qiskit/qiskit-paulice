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

"""A ROCm scoring engine for qiskit-paulice.

Enable it with ``PAULICE_ENGINE=rocm`` or
``coheriq.enable_engine(qiskit_paulice, "rocm")``. Importing this package has
no side effect; Coheriq loads :mod:`qiskit_paulice_rocm._overrides` when the
engine is enabled.
"""
