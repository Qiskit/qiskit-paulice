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

"""Optional scoring engines.

Candidate checks are scored on the host unless a scoring engine is enabled. An
engine is a separately installed package, selected through `Coheriq
<https://github.com/Qiskit/coheriq>`_ before the first call to
:func:`~qiskit_paulice.checks.add_pauli_checks`, either with
``coheriq.enable_engine("qiskit_paulice", "<engine>")`` or by setting
``PAULICE_ENGINE=<engine>``.

The engine overrides one Coheriq candidate, :func:`scoring_engine`, to return
its native scoring table. qiskit-paulice calls that table from native code, so
there is no Python call per batch. An engine returns exactly the scores the
host would, and declines, so that the host scores it instead, any work it
cannot score that way. :func:`run_engine_conformance` checks an engine against
the host.
"""

from __future__ import annotations

import os
import threading
import warnings

from ._internal._internal_r import register_scoring_engine as _register_scoring_engine
from ._internal._internal_r import run_engine_conformance as _run_engine_conformance
from ._internal._internal_r import scoring_engine_name as _scoring_engine_name

try:
    from coheriq import AccelerationDomain
except ImportError:  # Without Coheriq no engine can be enabled.
    AccelerationDomain = None


def _host_scoring_engine():
    """Return the native scoring table to install, or ``None`` to score on the host.

    An engine overrides this to return a PyCapsule named
    ``qiskit_paulice.engine`` that carries a ``PauliceEngine`` table, as
    declared in ``include/paulice_engine.h``.
    """
    return None


if AccelerationDomain is not None:
    _domain = AccelerationDomain("qiskit_paulice", env_prefix="PAULICE")
    scoring_engine = _domain.acceleration_candidate(name="scoring_engine")(_host_scoring_engine)
    _domain.materialize()
else:
    scoring_engine = _host_scoring_engine

_install_lock = threading.Lock()
_installed = False


def install_scoring_engine() -> None:
    """Hand the enabled engine's table to the native layer, once per process.

    Called by :func:`~qiskit_paulice.checks.add_pauli_checks`. This is the first
    use of the Coheriq domain, so the engine choice is final afterwards.
    """
    global _installed  # pylint: disable=global-statement
    if _installed:
        return
    with _install_lock:
        if _installed:
            return
        if AccelerationDomain is None and os.environ.get("PAULICE_ENGINE"):
            warnings.warn(
                "PAULICE_ENGINE is set but Coheriq is not installed, so candidate checks "
                "are scored on the host. Install it from https://github.com/Qiskit/coheriq.",
                stacklevel=3,
            )
        table = scoring_engine()
        if table is not None:
            _register_scoring_engine(table)
        _installed = True


def scoring_engine_name() -> str | None:
    """Return the name of the installed scoring engine, or ``None``."""
    name: str | None = _scoring_engine_name()
    return name


def run_engine_conformance() -> list[tuple[str, str, str]]:
    """Check the installed engine against the host, bit for bit.

    Installs the enabled engine first. Returns ``(check, outcome, detail)``
    for every check, where ``outcome`` is ``"passed"``, ``"failed"`` or
    ``"skipped"``. Every check is skipped when no engine is installed or it
    opens no device.
    """
    install_scoring_engine()
    results: list[tuple[str, str, str]] = _run_engine_conformance()
    return results
