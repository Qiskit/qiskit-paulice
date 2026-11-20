=====================================================
Accelerating candidate scoring with a scoring engine
=====================================================

Candidate checks for ``cost="gamma"`` and ``cost="LER"`` are scored on the CPU by default.
A separately installed scoring engine can score them instead. This page covers how an
engine is enabled, which work it takes, and what its results are guaranteed to match.

Enabling an engine
==================

An engine is a separate package, installed alongside qiskit-paulice. Engines are discovered
and enabled through `Coheriq <https://github.com/Qiskit/coheriq>`_, which qiskit-paulice
installs as a dependency. Select the engine by name before the first call to
:func:`~qiskit_paulice.checks.add_pauli_checks`, either by setting ``PAULICE_ENGINE=<name>`` before
starting Python, or in code:

.. code:: python

    import coheriq
    import qiskit_paulice

    coheriq.enable_engine(qiskit_paulice, "<name>")

The choice is final for the rest of the process: a second registration is refused.

``PAULICE_ENGINE_DEVICE`` sets the index of the device the engine opens. It defaults to ``0``,
and a value that is not an integer also means ``0``.

:func:`qiskit_paulice.acceleration.scoring_engine_name` reports which engine is registered,
and :func:`qiskit_paulice.acceleration.run_engine_conformance` checks the registered engine
against the CPU.

What the engine scores
======================

An engine returns the same costs as the CPU. Work it cannot score that way runs on the CPU
as before. Each engine declines what exceeds its own limits, such as a largest circuit width,
and its documentation lists them.

- **Gamma.** The engine takes a batch only when the search has no stabilizers, the noise
  models are uniform depolarizing and readout only, and the circuit has no more qubits than
  the engine's format allows. Searches with other noise models, such as layered or gate-wise
  noise, or with stabilizers, are scored on the CPU.
- **LER.** The engine draws the Monte Carlo sample as well as scoring it, from the same random
  streams and Poisson tables as the CPU. A seeded search therefore finds the same checks at the
  same costs with or without an engine.

What is reproducible
====================

Costs are held to two different bars.

- **An engine against the CPU, on the same machine: bit for bit.** For ``cost="gamma"`` the
  engine returns each candidate's summed fault rates, added in the same order as on the CPU.
  The CPU then applies ``exp`` to them, so the device's maths library never enters a score.
  For ``cost="LER"`` the engine draws from the same random streams and Poisson tables as the
  CPU and counts shots as integers. :func:`~qiskit_paulice.acceleration.run_engine_conformance`
  checks both.
- **The CPU across platforms: close, but not always bit for bit.** The last bit of a
  ``cost="gamma"`` score comes from the platform's ``exp``, so it can differ between operating
  systems or after an update of the maths library. The test suite pins these scores to a
  relative tolerance of ``1e-12``. ``cost="LER"`` counts are pinned exactly, on Linux, macOS
  and Windows. The Poisson tables are built with ``exp`` too, so another maths library could in
  principle move a single draw, though a last-bit difference almost never does.

A change to how qiskit-paulice computes a cost shows up in those pinned values. If engines have
to follow it, the engine ABI version is bumped, and an engine built for the old version is
refused.
