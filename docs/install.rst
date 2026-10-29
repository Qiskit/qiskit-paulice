Installation instructions
=========================


Prerequisites
^^^^^^^^^^^^^

First, create a minimal environment with only Python installed in it. We recommend using `Python virtual environments <https://docs.python.org/3.10/tutorial/venv.html>`__.

.. code:: sh

    python3 -m venv /path/to/virtual/environment

Activate your new environment.

.. code:: sh

    source /path/to/virtual/environment/bin/activate

There are two primary ways to install this package: from PyPI or from source. The preferred method is to install from PyPI:

Install from PyPI
^^^^^^^^^^^^^^^^^

.. code:: sh

    pip install qiskit-paulice


Install from source
^^^^^^^^^^^^^^^^^^^

If you plan to develop in the repository or run the notebooks locally, you should install from source.

First, clone the ``qiskit-paulice`` repository.

.. code:: sh

    git clone git@github.com:Qiskit/qiskit-paulice.git

Next, install the Rust toolchain, upgrade pip, and enter the repository. Refer to the `Rust documentation <https://www.rust-lang.org/tools/install>`__
for instructions on installing the toolchain.

.. code:: sh
    
    ### <INSTALL RUST HERE> ###
    pip install --upgrade pip
    cd qiskit-paulice

The next step is to install ``qiskit-paulice`` to the virtual environment. If you plan to run the notebooks and their visualizations, install the notebook dependencies.
If you plan on developing in the repository, install the ``dev`` dependencies.

Adjust the options below to suit your needs.

.. code:: sh

    pip install tox notebook -e '.[notebook-dependencies,dev]'

If you installed the notebook dependencies, you can get started by running the notebooks in the docs.

.. code::

    cd docs/
    jupyter lab

Optional: score candidate checks with an acceleration engine
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Candidate checks for ``cost="gamma"`` and ``cost="LER"`` can be scored by a separately
installed acceleration engine instead of on the CPU. Engines are discovered and enabled
through `Coheriq <https://github.com/Qiskit/coheriq>`_, so install it alongside the
engine package. Coheriq is an early prototype and is not yet released on PyPI, so it is
installed from its repository:

.. code:: sh

    pip install "coheriq @ git+https://github.com/Qiskit/coheriq.git"
    pip install <engine-package>

Then select the engine by name before the first call to
:func:`~qiskit_paulice.add_pauli_checks`, either by setting ``PAULICE_ENGINE=<name>``
before starting Python or in code:

.. code:: python

    import coheriq
    import qiskit_paulice  # registers the "qiskit_paulice" domain

    coheriq.enable_engine("qiskit_paulice", "<name>")

The choice is final for the rest of the process. ``PAULICE_ENGINE_DEVICE`` sets the
index of the device the engine opens (default ``0``; a value that is not an integer also
means ``0``).

An engine returns the same costs as the CPU, bit for bit, and anything it cannot score
that way runs on the CPU as before. For ``cost="gamma"`` that includes searches with
layered or gate-wise noise models, with stabilizers, or on more than 256 qubits. For
``cost="LER"`` the engine draws the Monte Carlo sample as well as scoring it, from the
same random streams as the CPU, so a seeded search finds the same checks at the same
costs either way. A candidate runs on the CPU instead if it has more than 65535 error
generators, if its packed scoring row exceeds 1024 words, or if the engine declines it.

:func:`qiskit_paulice.acceleration.run_engine_conformance` checks the enabled engine
against the CPU, and :func:`qiskit_paulice.acceleration.scoring_engine_name` reports
which engine is installed.
