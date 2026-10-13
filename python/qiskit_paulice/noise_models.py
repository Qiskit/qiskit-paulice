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

"""Noise model specification for evaluating spacetime Pauli checks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from qiskit.providers import BackendV2
from qiskit.quantum_info import Pauli, PauliLindbladMap

# Type aliases for gate noise specifications
UniformGateNoise = float
"""An average gate infidelity for every 2-qubit gate.

A depolarizing channel is applied after each 2-qubit gate. Each of the ``15`` non-identity
2-qubit Paulis occurs with the same probability, and the total probability that a gate suffers
any Pauli error is ``5 / 4`` of the infidelity, so the value must lie in ``[0, 0.75)``.
"""

LayeredGateNoise = dict[tuple[tuple[int, int], ...], list[tuple[Pauli | str, float]]]
"""Layered noise model mapping unique entangling layers to Pauli error generators.

.. note::
    The Pauli-Lindblad error associated with each entangling layer is assumed to be defined
    **before** the gates in the layer.

Keys are tuples of qubit index pairs (edges) defining a layer. Values are lists of
``(pauli, rate)`` tuples where ``pauli`` is a Pauli or Pauli string defined over all qubits
and ``rate`` is the associated error rate.
"""

GateWiseNoise = dict[tuple[int, int], list[tuple[str, float]]]
"""Gate-wise noise model mapping qubit pairs to a Pauli noise channel.

Keys are tuples of qubit index pairs representing edges. Values are lists of (pauli, rate)
tuples where ``pauli`` is a 2-character Pauli string and ``rate`` is the associated error
rate. The string is paired left-to-right with the edge tuple — matching
:class:`~qiskit.quantum_info.PauliLindbladMap`'s sparse form ``(pauli_str, indices)``: for
edge ``(a, b)``, ``"XZ"`` places ``X`` on ``a`` and ``Z`` on ``b``.
"""

GateNoise = UniformGateNoise | LayeredGateNoise | GateWiseNoise
"""Gate noise can be uniform (float), layered (dict), or gate-wise (dict)."""


_INFIDELITY_TO_ERROR_PROBABILITY = 5 / 4
"""Total Pauli error probability of a two-qubit depolarizing channel per unit of average gate
infidelity: ``(D + 1) / D`` for Hilbert-space dimension ``D = 4``."""


def _equal_generator_rate(paulis: Sequence[str], error_probability: float) -> float:
    """Common rate of two-qubit Pauli generators whose channel has a given total error probability.

    The generators fire independently, so the channel they produce is not ``paulis`` with
    probability ``error_probability / len(paulis)`` each: products of fired generators land on
    other Paulis, and some products cancel. The channel multiplies the expectation value of a
    Pauli ``Q`` by ``exp(-2 r n_Q)``, where ``n_Q`` counts the generators anticommuting with
    ``Q``, so the probability that it applies no error is the mean of ``exp(-2 r n_Q)`` over all
    ``16`` two-qubit Paulis. This function returns the rate ``r`` at which that probability equals
    ``1 - error_probability``, found by bisection since it decreases monotonically in ``r``. For
    the full basis of ``15`` Paulis every non-identity ``Q`` has ``n_Q = 8``, which gives
    ``-ln(1 - 16 p / 15) / 16``, the rate of :data:`.UniformGateNoise`.

    Args:
        paulis: Two-character labels over ``I``, ``X``, ``Y``, ``Z`` of the generators.
        error_probability: Total probability that the channel applies a non-identity Pauli.

    Returns:
        The rate shared by the generators; ``0.0`` when ``error_probability`` is ``0``.

    Raises:
        ValueError: ``error_probability`` is negative, or at least the fraction of two-qubit Paulis
            that anticommute with some generator, which is the most error the generators can
            produce.
    """
    labels = [a + b for a in "IXYZ" for b in "IXYZ"]
    x = np.array([[c in "XY" for c in label] for label in labels], dtype=int)
    z = np.array([[c in "ZY" for c in label] for label in labels], dtype=int)
    gen_x = np.array([[c in "XY" for c in label] for label in paulis], dtype=int)
    gen_z = np.array([[c in "ZY" for c in label] for label in paulis], dtype=int)
    anticommuting = (x @ gen_z.T + z @ gen_x.T) % 2  # [Q, generator]: do they anticommute?
    n_q = anticommuting.sum(axis=1)
    reachable = float(np.mean(n_q > 0))
    if not 0 <= error_probability < reachable:
        raise ValueError(
            f"Generators {list(paulis)} can produce a total error probability in [0, {reachable}),"
            f" not {error_probability!r}."
        )
    if error_probability == 0:
        return 0.0

    def no_error_probability(rate: float) -> float:
        return float(np.mean(np.exp(-2 * rate * n_q)))

    target = 1 - error_probability
    low, high = 0.0, 1.0
    while no_error_probability(high) > target:
        high *= 2
    while (mid := (low + high) / 2) not in (low, high):
        if no_error_probability(mid) > target:
            low = mid
        else:
            high = mid
    return high


@dataclass
class NoiseModel:
    """Noise model used to find optimal circuit locations for spacetime Pauli checks."""

    gate_noise: GateNoise | None = None
    """Errors that occur during 2-qubit gate operations. Can be:

       - :data:`.UniformGateNoise`: The same depolarizing channel after every gate, with every
         non-identity Pauli equally likely.
       - :data:`.LayeredGateNoise`: Pauli-Lindblad noise channel per unique entangling layer
       - :data:`.GateWiseNoise`: Pauli-Lindblad noise channel per unique entangling edge
    """

    readout_noise: float | None = None
    """Probability of bit-flip during measurement."""

    idling_noise: float | None = None
    """Qubit decay rate during idle time. Total error probability is given as ``1 - exp(-t / idling_noise)``.

    Not currently supported: both :func:`~qiskit_paulice.checks.add_pauli_checks` and
    :meth:`~qiskit_paulice.checked_circuit.CheckedCircuit.estimate_fault_rates` raise if this is set."""

    @classmethod
    def from_backend(
        cls,
        backend: BackendV2,
        layout: Sequence[int],
        uniform_gate_noise: bool = False,
        pauli_bases: Sequence[str] | None = None,
    ) -> NoiseModel:
        """Instantiate a :class:`.NoiseModel` from backend calibration data.

        Edge keys in the resulting :data:`.GateWiseNoise` dict use **virtual** qubit indices
        (positions in ``layout``). If the input circuit has a layout that maps virtual to physical
        consistently with ``layout``, the keys also serve as physical indices.

        ``idling_noise`` is always ``None`` from this method; if ``backend`` lacks readout
        calibration, ``readout_noise`` will also be ``None``.

        Args:
            backend: A backend containing two-qubit gate error probabilities for each coupling map edge
                as well as readout error information for each qubit in the layout.
            layout: Physical qubit indices on the backend to include in the noise model. The order
                defines the virtual qubit indexing (virtual qubit 0 maps to ``layout[0]``, etc.).
            uniform_gate_noise: If ``True``, the ``gate_noise`` field in the output :class:`.NoiseModel`
                will be a :data:`.UniformGateNoise`: the mean reported error over all edges, applied as
                the same depolarizing channel after every entangling gate.
                If ``False``, ``gate_noise`` will be a :data:`.GateWiseNoise` instance where each
                edge carries its own channel built from its calibration data. Either way the
                reported error is an average gate infidelity, and the channel on an edge applies
                a non-identity Pauli with total probability ``5 / 4`` of it.
            pauli_bases: For :data:`.GateWiseNoise` models, the Pauli generators of each edge's
                channel. Each basis is a 2-character Pauli string paired left-to-right with the
                edge tuple, so ``"XZ"`` on edge ``(a, b)`` places ``X`` on ``a`` and ``Z`` on
                ``b``. All generators of an edge share one rate, chosen so that the channel they
                produce has exactly the edge's total error probability. The default is the full
                2Q Pauli basis excluding ``"II"``, which makes every edge's channel the
                depolarizing channel of :data:`.UniformGateNoise`. For
                :data:`.UniformGateNoise`, the full basis is assumed, and this argument is ignored.

        Returns:
            A :class:`NoiseModel` instance containing gate and readout noise derived from backend
            calibration data.

        Raises:
            ValueError: Backend is missing calibration data, layout contains invalid qubit indices,
                a Pauli basis is malformed, or ``pauli_bases`` cannot produce an edge's error.
        """
        # Make sure the inputs are valid
        if not all(0 <= q < backend.num_qubits for q in layout):
            raise ValueError("Invalid qubits in layout")
        two_q_insts = [o.name for o in backend.target.operations if o.num_qubits == 2]

        # Use the full Pauli basis if unspecified. This argument is ignored if uniform_gate_noise,
        # since the full basis is always assumed in that case.
        if pauli_bases is None:
            pauli_bases = [p1 + p2 for p1 in "IXYZ" for p2 in "IXYZ" if (p1, p2) != ("I", "I")]
        if not uniform_gate_noise:
            for basis in pauli_bases:
                if not isinstance(basis, str) or len(basis) != 2 or set(basis) - set("IXYZ"):
                    raise ValueError(
                        "Each Pauli basis must be a 2-character string over I, X, Y, and Z. "
                        f"Got: {basis!r}"
                    )

        # Mapping from physical to virtual qubit indices.
        phys_to_virt = {phys: virt for virt, phys in enumerate(layout)}

        # Collect error data once per physical edge (coupling maps often list both
        # orientations). For each edge we populate both directed keys in the output dict:
        # the rust GateWiseNoiseModel looks up `(qbits[0], qbits[1])` straight off each gate
        # (see src/noise_model.rs:97), and we want the noise to land on the same physical
        # qubits regardless of which way the gate happens to be ordered.
        gate_noise_per_gate: dict[tuple[int, int], list[tuple[str, float]]] = {}
        gate_noise_per_edge: list[float] = []
        qubit_set = set(layout)
        seen_phys: set[tuple[int, int]] = set()
        for edge in backend.coupling_map:
            if edge[0] not in qubit_set or edge[1] not in qubit_set:
                continue
            canonical_phys = (min(edge), max(edge))
            if canonical_phys in seen_phys:
                continue
            seen_phys.add(canonical_phys)

            # Collect non-None errors for this edge across all 2Q instructions
            edge_errors = [
                backend.target[i][edge].error
                for i in two_q_insts
                if backend.target[i][edge].error is not None
            ]

            # Skip this edge if no valid error data
            if not edge_errors:
                continue

            mean_edge_error = float(np.mean(edge_errors))
            if uniform_gate_noise:
                gate_noise_per_edge.append(mean_edge_error)
            else:
                # The user's ``pauli_bases`` is placed on the canonical virtual direction
                # (lower virtual index first); the reverse direction stores the swapped
                # Pauli string so the noise lands on the same physical qubits regardless
                # of how the gate's `qbits` happen to be ordered at evaluation time.
                v0 = phys_to_virt[edge[0]]
                v1 = phys_to_virt[edge[1]]
                a, b = (v0, v1) if v0 < v1 else (v1, v0)
                error_probability = _INFIDELITY_TO_ERROR_PROBABILITY * mean_edge_error
                rate_per_basis = _equal_generator_rate(pauli_bases, error_probability)
                generators = [(g, rate_per_basis) for g in pauli_bases]
                mirrored = [(g[::-1], rate_per_basis) for g in pauli_bases]
                gate_noise_per_gate[(a, b)] = generators
                gate_noise_per_gate[(b, a)] = mirrored

        # Mean readout error probability across qubits in layout.
        valid_readout = [
            e for i in layout if (e := backend.target["measure"][(i,)].error) is not None
        ]
        readout_noise = sum(valid_readout) / len(valid_readout) if valid_readout else None

        # Prepare gate noise output, based on user input
        gate_noise: GateNoise | None
        if uniform_gate_noise:
            gate_noise = float(np.mean(gate_noise_per_edge)) if gate_noise_per_edge else None
        else:
            gate_noise = gate_noise_per_gate if gate_noise_per_gate else None

        return cls(gate_noise=gate_noise, readout_noise=readout_noise, idling_noise=None)

    @classmethod
    def from_pauli_lindblad_maps(
        cls,
        layer_noise: Sequence[PauliLindbladMap],
        readout_noise: PauliLindbladMap | None = None,
    ) -> NoiseModel:
        """Create a NoiseModel from Pauli-Lindblad maps.

        This method constructs a :class:`.NoiseModel` from ``PauliLindbladMap`` instances that
        represent noise channels affecting gates and, optionally, measurements. Gate noise
        is specified with the ``layer_noise`` argument. Each ``PauliLindbladMap`` instance is
        assumed to act on a unique layer of entangling gates. Readout noise is specified by a
        single ``PauliLindbladMap`` containing single-qubit Pauli-X generators and their associated
        rates.

        Args:
            layer_noise: A sequence of :class:`~qiskit.quantum_info.PauliLindbladMap` objects, where each map represents
                the noise channel for one unique entangling layer in the circuit.
            readout_noise: Optional :class:`~qiskit.quantum_info.PauliLindbladMap` containing Pauli X generators on each
                qubit for readout errors.

        Returns:
            A :class:`NoiseModel` instance reflecting the noise model(s) defined in the input
            ``PauliLindbladMap``s. ``idling_noise`` will always be ``None`` for this method.

        Raises:
            ValueError: Weight of error generator greater than ``2``.
            ValueError: Readout error generator is not Pauli-X.
        """
        # Build layered gate noise dictionary
        gate_noise_dict: dict[tuple[tuple[int, int], ...], list[tuple[str, float]]] = {}

        for plm in layer_noise:
            # Collect all generators for this layer and determine the layer structure
            layer_edges = set()
            generators = []
            num_qubits = plm.num_qubits

            for generator in plm:
                pauli_str = generator.pauli_labels()
                rate = generator.rate
                indices = generator.indices

                # Validate generator size
                if len(indices) > 2:
                    raise ValueError(
                        f"Generators with more than 2 qubits are not supported. "
                        f"Got generator with {len(indices)} qubits: {pauli_str}"
                    )

                full_pauli = ["I"] * num_qubits
                for pauli_char, qubit_idx in zip(pauli_str, indices, strict=True):
                    full_pauli[num_qubits - 1 - qubit_idx] = pauli_char

                full_pauli_str = "".join(full_pauli)

                # Add this as an edge (2-qubit generators only)
                if len(indices) == 2:
                    layer_edges.add(tuple(sorted(indices)))

                generators.append((full_pauli_str, rate))

            # Validate that the layer has at least one 2-qubit generator
            if not layer_edges:
                raise ValueError(
                    "Each PauliLindbladMap in layer_noise must contain at least one "
                    "2-qubit generator to define a valid entangling layer."
                )

            # Use the edges as the layer key
            layer_key = tuple(sorted(layer_edges))
            gate_noise_dict[layer_key] = generators

        # Extract readout noise if provided
        readout_prob = None
        if readout_noise is not None:
            # Collect all X generator rates and convert to probabilities
            x_probs = []
            for generator in readout_noise:
                pauli_str = generator.pauli_labels()
                if pauli_str != "X":
                    raise ValueError(
                        "Readout noise should be defined by a Pauli X generator and rate for each qubit."
                    )
                # Convert rate to probability: p = (1 - exp(-2*rate)) / 2
                rate = generator.rate
                prob = (1.0 - np.exp(-2.0 * rate)) / 2.0
                x_probs.append(prob)

            # Average the probabilities
            readout_prob = sum(x_probs) / len(x_probs) if x_probs else None

        return cls(gate_noise=gate_noise_dict, readout_noise=readout_prob, idling_noise=None)
