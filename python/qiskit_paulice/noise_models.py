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
"""A depolarizing probability applied uniformly to all 2-qubit gates.

This probability will be equally distributed among the ``15`` non-identity
Paulis in the 2-qubit Pauli basis to form the depolarizing channel. This noise channel
will be applied **after** each entangling gate.
"""

LayeredGateNoise = dict[tuple[tuple[int, int], ...], list[tuple[Pauli | str, float]]]
"""Layered noise model mapping unique entangling layers to Pauli error generators.

.. note::
    The Pauli-Lindblad error associated with each entangling layer is assumed to be defined
    **before** the gates in the layer.

Keys are tuples of qubit index pairs (edges) defining a layer. Values are lists of
``(pauli, rate)`` tuples where ``pauli`` is a Pauli or Pauli string defined over all qubits and
``rate`` is the associated error rate.
"""

GateWiseNoise = dict[tuple[int, int], list[tuple[Pauli | str, float]]]
"""Gate-wise noise model mapping qubit pairs to a Pauli noise channel.

Keys are tuples of qubit index pairs representing edges. Values are lists of (pauli, rate)
tuples where ``pauli`` is a 2-qubit :class:`qiskit.quantum_info.Pauli` or Pauli string and
``rate`` is the associated error rate. A Pauli generator, "XY", associated with an edge,
``(a, b)``, is interprted such that the ``X`` error is on ``a`` and the ``Y`` error is
on ``b``.
"""

GateNoise = UniformGateNoise | LayeredGateNoise | GateWiseNoise
"""Gate noise can be uniform (float), layered (dict), or gate-wise (dict)."""


@dataclass
class NoiseModel:
    """Noise model used to find optimal circuit locations for spacetime Pauli checks."""

    gate_noise: GateNoise | None = None
    """Errors that occur during 2-qubit gate operations. Can be:

       - :class:`.UniformNoise`: Same error probability for all gates. The error probability is
         equally distributed to each Pauli basis represented in the channel.
       - :class:`.LayeredNoise`: Pauli-Lindblad noise channel per unique entangling layer
       - :class:`.GateWiseNoise`: Pauli-Lindblad noise channel per unique entangling edge
    """

    readout_noise: float | None = None
    """Probability of bit-flip during measurement."""

    idling_noise: float | None = None
    """Qubit decay rate during idle time. Total error probability is given as ``1 - exp(-t / idling_noise)``."""

    @classmethod
    def from_backend(
        cls,
        backend: BackendV2,
        layout: Sequence[int],
        uniform_gate_noise: bool = False,
        pauli_bases: Sequence[Pauli | str] | None = None,
    ) -> NoiseModel:
        """Instantiate a :class:`.NoiseModel` from backend calibration data.

        Edge keys in the resulting :class:`.GateWiseNoise` dict use **virtual** qubit indices
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
                will be a :class:`.UniformGateNoise` and the error probability is assumed to be
                distributed uniformly among the ``15`` non-identity Pauli bases to form a uniform
                depolarizing channel which will affect all entangling gates equally.
                If ``False``, ``gate_noise`` will be a :class:`.GateWiseNoise` instance where each
                edge is associated with a custom noise channel based on backend calibration data.
            pauli_bases: For :class:`.GateWiseNoise` models, ``pauli_bases`` are the bases over which
                the error probability reported from the backend will be distributed. The default
                behavior is to use the full 2Q Pauli basis excluding ``II``. For :class:`UniformNoise`,
                the full basis is assumed, and this argument is ignored.

        Returns:
            A :class:`NoiseModel` instance containing gate and readout noise derived from backend
            calibration data.

        Raises:
            ValueError: Backend is missing calibration data or layout contains invalid qubit indices.
        """
        # Make sure the inputs are valid
        if not all(0 <= q < backend.num_qubits for q in layout):
            raise ValueError("Invalid qubits in layout")
        two_q_insts = [o.name for o in backend.target.operations if o.num_qubits == 2]

        # Use the full Pauli basis if unspecified. This argument is ignored if uniform_gate_noise,
        # since the full basis is always assumed in that case.
        if pauli_bases is None:
            pauli_bases = [p1 + p2 for p1 in "IXYZ" for p2 in "IXYZ" if (p1, p2) != ("I", "I")]
        pauli_bases = [p.to_label() if isinstance(p, Pauli) else p for p in pauli_bases]

        # Validate that all Pauli bases are 2-qubit
        if not uniform_gate_noise:
            for basis in pauli_bases:
                if len(basis) != 2:
                    raise ValueError(
                        f"All Pauli bases must be 2-qubit for gate-wise noise models. Got: '{basis}'"
                    )

        # Mapping from physical to virtual qubit indices.
        phys_to_virt = {phys: virt for virt, phys in enumerate(layout)}

        # Collect error data for edges. The rust GateWiseNoiseModel looks up a directed key
        # `(qbits[0], qbits[1])` straight off each gate (see src/noise_model.rs:97), so we
        # populate both orientations regardless of whether the coupling map provides both.
        gate_noise_per_gate: dict[tuple[int, int], list[tuple[str, float]]] = {}
        gate_noise_per_edge: list[float] = []
        qubit_set = set(layout)
        for edge in backend.coupling_map:
            if edge[0] not in qubit_set or edge[1] not in qubit_set:
                continue

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
                a, b = phys_to_virt[edge[0]], phys_to_virt[edge[1]]
                prob_per_basis = mean_edge_error / len(pauli_bases)
                rate_per_basis = -0.5 * float(np.log(1.0 - 2.0 * prob_per_basis))
                generators = [(g, rate_per_basis) for g in pauli_bases]
                gate_noise_per_gate[(a, b)] = generators
                gate_noise_per_gate[(b, a)] = generators

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
            layer_noise: A sequence of :class:`PauliLindbladMap` objects, where each map represents
                the noise channel for one unique entangling layer in the circuit.
            readout_noise: Optional :class:`PauliLindbladMap` containing Pauli X generators on each
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
