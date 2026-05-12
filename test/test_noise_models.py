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

"""Test noise_models module."""

from __future__ import annotations

import unittest

import numpy as np
from qiskit.quantum_info import Pauli, PauliLindbladMap
from qiskit_ibm_runtime.fake_provider import FakeFez
from qiskit_paulice.noise_models import NoiseModel


class TestNoiseModels(unittest.TestCase):
    """Tests covering the ``noise_models`` module."""

    def test_noise_model_creation(self):
        """Test NoiseModel creation with various noise types."""
        # Test individual noise types
        noise_gate = NoiseModel(gate_noise=0.001)
        self.assertEqual(noise_gate.gate_noise, 0.001)

        noise_readout = NoiseModel(readout_noise=0.01)
        self.assertEqual(noise_readout.readout_noise, 0.01)

        noise_idling = NoiseModel(idling_noise=1e5)
        self.assertEqual(noise_idling.idling_noise, 1e5)

        # Test combined noise
        noise_combined = NoiseModel(gate_noise=0.001, readout_noise=0.01, idling_noise=1e5)
        self.assertEqual(noise_combined.gate_noise, 0.001)
        self.assertEqual(noise_combined.readout_noise, 0.01)
        self.assertEqual(noise_combined.idling_noise, 1e5)

        # Test layered gate noise
        layered_noise = {((0, 1), (2, 3)): [("IXYZ", 0.001), ("ZZII", 0.0005)]}
        noise_layered = NoiseModel(gate_noise=layered_noise)
        self.assertEqual(noise_layered.gate_noise, layered_noise)

    def test_dataclass_features(self):
        """Test that NoiseModel has dataclass features."""
        noise1 = NoiseModel(gate_noise=0.001, readout_noise=0.01)
        noise2 = NoiseModel(gate_noise=0.001, readout_noise=0.01)
        noise3 = NoiseModel(gate_noise=0.002, readout_noise=0.01)

        # Test equality
        self.assertEqual(noise1, noise2)
        self.assertNotEqual(noise1, noise3)

        # Test repr
        repr_str = repr(noise1)
        self.assertIn("NoiseModel", repr_str)
        self.assertIn("gate_noise=0.001", repr_str)
        self.assertIn("readout_noise=0.01", repr_str)

    def test_from_backend_gatewise_manual_calculation(self):
        """Test from_backend gate-wise noise with manual calculation of expected values."""
        backend = FakeFez()
        layout = [0, 1, 2]

        # Get all edges and their error probabilities
        two_q_insts = [op.name for op in backend.target.operations if op.num_qubits == 2]
        edge_errors = {}
        for edge in backend.coupling_map:
            if edge[0] in layout and edge[1] in layout:
                errors = [
                    backend.target[inst][edge].error
                    for inst in two_q_insts
                    if backend.target[inst][edge].error is not None
                ]
                if errors:
                    edge_errors[edge] = np.mean(errors)

        # Create noise model
        noise = NoiseModel.from_backend(backend, layout=layout, uniform_gate_noise=False)

        # Verify each edge has correct rates
        for edge, backend_prob in edge_errors.items():
            self.assertIn(edge, noise.gate_noise)
            generators = noise.gate_noise[edge]

            # Should have 15 generators (full 2Q Pauli basis minus II)
            self.assertEqual(len(generators), 15)

            # Manual calculation: probability per basis and conversion to rate
            prob_per_basis = backend_prob / 15
            expected_rate = -0.5 * np.log(1.0 - 2.0 * prob_per_basis)

            # All generators should have the same rate
            for _, rate in generators:
                self.assertAlmostEqual(rate, expected_rate, places=10)

    def test_from_backend_uniform_manual_calculation(self):
        """Test from_backend uniform noise with manual calculation of expected value."""
        backend = FakeFez()
        layout = [0, 1, 2]

        # Manually calculate expected uniform noise
        two_q_insts = [op.name for op in backend.target.operations if op.num_qubits == 2]
        edge_probs = []
        for edge in backend.coupling_map:
            if edge[0] in layout and edge[1] in layout:
                errors = [
                    backend.target[inst][edge].error
                    for inst in two_q_insts
                    if backend.target[inst][edge].error is not None
                ]
                if errors:
                    edge_probs.append(np.mean(errors))

        expected_uniform = np.mean(edge_probs) if edge_probs else None

        # Create noise model
        noise = NoiseModel.from_backend(backend, layout=layout, uniform_gate_noise=True)

        # Verify the uniform noise matches our manual calculation
        if expected_uniform is not None:
            self.assertAlmostEqual(noise.gate_noise, expected_uniform, places=10)
        else:
            self.assertIsNone(noise.gate_noise)

    def test_from_backend_readout_manual_calculation(self):
        """Test from_backend readout noise with manual calculation of expected value."""
        backend = FakeFez()
        layout = [0, 1, 2]

        # Manually calculate expected readout noise
        readout_errors = []
        for qubit in layout:
            error = backend.target["measure"][(qubit,)].error
            if error is not None:
                readout_errors.append(error)

        expected_readout = np.mean(readout_errors) if readout_errors else None

        # Create noise model
        noise = NoiseModel.from_backend(backend, layout=layout)

        # Verify the readout noise matches our manual calculation
        if expected_readout is not None:
            self.assertAlmostEqual(noise.readout_noise, expected_readout, places=10)
        else:
            self.assertIsNone(noise.readout_noise)

    def test_from_backend_idling_noise_is_none(self):
        """from_backend always returns idling_noise=None (it does not consume T1)."""
        backend = FakeFez()
        noise = NoiseModel.from_backend(backend, layout=[0, 1, 2])
        self.assertIsNone(noise.idling_noise)

    def test_from_backend_custom_pauli_bases(self):
        """Test from_backend with custom Pauli bases and verify correct scaling."""
        backend = FakeFez()
        layout = [0, 1, 2]

        # Get all edges and their error probabilities
        two_q_insts = [op.name for op in backend.target.operations if op.num_qubits == 2]
        edge_errors = {}
        for edge in backend.coupling_map:
            if edge[0] in layout and edge[1] in layout:
                errors = [
                    backend.target[inst][edge].error
                    for inst in two_q_insts
                    if backend.target[inst][edge].error is not None
                ]
                if errors:
                    edge_errors[edge] = np.mean(errors)

        # Test with custom Pauli bases (only 3 bases)
        custom_bases = ["XX", "YY", "ZZ"]
        noise = NoiseModel.from_backend(
            backend, layout=layout, uniform_gate_noise=False, pauli_bases=custom_bases
        )

        # Verify each edge has correct rates with proper scaling
        for edge, backend_prob in edge_errors.items():
            self.assertIn(edge, noise.gate_noise)
            generators = noise.gate_noise[edge]

            # Check that each edge has only 3 generators
            self.assertEqual(len(generators), 3)

            # Check that only custom bases are used
            for pauli, rate in generators:
                self.assertIn(pauli, custom_bases)

                # Manual calculation: probability should be divided by 3 (not 15)
                # since we only have 3 custom bases
                prob_per_basis = backend_prob / 3
                expected_rate = -0.5 * np.log(1.0 - 2.0 * prob_per_basis)

                # Verify the rate matches the expected value
                self.assertAlmostEqual(rate, expected_rate, places=10)

    def test_from_backend_pauli_objects(self):
        """Test from_backend with Pauli objects as bases."""
        backend = FakeFez()
        layout = [0, 1, 2]

        # Test with Pauli objects
        custom_bases = [Pauli("XX"), Pauli("YY"), Pauli("ZZ")]
        noise = NoiseModel.from_backend(
            backend, layout=layout, uniform_gate_noise=False, pauli_bases=custom_bases
        )

        # Check that Pauli objects were converted to strings
        for _edge, generators in noise.gate_noise.items():
            for pauli, _rate in generators:
                self.assertIsInstance(pauli, str)
                self.assertIn(pauli, ["XX", "YY", "ZZ"])

    def test_from_backend_invalid_qubits(self):
        """Test from_backend with invalid qubit indices."""
        backend = FakeFez()

        # Test with qubit index out of range (too large)
        with self.assertRaises(ValueError) as cm:
            NoiseModel.from_backend(backend, layout=[0, 1, 999])
        self.assertIn("Invalid qubits", str(cm.exception))

        # Test with negative qubit index
        with self.assertRaises(ValueError) as cm:
            NoiseModel.from_backend(backend, layout=[0, 1, -1])
        self.assertIn("Invalid qubits", str(cm.exception))

    def test_from_backend_invalid_pauli_bases(self):
        """Test that non-2-qubit Pauli bases raise an error for gate-wise noise."""
        backend = FakeFez()
        layout = [0, 1, 2]

        # Test with 1-qubit Pauli basis (should fail)
        with self.assertRaises(ValueError) as cm:
            NoiseModel.from_backend(
                backend, layout=layout, uniform_gate_noise=False, pauli_bases=["X", "Y", "Z"]
            )
        self.assertIn("2-qubit", str(cm.exception))

    def test_from_backend_missing_data(self):
        """Test from_backend with missing calibration data."""
        backend = FakeFez()

        # Test all gate errors missing
        for op in backend.target.operations:
            if op.num_qubits == 2:
                for edge in backend.coupling_map:
                    backend.target[op.name][edge].error = None

        noise = NoiseModel.from_backend(backend, layout=[0, 1, 2], uniform_gate_noise=False)
        self.assertIsNone(noise.gate_noise)

        # Test all readout errors missing
        backend = FakeFez()
        for i in range(backend.num_qubits):
            backend.target["measure"][(i,)].error = None

        noise = NoiseModel.from_backend(backend, layout=[0, 1, 2])
        self.assertIsNone(noise.readout_noise)

    def test_from_backend_partial_data(self):
        """Test from_backend with partial calibration data and manual calculation."""
        backend = FakeFez()
        layout = [0, 1, 2]

        # Get the original readout errors before modifying
        error_0 = backend.target["measure"][(0,)].error
        error_2 = backend.target["measure"][(2,)].error

        # Set one readout error to None, keep others
        backend.target["measure"][(1,)].error = None

        # Manual calculation: should average only the non-None values
        expected_readout = (error_0 + error_2) / 2

        noise = NoiseModel.from_backend(backend, layout=layout)

        # Verify the readout noise matches our manual calculation
        self.assertIsNotNone(noise.readout_noise)
        self.assertAlmostEqual(noise.readout_noise, expected_readout, places=10)

    def test_from_backend_uses_virtual_indices(self):
        """from_backend keys gate_noise edges by virtual qubit indices (positions in layout)."""
        backend = FakeFez()
        # Pick a non-identity layout so virtual != physical.
        layout = [1, 2, 3]
        noise = NoiseModel.from_backend(backend, layout=layout, uniform_gate_noise=False)

        # All edge keys should be in the virtual range [0, len(layout)) and never use
        # the physical indices directly.
        n = len(layout)
        for edge in noise.gate_noise:
            self.assertIn(edge[0], range(n))
            self.assertIn(edge[1], range(n))

        # Both directions of any included physical link must appear (rust-side does no
        # canonicalization; see src/noise_model.rs:97).
        phys_to_virt = {phys: virt for virt, phys in enumerate(layout)}
        qubit_set = set(layout)
        for edge in backend.coupling_map:
            if edge[0] in qubit_set and edge[1] in qubit_set:
                a, b = phys_to_virt[edge[0]], phys_to_virt[edge[1]]
                if (a, b) in noise.gate_noise:
                    self.assertIn((b, a), noise.gate_noise)

    def test_from_pauli_lindblad_maps_readout_manual_calculation(self):
        """Test readout noise conversion and averaging with explicit manual calculation."""
        # Test with multiple generators with different rates to verify averaging
        rate1, rate2, rate3 = 0.001, 0.005, 0.01

        # Create readout noise with 3 X generators on different qubits
        readout_plm = PauliLindbladMap(
            [("XII", rate1), ("IXI", rate2), ("IIX", rate3)], num_qubits=3
        )
        noise = NoiseModel.from_pauli_lindblad_maps([], readout_noise=readout_plm)

        # Manual calculation: p = (1 - exp(-2*rate)) / 2 for each generator
        prob1 = (1.0 - np.exp(-2.0 * rate1)) / 2.0
        prob2 = (1.0 - np.exp(-2.0 * rate2)) / 2.0
        prob3 = (1.0 - np.exp(-2.0 * rate3)) / 2.0

        # The final readout noise should be the average of all three probabilities
        expected_readout = (prob1 + prob2 + prob3) / 3

        self.assertAlmostEqual(noise.readout_noise, expected_readout, places=10)

        # Also test single generator case to verify the conversion formula
        single_rate = 0.02
        single_plm = PauliLindbladMap([("X", single_rate)], num_qubits=1)
        single_noise = NoiseModel.from_pauli_lindblad_maps([], readout_noise=single_plm)

        expected_single_prob = (1.0 - np.exp(-2.0 * single_rate)) / 2.0
        self.assertAlmostEqual(single_noise.readout_noise, expected_single_prob, places=10)

        # Verify the conversion is correct by checking the inverse
        # rate = -0.5 * ln(1 - 2*p) should give us back the original rate
        recovered_rate = -0.5 * np.log(1.0 - 2.0 * expected_single_prob)
        self.assertAlmostEqual(recovered_rate, single_rate, places=10)

    def test_from_pauli_lindblad_maps_layer_structure_manual(self):
        """Test layer structure extraction with manual verification."""
        # Create a 4-qubit system with specific generators
        plm = PauliLindbladMap(
            [
                ("XYII", 0.01),
                ("YYII", 0.02),
                ("IIZZ", 0.03),
            ],
            num_qubits=4,
        )

        noise = NoiseModel.from_pauli_lindblad_maps([plm])

        # Manual verification: should have one layer with two edges
        self.assertEqual(len(noise.gate_noise), 1)

        # The layer key should contain both edges (0,1) and (2,3)
        layer_key = next(iter(noise.gate_noise.keys()))
        self.assertEqual(len(layer_key), 2)
        self.assertIn((0, 1), layer_key)
        self.assertIn((2, 3), layer_key)

        # Verify all generators are present with correct rates
        generators = noise.gate_noise[layer_key]
        self.assertEqual(len(generators), 3)

        # Check each generator
        gen_dict = {pauli: rate for pauli, rate in generators}
        self.assertAlmostEqual(gen_dict["XYII"], 0.01, places=10)
        self.assertAlmostEqual(gen_dict["YYII"], 0.02, places=10)
        self.assertAlmostEqual(gen_dict["IIZZ"], 0.03, places=10)

    def test_from_pauli_lindblad_maps_pauli_ordering_manual(self):
        """Test Pauli string ordering with manual verification of qubit indexing."""
        # Create a 3-qubit system with generator on qubits [0, 2]
        # In Qiskit convention, rightmost is qubit 0
        # So "ZIX" means: Z on qubit 2, I on qubit 1, X on qubit 0
        plm = PauliLindbladMap([("ZIX", 0.01)], num_qubits=3)

        noise = NoiseModel.from_pauli_lindblad_maps([plm])

        # Get the generator
        layer_key = next(iter(noise.gate_noise.keys()))
        generators = noise.gate_noise[layer_key]
        pauli_str, rate = generators[0]

        # Manual verification: the Pauli string should be preserved exactly
        self.assertEqual(pauli_str, "ZIX")
        self.assertAlmostEqual(rate, 0.01, places=10)

        # Verify the edge is correctly identified as (0, 2)
        # (sorted because edges are stored sorted)
        self.assertIn((0, 2), layer_key)

    def test_from_pauli_lindblad_maps_multiple_layers_manual(self):
        """Test multiple layers with manual verification of structure."""
        # Create three distinct layers
        # Note: Qiskit uses right-to-left ordering (rightmost is qubit 0)
        plm1 = PauliLindbladMap([("IIIXY", 0.01)], num_qubits=5)  # qubits [0, 1]
        plm2 = PauliLindbladMap([("IIXYI", 0.02)], num_qubits=5)  # qubits [1, 2]
        plm3 = PauliLindbladMap([("XYIII", 0.03)], num_qubits=5)  # qubits [3, 4]

        noise = NoiseModel.from_pauli_lindblad_maps([plm1, plm2, plm3])

        # Manual verification: should have three layers
        self.assertEqual(len(noise.gate_noise), 3)

        # Verify each layer has the correct edge and generator
        for layer_key, generators in noise.gate_noise.items():
            self.assertEqual(len(generators), 1)
            pauli_str, rate = generators[0]

            if (0, 1) in layer_key:
                self.assertEqual(pauli_str, "IIIXY")
                self.assertAlmostEqual(rate, 0.01, places=10)
            elif (1, 2) in layer_key:
                self.assertEqual(pauli_str, "IIXYI")
                self.assertAlmostEqual(rate, 0.02, places=10)
            elif (3, 4) in layer_key:
                self.assertEqual(pauli_str, "XYIII")
                self.assertAlmostEqual(rate, 0.03, places=10)
            else:
                self.fail(f"Unexpected layer key: {layer_key}")

    def test_from_pauli_lindblad_maps_readout_only_x(self):
        """Test that readout noise validates X generators."""
        # Create readout noise with non-X generator (should fail)
        readout_plm = PauliLindbladMap([("YI", 0.005)], num_qubits=2)

        with self.assertRaises(ValueError) as cm:
            NoiseModel.from_pauli_lindblad_maps([], readout_noise=readout_plm)
        self.assertIn("Pauli X generator", str(cm.exception))

    def test_from_pauli_lindblad_maps_readout_empty(self):
        """Test that readout noise with no generators raises ValueError."""
        # Create empty readout noise (no generators)
        readout_plm = PauliLindbladMap([], num_qubits=2)

        noise = NoiseModel.from_pauli_lindblad_maps([], readout_noise=readout_plm)
        self.assertEqual(None, noise.readout_noise)

    def test_from_pauli_lindblad_maps_single_qubit_generator_error(self):
        """Test that layers with only single-qubit generators raise ValueError."""
        # Create a layer with only single-qubit generator (no 2-qubit generators)
        plm = PauliLindbladMap([("XII", 0.01)], num_qubits=3)

        # Should raise ValueError since there are no 2-qubit generators to define a layer
        with self.assertRaises(ValueError) as cm:
            NoiseModel.from_pauli_lindblad_maps([plm])
        self.assertIn("at least one 2-qubit generator", str(cm.exception))

    def test_from_pauli_lindblad_maps_three_qubit_generator_error(self):
        """Test that 3+ qubit generators raise ValueError."""
        # Create a layer with 3-qubit generator
        plm = PauliLindbladMap([("XYZII", 0.01)], num_qubits=5)

        with self.assertRaises(ValueError) as cm:
            NoiseModel.from_pauli_lindblad_maps([plm])
        self.assertIn("more than 2 qubits", str(cm.exception))
        self.assertIn("3 qubits", str(cm.exception))

    def test_from_pauli_lindblad_maps_empty_layers(self):
        """Test with empty layer list."""
        noise = NoiseModel.from_pauli_lindblad_maps([])

        # Should return empty gate noise dict
        self.assertEqual(noise.gate_noise, {})
        self.assertIsNone(noise.readout_noise)
        self.assertIsNone(noise.idling_noise)


# Made with Bob
