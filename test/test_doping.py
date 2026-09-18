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

"""Tests for the ``doping`` module."""

from __future__ import annotations

import unittest

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import (
    Clifford,
    Pauli,
    SparsePauliOp,
    Statevector,
    random_clifford,
)
from qiskit_paulice import CheckedCircuit, DopingSite, dope_clifford_circuit
from qiskit_paulice.checks import add_pauli_checks
from qiskit_paulice.noise_models import NoiseModel


def _stabilizer_renyi_2(circuit: QuantumCircuit) -> float:
    """The 2-stabilizer Renyi entropy M2 = -log2( sum_P <P>^4 / 2^n ) of the output state.

    Zero exactly on stabilizer states, positive on magic states.
    """
    state = Statevector(circuit)
    rho = np.outer(state.data, state.data.conj())
    expvals = SparsePauliOp.from_operator(rho).coeffs.real * 2**state.num_qubits
    return float(-np.log2(np.sum(expvals**4) / 2**state.num_qubits))


def _paper_ansatz(num_qubits: int, seed: int) -> QuantumCircuit:
    """The paper's graph-state ansatz: H layer, then brickwork CZ + random S/sqrt(X) layers."""
    rng = np.random.default_rng(seed)
    circuit = QuantumCircuit(num_qubits)
    circuit.h(range(num_qubits))
    for layer in range(num_qubits):
        for a in range(layer % 2, num_qubits - 1, 2):
            circuit.cz(a, a + 1)
        for q in range(num_qubits):
            if rng.random() < 0.5:
                circuit.s(q)
            circuit.sx(q)
    return circuit


def _position(site: DopingSite) -> int:
    """The number of instructions preceding a site's wire boundary."""
    return 0 if site.after_instruction is None else site.after_instruction + 1


def _measure_positions(circuit: QuantumCircuit) -> dict[int, int]:
    """Each measured qubit's index in ``circuit.data`` of its measurement."""
    return {
        circuit.find_bit(inst.qubits[0]).index: index
        for index, inst in enumerate(circuit.data)
        if inst.operation.name == "measure"
    }


def _split(circuit: QuantumCircuit, position: int) -> tuple[QuantumCircuit, QuantumCircuit]:
    """Split a circuit into prefix/suffix around a wire boundary, dropping non-unitaries."""
    prefix = QuantumCircuit(circuit.num_qubits)
    suffix = QuantumCircuit(circuit.num_qubits)
    for index, inst in enumerate(circuit.data):
        if inst.operation.name not in ("measure", "barrier"):
            target = prefix if index < position else suffix
            target.append(inst.operation, [circuit.find_bit(q).index for q in inst.qubits])
    return prefix, suffix


def _propagated(circuit: QuantumCircuit, site: DopingSite) -> tuple[Pauli, Pauli]:
    """A site generator's forward (output) and backward (input) propagation via ``Pauli.evolve``.

    Deliberately a different implementation from the module's single tableau sweep.
    """
    z = Pauli("I" * circuit.num_qubits)
    z.z[site.qubit] = True
    prefix, suffix = _split(circuit, _position(site))
    return z.evolve(Clifford(suffix), frame="s"), z.evolve(Clifford(prefix), frame="h")


def _syndrome_values(checked: CheckedCircuit, circuit: QuantumCircuit) -> list[float]:
    """The expectation value of each check's syndrome operator on the pre-measurement state."""
    state = Statevector(circuit.remove_final_measurements(inplace=False))
    values = []
    for support in checked.check_support:
        label = "".join("Z" if q in support else "I" for q in reversed(range(circuit.num_qubits)))
        values.append(complex(state.expectation_value(Pauli(label))).real)
    return values


def _checked_circuit() -> CheckedCircuit:
    """A small checked Clifford circuit with one spacetime Pauli check."""
    circuit = QuantumCircuit(3)
    for _ in range(2):
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.cx(1, 2)
        circuit.s(0)
        circuit.s(2)
    circuit.measure_all()
    noise = NoiseModel(gate_noise=1e-3, readout_noise=1e-2)
    return add_pauli_checks(circuit, [1], noise, seed=0)[-1]


class TestDopeCliffordCircuit(unittest.TestCase):
    """Tests for :func:`dope_clifford_circuit`."""

    def _assert_irreducible(self, circuit: QuantumCircuit, sites: list[DopingSite]):
        """Independently verify that no pruning rewrite applies to the returned rotations."""
        forward = []
        backward = []
        for site in sites:
            gens = _propagated(circuit, site)
            forward.append(gens[0])
            backward.append(gens[1])
        for i, gen in enumerate(forward):
            if not backward[i].x.any():
                self.assertFalse(
                    all(gen.commutes(other) for other in forward[:i]),
                    msg="trivial rotation commutes into the input",
                )
            if not gen.x.any():
                self.assertFalse(
                    all(gen.commutes(other) for other in forward[i + 1 :]),
                    msg="diagonal rotation commutes into the measurement",
                )
            for j in range(i + 1, len(forward)):
                if np.array_equal(gen.x, forward[j].x) and np.array_equal(gen.z, forward[j].z):
                    self.assertTrue(
                        any(not gen.commutes(k) for k in forward[i + 1 : j]),
                        msg="equivalent rotations merge into a Clifford",
                    )

    def test_changes_distribution_and_injects_magic(self):
        """Doping alters the sampled distribution and injects magic."""
        circuit = _paper_ansatz(5, seed=1)
        doped, sites = dope_clifford_circuit(circuit)
        self.assertGreater(len(sites), 0)
        self.assertFalse(
            np.allclose(
                Statevector(doped).probabilities(), Statevector(circuit).probabilities(), atol=1e-6
            )
        )
        self.assertAlmostEqual(_stabilizer_renyi_2(circuit), 0.0, places=10)
        self.assertGreater(_stabilizer_renyi_2(doped), 0.5)

    def test_trivial_circuits_have_no_sites(self):
        """Circuits whose rotations all prune away (bare H, diagonal circuit) yield no site."""
        h_only = QuantumCircuit(1)
        h_only.h(0)
        diagonal = QuantumCircuit(2)
        diagonal.s(0)
        diagonal.cz(0, 1)
        for circuit in (h_only, diagonal):
            doped, sites = dope_clifford_circuit(circuit)
            self.assertEqual(sites, [])
            self.assertEqual(doped.count_ops().get("rz", 0), 0)
        with self.assertRaises(ValueError):
            dope_clifford_circuit(diagonal, num_sites=1)

    def test_sites_are_irreducible(self):
        """No pruning rewrite of the reference applies to the returned site set."""
        for seed in range(3):
            circuit = random_clifford(4, seed=seed).to_circuit()
            _, sites = dope_clifford_circuit(circuit)
            self.assertGreater(len(sites), 0, msg=f"seed {seed}")
            self._assert_irreducible(circuit, sites)
        circuit = _paper_ansatz(4, seed=2)
        _, sites = dope_clifford_circuit(circuit)
        self._assert_irreducible(circuit, sites)

    def test_num_sites_and_seed(self):
        """Random subsets are exact in size, seed-reproducible, and themselves irreducible."""
        circuit = _paper_ansatz(4, seed=0)
        _, all_sites = dope_clifford_circuit(circuit)
        self.assertGreater(len(all_sites), 3)
        doped, sites = dope_clifford_circuit(circuit, num_sites=3, seed=42)
        self.assertEqual(len(sites), 3)
        self.assertEqual(doped.count_ops()["rz"], 3)
        self.assertTrue(set(sites) <= set(all_sites))
        self._assert_irreducible(circuit, sites)
        _, again = dope_clifford_circuit(circuit, num_sites=3, seed=42)
        self.assertEqual(sites, again)
        with self.assertRaises(ValueError):
            dope_clifford_circuit(circuit, num_sites=len(all_sites) + 1)
        with self.assertRaises(ValueError):
            dope_clifford_circuit(circuit, num_sites=-1)

    def test_only_rz_gates_inserted(self):
        """The doped circuit is the original instruction sequence with only rz gates added."""
        circuit = _paper_ansatz(4, seed=3)
        doped, sites = dope_clifford_circuit(circuit, num_sites=3, seed=0)
        stripped = [
            (inst.name, tuple(doped.find_bit(q).index for q in inst.qubits))
            for inst in doped
            if inst.name != "rz"
        ]
        original = [
            (inst.name, tuple(circuit.find_bit(q).index for q in inst.qubits)) for inst in circuit
        ]
        self.assertEqual(stripped, original)
        self.assertEqual(len(doped), len(circuit) + len(sites))

    def test_non_clifford_raises(self):
        """A non-Clifford instruction is rejected."""
        circuit = QuantumCircuit(1)
        circuit.rx(0.3, 0)
        with self.assertRaises(ValueError):
            dope_clifford_circuit(circuit)

    def test_barriers_ignored(self):
        """Barriers are transparent to the propagation and preserved in the output."""
        circuit = _paper_ansatz(3, seed=1)
        circuit.barrier()
        circuit.sx(0)
        doped, sites = dope_clifford_circuit(circuit)
        self.assertGreater(len(sites), 0)
        self.assertIn("barrier", doped.count_ops())
        self.assertEqual(doped.count_ops()["rz"], len(sites))
        self.assertEqual(len(doped), len(circuit) + len(sites))

    def test_measured_circuit(self):
        """Terminal measurements are preserved and no site lies past a qubit's measurement."""
        circuit = _paper_ansatz(3, seed=1)
        circuit.measure_all()
        measure_pos = _measure_positions(circuit)
        doped, sites = dope_clifford_circuit(circuit)
        self.assertGreater(len(sites), 0)
        self.assertEqual(doped.count_ops()["measure"], 3)
        self.assertEqual(doped.count_ops()["rz"], len(sites))
        for site in sites:
            self.assertLessEqual(_position(site), measure_pos[site.qubit])
        circuit.x(0)
        with self.assertRaises(ValueError):
            dope_clifford_circuit(circuit)


class TestCheckedCircuitDoping(unittest.TestCase):
    """Tests for doping a :class:`.CheckedCircuit` without breaking its spacetime code."""

    def _assert_code_preserved(self, checked: CheckedCircuit, doped: CheckedCircuit, sites):
        """The doped circuit keeps the checks' metadata, wires, syndromes, and cumulants."""
        self.assertIsInstance(doped, CheckedCircuit)
        for field in ("target_qubits", "check_qubits", "check_support", "cost"):
            self.assertEqual(getattr(doped, field), getattr(checked, field))
        num_qubits = checked.circuit.num_qubits
        # Sites avoid ancilla wires and post-measurement wires.
        measure_pos = _measure_positions(checked.circuit)
        for site in sites:
            self.assertNotIn(site.qubit, checked.check_qubits)
            self.assertLessEqual(_position(site), measure_pos[site.qubit])
        # Every syndrome stays deterministic with its original sign.
        original = _syndrome_values(checked, checked.circuit)
        for value in original:
            self.assertAlmostEqual(abs(value), 1.0, places=10)
        np.testing.assert_allclose(_syndrome_values(doped, doped.circuit), original, atol=1e-10)
        # Z on every doped wire commutes with each check's back-cumulant there.
        for site in sites:
            site_z = Pauli("I" * num_qubits)
            site_z.z[site.qubit] = True
            suffix = Clifford(_split(checked.circuit, _position(site))[1])
            for support in checked.check_support:
                label = "".join("Z" if q in support else "I" for q in reversed(range(num_qubits)))
                cumulant = Pauli(label).evolve(suffix, frame="h")
                self.assertTrue(cumulant.commutes(site_z))

    def test_code_preserved(self):
        """Doping changes the payload distribution but never breaks a check."""
        checked = _checked_circuit()
        self.assertEqual(len(checked.check_qubits), 1)
        doped, sites = dope_clifford_circuit(checked)
        self.assertGreater(len(sites), 0)
        self._assert_code_preserved(checked, doped, sites)

    def test_template_preserves_code(self):
        """A parametrized template keeps every syndrome at any angles, for every wires rule."""
        checked = _checked_circuit()
        original = _syndrome_values(checked, checked.circuit)
        for wires in ("all", "after_entangling", "before_entangling"):
            with self.subTest(wires=wires):
                doped, sites = dope_clifford_circuit(checked, wires=wires, angle=None)
                self.assertGreater(len(sites), 0)
                angles = np.random.default_rng(0).uniform(0, 2 * np.pi, len(sites))
                bound = doped.circuit.assign_parameters(angles)
                np.testing.assert_allclose(_syndrome_values(checked, bound), original, atol=1e-10)


class TestHardwareStyle(unittest.TestCase):
    """Tests for the ``wires`` site restriction and ``angle=None`` templates."""

    def test_entangling_wires(self):
        """Sites are restricted to wires directly after, or directly before, an entangling gate."""
        circuit = _paper_ansatz(5, seed=1)
        for wires, offset in (("after_entangling", -1), ("before_entangling", 0)):
            with self.subTest(wires=wires):
                doped, sites = dope_clifford_circuit(circuit, wires=wires)
                self.assertGreater(len(sites), 0)
                self.assertEqual(doped.count_ops()["rz"], len(sites))
                for site in sites:
                    inst = circuit.data[_position(site) + offset]
                    self.assertGreater(inst.operation.num_qubits, 1)
                    self.assertIn(site.qubit, [circuit.find_bit(q).index for q in inst.qubits])

    def test_invalid_wires(self):
        """An unknown ``wires`` value is rejected."""
        with self.assertRaisesRegex(ValueError, "wires must be"):
            dope_clifford_circuit(_paper_ansatz(3, seed=1), wires="between")

    def test_angle(self):
        """Sites are angle-independent; a Clifford angle yields a Clifford doped circuit."""
        circuit = _paper_ansatz(4, seed=1)
        default, sites = dope_clifford_circuit(circuit)
        s_doped, s_sites = dope_clifford_circuit(circuit, angle=np.pi / 2)
        self.assertEqual(s_sites, sites)
        self.assertEqual(
            [inst.operation.params[0] for inst in s_doped.data if inst.operation.name == "rz"],
            [np.pi / 2] * len(sites),
        )
        Clifford(s_doped.remove_final_measurements(inplace=False))
        with self.assertRaises(QiskitError):
            Clifford(default.remove_final_measurements(inplace=False))

    def test_parametric_template(self):
        """One template reproduces the default doping at pi/4 and the base circuit at 0."""
        circuit = _paper_ansatz(4, seed=1)
        template, sites = dope_clifford_circuit(circuit, angle=None)
        self.assertEqual(len(template.parameters), len(sites))
        t_doped, t_sites = dope_clifford_circuit(circuit)
        self.assertEqual(sites, t_sites)
        bound = template.assign_parameters([np.pi / 4] * len(sites))
        self.assertTrue(Statevector(bound).equiv(Statevector(t_doped)))
        zero = template.assign_parameters([0.0] * len(sites))
        self.assertTrue(Statevector(zero).equiv(Statevector(circuit)))


if __name__ == "__main__":
    unittest.main()
