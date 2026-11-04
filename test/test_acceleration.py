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

"""Test the scoring-engine seam."""

from __future__ import annotations

import os
import unittest
from unittest import mock

import coheriq
import qiskit_paulice
from qiskit_paulice import acceleration
from qiskit_paulice._internal._internal_r import register_scoring_engine


class TestScoringEngineSeam(unittest.TestCase):
    """Test registration and conformance of a scoring engine."""

    def test_importing_the_package_registers_its_domain(self):
        """Coheriq finds the domain from the package module."""
        self.assertIn("qiskit_paulice", coheriq.known_domains())
        self.assertIn("reference", coheriq.available_engines(qiskit_paulice))

    def test_installs_the_enabled_engine_table_once(self):
        """The table the enabled engine returns is registered on the first install only."""
        table = object()
        with (
            mock.patch.object(acceleration, "_installed", False),
            mock.patch.object(acceleration, "scoring_engine", return_value=table) as engine,
            mock.patch.object(acceleration, "_register_scoring_engine") as register,
        ):
            acceleration.install_scoring_engine()
            acceleration.install_scoring_engine()
        engine.assert_called_once_with()
        register.assert_called_once_with(table)

    def test_rejects_an_object_that_is_not_an_engine_capsule(self):
        """Only a capsule named ``qiskit_paulice.engine`` is accepted."""
        with self.assertRaises((TypeError, ValueError)):
            register_scoring_engine(object())

    @unittest.skipIf(os.environ.get("PAULICE_ENGINE"), "an engine is selected")
    def test_conformance_skips_without_an_engine(self):
        """Every check reports skipped when no engine is registered."""
        self.assertIsNone(acceleration.scoring_engine_name())
        results = acceleration.run_engine_conformance()
        self.assertTrue(results)
        for name, outcome, _ in results:
            with self.subTest(check=name):
                self.assertEqual(outcome, "skipped")

    @unittest.skipUnless(os.environ.get("PAULICE_ENGINE"), "no engine is selected")
    def test_selected_engine_passes_conformance(self):
        """The engine chosen through ``PAULICE_ENGINE`` matches the host."""
        results = acceleration.run_engine_conformance()
        self.assertIsNotNone(acceleration.scoring_engine_name())
        for name, outcome, detail in results:
            with self.subTest(check=name):
                self.assertNotEqual(outcome, "failed", detail)
