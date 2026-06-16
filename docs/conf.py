# This code is a Qiskit project.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import inspect
import os
import re
import sys
from importlib.metadata import version as metadata_version

# The following line is required for autodoc to be able to find and import the code whose API should
# be documented.
sys.path.insert(0, os.path.abspath(".."))

project = "Qiskit Paulice"
project_copyright = "2026, Qiskit addons team"
description = "Library for implementing spacetime Pauli checks"
author = "Qiskit addons team"
language = "en"
release = metadata_version("qiskit-paulice")

html_theme = "qiskit-ecosystem"

# Make `docs/images/` available as a static asset directory; the qiskit-ecosystem
# theme resolves `dark_logo`/`light_logo` relative to `html_static_path`.
html_static_path = ["images"]

# This allows including custom CSS and HTML templates.
html_theme_options = {
    "dark_logo": "qiskit-dark-logo.svg",
    "light_logo": "qiskit-light-logo.svg",
    "sidebar_qiskit_ecosystem_member": False,
}
html_static_path = ["images"]
templates_path = ["_templates"]

# Sphinx should ignore these patterns when building.
exclude_patterns = [
    "_build",
    "_ecosystem_build",
    "_qiskit_build",
    "_pytorch_build",
    "**.ipynb_checkpoints",
    "jupyter_execute",
]

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.linkcode",
    "sphinx.ext.intersphinx",
    "matplotlib.sphinxext.plot_directive",
    "sphinx_copybutton",
    "sphinx_reredirects",
    "reno.sphinxext",
    "nbsphinx",
    "qiskit_sphinx_theme",
]

html_last_updated_fmt = "%Y/%m/%d"
html_title = f"{project} {release}"

# This allows RST files to put `|version|` in their file and
# have it updated with the release set in conf.py.
rst_prolog = f"""
.. |version| replace:: {release}
"""

# Options for autodoc. These reflect the values from Qiskit SDK and Runtime.
autosummary_generate = True
autosummary_generate_overwrite = False
autoclass_content = "both"
autodoc_typehints = "description"
autodoc_default_options = {
    "inherited-members": None,
    "show-inheritance": True,
}
autodoc_mock_imports = ["qiskit_paulice._internal_r"]

# Render these annotations as their alias names (linked to their autodata
# entry) instead of expanding them to the underlying raw types. Requires
# `from __future__ import annotations` in the source module so that the
# annotation is a string at autodoc time.
autodoc_type_aliases = {
    "UniformGateNoise": "UniformGateNoise",
    "LayeredGateNoise": "LayeredGateNoise",
    "GateWiseNoise": "GateWiseNoise",
    "GateNoise": "GateNoise",
}
napoleon_google_docstring = True
napoleon_numpy_docstring = False


# This adds numbers to the captions for figures, tables,
# and code blocks.
numfig = True
numfig_format = {"table": "Table %s"}

# Settings for Jupyter notebooks.
nbsphinx_execute = "never"

add_module_names = False

modindex_common_prefix = ["qiskit_paulice."]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "qiskit": ("https://quantum.cloud.ibm.com/docs/api/qiskit/", None),
    "rustworkx": ("https://www.rustworkx.org/", None),
}

plot_working_directory = "."
plot_html_show_source_link = False

# ----------------------------------------------------------------------------------
# Redirects
# ----------------------------------------------------------------------------------

_inlined_apis = [
    ("qiskit_paulice.checks", "add_pauli_checks"),
    ("qiskit_paulice.checked_circuit", "CheckedCircuit"),
    ("qiskit_paulice.checked_circuit", "UncoveredPauli"),
    ("qiskit_paulice.noise_models", "NoiseModel"),
    ("qiskit_paulice.noise_models", "GateNoise"),
    ("qiskit_paulice.noise_models", "UniformGateNoise"),
    ("qiskit_paulice.noise_models", "LayeredGateNoise"),
    ("qiskit_paulice.noise_models", "GateWiseNoise"),
    ("qiskit_paulice.layout", "get_low_overhead_ancillas"),
]

redirects = {
    "apidocs/qiskit_paulice": "./index.html",
    **{
        f"stubs/{module}.{name}": f"../apidocs/{module}.html#{module}.{name}"
        for module, name in _inlined_apis
    },
}

# ----------------------------------------------------------------------------------
# Source code links
# ----------------------------------------------------------------------------------


def determine_github_branch() -> str:
    """Determine the GitHub branch name to use for source code links.

    We need to decide whether to use `stable/<version>` vs. `main` for dev builds.
    Refer to https://docs.github.com/en/actions/learn-github-actions/variables
    for how we determine this with GitHub Actions.
    """
    # If CI env vars not set, default to `main`. This is relevant for local builds.
    if "GITHUB_REF_NAME" not in os.environ:
        return "main"

    # PR workflows set the branch they're merging into.
    if base_ref := os.environ.get("GITHUB_BASE_REF"):
        return base_ref

    ref_name = os.environ["GITHUB_REF_NAME"]

    # Check if the ref_name is a tag like `1.0.0` or `1.0.0rc1`. If so, we need
    # to transform it to a Git branch like `stable/1.0`.
    version_without_patch = re.match(r"(\d+\.\d+)", ref_name)
    return f"stable/{version_without_patch.group()}" if version_without_patch else ref_name


GITHUB_BRANCH = determine_github_branch()


def linkcode_resolve(domain, info):
    if domain != "py":
        return None

    module_name = info["module"]
    module = sys.modules.get(module_name)
    if module is None or "qiskit_paulice" not in module_name:
        return None

    def is_valid_code_object(obj):
        return inspect.isclass(obj) or inspect.ismethod(obj) or inspect.isfunction(obj)

    obj = module
    for part in info["fullname"].split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            return None
        if not is_valid_code_object(obj):
            return None

    # Unwrap decorators. This requires they used `functools.wrap()`.
    while hasattr(obj, "__wrapped__"):
        obj = obj.__wrapped__
        if not is_valid_code_object(obj):
            return None

    try:
        full_file_name = inspect.getsourcefile(obj)
    except TypeError:
        return None
    if full_file_name is None or "/qiskit_paulice/" not in full_file_name:
        return None
    file_name = full_file_name.split("/qiskit_paulice/")[-1]

    try:
        source, lineno = inspect.getsourcelines(obj)
    except (OSError, TypeError):
        linespec = ""
    else:
        ending_lineno = lineno + len(source) - 1
        linespec = f"#L{lineno}-L{ending_lineno}"
    return f"https://github.com/Qiskit/qiskit-paulice/tree/{GITHUB_BRANCH}/qiskit_paulice/{file_name}{linespec}"
