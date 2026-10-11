"""Task-oriented command line: ``photonix <command> [--json]``.

Commands (``photonix COMMAND --help`` documents every option):

``info``
    numerical runtime and optional dependencies (also the default command);
``models [NAME]``
    the compact-model registry with ports and parameter defaults;
``mode``
    full-vector FDE modes of a strip or a PDK cross-section, with an optional
    grid-convergence study;
``sweep MODEL``
    evaluate a model over wavelength; write Touchstone, NPZ, JSON or CSV;
``sparams check FILE`` / ``sparams convert IN OUT``
    passivity/reciprocity gate (exit 1 on violation) and format conversion;
``gds inspect FILE`` / ``gds precheck FILE``
    GDSII summary and the geometric DRC pre-check (optional gdstk);
``netlist simulate FILE``
    simulate a ``Netlist.to_json`` file over wavelength;
``validate [--quick]``
    analytic self-checks emitted with the runtime as a reproducibility manifest.

Exit codes are the :class:`ExitCode` values: 0 ok, 1 a check failed, 2 usage
or input error, 3 missing optional dependency, 4 internal error. ``--json``
prints one document whose ``schema`` field is ``photonix.cli/<command>/1``;
errors under ``--json`` print a ``photonix.cli/error/1`` document. The
commands wrap library calls and add no numerics of their own.

Examples
--------
>>> from photonix.cli import main
>>> main(["sparams", "check", "missing.s2p"])
2
"""
from __future__ import annotations

from ._common import SCHEMA_PREFIX, CliError, ExitCode, parse_params, parse_wavelengths, schema_id
from .main import build_parser, main

__all__ = [
    "main", "build_parser", "ExitCode", "CliError", "SCHEMA_PREFIX", "schema_id",
    "parse_wavelengths", "parse_params",
]
