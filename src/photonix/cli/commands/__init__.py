"""``photonix`` subcommands. Each module defines ``register(subparsers)``.

A registered parser sets ``handler``: a function ``(args) -> (payload, exit_code)``
whose payload is a JSON-compatible dictionary, and ``render``: a function
``(payload) -> str`` producing the text output for the same payload.
"""
from __future__ import annotations

from . import gds, info, mode, models, netlist, sparams, sweep, validate

#: Registration order is the order shown by ``photonix --help``.
COMMANDS = (info, models, mode, sweep, sparams, gds, netlist, validate)

__all__ = ["COMMANDS"]
