"""Command-line entry point for Photonix diagnostics."""
from __future__ import annotations

import argparse
from collections.abc import Sequence

from . import __version__
from .diagnostics import show_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="photonix",
        description="Inspect the Photonix numerical runtime and optional integrations.",
    )
    parser.add_argument("command", nargs="?", choices=("info",), default="info")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Photonix command-line interface."""
    args = _parser().parse_args(argv)
    show_config(json_output=args.json)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through ``python -m``
    raise SystemExit(main())
