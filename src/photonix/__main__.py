"""``python -m photonix`` and the ``photonix`` console script.

Both delegate to :func:`photonix.cli.main`; see :mod:`photonix.cli` for the
commands, the JSON schemas and the exit codes.
"""
from __future__ import annotations

from collections.abc import Sequence

from .cli import main as _cli_main

__all__ = ["main"]


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``photonix`` command line and return its exit code.

    Examples
    --------
    >>> from photonix.__main__ import main
    >>> main(["gds", "inspect", "no-such-file.gds"]) in (2, 3)   # 3 when gdstk is not installed
    True
    """
    return _cli_main(argv)


if __name__ == "__main__":  # pragma: no cover - exercised through ``python -m``
    raise SystemExit(main())
