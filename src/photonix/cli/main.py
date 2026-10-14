"""Argument parsing, dispatch, output and exit-code policy of the ``photonix`` command."""
from __future__ import annotations

import argparse
import os
import sys
import traceback
from collections.abc import Sequence
from typing import Any

from .. import __version__
from ._common import CliError, ExitCode, dump_json, schema_id
from .commands import COMMANDS

__all__ = ["build_parser", "main"]

_EPILOG = """\
exit codes: 0 ok, 1 a check failed, 2 usage or input error, 3 missing optional dependency,
4 internal error. Every command accepts --json (schema photonix.cli/<command>/1).
See docs/CLI.md."""


class _Parser(argparse.ArgumentParser):
    """An ArgumentParser that raises instead of exiting, so :func:`main` always returns."""

    def error(self, message: str):  # type: ignore[override]
        raise CliError(f"{self.prog}: {message}")

    def exit(self, status: int = 0, message: str | None = None):  # type: ignore[override]
        if message:
            print(message, file=sys.stderr, end="")
        raise _Exit(status)


class _Exit(Exception):
    def __init__(self, status: int):
        super().__init__(status)
        self.status = status


def build_parser() -> argparse.ArgumentParser:
    """The complete ``photonix`` argument parser (all subcommands).

    Examples
    --------
    >>> parser = build_parser()
    >>> args = parser.parse_args(["sweep", "straight", "--wl", "1.55", "-p", "length=25"])
    >>> args.model, args.wl, args.param
    ('straight', '1.55', ['length=25'])
    """
    parser = _Parser(
        prog="photonix",
        description="photonix: accuracy-first, differentiable photonic integrated circuit design. "
                    "Without a command, `photonix` prints the runtime information (`photonix info`).",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--json", action="store_true", default=False,
                        help="machine-readable JSON output (also accepted after the command)")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND", parser_class=_Parser)
    for module in COMMANDS:
        module.register(sub)
    return parser


def _error_payload(command: str | None, code: ExitCode, message: str, hint: str | None) -> dict[str, Any]:
    return {"schema": schema_id("error"), "command": command, "exit_code": int(code),
            "error": {"message": message, "hint": hint}}


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``photonix`` command line and return its exit code.

    ``argv`` defaults to ``sys.argv[1:]``. The function never calls
    :func:`sys.exit`, so it can be used from Python and from tests.

    Examples
    --------
    >>> main(["sweep", "straight", "--wl", "1.55", "--json"]) == 0  # doctest: +ELLIPSIS
    {...}
    True
    """
    args_list = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    wants_json = "--json" in args_list
    command: str | None = None
    try:
        try:
            args = parser.parse_args(args_list)
        except _Exit as exc:  # --help / --version
            return int(exc.status)
        command = args.command or "info"
        if args.command is None:
            args = parser.parse_args(["info", *args_list])
        wants_json = bool(getattr(args, "json", False))
        payload, code = args.handler(args)
        if wants_json:
            dump_json(payload)
        else:
            print(args.render(payload))
        return int(code)
    except CliError as exc:
        print(f"photonix: error: {exc}", file=sys.stderr)
        if exc.hint:
            print(f"photonix: hint: {exc.hint}", file=sys.stderr)
        if wants_json:
            dump_json(_error_payload(command, exc.code, str(exc), exc.hint))
        return int(exc.code)
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        print("photonix: interrupted", file=sys.stderr)
        return 130
    except BrokenPipeError:
        # The reader closed stdout early (``photonix ... --json | head``). Point
        # stdout at /dev/null so the interpreter's final flush cannot fail too,
        # and exit like a process killed by SIGPIPE.
        try:
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, sys.stdout.fileno())
        except (OSError, ValueError):  # pragma: no cover - stdout already gone
            pass
        return 128 + 13
    except Exception as exc:  # an unexpected failure is a bug: keep the traceback
        traceback.print_exc()
        print(f"photonix: internal error ({type(exc).__name__}); please report it with `photonix info --json`",
              file=sys.stderr)
        if wants_json:
            dump_json(_error_payload(command, ExitCode.INTERNAL_ERROR, f"{type(exc).__name__}: {exc}", None))
        return int(ExitCode.INTERNAL_ERROR)
