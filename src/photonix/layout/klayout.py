"""Optional KLayout batch integration for user-supplied DRC and LVS decks.

Photonix does not ship design-rule or extraction content.  These helpers only
construct and run KLayout's documented headless command line against paths the
caller provides.  In particular, deck files are never copied, rewritten, read,
or embedded in Photonix output.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "KLayoutResult",
    "KLayoutRunError",
    "find_klayout",
    "klayout_available",
    "run_klayout_deck",
    "run_drc",
    "run_lvs",
]

_EXECUTABLE_ENV = "KLAYOUT_EXECUTABLE"
_EXECUTABLE_NAMES = ("klayout", "klayout_app")
_VARIABLE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class KLayoutResult:
    """Completed KLayout batch invocation and its captured process output."""

    kind: str
    command: tuple[str, ...]
    layout_path: str
    deck_path: str
    report_path: str | None
    returncode: int
    stdout: str
    stderr: str

    @property
    def succeeded(self) -> bool:
        """Whether KLayout exited successfully.

        A zero exit status does not by itself certify a DRC/LVS-clean design;
        the supplied deck defines how violations are represented in its report.
        """
        return self.returncode == 0

    @property
    def report_exists(self) -> bool:
        """Whether the requested report path exists after the run."""
        return self.report_path is not None and Path(self.report_path).is_file()


class KLayoutRunError(RuntimeError):
    """Raised when a checked KLayout batch invocation exits unsuccessfully."""

    def __init__(self, result: KLayoutResult):
        detail = result.stderr.strip() or result.stdout.strip() or "no process output"
        super().__init__(f"KLayout {result.kind} failed with exit code {result.returncode}: {detail}")
        self.result = result


def find_klayout(executable: str | os.PathLike[str] | None = None) -> str | None:
    """Return the KLayout executable path without importing KLayout.

    Resolution order is an explicit ``executable``, the
    ``KLAYOUT_EXECUTABLE`` environment variable, then ``klayout`` and
    ``klayout_app`` on ``PATH``.  ``None`` means no runnable executable was
    found.
    """
    requested = executable if executable is not None else os.environ.get(_EXECUTABLE_ENV)
    if requested:
        return shutil.which(os.fspath(requested))
    for name in _EXECUTABLE_NAMES:
        found = shutil.which(name)
        if found is not None:
            return found
    return None


def klayout_available(executable: str | os.PathLike[str] | None = None) -> bool:
    """Return whether a KLayout executable is available for batch runs."""
    return find_klayout(executable) is not None


def _input_file(path: str | os.PathLike[str], label: str) -> str:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist or is not a file: {resolved}")
    return str(resolved)


def _output_file(path: str | os.PathLike[str], label: str) -> str:
    resolved = Path(path).expanduser().resolve()
    if not resolved.parent.is_dir():
        raise FileNotFoundError(f"parent directory for {label} does not exist: {resolved.parent}")
    return str(resolved)


def _runtime_value(value: object) -> str:
    if value is None:
        raise ValueError("KLayout runtime variable values cannot be None")
    rendered = os.fspath(value) if isinstance(value, os.PathLike) else str(value)
    if "\x00" in rendered:
        raise ValueError("KLayout runtime variable values cannot contain NUL characters")
    return rendered


def _validate_variable_name(name: object) -> str:
    if not isinstance(name, str) or not _VARIABLE_RE.fullmatch(name):
        raise ValueError(f"invalid KLayout runtime variable name: {name!r}")
    return name


def _run_deck(
    kind: str,
    layout_path: str | os.PathLike[str],
    deck_path: str | os.PathLike[str],
    *,
    report_path: str | os.PathLike[str] | None,
    variables: Mapping[str, object] | None,
    input_variable: str | None,
    report_variable: str | None,
    executable: str | os.PathLike[str] | None,
    extra_args: Sequence[str],
    cwd: str | os.PathLike[str] | None,
    timeout: float | None,
    check: bool,
) -> KLayoutResult:
    exe = find_klayout(executable)
    if exe is None:
        raise FileNotFoundError(
            "KLayout executable not found. Install KLayout, put it on PATH, set "
            f"{_EXECUTABLE_ENV}, or pass executable= explicitly."
        )
    layout = _input_file(layout_path, "layout_path")
    deck = _input_file(deck_path, "deck_path")
    report = _output_file(report_path, "report_path") if report_path is not None else None

    runtime: dict[str, str] = {}
    reserved = {name for name in (input_variable,) if name is not None}
    if report is not None and report_variable is not None:
        reserved.add(report_variable)
    reserved = {_validate_variable_name(name) for name in reserved}
    supplied = dict(variables or {})
    duplicate = reserved.intersection(supplied)
    if duplicate:
        names = ", ".join(sorted(duplicate))
        raise ValueError(f"variables duplicates managed KLayout variable(s): {names}")
    if input_variable is not None:
        runtime[input_variable] = layout
    if report_variable is not None and report is not None:
        runtime[report_variable] = report
    for name, value in supplied.items():
        runtime[_validate_variable_name(name)] = _runtime_value(value)

    command = [exe, "-b"]
    command.extend(_runtime_value(arg) for arg in extra_args)
    for name, value in runtime.items():
        command.extend(("-rd", f"{name}={value}"))
    command.extend(("-r", deck))
    completed = subprocess.run(
        command,
        cwd=os.fspath(cwd) if cwd is not None else None,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        shell=False,
    )
    result = KLayoutResult(
        kind=kind,
        command=tuple(command),
        layout_path=layout,
        deck_path=deck,
        report_path=report,
        returncode=int(completed.returncode),
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
    )
    if check and not result.succeeded:
        raise KLayoutRunError(result)
    return result


def run_klayout_deck(
    layout_path: str | os.PathLike[str],
    deck_path: str | os.PathLike[str],
    *,
    report_path: str | os.PathLike[str] | None = None,
    variables: Mapping[str, object] | None = None,
    input_variable: str | None = "input",
    report_variable: str | None = "report",
    executable: str | os.PathLike[str] | None = None,
    extra_args: Sequence[str] = (),
    cwd: str | os.PathLike[str] | None = None,
    timeout: float | None = None,
    check: bool = True,
) -> KLayoutResult:
    """Run a user-supplied KLayout deck in headless batch mode.

    The command follows KLayout's documented form ``klayout -b -rd name=value
    -r deck``.  By default the absolute layout and report paths are supplied as
    the runtime variables ``input`` and ``report``.  Change those names—or pass
    ``None`` to disable either variable—when a foundry deck uses a different
    contract. Additional deck-specific string variables belong in ``variables``.

    ``check=True`` raises :class:`KLayoutRunError` on a non-zero process exit.
    A successful exit is not interpreted as a clean sign-off result: each deck
    defines its own report schema and violation policy.
    """
    return _run_deck(
        "deck", layout_path, deck_path, report_path=report_path, variables=variables,
        input_variable=input_variable, report_variable=report_variable, executable=executable,
        extra_args=extra_args, cwd=cwd, timeout=timeout, check=check,
    )


def run_drc(
    layout_path: str | os.PathLike[str],
    deck_path: str | os.PathLike[str],
    *,
    report_path: str | os.PathLike[str] | None = None,
    variables: Mapping[str, object] | None = None,
    input_variable: str | None = "input",
    report_variable: str | None = "report",
    executable: str | os.PathLike[str] | None = None,
    extra_args: Sequence[str] = (),
    cwd: str | os.PathLike[str] | None = None,
    timeout: float | None = None,
    check: bool = True,
) -> KLayoutResult:
    """Run a user-supplied KLayout DRC deck in headless batch mode."""
    return _run_deck(
        "DRC", layout_path, deck_path, report_path=report_path, variables=variables,
        input_variable=input_variable, report_variable=report_variable, executable=executable,
        extra_args=extra_args, cwd=cwd, timeout=timeout, check=check,
    )


def run_lvs(
    layout_path: str | os.PathLike[str],
    deck_path: str | os.PathLike[str],
    *,
    report_path: str | os.PathLike[str] | None = None,
    variables: Mapping[str, object] | None = None,
    input_variable: str | None = "input",
    report_variable: str | None = "report",
    executable: str | os.PathLike[str] | None = None,
    extra_args: Sequence[str] = (),
    cwd: str | os.PathLike[str] | None = None,
    timeout: float | None = None,
    check: bool = True,
) -> KLayoutResult:
    """Run a user-supplied KLayout LVS deck in headless batch mode."""
    return _run_deck(
        "LVS", layout_path, deck_path, report_path=report_path, variables=variables,
        input_variable=input_variable, report_variable=report_variable, executable=executable,
        extra_args=extra_args, cwd=cwd, timeout=timeout, check=check,
    )
