"""Shared plumbing for the ``photonix`` command line: exit codes, errors, parsing, output.

Every command builds one JSON-compatible *payload* dictionary. ``--json``
prints it verbatim; the text renderer formats the same dictionary, so the two
outputs can never disagree. Each payload carries ``"schema":
"photonix.cli/<command>/<version>"``; the version changes only when a field is
removed or changes meaning (adding a field keeps it).
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from enum import IntEnum
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "ExitCode", "CliError", "SCHEMA_PREFIX", "schema_id", "require", "jsonable", "dump_json",
    "parse_wavelengths", "parse_params", "natural_key", "sort_ports", "load_dataset", "write_dataset",
    "dataset_document", "physicality", "add_json_flag",
]

#: Prefix of every payload's ``schema`` identifier.
SCHEMA_PREFIX = "photonix.cli"


class ExitCode(IntEnum):
    """Process exit status of every ``photonix`` command.

    ``OK`` (0) success; ``CHECK_FAILED`` (1) the command ran and a check it
    performs failed (non-passive data, DRC violations, a failed self-check);
    ``USAGE`` (2) bad arguments or unreadable input; ``MISSING_DEPENDENCY``
    (3) an optional package the command needs is not installed;
    ``INTERNAL_ERROR`` (4) an unexpected exception (a bug -- please report it
    with the printed traceback and ``photonix info --json``).
    """

    OK = 0
    CHECK_FAILED = 1
    USAGE = 2
    MISSING_DEPENDENCY = 3
    INTERNAL_ERROR = 4


class CliError(Exception):
    """An error reported to the user as one line, with an exit code (default: usage)."""

    def __init__(self, message: str, code: ExitCode = ExitCode.USAGE, *, hint: str | None = None):
        super().__init__(message)
        self.code = ExitCode(code)
        self.hint = hint


def schema_id(command: str, version: int = 1) -> str:
    """Schema identifier of a command's JSON payload.

    >>> schema_id("sparams-check")
    'photonix.cli/sparams-check/1'
    """
    return f"{SCHEMA_PREFIX}/{command}/{int(version)}"


_INSTALL_HINTS = {
    "gdstk": "pip install 'photonix[layout]'",
    "matplotlib": "pip install 'photonix[viz]'",
    "skrf": "pip install 'photonix[rf]'",
    "jax": "pip install 'photonix[jax]'",
}


def require(module: str, feature: str) -> Any:
    """Import an optional dependency or raise a :class:`CliError` with exit code 3."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        hint = _INSTALL_HINTS.get(module, f"pip install {module}")
        raise CliError(f"{feature} requires the optional package {module!r}", ExitCode.MISSING_DEPENDENCY,
                       hint=f"install it with: {hint}") from exc


# --------------------------------------------------------------------------- #
# JSON
# --------------------------------------------------------------------------- #
def jsonable(value: Any) -> Any:
    """Convert NumPy/JAX scalars and arrays, tuples and paths to strict-JSON values.

    Non-finite floats become the strings ``"inf"``, ``"-inf"`` and ``"nan"``
    (strict JSON has no such numbers); complex numbers become ``[re, im]``.

    >>> jsonable({"a": (1, 2.5), "b": float("inf"), "c": 1 + 2j})
    {'a': [1, 2.5], 'b': 'inf', 'c': [1.0, 2.0]}
    """
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (str, bool)) or value is None:
        return value
    if isinstance(value, (int, np.integer)) and not isinstance(value, np.bool_):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        x = float(value)
        if math.isfinite(x):
            return x
        return "nan" if math.isnan(x) else ("inf" if x > 0 else "-inf")
    if isinstance(value, (complex, np.complexfloating)):
        return [jsonable(complex(value).real), jsonable(complex(value).imag)]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple, set, frozenset)):
        items = sorted(value, key=repr) if isinstance(value, (set, frozenset)) else value
        return [jsonable(v) for v in items]
    if hasattr(value, "__array__") or isinstance(value, np.ndarray):
        arr = np.asarray(value)
        return jsonable(arr.item()) if arr.ndim == 0 else [jsonable(v) for v in arr.tolist()]
    if callable(value):
        name = getattr(value, "__qualname__", None) or type(value).__name__
        return f"<callable {getattr(value, '__module__', '?')}.{name}>"
    return repr(value)


def dump_json(payload: Mapping[str, Any], file=None) -> None:
    """Print a payload as indented, key-sorted, strict JSON."""
    print(json.dumps(jsonable(payload), indent=2, sort_keys=True, allow_nan=False), file=file or sys.stdout)


def add_json_flag(parser: argparse.ArgumentParser) -> None:
    """Add the ``--json`` switch every command shares."""
    parser.add_argument("--json", action="store_true", default=argparse.SUPPRESS,
                        help="print a machine-readable JSON document (versioned schema) instead of text")


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
def parse_wavelengths(spec: str) -> np.ndarray:
    """Parse a wavelength specification in micrometres.

    ``"1.5:1.6:201"`` is ``numpy.linspace(1.5, 1.6, 201)`` (both ends
    included), ``"1.5,1.55,1.6"`` a list and ``"1.55"`` a single point. The
    result must be positive, finite and strictly increasing.

    >>> parse_wavelengths("1.5:1.6:3")
    array([1.5 , 1.55, 1.6 ])
    >>> parse_wavelengths("1.31")
    array([1.31])
    """
    text = spec.strip()
    try:
        if ":" in text:
            parts = text.split(":")
            if len(parts) != 3:
                raise ValueError
            start, stop, count = float(parts[0]), float(parts[1]), int(parts[2])
            if count < 1:
                raise CliError(f"--wl {spec!r}: the point count must be at least 1")
            values = np.linspace(start, stop, count) if count > 1 else np.asarray([start])
        else:
            values = np.asarray([float(v) for v in text.split(",") if v.strip()], dtype=float)
    except ValueError as exc:
        raise CliError(f"--wl {spec!r}: expected START:STOP:COUNT, a comma-separated list, or one value (um)") \
            from exc
    if values.size == 0 or not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise CliError(f"--wl {spec!r}: wavelengths must be positive and finite (um)")
    if np.any(np.diff(values) <= 0):
        raise CliError(f"--wl {spec!r}: wavelengths must be strictly increasing")
    return values


_LITERALS = {"true": True, "false": False, "none": None, "null": None}


def _parse_value(text: str) -> Any:
    lowered = text.strip().lower()
    if lowered in _LITERALS:
        return _LITERALS[lowered]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def parse_params(items: Sequence[str] | None, allowed: Callable | None = None) -> dict[str, Any]:
    """Parse ``key=value`` settings; values are JSON literals, else strings.

    When ``allowed`` (a model) is given, keys it does not accept are rejected
    with the list of valid parameter names.

    >>> parse_params(["length=25", "neff=2.45", "apodization=gaussian", "seed=null"])
    {'length': 25, 'neff': 2.45, 'apodization': 'gaussian', 'seed': None}
    """
    out: dict[str, Any] = {}
    for item in items or ():
        key, sep, raw = item.partition("=")
        key = key.strip()
        if not sep or not key.isidentifier():
            raise CliError(f"-p {item!r}: expected NAME=VALUE")
        out[key] = _parse_value(raw)
    if allowed is not None and out:
        import inspect

        try:
            params = inspect.signature(allowed).parameters
        except (TypeError, ValueError):
            return out
        if not any(p.kind is p.VAR_KEYWORD for p in params.values()):
            names = [n for n, p in params.items() if p.kind is p.KEYWORD_ONLY and n != "wl"]
            unknown = sorted(set(out) - set(names))
            if unknown:
                raise CliError(f"unknown parameter(s) {', '.join(unknown)}; valid: {', '.join(names) or '(none)'}")
    return out


def natural_key(name: str) -> tuple:
    """Sort key that orders ``o2`` before ``o10``.

    >>> sorted(["o10", "o2", "o1"], key=natural_key)
    ['o1', 'o2', 'o10']
    """
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name))


def sort_ports(names) -> list[str]:
    """Port names in natural order."""
    return sorted({str(n) for n in names}, key=natural_key)


# --------------------------------------------------------------------------- #
# S-parameter files
# --------------------------------------------------------------------------- #
_TOUCHSTONE = re.compile(r"^\.s(\d+)p$", re.IGNORECASE)
DATA_FORMAT = "photonix.sparameters"


def _kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if _TOUCHSTONE.match(suffix):
        return "touchstone"
    if suffix in (".npz", ".json", ".csv"):
        return suffix[1:]
    raise CliError(f"{path}: unsupported file type {path.suffix or '(none)'!r}; "
                   "use .sNp (Touchstone), .npz, .json or (output only) .csv")


def dataset_document(ds) -> dict[str, Any]:
    """The JSON form of an :class:`~photonix.SParameterDataset` (``s[k, out, in]``)."""
    return {
        "format": DATA_FORMAT,
        "version": 1,
        "convention": "s[k][out][in]; ports in the listed order; wavelengths in um",
        "wavelengths_um": ds.wavelengths,
        "ports": list(ds.ports),
        "s_real": ds.s.real,
        "s_imag": ds.s.imag,
        "metadata": ds.metadata,
    }


def _from_document(doc: Mapping[str, Any], path: Path):
    from photonix.core import SParameterDataset

    if doc.get("format") != DATA_FORMAT or int(doc.get("version", 0)) != 1:
        raise CliError(f"{path}: not a {DATA_FORMAT} version 1 JSON document")
    try:
        s = np.asarray(doc["s_real"], dtype=float) + 1j * np.asarray(doc["s_imag"], dtype=float)
        return SParameterDataset(np.asarray(doc["wavelengths_um"], dtype=float), tuple(doc["ports"]), s,
                                 dict(doc.get("metadata") or {}))
    except (KeyError, TypeError) as exc:
        raise CliError(f"{path}: malformed {DATA_FORMAT} document ({exc})") from exc


def load_dataset(path: str | Path):
    """Read a Touchstone ``.sNp``, a dataset ``.npz`` or a ``photonix.sparameters`` JSON file."""
    from photonix.core import SParameterDataset

    source = Path(path)
    if not source.is_file():
        raise CliError(f"{source}: no such file")
    kind = _kind(source)
    try:
        if kind == "touchstone":
            return SParameterDataset.load_touchstone(source)
        if kind == "npz":
            return SParameterDataset.load_npz(source)
        if kind == "json":
            return _from_document(json.loads(source.read_text(encoding="utf-8")), source)
    except (ValueError, KeyError, OSError) as exc:
        raise CliError(f"{source}: {exc}") from exc
    raise CliError(f"{source}: CSV is an output-only format")


def _write_csv(ds, target: Path) -> None:
    columns = ["wl_um"]
    blocks = [ds.wavelengths[:, None]]
    for j, p_in in enumerate(ds.ports):
        for i, p_out in enumerate(ds.ports):
            columns += [f"{p_in}->{p_out}:re", f"{p_in}->{p_out}:im"]
            blocks.append(np.stack([ds.s[:, i, j].real, ds.s[:, i, j].imag], axis=1))
    np.savetxt(target, np.concatenate(blocks, axis=1), delimiter=",", header=",".join(columns),
               comments="", fmt="%.17g")


def write_dataset(ds, path: str | Path, *, frequency_unit: str = "ghz") -> str:
    """Write a dataset; the suffix picks the format. Returns the format written."""
    target = Path(path)
    kind = _kind(target)
    if kind == "touchstone":
        match = _TOUCHSTONE.match(target.suffix)
        assert match is not None
        if int(match.group(1)) != len(ds.ports):
            raise CliError(f"{target}: a .s{match.group(1)}p file holds {match.group(1)} ports but the data has "
                           f"{len(ds.ports)} ({', '.join(ds.ports)}); use the suffix .s{len(ds.ports)}p")
    if not target.parent.exists():
        raise CliError(f"{target}: directory {target.parent} does not exist")
    try:
        if kind == "touchstone":
            ds.save_touchstone(target, frequency_unit=frequency_unit)
        elif kind == "npz":
            ds.save_npz(target)
        elif kind == "json":
            target.write_text(json.dumps(jsonable(dataset_document(ds)), allow_nan=False) + "\n", encoding="utf-8")
        else:
            _write_csv(ds, target)
    except (ValueError, OSError) as exc:
        raise CliError(f"{target}: {exc}") from exc
    return kind


def physicality(ds, *, passivity_atol: float = 1e-9, reciprocity_atol: float = 1e-9,
                lossless_atol: float = 1e-9) -> dict[str, Any]:
    """Passivity, reciprocity and losslessness of a dataset (see :func:`photonix.analyze_sparameters`)."""
    from photonix.core import analyze_sparameters

    d = analyze_sparameters((ds.s, {p: i for i, p in enumerate(ds.ports)}), passivity_atol=passivity_atol,
                            reciprocity_atol=reciprocity_atol, lossless_atol=lossless_atol)
    sv = np.asarray(d.maximum_singular_value).reshape(-1)
    rec = np.asarray(d.reciprocity_error).reshape(-1)
    uni = np.asarray(d.unitarity_error).reshape(-1)
    return {
        "passive": d.passive,
        "reciprocal": d.reciprocal,
        "lossless": d.lossless,
        "max_singular_value": float(sv.max()),
        "max_singular_value_wl_um": float(ds.wavelengths[int(sv.argmax())]),
        "worst_passivity_violation": d.worst_passivity_violation,
        "worst_reciprocity_error": d.worst_reciprocity_error,
        "worst_reciprocity_wl_um": float(ds.wavelengths[int(rec.argmax())]),
        "worst_unitarity_error": d.worst_unitarity_error,
        "worst_unitarity_wl_um": float(ds.wavelengths[int(uni.argmax())]),
        "passivity_atol": d.passivity_atol,
        "reciprocity_atol": d.reciprocity_atol,
        "lossless_atol": d.lossless_atol,
    }
