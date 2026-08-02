"""Runtime diagnostics for reproducible Photonix simulations.

The helpers in this module inspect the active numerical backend and optional
dependencies without importing those dependencies.  This keeps diagnostics
safe to use when debugging a minimal installation or preparing a bug report.
"""
from __future__ import annotations

import json
import platform
import shutil
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from importlib import metadata, util
from types import MappingProxyType
from typing import TextIO

from . import __version__
from .core import backend_name, device_count, xp

__all__ = [
    "DependencyStatus",
    "RuntimeInfo",
    "format_runtime_info",
    "runtime_info",
    "show_config",
]


@dataclass(frozen=True)
class DependencyStatus:
    """Availability and installed version of an optional Python dependency."""

    available: bool
    version: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.available, bool):
            raise TypeError("available must be a bool")
        if self.version is not None and not isinstance(self.version, str):
            raise TypeError("version must be a string or None")
        if not self.available and self.version is not None:
            raise ValueError("an unavailable dependency cannot have an installed version")


@dataclass(frozen=True)
class RuntimeInfo:
    """Serializable snapshot of the environment relevant to a simulation.

    ``optional_dependencies`` is exposed as a read-only mapping so a captured
    snapshot cannot be modified accidentally.  Use :meth:`as_dict` when a plain
    JSON-serializable representation is needed.
    """

    photonix_version: str
    python_version: str
    python_implementation: str
    platform: str
    backend: str
    x64_enabled: bool
    default_real_dtype: str
    device_count: int
    optional_dependencies: Mapping[str, DependencyStatus]
    klayout_available: bool

    def __post_init__(self) -> None:
        dependencies = dict(self.optional_dependencies)
        if any(not isinstance(name, str) or not name for name in dependencies):
            raise TypeError("optional dependency names must be non-empty strings")
        if any(not isinstance(status, DependencyStatus) for status in dependencies.values()):
            raise TypeError("optional dependency values must be DependencyStatus objects")
        if isinstance(self.device_count, bool) or not isinstance(self.device_count, int) or self.device_count < 1:
            raise ValueError("device_count must be a positive integer")
        if not isinstance(self.x64_enabled, bool) or not isinstance(self.klayout_available, bool):
            raise TypeError("x64_enabled and klayout_available must be bools")
        object.__setattr__(self, "optional_dependencies", MappingProxyType(dependencies))

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable copy of this snapshot."""
        return {
            "photonix_version": self.photonix_version,
            "python_version": self.python_version,
            "python_implementation": self.python_implementation,
            "platform": self.platform,
            "backend": self.backend,
            "x64_enabled": self.x64_enabled,
            "default_real_dtype": self.default_real_dtype,
            "device_count": self.device_count,
            "optional_dependencies": {
                name: asdict(status) for name, status in self.optional_dependencies.items()
            },
            "klayout_available": self.klayout_available,
        }


# (display/import key, distribution metadata key).  Keep this aligned with the
# extras in pyproject.toml; Meep is intentionally present even though its conda
# package cannot be declared as a pip extra.
_OPTIONAL_DISTRIBUTIONS = {
    "gdstk": "gdstk",
    "jax": "jax",
    "matplotlib": "matplotlib",
    "meep": "meep",
    "meshio": "meshio",
    "networkx": "networkx",
    "skrf": "scikit-rf",
}


def _module_available(module: str) -> bool:
    try:
        return util.find_spec(module) is not None
    except (AttributeError, ImportError, ValueError):
        # A broken parent package or a manually injected sys.modules entry must
        # not make the diagnostics command itself fail.
        return False


def _distribution_version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def _dependency_status(module: str, distribution: str) -> DependencyStatus:
    available = _module_available(module)
    version = _distribution_version(distribution) if available else None
    return DependencyStatus(available=available, version=version)


def runtime_info() -> RuntimeInfo:
    """Capture the active backend, precision, devices, and optional features.

    Availability means that Python can discover an optional module; it does not
    claim that an external solver is licensed, configured, or numerically
    qualified.  In particular, an installed JAX package can be listed while the
    active backend remains NumPy when ``PHOTONIX_BACKEND=numpy`` is set.
    """
    default_dtype = str(xp.asarray(0.0).dtype)
    dependencies = {
        module: _dependency_status(module, distribution)
        for module, distribution in _OPTIONAL_DISTRIBUTIONS.items()
    }
    return RuntimeInfo(
        photonix_version=__version__,
        python_version=platform.python_version(),
        python_implementation=platform.python_implementation(),
        platform=platform.platform(),
        backend=backend_name(),
        x64_enabled=default_dtype in {"float64", "complex128"},
        default_real_dtype=default_dtype,
        device_count=device_count(),
        optional_dependencies=MappingProxyType(dependencies),
        klayout_available=shutil.which("klayout") is not None or shutil.which("klayout_app") is not None,
    )


def format_runtime_info(info: RuntimeInfo | None = None, *, json_output: bool = False) -> str:
    """Format a runtime snapshot as readable text or stable JSON."""
    info = runtime_info() if info is None else info
    if json_output:
        return json.dumps(info.as_dict(), indent=2, sort_keys=True)

    dependency_lines = []
    for name, status in info.optional_dependencies.items():
        if not status.available:
            value = "not installed"
        elif status.version is None:
            value = "available (version unknown)"
        else:
            value = status.version
        dependency_lines.append(f"    {name}: {value}")

    return "\n".join(
        [
            f"Photonix {info.photonix_version}",
            f"  Python: {info.python_version} ({info.python_implementation})",
            f"  Platform: {info.platform}",
            "Numerics",
            f"  Backend: {info.backend}",
            f"  Default real dtype: {info.default_real_dtype}",
            f"  64-bit enabled: {info.x64_enabled}",
            f"  Devices: {info.device_count}",
            "Optional Python dependencies",
            *dependency_lines,
            "External tools",
            f"  KLayout: {'available' if info.klayout_available else 'not found'}",
        ]
    )


def show_config(*, file: TextIO | None = None, json_output: bool = False) -> None:
    """Print runtime configuration for a bug report or simulation manifest."""
    destination = sys.stdout if file is None else file
    print(format_runtime_info(json_output=json_output), file=destination)
