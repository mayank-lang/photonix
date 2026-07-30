"""Import-safe command adapters for external multiphysics solvers."""
from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from .contracts import MultiphysicsJob, Physics, SolverCapability

__all__ = [
    "SolverAdapter",
    "ElmerAdapter",
    "DevsimAdapter",
    "LumericalDeviceAdapter",
    "ExternalSolverAdapter",
    "detect_capabilities",
]


def _resolve_executable(executable: str) -> str | None:
    candidate = Path(executable).expanduser()
    if candidate.is_absolute() or candidate.parent != Path("."):
        return str(candidate.resolve()) if candidate.is_file() else None
    return shutil.which(executable)


def _validate_supported(name: str, supported: frozenset[Physics], job: MultiphysicsJob) -> None:
    missing = job.physics - supported
    if missing:
        labels = ", ".join(sorted(item.value for item in missing))
        raise ValueError(f"solver {name!r} does not declare support for: {labels}")


@runtime_checkable
class SolverAdapter(Protocol):
    """Structural interface implemented by every external solver adapter."""

    name: str
    supported_physics: frozenset[Physics]

    def capability(self, environment: dict[str, str] | None = None) -> SolverCapability: ...

    def build_command(self, job: MultiphysicsJob) -> tuple[str, ...]: ...


@dataclass(frozen=True)
class ElmerAdapter:
    """Adapter for a prepared Elmer ``.sif`` case run by ``ElmerSolver``.

    Mesh generation and scientific deck authoring stay explicit upstream steps;
    this adapter neither guesses materials nor synthesizes boundary conditions.
    """

    executable: str = "ElmerSolver"
    extra_arguments: tuple[str, ...] = ()
    name: str = field(default="elmer", init=False)
    supported_physics: frozenset[Physics] = field(
        default_factory=lambda: frozenset({Physics.THERMAL, Physics.ELECTRICAL, Physics.ELECTROTHERMAL}),
        init=False,
    )

    def capability(self, environment: dict[str, str] | None = None) -> SolverCapability:
        del environment
        executable = _resolve_executable(self.executable)
        reason = None if executable else f"executable {self.executable!r} was not found"
        return SolverCapability(self.name, executable is not None, executable, self.supported_physics, reason)

    def build_command(self, job: MultiphysicsJob) -> tuple[str, ...]:
        _validate_supported(self.name, self.supported_physics, job)
        if job.input_file.suffix.lower() != ".sif":
            raise ValueError("Elmer input_file must be a .sif solver input deck")
        executable = _resolve_executable(self.executable) or self.executable
        return (executable, *map(str, self.extra_arguments), str(job.input_file))


@dataclass(frozen=True)
class DevsimAdapter:
    """Adapter for a prepared Python script using the open-source DEVSIM module."""

    python_executable: str = sys.executable
    module_name: str = "devsim"
    extra_arguments: tuple[str, ...] = ()
    name: str = field(default="devsim", init=False)
    supported_physics: frozenset[Physics] = field(
        default_factory=lambda: frozenset({Physics.ELECTRICAL, Physics.CARRIER}),
        init=False,
    )

    def capability(self, environment: dict[str, str] | None = None) -> SolverCapability:
        del environment
        executable = _resolve_executable(self.python_executable)
        try:
            module_found = importlib.util.find_spec(self.module_name) is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            module_found = False
        available = executable is not None and module_found
        if executable is None:
            reason = f"Python executable {self.python_executable!r} was not found"
        elif not module_found:
            reason = f"Python module {self.module_name!r} is not installed"
        else:
            reason = None
        return SolverCapability(self.name, available, executable, self.supported_physics, reason)

    def build_command(self, job: MultiphysicsJob) -> tuple[str, ...]:
        _validate_supported(self.name, self.supported_physics, job)
        if job.input_file.suffix.lower() != ".py":
            raise ValueError("DEVSIM input_file must be a Python script")
        executable = _resolve_executable(self.python_executable) or self.python_executable
        return (executable, *map(str, self.extra_arguments), str(job.input_file))


@dataclass(frozen=True)
class LumericalDeviceAdapter:
    """Adapter for a prepared Python script that uses Ansys ``lumapi.DEVICE``.

    The caller declares ``supported_physics`` from the prepared, licensed DEVICE
    workflow; this class does not infer enabled products, models, or license
    features.  ``lumapi_file`` is an optional side-effect-free installation hint
    for setups where ``lumapi`` is loaded explicitly rather than on ``sys.path``.
    The prepared script or its job environment remains responsible for making
    that module importable when the subprocess runs.
    """

    supported_physics: frozenset[Physics]
    python_executable: str = sys.executable
    module_name: str = "lumapi"
    lumapi_file: Path | None = None
    license_environment: str | None = None
    extra_arguments: tuple[str, ...] = ()
    name: str = field(default="lumerical-device", init=False)

    def __post_init__(self) -> None:
        supported = frozenset(
            value if isinstance(value, Physics) else Physics(value)
            for value in self.supported_physics
        )
        if not supported:
            raise ValueError("Lumerical DEVICE adapter must declare supported physics")
        lumapi_file = None if self.lumapi_file is None else Path(self.lumapi_file).expanduser().resolve(strict=False)
        if self.license_environment is not None and not self.license_environment:
            raise ValueError("license_environment must be non-empty or None")
        object.__setattr__(self, "supported_physics", supported)
        object.__setattr__(self, "lumapi_file", lumapi_file)
        object.__setattr__(self, "extra_arguments", tuple(map(str, self.extra_arguments)))

    def capability(self, environment: dict[str, str] | None = None) -> SolverCapability:
        executable = _resolve_executable(self.python_executable)
        if self.lumapi_file is not None:
            module_found = self.lumapi_file.is_file()
            module_reason = f"lumapi file {str(self.lumapi_file)!r} was not found"
        else:
            try:
                module_found = importlib.util.find_spec(self.module_name) is not None
            except (ImportError, ModuleNotFoundError, ValueError):
                module_found = False
            module_reason = f"Python module {self.module_name!r} is not discoverable"
        env = os.environ if environment is None else environment
        license_ready = self.license_environment is None or bool(env.get(self.license_environment))
        available = executable is not None and module_found and license_ready
        if executable is None:
            reason = f"Python executable {self.python_executable!r} was not found"
        elif not module_found:
            reason = module_reason
        elif not license_ready:
            reason = f"required license environment variable {self.license_environment!r} is not set"
        else:
            reason = None
        return SolverCapability(self.name, available, executable, self.supported_physics, reason)

    def build_command(self, job: MultiphysicsJob) -> tuple[str, ...]:
        _validate_supported(self.name, self.supported_physics, job)
        if job.input_file.suffix.lower() != ".py":
            raise ValueError("Lumerical DEVICE input_file must be a prepared Python script")
        executable = _resolve_executable(self.python_executable) or self.python_executable
        return (executable, *self.extra_arguments, str(job.input_file))


@dataclass(frozen=True)
class ExternalSolverAdapter:
    """Configurable non-shell adapter for licensed or site-local solvers.

    ``arguments`` is an argv template, not a shell command.  Supported tokens
    are ``{input}``, ``{workdir}``, and exact names from ``job.parameters``.
    ``license_environment`` only checks that a variable is configured; it does
    not contact a license server or consume a license during capability probes.
    """

    name: str
    executable: str
    supported_physics: frozenset[Physics]
    arguments: tuple[str, ...] = ("{input}",)
    license_environment: str | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("external solver name must be non-empty")
        if not self.executable:
            raise ValueError("external solver executable must be non-empty")
        supported = frozenset(
            value if isinstance(value, Physics) else Physics(value)
            for value in self.supported_physics
        )
        if not supported:
            raise ValueError("external solver must declare supported physics")
        arguments = tuple(str(value) for value in self.arguments)
        if self.license_environment is not None and not self.license_environment:
            raise ValueError("license_environment must be non-empty or None")
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "supported_physics", supported)
        object.__setattr__(self, "arguments", arguments)

    def capability(self, environment: dict[str, str] | None = None) -> SolverCapability:
        executable = _resolve_executable(self.executable)
        env = os.environ if environment is None else environment
        license_ready = self.license_environment is None or bool(env.get(self.license_environment))
        available = executable is not None and license_ready
        if executable is None:
            reason = f"executable {self.executable!r} was not found"
        elif not license_ready:
            reason = f"required license environment variable {self.license_environment!r} is not set"
        else:
            reason = None
        return SolverCapability(self.name, available, executable, self.supported_physics, reason)

    def build_command(self, job: MultiphysicsJob) -> tuple[str, ...]:
        _validate_supported(self.name, self.supported_physics, job)
        values = {
            "input": str(job.input_file),
            "workdir": str(job.workdir),
            **{name: str(value) for name, value in job.parameters.items()},
        }
        rendered: list[str] = []
        for raw_argument in self.arguments:
            argument = raw_argument
            for name, value in values.items():
                argument = argument.replace("{" + name + "}", value)
            if "{" in argument or "}" in argument:
                raise ValueError(f"unresolved command-template token in {raw_argument!r}")
            rendered.append(argument)
        executable = _resolve_executable(self.executable) or self.executable
        return (executable, *rendered)


def detect_capabilities(
    adapters,
    *,
    environment: dict[str, str] | None = None,
) -> dict[str, SolverCapability]:
    """Probe adapters without importing solver packages or launching processes."""
    capabilities: dict[str, SolverCapability] = {}
    for adapter in adapters:
        if adapter.name in capabilities:
            raise ValueError(f"duplicate solver adapter name {adapter.name!r}")
        capabilities[adapter.name] = adapter.capability(environment)
    return capabilities
