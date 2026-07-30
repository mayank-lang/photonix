"""Solver-neutral configuration and result contracts for multiphysics jobs."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

__all__ = ["Physics", "MultiphysicsJob", "SolverCapability", "SimulationResult"]


class Physics(str, Enum):
    """Declared equation family required by an external job."""

    THERMAL = "thermal"
    ELECTRICAL = "electrical"
    CARRIER = "carrier"
    ELECTROTHERMAL = "electrothermal"


def _physics_set(values) -> frozenset[Physics]:
    if isinstance(values, (str, Physics)):
        values = (values,)
    try:
        result = frozenset(value if isinstance(value, Physics) else Physics(value) for value in values)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in Physics)
        raise ValueError(f"physics entries must be one of: {allowed}") from exc
    if not result:
        raise ValueError("physics must contain at least one equation family")
    return result


@dataclass(frozen=True)
class MultiphysicsJob:
    """Prepared input deck/script plus execution and output expectations.

    Paths are resolved at construction so later execution does not depend on a
    changed Python current directory.  Scientific output interpretation remains
    solver-specific and is deliberately outside this transport contract.
    """

    name: str
    physics: frozenset[Physics]
    input_file: Path
    workdir: Path | None = None
    required_outputs: tuple[Path, ...] = ()
    parameters: dict[str, str | int | float | bool] = field(default_factory=dict)
    environment: dict[str, str] = field(default_factory=dict)
    timeout_s: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("job name must be a non-empty string")
        physics = _physics_set(self.physics)
        raw_input = Path(self.input_file).expanduser()
        if self.workdir is None:
            workdir = raw_input.parent if raw_input.is_absolute() else Path.cwd()
        else:
            workdir = Path(self.workdir).expanduser()
        workdir = workdir.resolve(strict=False)
        input_file = (
            raw_input.resolve(strict=False)
            if raw_input.is_absolute()
            else (workdir / raw_input).resolve(strict=False)
        )
        outputs = tuple(
            path.resolve(strict=False) if path.is_absolute() else (workdir / path).resolve(strict=False)
            for path in (Path(item).expanduser() for item in self.required_outputs)
        )
        if len(outputs) != len(set(outputs)):
            raise ValueError("required_outputs must not contain duplicates")
        parameters = dict(self.parameters)
        for key, value in parameters.items():
            if not isinstance(key, str) or not key or key != key.strip():
                raise ValueError("parameter names must be non-empty and trimmed")
            if not isinstance(value, (str, int, float, bool)):
                raise ValueError(f"parameter {key!r} must be a string, integer, float, or bool")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"parameter {key!r} must be finite")
        environment = dict(self.environment)
        if any(not isinstance(key, str) or not key or not isinstance(value, str)
               for key, value in environment.items()):
            raise ValueError("environment must map non-empty string names to string values")
        if self.timeout_s is not None and (not math.isfinite(self.timeout_s) or self.timeout_s <= 0):
            raise ValueError("timeout_s must be positive and finite, or None")
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "physics", physics)
        object.__setattr__(self, "input_file", input_file)
        object.__setattr__(self, "workdir", workdir)
        object.__setattr__(self, "required_outputs", outputs)
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "environment", environment)


@dataclass(frozen=True)
class SolverCapability:
    """Side-effect-free solver availability probe."""

    solver: str
    available: bool
    executable: str | None
    supported_physics: frozenset[Physics]
    reason: str | None = None


@dataclass(frozen=True)
class SimulationResult:
    """Unparsed process result and validated output-file presence."""

    solver: str
    job_name: str
    command: tuple[str, ...]
    cwd: Path
    returncode: int | None
    stdout: str
    stderr: str
    elapsed_s: float
    outputs: tuple[Path, ...]
    missing_outputs: tuple[Path, ...] = ()
    timed_out: bool = False

    @property
    def succeeded(self) -> bool:
        return self.returncode == 0 and not self.timed_out and not self.missing_outputs

    def output(self, name: str) -> Path:
        """Return one declared output by basename, rejecting missing/ambiguous names."""
        matches = [path for path in self.outputs if path.name == name]
        if len(matches) != 1:
            raise KeyError(f"expected exactly one output named {name!r}, found {len(matches)}")
        return matches[0]
