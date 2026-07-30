"""Vendor-neutral metadata for handing Photonix artifacts to external solvers.

This module describes an integration boundary; it does not bundle proprietary
SDKs, process decks, license credentials, or guessed command lines.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

__all__ = ["ExternalSolverHandoff"]

_SCHEMA = "photonix.external-solver-handoff/v1"


def _nonempty(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True)
class ExternalSolverHandoff:
    """JSON-serializable description of an external solver integration.

    At least one of ``required_executable`` and ``required_module`` must be
    supplied. Names identify prerequisites only; invocation and license setup
    remain the adapter/user's responsibility.
    """

    solver: str
    interface: str
    artifact_format: str
    required_executable: str | None = None
    required_module: str | None = None
    license_required: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "solver", _nonempty(self.solver, "solver"))
        object.__setattr__(self, "interface", _nonempty(self.interface, "interface"))
        object.__setattr__(self, "artifact_format", _nonempty(self.artifact_format, "artifact_format"))
        executable = None
        if self.required_executable is not None:
            executable = _nonempty(self.required_executable, "required_executable")
        module = None
        if self.required_module is not None:
            module = _nonempty(self.required_module, "required_module")
        if executable is None and module is None:
            raise ValueError("required_executable or required_module must be provided")
        if not isinstance(self.license_required, bool):
            raise ValueError("license_required must be a boolean")
        metadata = dict(self.metadata)
        try:
            json.dumps(metadata)
        except (TypeError, ValueError) as exc:
            raise ValueError("metadata must be JSON-serializable") from exc
        object.__setattr__(self, "required_executable", executable)
        object.__setattr__(self, "required_module", module)
        object.__setattr__(self, "metadata", metadata)

    def as_dict(self) -> dict[str, Any]:
        """Return the stable metadata contract for manifests or datasets."""
        return {
            "schema": _SCHEMA,
            "solver": self.solver,
            "interface": self.interface,
            "artifact_format": self.artifact_format,
            "required_executable": self.required_executable,
            "required_module": self.required_module,
            "license_required": self.license_required,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ExternalSolverHandoff:
        """Validate and reconstruct :meth:`as_dict` output."""
        if value.get("schema") != _SCHEMA:
            raise ValueError(f"unsupported external-solver handoff schema {value.get('schema')!r}")
        solver = value.get("solver")
        interface = value.get("interface")
        artifact_format = value.get("artifact_format")
        executable = value.get("required_executable")
        module = value.get("required_module")
        license_required = value.get("license_required")
        metadata = value.get("metadata", {})
        if not isinstance(solver, str) or not isinstance(interface, str) or not isinstance(artifact_format, str):
            raise ValueError("solver, interface, and artifact_format must be strings")
        if executable is not None and not isinstance(executable, str):
            raise ValueError("required_executable must be a string or null")
        if module is not None and not isinstance(module, str):
            raise ValueError("required_module must be a string or null")
        if not isinstance(license_required, bool):
            raise ValueError("license_required must be a boolean")
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be a JSON object")
        return cls(
            solver=solver,
            interface=interface,
            artifact_format=artifact_format,
            required_executable=executable,
            required_module=module,
            license_required=license_required,
            metadata=metadata,
        )
