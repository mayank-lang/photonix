"""Tests for vendor-neutral, license-safe external solver handoff metadata."""
from __future__ import annotations

import json

import pytest

from photonix.interop import ExternalSolverHandoff


def test_external_solver_handoff_is_json_serializable_and_roundtrips():
    handoff = ExternalSolverHandoff(
        solver="external-em-solver",
        interface="file-exchange",
        artifact_format="Touchstone 1.0 RI",
        required_module="vendor_api",
        license_required=True,
        metadata={"ports": ["o1", "o2"], "reference_impedance_ohm": 50.0},
    )
    encoded = json.dumps(handoff.as_dict())
    restored = ExternalSolverHandoff.from_dict(json.loads(encoded))
    assert restored == handoff
    assert restored.as_dict()["schema"] == "photonix.external-solver-handoff/v1"


def test_external_solver_handoff_requires_declared_prerequisite_and_safe_metadata():
    with pytest.raises(ValueError, match="required_executable or required_module"):
        ExternalSolverHandoff("solver", "files", "OASIS")
    with pytest.raises(ValueError, match="JSON-serializable"):
        ExternalSolverHandoff(
            "solver", "files", "OASIS", required_executable="solver-bin", metadata={"bad": object()}
        )
