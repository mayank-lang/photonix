"""Tests for the release and runtime-diagnostics surface."""
from __future__ import annotations

import json
from importlib import metadata
from pathlib import Path

import pytest

import photonix as px
from photonix.__main__ import main
from photonix.diagnostics import DependencyStatus, RuntimeInfo, format_runtime_info, runtime_info


def test_runtime_info_is_immutable_and_serializable():
    info = runtime_info()

    assert info.photonix_version == px.__version__
    assert info.default_real_dtype in {"float32", "float64"}
    assert info.device_count >= 1
    assert info.x64_enabled == (info.default_real_dtype == "float64")
    assert set(info.optional_dependencies) == {
        "gdstk", "jax", "matplotlib", "meep", "meshio", "networkx", "skrf"
    }
    json.dumps(info.as_dict())

    with pytest.raises(TypeError):
        info.optional_dependencies["jax"] = DependencyStatus(False)  # type: ignore[index]


def test_runtime_info_format_has_text_and_json_forms():
    info = RuntimeInfo(
        photonix_version="1.2.3",
        python_version="3.12.0",
        python_implementation="CPython",
        platform="test-platform",
        backend="jax:cpu",
        x64_enabled=True,
        default_real_dtype="float64",
        device_count=2,
        optional_dependencies={
            "jax": DependencyStatus(True, "0.4.99"),
            "meep": DependencyStatus(False),
        },
        klayout_available=False,
    )

    text = format_runtime_info(info)
    assert "Photonix 1.2.3" in text
    assert "Backend: jax:cpu" in text
    assert "jax: 0.4.99" in text
    assert "meep: not installed" in text

    payload = json.loads(format_runtime_info(info, json_output=True))
    assert payload["device_count"] == 2
    assert payload["optional_dependencies"]["jax"]["available"] is True
    with pytest.raises(TypeError):
        info.optional_dependencies["jax"] = DependencyStatus(False)  # type: ignore[index]


def test_diagnostic_records_reject_inconsistent_state():
    with pytest.raises(ValueError, match="unavailable dependency"):
        DependencyStatus(False, "1.0")
    with pytest.raises(ValueError, match="device_count"):
        RuntimeInfo(
            photonix_version="1.2.3",
            python_version="3.12.0",
            python_implementation="CPython",
            platform="test-platform",
            backend="numpy",
            x64_enabled=True,
            default_real_dtype="float64",
            device_count=0,
            optional_dependencies={},
            klayout_available=False,
        )


def test_cli_emits_machine_readable_runtime_info(capsys):
    assert main(["info", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["photonix_version"] == px.__version__
    assert payload["backend"] == px.backend_name()


def test_distribution_is_marked_typed_and_version_is_synchronized():
    package_dir = Path(px.__file__).parent
    assert (package_dir / "py.typed").is_file()
    assert metadata.version("photonix") == px.__version__
