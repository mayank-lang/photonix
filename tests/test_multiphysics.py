"""Pure multiphysics adapter contracts; no external solver installation required."""
from __future__ import annotations

import sys

import pytest

import photonix as px
import photonix.multiphysics as mp


def test_multiphysics_namespace_is_import_safe_and_exported():
    assert px.multiphysics is mp
    assert mp.ElmerAdapter().name == "elmer"
    assert mp.DevsimAdapter().name == "devsim"


def test_elmer_capability_and_command_without_elmer(monkeypatch, tmp_path):
    import photonix.multiphysics.adapters as adapters

    deck = tmp_path / "thermal.sif"
    deck.write_text("! prepared by test\n", encoding="utf-8")
    job = mp.MultiphysicsJob("thermal", {mp.Physics.THERMAL}, deck)
    monkeypatch.setattr(adapters.shutil, "which", lambda _name: None)
    adapter = mp.ElmerAdapter()
    capability = adapter.capability()
    assert not capability.available
    assert "not found" in capability.reason
    assert adapter.build_command(job) == ("ElmerSolver", str(deck.resolve()))
    with pytest.raises(mp.SolverUnavailableError, match="not found"):
        mp.run_job(adapter, job)


def test_devsim_capability_does_not_import_devsim(monkeypatch, tmp_path):
    import photonix.multiphysics.adapters as adapters

    script = tmp_path / "device.py"
    script.write_text("# prepared DEVSIM script\n", encoding="utf-8")
    monkeypatch.setattr(adapters.importlib.util, "find_spec", lambda _name: None)
    adapter = mp.DevsimAdapter(python_executable=sys.executable)
    capability = adapter.capability()
    assert not capability.available
    assert "not installed" in capability.reason
    job = mp.MultiphysicsJob("carrier", {mp.Physics.CARRIER}, script)
    assert adapter.build_command(job)[-1] == str(script.resolve())


def test_external_solver_license_probe_has_no_checkout_side_effect(monkeypatch):
    import photonix.multiphysics.adapters as adapters

    monkeypatch.setattr(adapters.shutil, "which", lambda _name: "/opt/vendor/bin/solver")
    adapter = mp.ExternalSolverAdapter(
        "vendor",
        "vendor-solver",
        frozenset({mp.Physics.ELECTROTHERMAL}),
        ("--batch", "{input}"),
        license_environment="VENDOR_LICENSE",
    )
    unavailable = adapter.capability({})
    available = adapter.capability({"VENDOR_LICENSE": "configured"})
    assert not unavailable.available and "VENDOR_LICENSE" in unavailable.reason
    assert available.available


def test_lumerical_device_probe_never_imports_lumapi(monkeypatch, tmp_path):
    import photonix.multiphysics.adapters as adapters

    script = tmp_path / "device_workflow.py"
    script.write_text("# prepared lumapi.DEVICE workflow\n", encoding="utf-8")
    calls = []

    def no_module_import(name):
        calls.append(name)
        return None

    monkeypatch.setattr(adapters.importlib.util, "find_spec", no_module_import)
    adapter = mp.LumericalDeviceAdapter(
        frozenset({mp.Physics.CARRIER}),
        python_executable=sys.executable,
        license_environment="ANSYSLMD_LICENSE_FILE",
    )
    capability = adapter.capability({"ANSYSLMD_LICENSE_FILE": "configured"})
    assert not capability.available
    assert calls == ["lumapi"]
    assert "not discoverable" in capability.reason
    job = mp.MultiphysicsJob("device", {mp.Physics.CARRIER}, script)
    assert adapter.build_command(job)[-1] == str(script.resolve())


def test_lumerical_device_accepts_explicit_installed_lumapi_file(tmp_path):
    lumapi_file = tmp_path / "lumapi.py"
    lumapi_file.write_text("# installation-owned API module placeholder\n", encoding="utf-8")
    adapter = mp.LumericalDeviceAdapter(
        frozenset({mp.Physics.ELECTROTHERMAL}),
        lumapi_file=lumapi_file,
        license_environment="ANSYSLMD_LICENSE_FILE",
    )
    capability = adapter.capability({"ANSYSLMD_LICENSE_FILE": "configured"})
    assert capability.available
    assert capability.executable


def test_external_command_template_is_argv_not_shell_text(tmp_path):
    deck = tmp_path / "case.in"
    deck.write_text("prepared input\n", encoding="utf-8")
    adapter = mp.ExternalSolverAdapter(
        "vendor",
        sys.executable,
        frozenset({mp.Physics.THERMAL}),
        ("--input", "{input}", "--label", "{label}"),
    )
    job = mp.MultiphysicsJob(
        "case", {mp.Physics.THERMAL}, deck, parameters={"label": "two words"}
    )
    command = adapter.build_command(job)
    assert command[1:] == ("--input", str(deck.resolve()), "--label", "two words")


def test_runner_executes_generic_adapter_and_validates_outputs(tmp_path):
    script = tmp_path / "fake_solver.py"
    script.write_text(
        "from pathlib import Path\n"
        "import sys\n"
        "Path(sys.argv[1]).write_text('solver-owned output', encoding='utf-8')\n"
        "print('fake solver complete')\n",
        encoding="utf-8",
    )
    adapter = mp.ExternalSolverAdapter(
        "fake-carrier",
        sys.executable,
        frozenset({mp.Physics.CARRIER}),
        ("{input}", "{output}"),
    )
    job = mp.MultiphysicsJob(
        "carrier-case",
        {mp.Physics.CARRIER},
        script,
        workdir=tmp_path,
        required_outputs=("result.dat",),
        parameters={"output": "result.dat"},
        timeout_s=10.0,
    )
    result = mp.run_job(adapter, job)
    assert result.succeeded
    assert result.returncode == 0
    assert "fake solver complete" in result.stdout
    assert result.output("result.dat").read_text(encoding="utf-8") == "solver-owned output"


def test_runner_reports_missing_required_output(tmp_path):
    script = tmp_path / "no_output.py"
    script.write_text("print('done')\n", encoding="utf-8")
    adapter = mp.ExternalSolverAdapter(
        "fake", sys.executable, frozenset({mp.Physics.ELECTRICAL}), ("{input}",)
    )
    job = mp.MultiphysicsJob(
        "missing-output",
        {mp.Physics.ELECTRICAL},
        script,
        workdir=tmp_path,
        required_outputs=("expected.dat",),
    )
    with pytest.raises(mp.SolverExecutionError) as error:
        mp.run_job(adapter, job)
    assert error.value.result.returncode == 0
    assert error.value.result.missing_outputs == ((tmp_path / "expected.dat").resolve(),)


def test_job_and_adapter_reject_invalid_contracts(tmp_path):
    deck = tmp_path / "case.in"
    deck.write_text("input\n", encoding="utf-8")
    with pytest.raises(ValueError, match="physics"):
        mp.MultiphysicsJob("empty", set(), deck)
    adapter = mp.ExternalSolverAdapter(
        "thermal-only", sys.executable, frozenset({mp.Physics.THERMAL})
    )
    carrier_job = mp.MultiphysicsJob("carrier", {mp.Physics.CARRIER}, deck)
    with pytest.raises(ValueError, match="does not declare support"):
        adapter.build_command(carrier_job)


def test_detect_capabilities_rejects_duplicate_names():
    one = mp.ExternalSolverAdapter("same", sys.executable, frozenset({mp.Physics.THERMAL}))
    two = mp.ExternalSolverAdapter("same", sys.executable, frozenset({mp.Physics.CARRIER}))
    with pytest.raises(ValueError, match="duplicate"):
        mp.detect_capabilities((one, two))
