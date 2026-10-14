"""The ``photonix`` command line: output, frozen JSON schemas, exit codes and equivalence to the library.

Every ``--json`` document is validated against its frozen schema in
``tests/data/cli_schema/`` (with ``jsonschema`` when installed, and always with
the small validator below, so the minimal-install CI job checks them too).
Numbers the CLI reports are compared with direct library calls on identical
arguments, so the command line is shown to add no numerics of its own.
"""
from __future__ import annotations

import importlib.util
import inspect
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from conftest import requires_gdstk

import photonix as px
from photonix.cli import ExitCode, main, parse_params, parse_wavelengths
from photonix.cli._common import CliError, jsonable, natural_key
from photonix.cli.commands import validate as validate_cmd

DATA = Path(__file__).parent / "data"
SCHEMAS = DATA / "cli_schema"
FIXTURES = DATA / "cli"


# --------------------------------------------------------------------------- #
# Schema validation
# --------------------------------------------------------------------------- #
_TYPES = {
    "object": dict, "array": list, "string": str, "boolean": bool, "null": type(None),
    "number": (int, float), "integer": int,
}


def _is(value, name: str) -> bool:
    if name in ("number", "integer") and isinstance(value, bool):
        return False
    return isinstance(value, _TYPES[name])


def _validate(value, schema: dict, path: str = "$") -> None:
    """Minimal JSON Schema subset used by the frozen CLI schemas."""
    if "anyOf" in schema:
        errors = []
        for option in schema["anyOf"]:
            try:
                _validate(value, option, path)
                break
            except AssertionError as exc:
                errors.append(str(exc))
        else:
            raise AssertionError(f"{path}: matches no anyOf option ({'; '.join(errors)})")
    if "const" in schema:
        assert value == schema["const"], f"{path}: {value!r} != const {schema['const']!r}"
    if "enum" in schema:
        assert value in schema["enum"], f"{path}: {value!r} not in {schema['enum']!r}"
    if "type" in schema:
        types = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        assert any(_is(value, t) for t in types), f"{path}: {value!r} is not {types}"
    if "minimum" in schema and _is(value, "number"):
        assert value >= schema["minimum"], f"{path}: {value} < {schema['minimum']}"
    if isinstance(value, dict):
        props = schema.get("properties", {})
        for key in schema.get("required", []):
            assert key in value, f"{path}: missing required key {key!r}"
        extra = schema.get("additionalProperties", True)
        for key, item in value.items():
            if key in props:
                _validate(item, props[key], f"{path}.{key}")
            elif extra is False:
                raise AssertionError(f"{path}: unexpected key {key!r} (update the frozen schema deliberately)")
            elif isinstance(extra, dict):
                _validate(item, extra, f"{path}.{key}")
    if isinstance(value, list):
        if "minItems" in schema:
            assert len(value) >= schema["minItems"], f"{path}: fewer than {schema['minItems']} items"
        if "maxItems" in schema:
            assert len(value) <= schema["maxItems"], f"{path}: more than {schema['maxItems']} items"
        if "items" in schema:
            for i, item in enumerate(value):
                _validate(item, schema["items"], f"{path}[{i}]")


def _schema(name: str) -> dict:
    return json.loads((SCHEMAS / f"{name}.json").read_text(encoding="utf-8"))


def check_schema(payload: dict) -> None:
    """Validate a payload against the frozen schema its ``schema`` field names."""
    sid = payload["schema"]
    prefix, name, version = sid.split("/")
    assert prefix == "photonix.cli" and version == "1", sid
    schema = _schema(name)
    _validate(payload, schema)
    if importlib.util.find_spec("jsonschema") is not None:
        import jsonschema

        jsonschema.Draft202012Validator(schema).validate(payload)


def test_frozen_schemas_are_valid_and_the_validator_has_teeth():
    names = sorted(p.stem for p in SCHEMAS.glob("*.json"))
    assert names == sorted(["error", "gds-inspect", "gds-precheck", "info", "mode", "models", "netlist-simulate",
                            "sparams-check", "sparams-convert", "sweep", "validate"])
    for name in names:
        schema = _schema(name)
        assert schema["properties"]["schema"]["const"] == f"photonix.cli/{name}/1"
        if importlib.util.find_spec("jsonschema") is not None:
            import jsonschema

            jsonschema.Draft202012Validator.check_schema(schema)
    good = {"schema": "photonix.cli/sparams-convert/1", "input": "a.s2p", "output": "b.npz", "format": "npz",
            "ports": ["o1", "o2"], "n_wavelengths": 3}
    check_schema(good)
    for bad in ({**good, "format": "xlsx"}, {**good, "n_wavelengths": -1}, {**good, "extra": 1},
                {k: v for k, v in good.items() if k != "ports"}, {**good, "n_wavelengths": True}):
        with pytest.raises(AssertionError):
            _validate(bad, _schema("sparams-convert"))


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def run(capsys, *argv: str) -> tuple[int, str, str]:
    code = main(list(argv))
    out, err = capsys.readouterr()
    return code, out, err


def run_json(capsys, *argv: str) -> tuple[int, dict]:
    code, out, err = run(capsys, *argv, "--json")
    payload = json.loads(out)
    check_schema(payload)
    return code, payload


def _dense(ds_ports, sdict):
    """S[out, in] matrix of an SDict in the given port order (direct, independent of the CLI)."""
    n = len(ds_ports)
    sample = np.asarray(next(iter(sdict.values())))
    out = np.zeros((*sample.shape, n, n), dtype=complex)
    for (p_in, p_out), value in sdict.items():
        out[..., ds_ports.index(p_out), ds_ports.index(p_in)] = np.asarray(value)
    return out


# --------------------------------------------------------------------------- #
# info, entry points, exit codes
# --------------------------------------------------------------------------- #
def test_info_text_and_json_match_runtime_info(capsys):
    code, payload = run_json(capsys, "info")
    assert code == ExitCode.OK
    expected = px.runtime_info().as_dict()
    assert {k: v for k, v in payload.items() if k != "schema"} == json.loads(json.dumps(expected))
    code, out, _ = run(capsys, "info")
    assert code == 0 and out.startswith(f"Photonix {px.__version__}") and "Backend:" in out
    # 0.1 behaviour: no command means `info`, and a leading --json still works.
    assert run(capsys)[1] == out
    code, out, _ = run(capsys, "--json")
    assert code == 0 and json.loads(out)["backend"] == px.backend_name()


def test_help_version_and_usage_errors(capsys):
    code, out, _ = run(capsys, "--version")
    assert code == 0 and out.strip() == f"photonix {px.__version__}"
    code, out, _ = run(capsys, "--help")
    assert code == 0 and "exit codes" in out and "sparams" in out and "validate" in out
    code, out, _ = run(capsys, "sweep", "--help")
    assert code == 0 and "--wl" in out
    for argv in (["frobnicate"], ["sweep"], ["sweep", "straight", "--wl", "1.6:1.5:3"],
                 ["sweep", "straight", "--wl", "abc"], ["mode", "--modes", "0"], ["sparams"],
                 ["sweep", "straight", "-p", "length"]):
        code, _out, err = run(capsys, *argv)
        assert code == ExitCode.USAGE, argv
        assert "photonix: error:" in err and "Traceback" not in err, argv


def test_errors_under_json_emit_an_error_document(capsys):
    code, out, err = run(capsys, "sweep", "no_such_model", "--json")
    assert code == ExitCode.USAGE and "did you mean" not in err
    doc = json.loads(out)
    check_schema(doc)
    assert doc["exit_code"] == 2 and doc["command"] == "sweep" and "no_such_model" in doc["error"]["message"]
    code, out, err = run(capsys, "sweep", "straigt", "--json")
    assert "did you mean straight" in json.loads(out)["error"]["message"]


def test_unexpected_exception_is_exit_4_with_traceback(capsys, monkeypatch):
    def boom(args):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(validate_cmd, "run", boom)   # register() binds the handler when the parser is built
    code, out, err = run(capsys, "validate", "--json")
    assert code == ExitCode.INTERNAL_ERROR
    assert "Traceback" in err and "synthetic failure" in err
    doc = json.loads(out)
    check_schema(doc)
    assert doc["exit_code"] == 4


def test_console_script_and_python_dash_m_are_equivalent():
    root = Path(px.__file__).resolve().parents[1]
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    if pyproject.is_file():
        text = pyproject.read_text(encoding="utf-8")
        assert 'photonix = "photonix.__main__:main"' in text
    from photonix.__main__ import main as script_main

    assert inspect.signature(script_main).parameters.keys() == {"argv"}
    env = {**__import__("os").environ, "PYTHONPATH": str(root)}
    commands = [
        [sys.executable, "-m", "photonix", "sparams", "check", str(FIXTURES / "passive.s2p"), "--json"],
        [sys.executable, "-c", "import sys; from photonix.__main__ import main; sys.exit(main())",
         "sparams", "check", str(FIXTURES / "passive.s2p"), "--json"],
    ]
    results = [subprocess.run(c, capture_output=True, text=True, env=env, timeout=300) for c in commands]
    assert [r.returncode for r in results] == [0, 0], [r.stderr for r in results]
    assert json.loads(results[0].stdout) == json.loads(results[1].stdout)
    bad = subprocess.run([sys.executable, "-m", "photonix", "sparams", "check", str(FIXTURES / "nonpassive.s2p")],
                         capture_output=True, text=True, env=env, timeout=300)
    assert bad.returncode == ExitCode.CHECK_FAILED and "FAIL" in bad.stdout


# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #
def test_models_lists_the_registry_with_signature_defaults(capsys):
    code, payload = run_json(capsys, "models")
    assert code == 0
    by_name = {m["name"]: m for m in payload["models"]}
    assert set(by_name) == set(px.components.MODELS) and payload["count"] == len(by_name)
    for name, fn in px.components.MODELS.items():
        entry = by_name[name]
        assert entry["error"] is None, (name, entry["error"])
        sig = inspect.signature(fn).parameters
        assert [p["name"] for p in entry["parameters"]] == [n for n in sig if n != "wl"]
        for p in entry["parameters"]:
            assert p["default"] == jsonable(sig[p["name"]].default)
        ports = sorted({q for pair in fn(wl=1.55) for q in pair}, key=natural_key)
        assert entry["ports"] == ports
    assert by_name["straight"]["waveguide_parameters"] == list(px.components.WAVEGUIDE_PARAMETERS)
    code, out, _ = run(capsys, "models", "--no-ports")
    assert code == 0 and "straight" in out and "ports]" not in out


def test_models_detail_and_unknown(capsys):
    code, payload = run_json(capsys, "models", "bragg_grating")
    assert code == 0 and payload["count"] == 1
    entry = payload["models"][0]
    assert entry["ports"] == ["o1", "o2"] and "tanh" in entry["doc"]
    code, out, _ = run(capsys, "models", "ring_single")
    assert code == 0 and "radius = " in out and "ports: o1 o2" in out
    code, _out, err = run(capsys, "models", "ring_singel")
    assert code == 2 and "did you mean ring_single" in err


# --------------------------------------------------------------------------- #
# sweep and S-parameter files
# --------------------------------------------------------------------------- #
def test_sweep_json_equals_the_direct_model_evaluation(capsys):
    code, payload = run_json(capsys, "sweep", "bragg_grating", "--wl", "1.54:1.56:41", "-p", "dn=0.02",
                             "-p", "n_periods=250", "-p", "apodization=gaussian", "-p", "n_sections=20")
    assert code == 0 and payload["output"] is None
    wl = np.linspace(1.54, 1.56, 41)
    direct = px.components.bragg_grating(wl=wl, dn=0.02, n_periods=250, apodization="gaussian", n_sections=20)
    data = payload["data"]
    assert data["ports"] == payload["ports"] == ["o1", "o2"]
    s = np.asarray(data["s_real"]) + 1j * np.asarray(data["s_imag"])
    np.testing.assert_allclose(data["wavelengths_um"], wl, rtol=0, atol=0)
    np.testing.assert_allclose(s, _dense(data["ports"], direct), rtol=0, atol=1e-15)
    assert payload["settings"] == {"dn": 0.02, "n_periods": 250, "apodization": "gaussian", "n_sections": 20}
    assert payload["physicality"]["passive"] and payload["physicality"]["reciprocal"]
    reflect = next(p for p in payload["paths"] if p["in"] == "o1" and p["out"] == "o1")
    power = np.abs(np.asarray(direct[("o1", "o1")])) ** 2
    assert reflect["max_db"] == pytest.approx(10 * math.log10(power.max()), abs=1e-12)


def test_sweep_writes_every_format_losslessly(capsys, tmp_path):
    wl = np.linspace(1.5, 1.6, 11)
    direct = px.components.mmi_nxn(wl=wl, n=3)
    ports = sorted({q for pair in direct for q in pair}, key=natural_key)
    expected = _dense(ports, direct)
    for name in ("m.s6p", "m.npz", "m.json"):
        code, payload = run_json(capsys, "sweep", "mmi_nxn", "-p", "n=3", "--wl", "1.5:1.6:11", "-o",
                                 str(tmp_path / name))
        assert code == 0 and payload["data"] is None and payload["output"]["path"] == str(tmp_path / name)
        ds = px.cli._common.load_dataset(tmp_path / name)
        assert list(ds.ports) == ports
        np.testing.assert_allclose(ds.wavelengths, wl, rtol=1e-15)
        np.testing.assert_allclose(ds.s, expected, rtol=0, atol=1e-15)
    code, payload = run_json(capsys, "sweep", "mmi_nxn", "-p", "n=3", "--wl", "1.5:1.6:11", "-o",
                             str(tmp_path / "m.csv"))
    assert code == 0 and payload["output"]["format"] == "csv"
    header = (tmp_path / "m.csv").read_text().splitlines()[0].split(",")
    table = np.loadtxt(tmp_path / "m.csv", delimiter=",", skiprows=1)
    assert header[0] == "wl_um" and len(header) == 1 + 2 * 36
    col = header.index("o1->o5:re")
    np.testing.assert_allclose(table[:, col] + 1j * table[:, col + 1], expected[:, ports.index("o5"), 0], atol=1e-16)
    code, _out, err = run(capsys, "sweep", "mmi_nxn", "-p", "n=3", "-o", str(tmp_path / "m.s4p"))
    assert code == 2 and "use the suffix .s6p" in err
    code, _out, err = run(capsys, "sweep", "straight", "-o", str(tmp_path / "missing_dir" / "x.s2p"))
    assert code == 2 and "does not exist" in err
    code, _out, err = run(capsys, "sweep", "straight", "-o", str(tmp_path / "x.xlsx"))
    assert code == 2 and "unsupported file type" in err


def test_sweep_orders_ports_naturally(capsys):
    code, payload = run_json(capsys, "sweep", "awg", "-p", "n_channels=10", "--wl", "1.55")
    assert code == 0 and payload["ports"] == [f"o{i}" for i in range(1, 12)]


def test_sweep_then_check_round_trip_returns_zero(capsys, tmp_path):
    target = str(tmp_path / "x.s2p")
    assert run(capsys, "sweep", "straight", "-p", "length=25", "-p", "loss_db_cm=3", "-o", target)[0] == 0
    code, payload = run_json(capsys, "sparams", "check", target)
    assert code == 0 and payload["ok"]
    t = abs(complex(px.components.straight(wl=1.55, length=25.0, loss_db_cm=3.0)[("o1", "o2")]))
    assert payload["max_singular_value"] == pytest.approx(t, rel=1e-12)


def test_sweep_with_pdk_binds_the_cross_section(capsys):
    code, payload = run_json(capsys, "sweep", "straight", "--pdk", "demo", "--wl", "1.55", "-p", "length=1000")
    assert code == 0
    pdk = px.pdk.demo_pdk()
    direct = pdk.models()["straight"](wl=np.asarray([1.55]), length=1000.0)
    s = complex(payload["data"]["s_real"][0][1][0], payload["data"]["s_imag"][0][1][0])
    assert s == pytest.approx(complex(np.asarray(direct[("o1", "o2")])[0]), abs=1e-15)
    assert abs(s) ** 2 == pytest.approx(10 ** (-2.0 * 0.1 / 10), rel=1e-12)  # 2 dB/cm over 1 mm
    assert run(capsys, "sweep", "straight", "--xs", "strip")[0] == 2
    assert run(capsys, "sweep", "straight", "--pdk", "nope")[0] == 2
    assert run(capsys, "sweep", "straight", "--pdk", "demo", "--xs", "rib")[0] == 2


def test_sparams_check_exit_codes_on_handwritten_fixtures(capsys):
    code, payload = run_json(capsys, "sparams", "check", str(FIXTURES / "passive.s2p"))
    assert code == 0 and payload["ok"] and payload["ports"] == ["o1", "o2"] and payload["n_wavelengths"] == 3
    assert payload["max_singular_value"] == pytest.approx(abs(0.1 + 0.9j), rel=1e-15)
    c = 299792458e6
    assert payload["wl_min_um"] == pytest.approx(c / 196e12, rel=1e-15)
    code, payload = run_json(capsys, "sparams", "check", str(FIXTURES / "nonpassive.s2p"))
    assert code == ExitCode.CHECK_FAILED and not payload["ok"]
    passivity = payload["checks"][0]
    assert passivity["name"] == "passivity" and not passivity["passed"]
    assert passivity["value"] == pytest.approx(0.05, rel=1e-12)
    assert passivity["at_wl_um"] == pytest.approx(c / 193e12, rel=1e-12)
    # A tolerance above the violation lets it pass; losslessness gates only on request.
    assert run(capsys, "sparams", "check", str(FIXTURES / "nonpassive.s2p"), "--passivity-atol", "0.06")[0] == 0
    assert run(capsys, "sparams", "check", str(FIXTURES / "passive.s2p"), "--lossless")[0] == 1
    code, out, _ = run(capsys, "sparams", "check", str(FIXTURES / "nonpassive.s2p"))
    assert code == 1 and "passivity     FAIL" in out and out.rstrip().endswith("FAILED")


def test_sparams_check_reciprocity_gate(capsys, tmp_path):
    s = np.zeros((3, 2, 2), dtype=complex)
    s[:, 1, 0] = 0.8          # o1 -> o2
    s[:, 0, 1] = 0.2          # o2 -> o1: an isolator-like, non-reciprocal device
    ds = px.SParameterDataset(np.array([1.5, 1.55, 1.6]), ("o1", "o2"), s)
    ds.save_npz(tmp_path / "iso.npz")
    code, payload = run_json(capsys, "sparams", "check", str(tmp_path / "iso.npz"))
    assert code == 1 and payload["checks"][1]["value"] == pytest.approx(0.6)
    code, payload = run_json(capsys, "sparams", "check", str(tmp_path / "iso.npz"), "--no-reciprocity")
    assert code == 0 and payload["ok"] and not payload["checks"][1]["passed"] and not payload["checks"][1]["gated"]


def test_sparams_convert_round_trip_and_errors(capsys, tmp_path):
    src = FIXTURES / "passive.s2p"
    original = px.SParameterDataset.load_touchstone(src)
    chain = [src, tmp_path / "a.npz", tmp_path / "b.json", tmp_path / "c.s2p", tmp_path / "d.npz"]
    for a, b in zip(chain, chain[1:], strict=False):
        code, payload = run_json(capsys, "sparams", "convert", str(a), str(b))
        assert code == 0 and payload["n_wavelengths"] == 3
    final = px.SParameterDataset.load_npz(chain[-1])
    np.testing.assert_allclose(final.wavelengths, original.wavelengths, rtol=1e-15)
    np.testing.assert_array_equal(final.s, original.s)
    assert final.ports == original.ports
    assert run(capsys, "sparams", "convert", str(src), str(tmp_path / "e.csv"))[0] == 0
    code, _out, err = run(capsys, "sparams", "convert", str(tmp_path / "e.csv"), str(tmp_path / "f.npz"))
    assert code == 2 and "output-only" in err
    (tmp_path / "broken.s2p").write_text("# GHZ S RI R 50\n1 2 3\n")
    code, _out, err = run(capsys, "sparams", "check", str(tmp_path / "broken.s2p"))
    assert code == 2 and "truncated" in err
    (tmp_path / "other.json").write_text('{"format": "something-else"}')
    assert run(capsys, "sparams", "check", str(tmp_path / "other.json"))[0] == 2


# --------------------------------------------------------------------------- #
# mode
# --------------------------------------------------------------------------- #
def test_mode_matches_the_library_to_machine_precision(capsys):
    code, payload = run_json(capsys, "mode")
    assert code == 0
    assert payload["modes"][0]["n_eff"] == pytest.approx(px.em.n_eff_fullvector(), abs=1e-12)
    args = dict(wl=1.31, width=0.45, thickness=0.22, resolution=24, margin=1.2)
    code, payload = run_json(capsys, "mode", "--wl", "1.31", "--width", "0.45", "--resolution", "24",
                             "--margin", "1.2", "--core", "Si", "--clad", "SiO2", "--modes", "2")
    n_si, n_ox = float(px.em.silicon(1.31)), float(px.em.silica(1.31))
    direct = px.em.solve_modes_fullvector(n_core=n_si, n_clad=n_ox, num_modes=2, **args)
    assert payload["inputs"]["n_core"] == pytest.approx(n_si, abs=0) and payload["inputs"]["n_clad"] == n_ox
    for m, n, te, guided in zip(payload["modes"], direct.n_eff, direct.te_fraction, direct.guided, strict=True):
        assert m["n_eff"] == pytest.approx(float(np.real(n)), abs=1e-12)
        assert m["te_fraction"] == pytest.approx(float(te), abs=1e-12) and m["guided"] == bool(guided)
    assert [m["polarization"] for m in payload["modes"]] == ["TE-like", "TM-like"]


def test_mode_from_the_pdk_layer_stack(capsys):
    code, payload = run_json(capsys, "mode", "--pdk", "demo", "--resolution", "20")
    assert code == 0 and payload["inputs"]["cross_section"] == "strip" and payload["inputs"]["layer"] == [1, 0]
    direct = px.pdk.demo_pdk().layer_stack.solve_modes({(1, 0): 0.5}, resolution=20)
    assert payload["modes"][0]["n_eff"] == pytest.approx(direct.neff0, abs=1e-12)
    for argv in (["mode", "--pdk", "demo", "--thickness", "0.3"], ["mode", "--xs", "strip"],
                 ["mode", "--pdk", "demo", "--xs", "rib"], ["mode", "--core", "1.4", "--clad", "1.5"],
                 ["mode", "--core", "unobtainium"], ["mode", "--width", "-1"]):
        assert run(capsys, *argv)[0] == 2, argv


def test_mode_grid_study_reports_the_library_estimate(capsys):
    code, payload = run_json(capsys, "mode", "--resolution", "12", "--converge")
    conv = payload["convergence"]
    assert conv["resolutions"] == [12, 18, 27]
    values = [px.em.n_eff_fullvector(resolution=r) for r in (12, 18, 27)]
    np.testing.assert_allclose(conv["values"], values, rtol=0, atol=1e-12)
    study = px.em.estimate_convergence([12, 18, 27], values, rtol=1e-3)
    # Non-finite numbers are encoded as strings in strict JSON ("nan" when the study cannot extrapolate).
    assert conv["extrapolated"] == jsonable(float(study.extrapolated)) or \
        conv["extrapolated"] == pytest.approx(float(study.extrapolated), abs=1e-12)
    assert conv["converged"] == study.converged and code == 0
    code, payload = run_json(capsys, "mode", "--resolution", "12", "--require-converged")
    assert code == (0 if study.converged else 1)


# --------------------------------------------------------------------------- #
# gds
# --------------------------------------------------------------------------- #
@pytest.fixture
def gds_files(tmp_path):
    pdk = px.pdk.demo_pdk()
    clean = pdk.get_layout("ring_single", gap=0.2)
    bad = pdk.get_layout("ring_single", gap=0.1)
    px.layout.write_gds(clean, tmp_path / "clean.gds")
    px.layout.write_gds(bad, tmp_path / "bad.gds")
    return tmp_path, clean, bad, pdk


@requires_gdstk
def test_gds_inspect_reports_cells_layers_and_bbox(capsys, gds_files):
    tmp, clean, _bad, _pdk = gds_files
    code, payload = run_json(capsys, "gds", "inspect", str(tmp / "clean.gds"))
    assert code == 0 and payload["top_cells"] == ["ring_single"] and payload["cell"] == "ring_single"
    # GDSII snaps to the 1 nm database grid and gdstk fractures polygons above 199 vertices on write,
    # so compare the bbox to the grid and the drawn area (fracturing preserves it), not polygon counts.
    np.testing.assert_allclose(payload["bbox_um"], clean.bbox(), atol=0.5e-3)
    assert [(la["layer"], la["datatype"]) for la in payload["layers"]] == sorted(clean.layers)

    def shoelace(p):
        x, y = p[:, 0], p[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    area = sum(shoelace(pts) for pts, _layer in clean.get_polygons())
    assert sum(la["drawn_area_um2"] for la in payload["layers"]) == pytest.approx(area, rel=1e-4)
    assert payload["labels"] == ["o1", "o2"] and payload["unit_m"] == 1e-6
    code, out, _ = run(capsys, "gds", "inspect", str(tmp / "clean.gds"))
    assert code == 0 and "layer 1/0" in out


@requires_gdstk
def test_gds_inspect_converts_library_units_and_needs_a_single_top(capsys, tmp_path):
    import gdstk

    lib = gdstk.Library(unit=1e-9, precision=1e-12)          # coordinates in nm
    a = lib.new_cell("A")
    a.add(gdstk.rectangle((0, 0), (2000, 500), layer=3, datatype=1))
    lib.new_cell("B").add(gdstk.rectangle((0, 0), (100, 100), layer=1))
    lib.write_gds(str(tmp_path / "nm.gds"))
    code, _out, err = run(capsys, "gds", "inspect", str(tmp_path / "nm.gds"))
    assert code == 2 and "2 top cells" in err and "--cell" in err
    code, payload = run_json(capsys, "gds", "inspect", str(tmp_path / "nm.gds"), "--cell", "A")
    assert code == 0 and payload["bbox_um"] == [[0.0, 0.0], [2.0, 0.5]]
    assert payload["layers"] == [{"layer": 3, "datatype": 1, "polygons": 1, "drawn_area_um2": pytest.approx(1.0)}]
    assert run(capsys, "gds", "inspect", str(tmp_path / "nm.gds"), "--cell", "Z")[0] == 2
    (tmp_path / "junk.gds").write_bytes(b"not a gds file")
    assert run(capsys, "gds", "inspect", str(tmp_path / "junk.gds"))[0] == 2
    assert run(capsys, "gds", "inspect", str(tmp_path / "absent.gds"))[0] == 2


@requires_gdstk
def test_gds_precheck_clean_and_violating(capsys, gds_files):
    tmp, clean, bad, pdk = gds_files
    code, payload = run_json(capsys, "gds", "precheck", str(tmp / "clean.gds"), "--pdk", "demo")
    assert code == 0 and payload["clean"] and payload["violation_count"] == 0
    assert pdk.drc_precheck(clean).clean
    code, payload = run_json(capsys, "gds", "precheck", str(tmp / "bad.gds"), "--pdk", "demo")
    assert code == ExitCode.CHECK_FAILED and not payload["clean"]
    expected = pdk.drc_precheck(bad).summary()
    assert {(s["rule"], tuple(s["layer"])): s["count"] for s in payload["summary"]} == expected
    assert payload["violation_count"] == sum(expected.values())
    code, out, _ = run(capsys, "gds", "precheck", str(tmp / "bad.gds"), "--pdk", "demo")
    assert code == 1 and "min_space" in out


@requires_gdstk
def test_gds_precheck_rules_file(capsys, gds_files):
    tmp, clean, _bad, _pdk = gds_files
    rules = tmp / "rules.json"
    rules.write_text(json.dumps({"format": "photonix.drc_rules", "version": 1,
                                 "rules": {"1/0": {"min_width": 0.6}, "11": {"min_space": 2.0}}}))
    code, payload = run_json(capsys, "gds", "precheck", str(tmp / "clean.gds"), "--rules", str(rules),
                             "--max-violations", "1")
    direct = px.layout.run_drc_precheck(clean, {(1, 0): {"min_width": 0.6}})
    assert code == 1 and payload["violation_count"] == len(direct.violations) >= 2
    assert len(payload["violations"]) == 1 and payload["truncated"]
    assert {"layer": [11, 0], "rule": "min_space", "value": 2.0} in payload["rules"]
    # The file overrides the PDK rule on the same layer and rule.
    rules.write_text(json.dumps({"1/0": {"min_space": 0.05}}))
    code, payload = run_json(capsys, "gds", "precheck", str(tmp / "clean.gds"), "--pdk", "demo", "--rules",
                             str(rules))
    assert code == 0 and {"layer": [1, 0], "rule": "min_space", "value": 0.05} in payload["rules"]
    for body in ('{"1/0": {"min_widht": 1}}', '{"x/y": {"min_width": 1}}', '{"1/0": {"min_width": -1}}', "[]",
                 "not json"):
        rules.write_text(body)
        code, _out, err = run(capsys, "gds", "precheck", str(tmp / "clean.gds"), "--rules", str(rules))
        assert code == 2 and "Traceback" not in err, body
    assert run(capsys, "gds", "precheck", str(tmp / "clean.gds"))[0] == 2   # no rules given


def test_gds_commands_exit_3_without_gdstk(capsys, monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "gdstk", None)      # makes `import gdstk` raise ImportError
    (tmp_path / "x.gds").write_bytes(b"")
    for argv in (["gds", "inspect", str(tmp_path / "x.gds")],
                 ["gds", "precheck", str(tmp_path / "x.gds"), "--pdk", "demo"]):
        code, out, err = run(capsys, *argv)
        assert code == ExitCode.MISSING_DEPENDENCY, argv
        assert "Traceback" not in err and "pip install 'photonix[layout]'" in err
    code, out, err = run(capsys, "gds", "inspect", str(tmp_path / "x.gds"), "--json")
    doc = json.loads(out)
    check_schema(doc)
    assert doc["exit_code"] == 3 and "gdstk" in doc["error"]["message"]


# --------------------------------------------------------------------------- #
# netlist simulate
# --------------------------------------------------------------------------- #
def _netlist(tmp_path) -> Path:
    nl = px.circuit.Netlist(name="ring_after_wg")
    nl.add("wg", "straight", length=40.0).add("ring", "ring_single", radius=6.0, gap=0.25)
    nl.connect(("wg", "o2"), ("ring", "o1"))
    nl.expose("in", ("wg", "o1")).expose("out", ("ring", "o2"))
    path = tmp_path / "circuit.json"
    nl.to_json(path)
    return path


def test_netlist_simulate_equals_circuit_from_netlist(capsys, tmp_path):
    path = _netlist(tmp_path)
    code, payload = run_json(capsys, "netlist", "simulate", str(path), "--wl", "1.54:1.56:51",
                             "-s", "ring.radius=7.5")
    assert code == 0 and payload["ports"] == ["in", "out"] and payload["instances"] == 2
    circuit = px.circuit.circuit_from_netlist(px.circuit.Netlist.from_json(path))
    wl = np.linspace(1.54, 1.56, 51)
    direct = circuit(wl=wl, ring={"radius": 7.5})
    s = np.asarray(payload["data"]["s_real"]) + 1j * np.asarray(payload["data"]["s_imag"])
    np.testing.assert_allclose(s, _dense(["in", "out"], direct), rtol=0, atol=1e-14)
    assert payload["settings"] == {"ring": {"radius": 7.5}}
    code, payload = run_json(capsys, "netlist", "simulate", str(path), "--pdk", "demo", "--wl", "1.55",
                             "-o", str(tmp_path / "c.s2p"))
    assert code == 0 and payload["output"]["format"] == "touchstone"
    direct = px.circuit.circuit_from_netlist(px.circuit.Netlist.from_json(path), px.pdk.demo_pdk().models())(
        wl=np.asarray([1.55]))
    ds = px.SParameterDataset.load_touchstone(tmp_path / "c.s2p")
    np.testing.assert_allclose(ds.s, _dense(["in", "out"], direct), atol=1e-15)


def test_netlist_simulate_errors(capsys, tmp_path):
    path = _netlist(tmp_path)
    assert run(capsys, "netlist", "simulate", str(path), "-s", "nosuch.radius=1")[0] == 2
    assert run(capsys, "netlist", "simulate", str(path), "-s", "ring_radius=1")[0] == 2
    assert run(capsys, "netlist", "simulate", str(tmp_path / "absent.json"))[0] == 2
    (tmp_path / "bad.json").write_text('{"format": "photonix.netlist", "version": 1, "instances": {"a": '
                                       '{"model": "warp_drive"}}, "connections": [], "ports": {}}')
    code, _out, err = run(capsys, "netlist", "simulate", str(tmp_path / "bad.json"))
    assert code == 2 and "warp_drive" in err
    (tmp_path / "garbage.json").write_text("{nope")
    assert run(capsys, "netlist", "simulate", str(tmp_path / "garbage.json"))[0] == 2


# --------------------------------------------------------------------------- #
# validate
# --------------------------------------------------------------------------- #
def test_validate_quick_passes_and_is_a_manifest(capsys):
    code, payload = run_json(capsys, "validate", "--quick")
    assert code == 0 and payload["passed"] and payload["quick"]
    names = [c["name"] for c in payload["checks"]]
    assert names == [c.name for c in validate_cmd.CHECKS if c.quick]
    assert all(c["passed"] and c["error"] <= c["tolerance"] for c in payload["checks"])
    assert payload["runtime"]["photonix_version"] == px.__version__
    code, out, _ = run(capsys, "validate", "--check", "bragg_peak_reflectance")
    assert code == 0 and "bragg_peak_reflectance" in out and "ring_allpass" not in out


def test_validate_checks_are_independent_of_the_quantities_they_test():
    """The references are computed from closed forms, not from the code under test."""
    # Bragg: tanh^2(kappa L), kappa = 2 dn / wl_B for a 50 % duty square corrugation of peak-to-peak dn
    # (first Fourier harmonic, pi/wl * (2/pi) * dn).
    wl_b = float(px.components.bragg_wavelength(0.3229))
    kappa = float(px.components.grating_kappa(0.03, wl_b))
    assert kappa == pytest.approx(2 * 0.03 / wl_b, rel=1e-12)
    value, expected, error = validate_cmd._bragg_peak()
    assert expected == pytest.approx(math.tanh(kappa * 0.3229 * 300) ** 2, rel=1e-15) and error < 1e-6
    # Slab: the transcendental-equation reference is checked against its own definition.
    k0, d, n1, n2 = 2 * math.pi / 1.55, 0.22, 3.4757, 1.444
    neff = px.em.slab_neff_analytic()
    kx, g = k0 * math.sqrt(n1**2 - neff**2), k0 * math.sqrt(neff**2 - n2**2)
    assert math.tan(kx * d / 2) == pytest.approx(g / kx, rel=1e-9)


def test_validate_reports_failures_and_crashes(capsys, monkeypatch):
    broken = validate_cmd.Check("always_wrong", "a check that fails", "none", "0", 1e-9, True,
                                lambda: (2.0, 1.0, 1.0))
    crashing = validate_cmd.Check("crashes", "a check that raises", "none", "0", 1e-9, True,
                                  lambda: (_ for _ in ()).throw(ZeroDivisionError("bad")))
    monkeypatch.setattr(validate_cmd, "CHECKS", (broken, crashing))
    code, payload = run_json(capsys, "validate")
    assert code == ExitCode.CHECK_FAILED and payload["n_failed"] == 2
    assert payload["checks"][1]["exception"] == "ZeroDivisionError: bad"
    code, out, _ = run(capsys, "validate")
    assert code == 1 and "FAIL" in out and "FAILED" in out


# --------------------------------------------------------------------------- #
# parsing helpers
# --------------------------------------------------------------------------- #
def test_parse_wavelengths_and_params():
    np.testing.assert_array_equal(parse_wavelengths("1.5:1.6:3"), [1.5, 1.55, 1.6])
    np.testing.assert_array_equal(parse_wavelengths("1.3, 1.55"), [1.3, 1.55])
    np.testing.assert_array_equal(parse_wavelengths("1.5:1.5:1"), [1.5])
    for bad in ("", "1.5:1.6", "1.5:1.6:0", "-1", "1.6,1.5", "1.5:1.5:2", "inf", "a:b:c"):
        with pytest.raises(CliError):
            parse_wavelengths(bad)
    assert parse_params(["a=1", "b=2.5", "c=true", "d=None", "e=[1, 2]", "f=gaussian", 'g="3"']) == {
        "a": 1, "b": 2.5, "c": True, "d": None, "e": [1, 2], "f": "gaussian", "g": "3"}
    with pytest.raises(CliError, match="valid: length"):
        parse_params(["lenght=1"], allowed=px.components.straight)
    for bad in ("=1", "a", "1a=2"):
        with pytest.raises(CliError):
            parse_params([bad])
    assert jsonable({"x": np.float64("nan"), "y": np.arange(2), "z": (1 + 1j,)}) == {
        "x": "nan", "y": [0, 1], "z": [[1.0, 1.0]]}


# --------------------------------------------------------------------------- #
# Review regressions: input validation, output safety, broken pipes
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text", ["[1, 2]", '{"format": "photonix.sparameters", "version": null}',
                                  '{"format": "photonix.sparameters", "version": "1"}',
                                  '{"format": "photonix.sparameters", "version": 1, "ports": "ab", '
                                  '"wavelengths_um": [1.55], "s_real": [[[0, 1], [1, 0]]], '
                                  '"s_imag": [[[0, 0], [0, 0]]]}'])
def test_malformed_dataset_json_is_an_input_error(capsys, tmp_path, text):
    path = tmp_path / "bad.json"
    path.write_text(text)
    code, _, err = run(capsys, "sparams", "check", str(path))
    assert code == ExitCode.USAGE and "Traceback" not in err


@pytest.mark.parametrize("text", ["[1]", '{"rules": {"1/0": {"min_width": 1}}, "version": null}'])
def test_malformed_netlist_and_rules_are_input_errors(capsys, tmp_path, text):
    path = tmp_path / "bad.json"
    path.write_text(text)
    assert run(capsys, "netlist", "simulate", str(path))[0] == ExitCode.USAGE
    from photonix.cli.commands.gds import load_rules

    with pytest.raises(CliError):
        load_rules(path)


def test_netlist_path_starting_with_a_brace_is_read_as_a_file(capsys, tmp_path, monkeypatch):
    nl = px.circuit.Netlist(name="n")
    nl.add("a", "straight", length=5.0)
    nl.expose("i", ("a", "o1"))
    nl.expose("o", ("a", "o2"))
    monkeypatch.chdir(tmp_path)
    nl.to_json("{run1}.json")
    code, _, _ = run(capsys, "netlist", "simulate", "{run1}.json", "--wl", "1.55")
    assert code == ExitCode.OK


def test_existing_outputs_need_force_and_are_checked_before_work(capsys, tmp_path):
    notes = tmp_path / "notes.json"
    notes.write_text('{"important": "user data"}')
    code, _, err = run(capsys, "sweep", "straight", "--wl", "1.55", "-o", str(notes))
    assert code == ExitCode.USAGE and "--force" in err
    assert json.loads(notes.read_text()) == {"important": "user data"}
    assert run(capsys, "sweep", "straight", "--wl", "1.55", "-o", str(notes), "--force")[0] == ExitCode.OK
    assert json.loads(notes.read_text())["format"] == "photonix.sparameters"
    # A bad suffix or a missing directory fails before the model is evaluated.
    assert run(capsys, "sweep", "straight", "-o", str(tmp_path / "out.txt"))[0] == ExitCode.USAGE
    assert run(capsys, "sweep", "straight", "-o", str(tmp_path / "nodir" / "o.json"))[0] == ExitCode.USAGE
    src = tmp_path / "a.json"
    run(capsys, "sweep", "straight", "--wl", "1.55", "-o", str(src))
    report = tmp_path / "report.csv"
    report.write_text("precious")
    assert run(capsys, "sparams", "convert", str(src), str(report))[0] == ExitCode.USAGE
    assert report.read_text() == "precious"
    assert run(capsys, "sparams", "convert", str(src), str(report), "--force")[0] == ExitCode.OK


def test_json_output_into_a_closed_pipe_exits_quietly():
    code = ("import subprocess, sys; p = subprocess.Popen([sys.executable, '-m', 'photonix', 'sweep', 'straight', "
            "'--wl', '1.5:1.6:4001', '--json'], stdout=subprocess.PIPE, stderr=subprocess.PIPE); "
            "p.stdout.read(10); p.stdout.close(); err = p.stderr.read(); rc = p.wait(); "
            "print(rc, b'Traceback' in err)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300, check=True)
    rc, traceback_seen = out.stdout.split()
    assert rc in ("141", "-13") and traceback_seen == "False"
