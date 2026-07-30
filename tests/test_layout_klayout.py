"""Binary-free tests for optional OASIS and KLayout integration."""
from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

import photonix.layout as lay
import photonix.layout.gds as layout_io
import photonix.layout.klayout as klayout


def test_missing_gdstk_is_reported_only_at_io_boundary(monkeypatch):
    def missing(_name):
        raise ImportError("not installed")

    monkeypatch.setattr(layout_io.importlib, "import_module", missing)
    assert not lay.gdstk_available()
    with pytest.raises(ImportError, match=r"photonix\[layout\]"):
        lay.write_oas(lay.Cell("optional"), "unused.oas")


def test_klayout_capability_check_does_not_import_bindings(monkeypatch):
    calls = []

    def fake_which(name):
        calls.append(name)
        return "/opt/klayout/klayout" if name == "klayout" else None

    monkeypatch.delenv("KLAYOUT_EXECUTABLE", raising=False)
    monkeypatch.setattr(klayout.shutil, "which", fake_which)
    assert klayout.klayout_available()
    assert klayout.find_klayout() == "/opt/klayout/klayout"
    assert calls == ["klayout", "klayout"]


def test_drc_builds_headless_command_and_preserves_user_deck(tmp_path, monkeypatch):
    layout_path = tmp_path / "chip.oas"
    deck_path = tmp_path / "licensed.lydrc"
    report_path = tmp_path / "drc.lyrdb"
    layout_path.write_bytes(b"layout")
    deck_text = "# supplied externally\nreport($report)\n"
    deck_path.write_text(deck_text, encoding="utf-8")
    monkeypatch.setattr(klayout, "find_klayout", lambda executable=None: "/tools/klayout")

    seen = {}

    def fake_run(command, **kwargs):
        seen["command"] = command
        seen["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stdout="clean\n", stderr="")

    monkeypatch.setattr(klayout.subprocess, "run", fake_run)
    result = lay.run_drc(
        layout_path,
        deck_path,
        report_path=report_path,
        variables={"threads": 4, "mode": "signoff"},
    )

    assert result.succeeded
    assert result.kind == "DRC"
    assert seen["command"] == [
        "/tools/klayout",
        "-b",
        "-rd",
        f"input={layout_path.resolve()}",
        "-rd",
        f"report={report_path.resolve()}",
        "-rd",
        "threads=4",
        "-rd",
        "mode=signoff",
        "-r",
        str(deck_path.resolve()),
    ]
    assert seen["kwargs"]["check"] is False
    assert seen["kwargs"]["shell"] is False
    assert seen["kwargs"]["capture_output"] is True
    assert deck_path.read_text(encoding="utf-8") == deck_text


def test_klayout_errors_are_local_and_return_output_when_unchecked(tmp_path, monkeypatch):
    layout_path = tmp_path / "chip.gds"
    deck_path = tmp_path / "deck.lvs"
    layout_path.write_bytes(b"layout")
    deck_path.write_text("# lvs", encoding="utf-8")

    monkeypatch.setattr(klayout.shutil, "which", lambda _name: None)
    with pytest.raises(FileNotFoundError, match="KLayout executable"):
        lay.run_lvs(layout_path, deck_path)

    monkeypatch.setattr(klayout, "find_klayout", lambda executable=None: "/tools/klayout")
    monkeypatch.setattr(
        klayout.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=7, stdout="", stderr="deck error"),
    )
    with pytest.raises(lay.KLayoutRunError, match="deck error") as exc_info:
        lay.run_lvs(layout_path, deck_path)
    assert exc_info.value.result.returncode == 7

    result = lay.run_lvs(layout_path, deck_path, check=False)
    assert not result.succeeded
    assert result.stderr == "deck error"


def test_klayout_runtime_variable_contract_is_validated(tmp_path, monkeypatch):
    layout_path = tmp_path / "chip.gds"
    deck_path = tmp_path / "deck.drc"
    layout_path.write_bytes(b"layout")
    deck_path.write_text("# drc", encoding="utf-8")
    monkeypatch.setattr(klayout, "find_klayout", lambda executable=None: "/tools/klayout")

    with pytest.raises(ValueError, match="duplicates"):
        lay.run_drc(layout_path, deck_path, variables={"input": "other.gds"})
    with pytest.raises(ValueError, match="variable name"):
        lay.run_drc(layout_path, deck_path, variables={"not-valid": 1})


class _FakePolygon:
    def __init__(self, points, layer=0, datatype=0):
        self.points = np.asarray(points)
        self.layer = layer
        self.datatype = datatype


class _FakeLabel:
    def __init__(self, text, origin, layer=0):
        self.text = text
        self.origin = origin
        self.layer = layer


class _FakeCell:
    def __init__(self, name):
        self.name = name
        self.polygons = []
        self.labels = []

    def add(self, item):
        if isinstance(item, _FakeLabel):
            self.labels.append(item)
        else:
            self.polygons.append(item)

    def get_polygons(self):
        return self.polygons


class _FakeLibrary:
    def __init__(self, owner):
        self.owner = owner
        self.cells = []
        owner.latest = self

    def new_cell(self, name):
        cell = _FakeCell(name)
        self.cells.append(cell)
        return cell

    def top_level(self):
        return self.cells

    def write_oas(self, path, **kwargs):
        self.owner.write_call = (path, kwargs)


class _FakeGdstk:
    Polygon = _FakePolygon
    Label = _FakeLabel
    Reference = SimpleNamespace

    def __init__(self):
        self.latest = None
        self.write_call = None

    def Library(self):
        return _FakeLibrary(self)

    def read_oas(self, path):
        self.read_path = path
        return self.latest


def test_oasis_roundtrip_adapter_uses_gdstk_api_without_native_binary(tmp_path, monkeypatch):
    fake = _FakeGdstk()
    monkeypatch.setitem(sys.modules, "gdstk", fake)

    cell = lay.Cell("oas_top")
    cell.add_polygon([(0, 0), (2, 0), (2, 1), (0, 1)], layer=(7, 3))
    cell.add_port("o1", center=(0, 0), layer=(7, 3))
    path = tmp_path / "chip.oas"
    assert lay.gdstk_available()
    assert lay.write_oas(cell, path, compression_level=9, validation="crc32") == str(path)
    assert fake.write_call == (str(path), {"compression_level": 9, "validation": "crc32"})

    restored = lay.read_oas(path)
    assert restored.name == "oas_top"
    assert len(restored.polygons) == 1
    assert restored.polygons[0][1] == (7, 3)
    assert "o1" in restored.ports


def test_oasis_options_are_validated_without_importing_gdstk(tmp_path):
    with pytest.raises(ValueError, match="compression_level"):
        lay.write_oas(lay.Cell("bad"), tmp_path / "bad.oas", compression_level=10)
    with pytest.raises(ValueError, match="validation"):
        lay.write_oas(lay.Cell("bad"), tmp_path / "bad.oas", validation="md5")
