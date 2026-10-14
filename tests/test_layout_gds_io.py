"""Lossless GDSII/OASIS I/O: units, hierarchy, arrays, ports, sidecar semantics and determinism.

Independent references used here:
* gdstk's own ``Cell.flatten`` (a different code path from Photonix's hierarchy
  rebuild) for flattened geometry;
* a minimal GDSII stream decoder written in this file from the record layout
  of the GDSII Stream Format manual (record header = 2-byte length, 1-byte
  record type, 1-byte data type; 8-byte excess-64 base-16 reals), so the bytes
  on disk are checked without gdstk;
* KLayout's reader when the ``klayout`` Python package is installed.
"""
from __future__ import annotations

import datetime
import math
import os
import struct
import warnings

import numpy as np
import pytest
from conftest import finite_diff, requires_jax

import photonix as px
import photonix.layout as lay
from photonix.layout import components as lc
from photonix.layout.gds import _fingerprint
from photonix.layout.gds_io import (
    INSTANCE_NAME_ATTR,
    PORT_LAYER_ATTR,
    _content_hash,
    _decode,
    _encode,
)

gdstk = pytest.importorskip("gdstk")


def _fp(cell):
    return _fingerprint(cell, {})


def _area(pts):
    x, y = np.asarray(pts, dtype=float).T
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _layer_areas(polys):
    out: dict = {}
    for pts, layer in polys:
        out[tuple(layer)] = out.get(tuple(layer), 0.0) + _area(pts)
    return out


def _gdstk_flat(path, top, fmt="gds"):
    """gdstk's own flattening (OASIS element repetitions applied explicitly)."""
    lib = gdstk.read_gds(path, unit=1e-6) if fmt == "gds" else gdstk.read_oas(path, unit=1e-6)
    cell = next(c for c in lib.cells if c.name == top).copy("__flat__").flatten()
    polys = list(cell.polygons)
    for path_ in cell.paths:
        polys += path_.to_polygons()
    expanded = []
    for p in polys:
        expanded.append(p)
        expanded += p.apply_repetition()
    return [(np.asarray(p.points), (p.layer, p.datatype)) for p in expanded]


def _port_tuple(p):
    return (tuple(np.round(p.center, 9)), round(p.orientation % 360.0, 9), round(p.width, 9), p.layer, p.port_type)


def _route_circuit():
    top = lay.Cell("routed")
    a = top.add_ref(lc.straight(20.0), name="a")
    b = top.add_ref(lc.mmi1x2(), origin=(60.0, 40.0), name="b")
    r = lay.route_manhattan(a.port("o2"), b.port("o1"), radius=5.0)
    top.add_ref(r, name="r")
    top.expose_port(a, "o1", "in")
    return top


def _rotated_circuit():
    top = lay.Cell("rotated")
    s = top.add_ref(lc.straight(10.0), origin=(1.0, 2.0), rotation=30.0, name="s")
    top.add_ref(lc.bend_euler(radius=5.0), name="b").connect("o1", s.port("o2"))
    top.add_ref(lc.taper(), origin=(5.0, -20.0), rotation=180.0, mirror=True, name="m")
    top.expose_port(s, "o1", "o1")
    return top


CIRCUITS = {"mzi": lambda: lc.mzi(delta_length=20.0), "routed": _route_circuit, "rotated": _rotated_circuit}


# --------------------------------------------------------------------------- #
# A minimal, independent GDSII stream decoder
# --------------------------------------------------------------------------- #
def _real8(b: bytes) -> float:
    sign = -1.0 if b[0] & 0x80 else 1.0
    return sign * int.from_bytes(b[1:8], "big") / 2.0 ** 56 * 16.0 ** ((b[0] & 0x7F) - 64)


def _decode_gds(path):
    data = open(path, "rb").read()
    i, units, structs = 0, None, {}
    cur = el = None
    while i + 4 <= len(data):
        n, rtype, _dtype = struct.unpack(">HBB", data[i:i + 4])
        if n < 4:
            break
        payload = data[i + 4:i + n]
        i += n
        if rtype == 0x03:                                   # UNITS
            units = (_real8(payload[:8]), _real8(payload[8:16]))
        elif rtype == 0x06:                                 # STRNAME
            cur = {"boundaries": [], "srefs": [], "paths": [], "texts": [], "arefs": []}
            structs[payload.rstrip(b"\0").decode()] = cur
        elif rtype in (0x08, 0x0A, 0x09, 0x0C, 0x0B):       # BOUNDARY, SREF, PATH, TEXT, AREF
            el = {"kind": rtype, "props": {}}
        elif rtype == 0x13:                                 # COLROW
            el["colrow"] = struct.unpack(">hh", payload)
        elif rtype == 0x1A:                                 # STRANS
            el["reflect"] = bool(struct.unpack(">H", payload)[0] & 0x8000)
        elif rtype == 0x1C:                                 # ANGLE (degrees)
            el["angle"] = _real8(payload)
        elif rtype == 0x0D:
            el["layer"] = struct.unpack(">h", payload)[0]
        elif rtype in (0x0E, 0x16):                         # DATATYPE, TEXTTYPE
            el["datatype"] = struct.unpack(">h", payload)[0]
        elif rtype == 0x12:                                 # SNAME
            el["sname"] = payload.rstrip(b"\0").decode()
        elif rtype == 0x10:                                 # XY
            el["xy"] = np.frombuffer(payload, dtype=">i4").reshape(-1, 2).astype(np.int64)
        elif rtype == 0x19:                                 # STRING
            el["string"] = payload.rstrip(b"\0").decode()
        elif rtype == 0x2B:                                 # PROPATTR
            el["_attr"] = struct.unpack(">h", payload)[0]
        elif rtype == 0x2C:                                 # PROPVALUE
            el["props"][el.pop("_attr")] = payload.rstrip(b"\0").decode()
        elif rtype == 0x11:                                 # ENDEL
            key = {0x08: "boundaries", 0x0A: "srefs", 0x09: "paths", 0x0C: "texts", 0x0B: "arefs"}[el["kind"]]
            cur[key].append(el)
        elif rtype == 0x04:
            break
    return units, structs


def _int_area(xy):
    x, y = xy[:-1, 0], xy[:-1, 1]              # GDS boundaries repeat the first point
    return abs(int(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))) / 2


# --------------------------------------------------------------------------- #
# Units and top-cell selection
# --------------------------------------------------------------------------- #
def test_nanometre_user_unit_reads_in_micrometres(tmp_path):
    path = os.fspath(tmp_path / "nm.gds")
    lib = gdstk.Library(unit=1e-9, precision=1e-9)
    lib.new_cell("wg").add(gdstk.rectangle((0, 0), (10000, 500)))
    lib.write_gds(path)
    assert lay.layout_units(path) == (1e-9, 1e-9)
    cell = lay.read_layout(path)
    assert cell.bbox() == ((0.0, 0.0), (10.0, 0.5))
    assert lay.read_gds(path).bbox()[1][0] == pytest.approx(10000.0)   # the legacy reader's unit bug


def test_coarse_database_grid_warns(tmp_path):
    path = os.fspath(tmp_path / "coarse.gds")
    lib = gdstk.Library(unit=1e-6, precision=5e-9)
    lib.new_cell("c").add(gdstk.rectangle((0, 0), (1, 1)))
    lib.write_gds(path)
    with pytest.warns(UserWarning, match="5 nm database grid"):
        lay.read_layout(path)


def test_several_top_cells_must_be_chosen(tmp_path):
    path = os.fspath(tmp_path / "two.gds")
    lib = gdstk.Library()
    for name in ("B", "A"):
        lib.new_cell(name).add(gdstk.rectangle((0, 0), (1, 1)))
    lib.write_gds(path)
    assert lay.list_top_cells(path) == ["A", "B"]
    with pytest.raises(ValueError, match=r"\['A', 'B'\]"):
        lay.read_layout(path)
    assert lay.read_layout(path, top="B").name == "B"
    with pytest.raises(ValueError, match="no structure 'C'"):
        lay.read_layout(path, top="C")


# --------------------------------------------------------------------------- #
# Round trips
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fmt", ["gds", "oas"])
@pytest.mark.parametrize("name", sorted(CIRCUITS))
def test_round_trip_reproduces_tree_and_netlist(tmp_path, name, fmt):
    cell = CIRCUITS[name]()
    path = os.fspath(tmp_path / f"{name}.{fmt}")
    art = lay.write_layout(cell, path)
    back = lay.read_layout(path)
    snapped = lay.snap_to_grid(cell)
    assert _fp(back) == _fp(snapped) == art.structures[art.top]
    assert art.snap_error <= math.sqrt(2) / 2 * 1e-3 + 1e-15
    nl0 = lay.extract_netlist(cell)
    nl1 = lay.extract_netlist(back)
    assert nl1.instances == nl0.instances
    assert nl1.connections == nl0.connections
    assert nl1.settings == nl0.settings
    assert nl1.ports == nl0.ports


def test_round_trip_shares_structures_and_restores_names(tmp_path):
    top = lay.Cell("top")
    s10 = lc.straight(10.0)
    top.add_ref(s10, name="a")
    top.add_ref(s10, origin=(0, 5), name="b")
    top.add_ref(lc.straight(25.0), origin=(0, 10), name="c")     # same name, different content
    path = os.fspath(tmp_path / "t.gds")
    art = lay.write_layout(top, path)
    assert sorted(art.structures) == ["straight", "straight$1", "top"]
    back = lay.read_layout(path)
    assert back.references[0].cell is back.references[1].cell
    assert [r.cell.name for r in back.references] == ["straight"] * 3
    assert [r.name for r in back.references] == ["a", "b", "c"]
    without = lay.read_layout(path, sidecar=False)
    assert [r.cell.name for r in without.references] == ["straight", "straight", "straight$1"]


def test_rotated_reference_ports_are_exact(tmp_path):
    path = os.fspath(tmp_path / "rot.gds")
    lay.write_layout(_rotated_circuit(), path, sidecar=False)
    back = lay.read_layout(path)
    ref = back.references[0]
    assert ref.rotation == 30.0
    assert ref.port("o2").orientation == pytest.approx(30.0, abs=1e-9)
    assert back.references[2].mirror is True
    assert back.ports["o1"].orientation == pytest.approx(210.0, abs=1e-9)


def test_identical_geometry_with_different_settings_stays_distinct(tmp_path):
    a, b = lc.straight(10.0), lc.straight(10.0)
    b.model_settings["loss_db_cm"] = 3.0
    top = lay.Cell("top")
    top.add_ref(a, name="a")
    top.add_ref(b, origin=(0, 5), name="b")
    path = os.fspath(tmp_path / "s.gds")
    lay.write_layout(top, path)
    nl = lay.extract_netlist(lay.read_layout(path))
    assert nl.settings == {"a": {"length": 10.0}, "b": {"length": 10.0, "loss_db_cm": 3.0}}


def test_model_settings_encoding_is_exact():
    value = {"t": (1, 2.5), "c": 1 + 2j, "a": np.arange(6.0).reshape(2, 3), "n": np.float32(0.25),
             "nested": {"x": [1, None, True, "s"]}, "nan": float("nan"), "inf": -float("inf")}
    import json

    back = _decode(json.loads(json.dumps(_encode(value, "test"))))
    assert back["t"] == (1, 2.5) and back["c"] == 1 + 2j and back["nested"] == value["nested"]
    np.testing.assert_array_equal(back["a"], value["a"])
    assert back["a"].dtype == value["a"].dtype
    assert back["n"] == 0.25 and math.isnan(back["nan"]) and back["inf"] == -float("inf")
    with pytest.raises(TypeError, match="cannot be stored"):
        _encode({"neff": lambda wl: 2.4}, "cell 'x'")


def test_labels_round_trip(tmp_path):
    c = lay.LabeledCell.from_cell(lc.straight(10.0))
    c.add_label("opt_in_TE_1550_device_me_wg", (1.0, 2.0), (10, 0), rotation=90.0, magnification=2.0, anchor="sw")
    c.add_label("die A1", (3.0004, -1.0), (66, 3))
    c.add_label("north", (5.0, 0.5), (10, 0), rotation=45.0, magnification=2.5, anchor="n")
    want = [("opt_in_TE_1550_device_me_wg", (1.0, 2.0), (10, 0), 90.0, 2.0, "sw"),
            ("die A1", (3.0, -1.0), (66, 3), 0.0, 1.0, "o"), ("north", (5.0, 0.5), (10, 0), 45.0, 2.5, "n")]
    for fmt in ("gds", "oas"):
        path = os.fspath(tmp_path / f"l.{fmt}")
        lay.write_layout(c, path)
        back = lay.read_layout(path)
        # OASIS TEXT has no anchor/rotation/magnification: the sidecar restores them
        got = [(lb.text, lb.position, lb.layer, lb.rotation, lb.magnification, lb.anchor) for lb in back.labels]
        assert got == want
        assert set(back.ports) == {"o1", "o2"}  # pin name labels were consumed, not left behind
    # Without a sidecar an OASIS file keeps only text, position and layer.
    path = os.fspath(tmp_path / "bare.oas")
    lay.write_layout(c, path, sidecar=False)
    bare = lay.read_layout(path)
    assert [(lb.text, lb.rotation, lb.magnification, lb.anchor) for lb in bare.labels if lb.text == "north"] == \
        [("north", 0.0, 1.0, "sw")]


def test_label_markers_with_sidecar_leave_no_stray_labels(tmp_path):
    cell = lc.mzi()
    path = os.fspath(tmp_path / "lm.gds")
    lay.write_layout(cell, path, port_markers="label")
    back = lay.read_layout(path)
    assert _fp(back) == _fp(lay.snap_to_grid(cell))
    assert not back.labels and not any(r.cell.labels for r in back.references)


def test_unserialisable_settings_need_no_sidecar(tmp_path):
    cell = lc.straight(10.0)
    cell.model_settings["neff"] = lambda wl: 2.4
    path = os.fspath(tmp_path / "c.gds")
    assert lay.write_layout(cell, path, sidecar=False).sidecar_path is None
    with pytest.raises(TypeError, match="cannot be stored in the JSON sidecar"):
        lay.write_layout(cell, path)


def test_writing_without_sidecar_removes_a_stale_default_one(tmp_path):
    path = os.fspath(tmp_path / "w.gds")
    art = lay.write_layout(lc.straight(), path)
    assert os.path.exists(art.sidecar_path)
    assert lay.write_layout(lc.straight(), path, sidecar=False).sidecar_path is None
    assert not os.path.exists(art.sidecar_path)
    assert lay.read_layout(path).model == lay.GEOMETRY_ONLY
    with pytest.raises(FileNotFoundError):
        lay.read_layout(path, sidecar=True)


def test_explicit_sidecar_path(tmp_path):
    path = os.fspath(tmp_path / "x.gds")
    side = os.fspath(tmp_path / "meta.json")
    lay.write_layout(lc.straight(7.0), path, sidecar=side)
    assert lay.read_layout(path).model == lay.GEOMETRY_ONLY        # not at the default location
    assert lay.read_layout(path, sidecar=side).model_settings == {"length": 7.0}


# --------------------------------------------------------------------------- #
# Sidecar staleness
# --------------------------------------------------------------------------- #
def _rewrite(path, edit=None, reverse=False):
    lib = gdstk.read_gds(path, unit=1e-6)
    for cell in lib.cells:
        if edit is not None:
            edit(cell)
        if reverse:
            polys = list(cell.polygons)
            cell.remove(*polys)
            cell.add(*polys[::-1])
    lib.write_gds(path)


def test_rewritten_but_unchanged_file_keeps_sidecar(tmp_path):
    path = os.fspath(tmp_path / "m.gds")
    lay.write_layout(lc.mzi(), path)
    _rewrite(path, reverse=True)                    # polygon order and file bytes change, geometry does not
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        back = lay.read_layout(path)
    assert lay.extract_netlist(back).settings["top_s1"] == {"length": 10.0}


def test_edited_structure_invalidates_only_its_sidecar_entry(tmp_path):
    path = os.fspath(tmp_path / "m.gds")
    art = lay.write_layout(lc.mzi(), path)
    edited = next(n for n in art.structures if n.startswith("mmi1x2"))

    def stretch(cell):
        if cell.name == edited:
            cell.add(gdstk.rectangle((0, 3), (1, 4)))

    _rewrite(path, edit=stretch)
    with pytest.warns(lay.StaleSidecarWarning, match=edited.replace("$", r"\$")):
        back = lay.read_layout(path)
    split = next(r for r in back.references if r.name == "splitter").cell
    bend = next(r for r in back.references if r.name == "top_b1").cell
    assert split.model == lay.GEOMETRY_ONLY and set(split.ports) == {"o1", "o2", "o3"}   # ports from pins
    assert bend.model == "bend" and bend.model_settings == {"radius": 5.0, "angle": 90.0}
    with pytest.raises(ValueError, match="splitter"):
        lay.extract_netlist(back)


@pytest.mark.parametrize("fmt", ["gds", "oas"])
def test_editing_a_sub_structure_invalidates_every_structure_above_it(tmp_path, fmt):
    """A model describing geometry drawn in a sub-cell must not survive an edit of that sub-cell."""
    core = lay.Cell("wg_core").add_polygon([(0, -0.25), (10, -0.25), (10, 0.25), (0, 0.25)], (1, 0))
    dev = lay.Cell("wg10", model="straight", model_settings={"length": 10.0})
    dev.add_ref(core, name="core")
    dev.add_port("o1", (0, 0), 180, 0.5, (1, 0))
    dev.add_port("o2", (10, 0), 0, 0.5, (1, 0))
    top = lay.Cell("top")
    top.add_ref(dev, name="u1")
    other = lay.Cell("other", model="straight", model_settings={"length": 5.0})
    other.add_polygon([(0, -0.25), (5, -0.25), (5, 0.25), (0, 0.25)], (1, 0))
    other.add_port("o1", (0, 0), 180, 0.5, (1, 0))
    other.add_port("o2", (5, 0), 0, 0.5, (1, 0))
    top.add_ref(other, origin=(0, 20), name="u2")
    path = os.fspath(tmp_path / f"chip.{fmt}")
    lay.write_layout(top, path)
    lib = gdstk.read_gds(path) if fmt == "gds" else gdstk.read_oas(path)
    c = next(c for c in lib.cells if c.name == "wg_core")
    c.remove(*c.polygons)
    c.add(gdstk.rectangle((0, -0.25), (30, 0.25), layer=1, datatype=0))     # stretched to 30 um
    lib.write_gds(path) if fmt == "gds" else lib.write_oas(path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        back = lay.read_layout(path)
    stale = {str(w.message).split("'")[1] for w in caught if issubclass(w.category, lay.StaleSidecarWarning)}
    assert stale == {"wg_core", "wg10", "top"}
    u1 = next(r for r in back.references if r.name == "u1").cell
    u2 = next(r for r in back.references if r.name == "u2").cell
    assert u1.model != "straight" and u1.model_settings == {}               # the stale length is not applied
    assert u2.model == "straight" and u2.model_settings == {"length": 5.0}   # untouched siblings keep theirs


def test_content_hash_covers_the_hierarchy_below():
    def make(child_width):
        lib = gdstk.Library()
        leaf = lib.new_cell("leaf")
        leaf.add(gdstk.rectangle((0, 0), (child_width, 1)))
        mid = lib.new_cell("mid")
        mid.add(gdstk.Reference(leaf, (5, 0)))
        top = lib.new_cell("top")
        top.add(gdstk.Reference(mid, (0, 3)))
        return top, mid

    (t1, m1), (t2, m2) = make(1.0), make(1.0)
    assert _content_hash(t1, 1e-3) == _content_hash(t2, 1e-3)
    (t3, m3) = make(2.0)
    assert _content_hash(m3, 1e-3) != _content_hash(m1, 1e-3)
    assert _content_hash(t3, 1e-3) != _content_hash(t1, 1e-3)


def test_content_hash_ignores_text_and_order_but_not_geometry():
    def make(order, extra_text=False, shift=0.0):
        lib = gdstk.Library()
        c = lib.new_cell("c")
        shapes = [gdstk.rectangle((0, 0), (1, 1)), gdstk.rectangle((2, 0), (3 + shift, 1), layer=2)]
        c.add(*(shapes if order else shapes[::-1]))
        if extra_text:
            c.add(gdstk.Label("note", (0, 0)))
        return c

    h = _content_hash(make(True), 1e-3)
    assert _content_hash(make(False), 1e-3) == h
    assert _content_hash(make(True, extra_text=True), 1e-3) == h
    assert _content_hash(make(True, shift=0.001), 1e-3) != h


# --------------------------------------------------------------------------- #
# Ports without a sidecar
# --------------------------------------------------------------------------- #
def test_pinrec_markers_restore_ports_without_sidecar(tmp_path):
    cell = lc.heater_straight(50.0)
    path = os.fspath(tmp_path / "h.gds")
    lay.write_layout(cell, path, sidecar=False)
    back = lay.read_layout(path)
    assert {k: _port_tuple(p) for k, p in back.ports.items()} == {k: _port_tuple(p) for k, p in cell.ports.items()}
    assert back.ports["e1"].port_type == lay.ELECTRICAL and back.ports["e1"].layer == (11, 0)
    assert back.model == lay.GEOMETRY_ONLY and not back.labels
    assert [len(b["xy"]) for b in _decode_gds(path)[1]["heater_straight"]["boundaries"]] == [5, 5]  # no pin shapes


def test_non_manhattan_pin_orientation_is_exact_from_property(tmp_path):
    cell = lc.bend_circular(radius=5.0, angle=37.0)
    path = os.fspath(tmp_path / "b.gds")
    lay.write_layout(cell, path, sidecar=False)
    back = lay.read_layout(path)
    assert back.ports["o2"].orientation == 37.0
    assert back.ports["o2"].center == lay.snap_to_grid(cell).ports["o2"].center


def test_foreign_siepic_style_pins(tmp_path):
    """Pins drawn without Photonix properties: layer from the shape under the port, direction from the path."""
    path = os.fspath(tmp_path / "siepic.gds")
    lib = gdstk.Library()
    c = lib.new_cell("ebeam_wg")
    c.add(gdstk.rectangle((0, -0.25), (10, 0.25), layer=1))
    c.add(gdstk.rectangle((-2, -3), (12, 3), layer=68))             # DevRec box (larger, also under the port)
    for name, (x0, x1) in {"opt1": (0.05, -0.05), "opt2": (9.95, 10.05)}.items():
        c.add(gdstk.FlexPath([(x0, 0), (x1, 0)], 0.5, layer=1, datatype=10, simple_path=True))
        c.add(gdstk.Label(name, (0.5 * (x0 + x1), 0), layer=1, texttype=10))
    lib.write_gds(path)
    back = lay.read_layout(path)
    assert _port_tuple(back.ports["opt1"]) == ((0.0, 0.0), 180.0, 0.5, (1, 0), lay.OPTICAL)
    assert _port_tuple(back.ports["opt2"]) == ((10.0, 0.0), 0.0, 0.5, (1, 0), lay.OPTICAL)
    assert lay.read_layout(path, port_layers=[(68, 0)]).ports["opt1"].layer == (68, 0)


def test_unnamed_pin_gets_a_name_with_warning(tmp_path):
    path = os.fspath(tmp_path / "p.gds")
    lib = gdstk.Library()
    c = lib.new_cell("c")
    c.add(gdstk.rectangle((0, -0.25), (10, 0.25), layer=1))
    c.add(gdstk.FlexPath([(9.95, 0), (10.05, 0)], 0.5, layer=1, datatype=10, simple_path=True))
    lib.write_gds(path)
    with pytest.warns(UserWarning, match="no name label"):
        back = lay.read_layout(path)
    assert list(back.ports) == ["pin1"]


LABEL_EXACT = ["straight", "bend_circular", "bend_euler", "sbend", "taper", "ring_single", "ring_double",
               "directional_coupler", "coupler", "crossing", "edge_coupler", "pn_modulator"]


@pytest.mark.parametrize("name", LABEL_EXACT)
def test_label_convention_recovers_face_ports(tmp_path, name):
    cell = getattr(lc, name)()
    path = os.fspath(tmp_path / f"{name}.gds")
    lay.write_layout(cell, path, sidecar=False, port_markers="label")
    back = lay.read_layout(path, port_convention="label", electrical_layers=[(2, 0)])
    want = {k: _port_tuple(p) for k, p in lay.snap_to_grid(cell).ports.items()}
    assert {k: _port_tuple(p) for k, p in back.ports.items()} == want
    assert not back.labels


def test_label_convention_documented_limits(tmp_path):
    def read(cell):
        path = os.fspath(tmp_path / "x.gds")
        lay.write_layout(cell, path, sidecar=False, port_markers="label")
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            back = lay.read_layout(path, port_convention="label")
        return back, [str(w.message) for w in rec]

    back, _ = read(lc.mmi1x2())                      # ports on the 2 um body face read the face width
    assert [p.width for p in back.ports.values()] == [2.0, 2.0, 2.0]
    back, _ = read(lc.bragg_grating(n_periods=10))   # faces are the corrugation teeth (w +/- dw/2)
    assert (back.ports["o1"].width, back.ports["o2"].width) == (0.55, 0.45)
    back, msgs = read(lc.heater_straight())          # e1/e2 sit on heater corners: ambiguous, kept as labels
    assert set(back.ports) == {"o1", "o2"} and any("corner" in m for m in msgs)
    assert sorted(lb.text for lb in back.labels) == ["e1", "e2"]
    back, _ = read(lc.pad())                         # a label inside a shape is not a port
    assert not back.ports and [lb.text for lb in back.labels] == ["e1"]
    back, _ = read(lc.grating_coupler())             # the fibre port has no geometry on its layer
    assert set(back.ports) == {"o1"}


def test_non_manhattan_label_port_within_documented_bound(tmp_path):
    cell = lc.bend_circular(radius=5.0, angle=37.0)
    path = os.fspath(tmp_path / "b.gds")
    lay.write_layout(cell, path, sidecar=False, port_markers="label")
    got = lay.read_layout(path).ports["o2"]
    bound = math.degrees(math.atan(math.sqrt(2) * 1e-3 / 0.5))
    assert abs(got.orientation - 37.0) <= bound
    assert abs(got.width - 0.5) <= 1e-3
    assert got.center == lay.snap_to_grid(cell).ports["o2"].center


def test_labels_inside_rotated_references_are_inferred_in_the_child_frame(tmp_path):
    top = _rotated_circuit()
    path = os.fspath(tmp_path / "r.gds")
    lay.write_layout(top, path, sidecar=False, port_markers="label")
    back = lay.read_layout(path)
    p = back.references[0].port("o2")                       # inferred in the child: exact
    assert p.orientation == pytest.approx(30.0, abs=1e-9) and p.width == 0.5
    top_port = back.ports["o1"]                               # inferred on the rotated, flattened geometry
    assert top_port.orientation == pytest.approx(210.0, abs=1e-6)
    assert top_port.width == pytest.approx(0.5, abs=1e-9)


def test_legacy_write_gds_labels_are_understood(tmp_path):
    path = os.fspath(tmp_path / "legacy.gds")
    lay.write_gds(lc.straight(12.0), path)
    back = lay.read_layout(path)
    assert {k: _port_tuple(p) for k, p in back.ports.items()} == {
        "o1": ((0.0, 0.0), 180.0, 0.5, (1, 0), lay.OPTICAL), "o2": ((12.0, 0.0), 0.0, 0.5, (1, 0), lay.OPTICAL)}


def _xor_area(a, b):
    pa = [gdstk.Polygon(p) for p in a]
    pb = [gdstk.Polygon(p) for p in b]
    return sum(p.area() for p in gdstk.boolean(pa, pb, "xor", precision=1e-6))


@pytest.mark.parametrize("make", [lc.mzi, lambda: lc.ring_double(radius=12.0), _route_circuit, _rotated_circuit])
def test_legacy_write_gds_round_trip_reproduces_tree_and_flat_geometry(tmp_path, make):
    """The plain write_gds output (no sidecar, no pins, 199-vertex fracturing) still reads back faithfully."""
    cell = make()
    path = os.fspath(tmp_path / "legacy.gds")
    lay.write_gds(cell, path)
    back = lay.read_layout(path)
    snapped = lay.snap_to_grid(cell)

    def tree(c):
        return (c.name.split("$")[0], [(r.cell.name.split("$")[0], tuple(np.round(r.origin, 9)), round(r.rotation, 9),
                                        r.mirror, tree(r.cell)) for r in c.references])

    assert tree(back) == tree(snapped)
    for layer in {lay_ for _p, lay_ in snapped.get_polygons()}:
        ours = [p for p, lay_ in back.get_polygons() if lay_ == layer]
        want = [p for p, lay_ in snapped.get_polygons() if lay_ == layer]
        perimeter = sum(np.linalg.norm(np.diff(np.vstack([p, p[:1]]), axis=0), axis=1).sum() for p in want)
        # write_gds fractures polygons above 199 vertices; the cut vertices land on the 1 nm grid,
        # so the flattened geometry agrees to well within one grid step along every edge
        assert _xor_area(ours, want) <= 0.1 * 1e-3 * perimeter
    for name, prt in snapped.ports.items():      # face ports of the top cell, inferred from the labels
        got = back.ports.get(name)
        if got is not None and prt.width == got.width:
            assert got.center == pytest.approx(prt.center, abs=1e-9) and got.layer == prt.layer
            assert abs((got.orientation - prt.orientation + 180) % 360 - 180) < 1e-6


def test_port_convention_none_keeps_markers_as_geometry(tmp_path):
    path = os.fspath(tmp_path / "n.gds")
    lay.write_layout(lc.straight(), path)
    back = lay.read_layout(path, port_convention="none", sidecar=False)
    assert not back.ports
    assert {lay_ for _p, lay_ in back.polygons} == {(1, 0), (1, 10)}
    assert sorted(lb.text for lb in back.labels) == ["o1", "o2"]


# --------------------------------------------------------------------------- #
# Arrays, magnification, paths, instance names
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fmt", ["gds", "oas"])
def test_arrays_expand_and_match_gdstk_flatten(tmp_path, fmt):
    path = os.fspath(tmp_path / f"a.{fmt}")
    lib = gdstk.Library()
    unit = lib.new_cell("unit")
    unit.add(gdstk.Polygon([(0, 0), (2, 0), (2, 1), (0.5, 1.5)], layer=1))
    unit.add(gdstk.rectangle((0, 0), (0.3, 0.3), layer=2))
    top = lib.new_cell("top")
    top.add(gdstk.Reference(unit, origin=(1, 2), columns=3, rows=2, spacing=(5, 7)))
    skew = gdstk.Reference(unit, origin=(-40, 3), rotation=np.pi / 2, x_reflection=True, columns=2, rows=3,
                           spacing=(4, 6))
    skew.set_gds_property(INSTANCE_NAME_ATTR, "sk")
    top.add(skew)
    dots = gdstk.rectangle((0, 50), (1, 51), layer=3)
    dots.repetition = gdstk.Repetition(offsets=[(3, 0), (5, 2)])
    top.add(dots)
    lib.write_gds(path) if fmt == "gds" else lib.write_oas(path)
    back = lay.read_layout(path)
    names = [r.name for r in back.references]
    assert names[:6] == [f"unit_{i}_{j}" for i in range(3) for j in range(2)]
    assert names[6:] == [f"sk_{i}_{j}" for i in range(2) for j in range(3)]
    ours = back.get_polygons()
    ref = _gdstk_flat(path, "top", fmt)
    assert len(ours) == len(ref) == 2 * 12 + 3
    oa, ra = _layer_areas(ours), _layer_areas(ref)
    assert oa.keys() == ra.keys() and all(oa[k] == pytest.approx(ra[k], rel=1e-12) for k in oa)
    key = lambda p: (p[1], tuple(np.round(np.mean(p[0], axis=0), 6)))   # noqa: E731
    assert sorted(map(key, ours)) == sorted(map(key, ref))


def test_aref_instances_follow_the_gdsii_lattice_definition(tmp_path):
    """AREF XY = origin, origin + cols*v_col, origin + rows*v_row (parent frame), decoded without gdstk."""
    path = os.fspath(tmp_path / "aref.gds")
    lib = gdstk.Library()
    unit = lib.new_cell("u")
    unit.add(gdstk.rectangle((0, 0), (1, 2)))
    lib.new_cell("top").add(gdstk.Reference(unit, origin=(3, -4), rotation=np.radians(30), x_reflection=True,
                                            columns=4, rows=3, spacing=(5, 9)))
    lib.write_gds(path)
    (aref,) = _decode_gds(path)[1]["top"]["arefs"]
    cols, rows = aref["colrow"]
    p1, p2, p3 = aref["xy"] * 1e-3
    want = sorted(tuple(np.round(p1 + i * (p2 - p1) / cols + j * (p3 - p1) / rows, 9))
                  for i in range(cols) for j in range(rows))
    back = lay.read_layout(path)
    assert sorted(tuple(np.round(r.origin, 9)) for r in back.references) == want
    assert {(r.rotation, r.mirror) for r in back.references} == {(round(aref["angle"], 9), aref["reflect"])}


def test_magnified_reference_is_flattened_or_rejected(tmp_path):
    path = os.fspath(tmp_path / "mag.gds")
    lib = gdstk.Library()
    child = lib.new_cell("child")
    child.add(gdstk.rectangle((0, 0), (1, 2), layer=1))
    lib.new_cell("top").add(gdstk.Reference(child, origin=(10, 0), rotation=np.pi / 4, magnification=2.5))
    lib.write_gds(path)
    with pytest.warns(UserWarning, match="magnification 2.5"):
        back = lay.read_layout(path)
    ref = back.references[0]
    assert ref.cell.model == lay.GEOMETRY_ONLY and ref.cell.name == "child_mag2.5"
    assert _layer_areas(back.get_polygons())[(1, 0)] == pytest.approx(2.0 * 2.5 ** 2, rel=1e-12)
    assert _layer_areas(_gdstk_flat(path, "top"))[(1, 0)] == pytest.approx(2.0 * 2.5 ** 2, rel=1e-9)
    with pytest.raises(ValueError, match="cannot scale"):
        lay.read_layout(path, magnification="error")


def test_paths_become_polygons(tmp_path):
    path = os.fspath(tmp_path / "p.gds")
    lib = gdstk.Library()
    c = lib.new_cell("c")
    c.add(gdstk.FlexPath([(0, 0), (10, 0), (10, 5)], 0.6, ends="round", layer=4))
    c.add(gdstk.RobustPath((0, 10), 0.4, layer=5).segment((8, 12)))
    lib.write_gds(path)
    back = lay.read_layout(path)
    ra, oa = _layer_areas(_gdstk_flat(path, "c")), _layer_areas(back.polygons)
    assert oa.keys() == ra.keys() == {(4, 0), (5, 0)}
    assert all(oa[k] == pytest.approx(ra[k], rel=1e-9) for k in oa)


def test_instance_names_and_duplicates(tmp_path):
    path = os.fspath(tmp_path / "n.gds")
    lib = gdstk.Library()
    child = lib.new_cell("child")
    child.add(gdstk.rectangle((0, 0), (1, 1)))
    top = lib.new_cell("top")
    for x, name in ((0, "dut"), (5, "dut"), (10, None)):
        ref = gdstk.Reference(child, origin=(x, 0))
        if name:
            ref.set_gds_property(INSTANCE_NAME_ATTR, name)
        top.add(ref)
    lib.write_gds(path)
    with pytest.warns(UserWarning, match="renamed"):
        back = lay.read_layout(path)
    assert [r.name for r in back.references] == ["dut", "dut$1", None]


@pytest.mark.filterwarnings("ignore:Missing reference")
def test_undefined_structure_is_an_error(tmp_path):
    path = os.fspath(tmp_path / "u.gds")
    lib = gdstk.Library()
    lib.new_cell("top").add(gdstk.Reference("missing", origin=(0, 0)))
    lib.write_gds(path)
    with pytest.raises(ValueError, match="'missing'"):
        lay.read_layout(path)


def test_flatten_keeps_geometry_ports_and_labels(tmp_path):
    cell = lay.LabeledCell.from_cell(lc.mzi())
    cell.add_label("chip", (1.0, 1.0), (10, 0))
    path = os.fspath(tmp_path / "f.gds")
    lay.write_layout(cell, path)
    flat = lay.read_layout(path, flatten=True)
    assert not flat.references and flat.model == lay.GEOMETRY_ONLY
    ra = _layer_areas([(p, lay_) for p, lay_ in _gdstk_flat(path, "mzi") if lay_ == (1, 0)])
    assert _layer_areas(flat.polygons)[(1, 0)] == pytest.approx(ra[(1, 0)], rel=1e-12)
    assert set(flat.ports) == {"o1", "o2"} and [lb.text for lb in flat.labels] == ["chip"]


# --------------------------------------------------------------------------- #
# Snapping, determinism and the bytes on disk
# --------------------------------------------------------------------------- #
def test_snap_error_and_degenerate_polygons(tmp_path):
    c = lay.Cell("c")
    c.add_polygon([(0, 0), (1.0004, 0), (1.0004, 0.5), (0, 0.5)])
    c.add_polygon([(0, 0), (1, 0.0002), (2, 0.0001)])              # collapses to a line
    path = os.fspath(tmp_path / "s.gds")
    with pytest.warns(UserWarning, match="1 polygon"):
        art = lay.write_layout(c, path)
    assert art.snap_error == pytest.approx(4e-4, abs=1e-12)
    assert len(lay.read_layout(path).polygons) == 1
    assert lay.write_layout(lc.straight(), path).snap_error == 0.0


@pytest.mark.parametrize("fmt", ["gds", "oas"])
def test_output_is_byte_reproducible(tmp_path, fmt):
    a, b = os.fspath(tmp_path / f"a.{fmt}"), os.fspath(tmp_path / f"b.{fmt}")
    art_a = lay.write_layout(lc.mzi(), a)
    art_b = lay.write_layout(lc.mzi(), b)
    assert art_a.sha256 == art_b.sha256
    assert open(a, "rb").read() == open(b, "rb").read()
    side_a = open(art_a.sidecar_path).read().replace("a." + fmt, "X")
    assert side_a == open(art_b.sidecar_path).read().replace("b." + fmt, "X")
    if fmt == "gds":
        other = lay.write_layout(lc.mzi(), a, timestamp=datetime.datetime(2020, 5, 17))
        assert other.sha256 != art_a.sha256 and other.structures == art_a.structures


def test_independent_decoder_agrees_with_read_layout(tmp_path):
    cell = _route_circuit()
    path = os.fspath(tmp_path / "d.gds")
    lay.write_layout(cell, path)
    units, structs = _decode_gds(path)
    assert units[0] == pytest.approx(1e-3, rel=1e-12) and units[1] == pytest.approx(1e-9, rel=1e-12)
    back = lay.read_layout(path)
    cells = {back.name: back}
    for ref in back.references:
        cells.setdefault(ref.cell.name, ref.cell)
    assert set(structs) == {"routed", "straight", "mmi1x2", "route"}
    for name, rec in structs.items():
        ours = cells[name]
        assert len(rec["boundaries"]) == len(ours.polygons)
        assert sum(_int_area(b["xy"]) for b in rec["boundaries"]) * 1e-6 == pytest.approx(
            sum(_area(p) for p, _l in ours.polygons), rel=1e-12)
        assert [s["props"].get(INSTANCE_NAME_ATTR) for s in rec["srefs"]] == [r.name for r in ours.references]
        pins = [p for p in rec["paths"] if (p["layer"], p["datatype"]) == (1, 10)]
        assert len(pins) == len(ours.ports)
        assert all(p["props"][PORT_LAYER_ATTR] == "1/0" for p in pins)
        assert sorted(t["string"] for t in rec["texts"]) == sorted(ours.ports)


def test_klayout_reader_agrees(tmp_path):
    kdb = pytest.importorskip("klayout.db")
    path = os.fspath(tmp_path / "k.gds")
    lay.write_layout(lc.mzi(), path)
    layout = kdb.Layout()
    layout.read(path)
    back = lay.read_layout(path)
    top = layout.top_cell()
    assert top.name == "mzi" and top.child_instances() == len(back.references)
    region = kdb.Region(top.begin_shapes_rec(layout.layer(1, 0)))
    ours = _layer_areas(back.get_polygons())[(1, 0)]
    merged = gdstk.boolean([gdstk.Polygon(p) for p, lay_ in back.get_polygons() if lay_ == (1, 0)], [], "or")
    assert region.merged().area() * layout.dbu ** 2 == pytest.approx(sum(p.area() for p in merged), rel=1e-6)
    assert ours > 0


# --------------------------------------------------------------------------- #
# Simulation semantics survive the round trip, gradients included
# --------------------------------------------------------------------------- #
@requires_jax
def test_round_tripped_layout_simulates_and_differentiates_identically(tmp_path):
    cell = lc.mzi(delta_length=20.0)
    path = os.fspath(tmp_path / "g.gds")
    lay.write_layout(cell, path)
    circ0 = px.circuit.circuit_from_netlist(lay.extract_netlist(cell))
    circ1 = px.circuit.circuit_from_netlist(lay.extract_netlist(lay.read_layout(path)))

    def fom(circ):
        return lambda wl: px.power(circ(wl=wl)[("o1", "o2")])

    wl = 1.551
    assert float(fom(circ1)(wl)) == pytest.approx(float(fom(circ0)(wl)), abs=1e-12)
    g1 = float(px.grad(fom(circ1))(wl))
    assert g1 == pytest.approx(float(px.grad(fom(circ0))(wl)), rel=1e-12)
    assert g1 == pytest.approx(float(finite_diff(fom(circ1), wl, h=1e-6)), rel=1e-5)
    assert abs(g1) > 1.0
