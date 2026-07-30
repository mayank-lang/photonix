"""Tests for layout primitives, GDS I/O, and netlist extraction."""
from __future__ import annotations

import os
import tempfile

import photonix.layout as lay
from photonix.layout import components as lc


def test_cell_build_and_bbox():
    c = lc.straight(10.0, width=0.5)
    assert len(c.ports) == 2
    (xmin, ymin), (xmax, ymax) = c.bbox()
    assert xmax - xmin >= 10.0 - 1e-9
    assert abs((ymax - ymin) - 0.5) < 1e-9


def test_gds_roundtrip():
    import pytest

    pytest.importorskip("gdstk")
    top = lay.Cell("demo")
    top.add_ref(lc.straight(10.0), origin=(0, 0), name="a")
    top.add_ref(lc.straight(10.0), origin=(10, 0), name="b")
    path = os.path.join(tempfile.mkdtemp(), "demo.gds")
    lay.write_gds(top, path)
    assert os.path.getsize(path) > 0
    rd = lay.read_gds(path)
    assert len(rd.polygons) >= 2


def test_oasis_roundtrip_when_gdstk_is_installed():
    """Exercise the real serializer when available; minimal installs skip locally."""
    import pytest

    pytest.importorskip("gdstk")
    top = lay.Cell("oas_demo")
    top.add_ref(lc.straight(10.0), origin=(0, 0), name="a")
    path = os.path.join(tempfile.mkdtemp(), "demo.oas")
    lay.write_oas(top, path, validation="crc32")
    assert os.path.getsize(path) > 0
    rd = lay.read_oas(path)
    assert len(rd.polygons) >= 1


def test_route_connects_ports():
    a = lay.Port("a", (0, 0), 0)
    b = lay.Port("b", (20, 10), 180)
    r = lay.route(a, b)
    assert len(r.polygons) == 1


def test_extract_netlist_finds_connection():
    top = lay.Cell("top")
    top.add_ref(lc.straight(10.0), origin=(0, 0), name="a")
    top.add_ref(lc.straight(10.0), origin=(10, 0), name="b")
    nl = lay.extract_netlist(top)
    insts = getattr(nl, "instances", None) or nl["instances"]
    conns = getattr(nl, "connections", None) or nl["connections"]
    assert set(insts) == {"a", "b"}
    assert ("a", "o2") in conns and conns[("a", "o2")] == ("b", "o1")
