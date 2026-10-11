"""The API reference documents every public name.

The pages are planned by ``docs/_ext/photonix_apigen.py`` (standard library
plus photonix only, so this runs without Sphinx). The plan is checked here for
completeness: every name in every public module's ``__all__`` must be described
on some page. When a built site is available (``docs/_build/html`` or the
directory named by ``PHOTONIX_DOCS_HTML``), its ``objects.inv`` is also checked
to contain every planned object, i.e. autodoc really rendered it.
"""
from __future__ import annotations

import importlib
import os
import re
import sys
import zlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
sys.path.insert(0, str(DOCS / "_ext"))

import photonix_apigen  # noqa: E402

# Implementation modules of the command-line interface: the commands are
# documented as a command line in docs/CLI.md, the Python entry points
# (photonix.cli) in the API reference.
UNDOCUMENTED_PREFIXES = ("photonix.cli.commands",)

_PRIMITIVE = (int, float, complex, str, bytes, bool, type(None))


@pytest.fixture(scope="module")
def pages() -> dict[str, photonix_apigen.Page]:
    return photonix_apigen.plan()


@pytest.fixture(scope="module")
def documented(pages) -> set[str]:
    return {e.target for p in pages.values() for e in p.documented}


def _resolve(name: str) -> object:
    parts = name.split(".")
    for split in range(len(parts), 0, -1):
        try:
            obj: object = importlib.import_module(".".join(parts[:split]))
        except ImportError:
            continue
        for attr in parts[split:]:
            obj = getattr(obj, attr)
        return obj
    raise ImportError(name)


def _public_modules() -> dict[str, object]:
    return {name: mod for name, mod in photonix_apigen._all_modules().items()
            if not photonix_apigen._private(name) and hasattr(mod, "__all__")
            and not name.startswith(UNDOCUMENTED_PREFIXES)}


def test_every_exported_name_has_a_documented_target(pages, documented):
    missing = [(p.module, e.name, e.target) for p in pages.values() for e in p.exports
               if e.kind != "module" and e.target not in documented]
    assert not missing


def test_every_page_module_is_listed_by_its_parent(pages):
    children = {m for p in pages.values() for m in p.submodules}
    orphans = sorted(set(pages) - children - {photonix_apigen.ROOT_PACKAGE})
    assert not orphans


def test_every_public_all_is_covered(pages, documented):
    """Each name of each public ``__all__`` is documented, here or at its home."""
    by_id: dict[int, str] = {}
    for target in documented:
        obj = _resolve(target)
        if not isinstance(obj, _PRIMITIVE):
            by_id[id(obj)] = target
    uncovered = []
    for modname, mod in _public_modules().items():
        planned = {e.name for e in pages[modname].exports} if modname in pages else set()
        for name in mod.__all__:
            if name in planned:
                continue
            value = getattr(mod, name, None)
            if value is None and importlib.util.find_spec(f"{modname}.{name}") is not None:
                value = importlib.import_module(f"{modname}.{name}")
            if hasattr(value, "__name__") and getattr(value, "__name__", None) in pages:
                continue  # a module with its own page
            if isinstance(value, _PRIMITIVE) or id(value) not in by_id:
                uncovered.append(f"{modname}.{name}")
    assert not uncovered, f"public names missing from the API reference: {uncovered}"


def test_targets_resolve_to_objects(documented):
    for target in sorted(documented):
        _resolve(target)


def _inventory(path: Path) -> set[str]:
    """Python-domain names in a Sphinx ``objects.inv`` (format version 2)."""
    raw = path.read_bytes()
    header_end = 0
    for _ in range(4):
        header_end = raw.index(b"\n", header_end) + 1
    assert raw.startswith(b"# Sphinx inventory version 2")
    body = zlib.decompress(raw[header_end:]).decode("utf-8")
    names = set()
    for line in body.splitlines():
        match = re.match(r"(.+?)\s+(py:\w+)\s+-?\d+\s+", line)
        if match:
            names.add(match.group(1))
    return names


def _built_inventory() -> Path | None:
    html = Path(os.environ.get("PHOTONIX_DOCS_HTML", DOCS / "_build" / "html"))
    inv = html / "objects.inv"
    return inv if inv.is_file() else None


def test_built_site_contains_every_documented_object(pages, documented):
    inv = _built_inventory()
    if inv is None:
        pytest.skip("no built site (run: python -m sphinx -W -b html docs docs/_build/html)")
    names = _inventory(inv)
    missing = sorted((documented | set(pages)) - names)
    assert not missing, f"planned but not rendered: {missing[:20]} ({len(missing)} total)"
