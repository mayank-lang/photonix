"""External-solver adapters.

Each adapter exposes ``run_all() -> dict[case_key, value]`` and must raise (not
silently return) if its solver isn't importable, so ``run.py`` can report the
skip. Adapters are intentionally optional: the benchmark runs photonix-only by
default and only touches these when invoked with ``--external``.
"""
from __future__ import annotations

from . import meep_adapter, tidy3d_adapter

ADAPTERS = {
    "meep": meep_adapter,
    "tidy3d": tidy3d_adapter,
}
