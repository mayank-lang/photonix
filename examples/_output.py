"""Shared output-path helper for the example scripts.

Every example writes its figure to ``examples/outputs/``, regardless of the
directory it is launched from. Using a bare ``fig.savefig("name.png")`` instead
resolves against the *current working directory*, so running an example from the
repository root silently scattered PNGs there.
"""
from __future__ import annotations

import os

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def save(fig, filename: str, *, dpi: int = 130) -> str:
    """Save ``fig`` into ``examples/outputs/`` and return the full path."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=dpi)
    return path
