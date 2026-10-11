"""Example: a scripted hand-off with the ``photonix`` command line.

1. ``photonix sweep``: evaluate a ring model over the C band into a Touchstone file.
2. ``photonix sparams check``: gate the file on passivity and reciprocity (exit 0).
3. ``photonix sparams check`` on a deliberately non-passive copy (exit 1), as a
   CI job or tape-out hand-off script would see it.
4. Read the file back with the library and plot it.

The commands are called through :func:`photonix.cli.main`, which returns the
exit code instead of exiting; the shell equivalent is shown for each step.

Run:  python examples/cli_handoff.py
Saves: examples/outputs/cli_handoff.png
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from _output import save  # noqa: E402

import photonix as px  # noqa: E402
from photonix.cli import main  # noqa: E402


def run(*argv: str) -> int:
    print("$ photonix", " ".join(argv))
    code = main(list(argv))
    print(f"(exit {code})\n")
    return code


def main_example() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        ring = str(Path(tmp) / "ring.s2p")
        assert run("sweep", "ring_single", "--wl", "1.545:1.555:2001", "-p", "radius=8", "-p", "gap=0.22",
                   "-o", ring) == 0
        assert run("sparams", "check", ring) == 0

        ds = px.SParameterDataset.load_touchstone(ring)
        gain = px.SParameterDataset(ds.wavelengths, ds.ports, 1.02 * ds.s, ds.metadata)  # 2 % amplitude gain
        bad = str(Path(tmp) / "ring_gain.s2p")
        gain.save_touchstone(bad)
        assert run("sparams", "check", bad) == 1

        through = np.abs(ds.s[:, ds.ports.index("o2"), ds.ports.index("o1")]) ** 2
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(ds.wavelengths * 1e3, 10 * np.log10(through))
        ax.set_xlabel("wavelength (nm)")
        ax.set_ylabel("through |S21|$^2$ (dB)")
        ax.set_title("ring_single (R = 8 um, gap = 0.22 um) read back from Touchstone")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        print("Saved", save(fig, "cli_handoff.png"))


if __name__ == "__main__":
    main_example()
