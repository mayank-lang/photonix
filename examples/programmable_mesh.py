"""Example: programming a 6-mode discrete Fourier transform onto an MZI mesh.

``clements_decompose`` returns the heater phases of a rectangular MZI mesh that
implements a target unitary; ``mzi_mesh`` simulates the mesh as a full
component-level circuit (two couplers and two phase shifters per MZI), so the
effect of fabrication errors on the realized transform can be studied
directly.

Run:  python examples/programmable_mesh.py
Saves: examples/outputs/programmable_mesh.png
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from _output import save

import photonix as px

N = 6


def realized(model) -> np.ndarray:
    s = model(wl=1.55)
    return np.array([[complex(s[(f"o{i + 1}", f"o{2 * N - j}")]) for i in range(N)] for j in range(N)])


def fidelity(target: np.ndarray, actual: np.ndarray) -> float:
    """Phase-insensitive overlap |Tr(U^H S)|^2 / (N Tr(S^H S))."""
    return float(abs(np.trace(target.conj().T @ actual)) ** 2 / (N * np.trace(actual.conj().T @ actual).real))


def main() -> None:
    k = np.arange(N)
    dft = np.exp(-2j * np.pi * np.outer(k, k) / N) / np.sqrt(N)
    program = px.circuit.clements_decompose(dft)
    print(f"{program.n_mzis} MZIs; first three (mode, theta, phi): "
          + ", ".join(f"({m}, {t:.3f}, {p:.3f})" for m, t, p in program.mzis[:3]))

    ideal = realized(px.circuit.mzi_mesh(program))
    print(f"ideal mesh: max |S - U| = {np.max(np.abs(ideal - dft)):.2e}")

    splits = np.linspace(0.40, 0.50, 11)
    fid = [fidelity(dft, realized(px.circuit.mzi_mesh(program, splitter_coupling=float(c)))) for c in splits]
    for c_, f_ in zip(splits[::5], fid[::5], strict=True):
        print(f"splitter {c_:.2f}: fidelity {f_:.5f}")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, data, title in ((axes[0], np.abs(dft) ** 2, "target |U|^2 (DFT)"),
                            (axes[1], np.abs(realized(px.circuit.mzi_mesh(program, splitter_coupling=0.44))) ** 2,
                             "realized |S|^2, 44:56 splitters")):
        im = ax.imshow(data, vmin=0, vmax=0.35, cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("input mode")
        ax.set_ylabel("output mode")
        fig.colorbar(im, ax=ax, fraction=0.046)
    axes[2].plot(splits, 1 - np.asarray(fid), "o-")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("coupler power splitting")
    axes[2].set_ylabel("1 - fidelity")
    axes[2].set_title("Infidelity from splitter imbalance")
    fig.tight_layout()
    print("Saved", save(fig, "programmable_mesh.png"))
    assert np.allclose(ideal, dft, atol=1e-10)


if __name__ == "__main__":
    main()
