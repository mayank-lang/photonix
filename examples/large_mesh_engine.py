"""Example: splitter-error tolerance of programmable meshes up to 32x32.

A 32x32 Clements mesh simulated at component level has 496 MZIs, 2016
instances and 6016 terminals. The circuit engine's sparse backend (the
``auto`` choice at this size) solves it in milliseconds per wavelength, which
makes a design study across mesh sizes practical: here, how fast the
fidelity of a Haar-random unitary degrades with coupler imbalance, plus a
reverse-mode gradient through the 32x32 mesh checked by finite differences.

Run:  python examples/large_mesh_engine.py
Saves: examples/outputs/large_mesh_engine.png
"""
from __future__ import annotations

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from _output import save
from scipy.stats import unitary_group

import photonix as px

SIZES = (4, 8, 16, 32)
SPLITS = (0.50, 0.49, 0.48, 0.46, 0.44)


def realized(model, n: int) -> np.ndarray:
    s = model(wl=1.55)
    return np.array([[complex(s[(f"o{i + 1}", f"o{2 * n - j}")]) for i in range(n)] for j in range(n)])


def fidelity(target: np.ndarray, actual: np.ndarray) -> float:
    """Phase-insensitive overlap |Tr(U^H S)|^2 / (N Tr(S^H S))."""
    n = target.shape[0]
    return float(abs(np.trace(target.conj().T @ actual)) ** 2 / (n * np.trace(actual.conj().T @ actual).real))


def main() -> None:
    infidelity = {}
    timing = {}
    for n in SIZES:
        U = unitary_group.rvs(n, random_state=n)
        program = px.circuit.clements_decompose(U)
        rows = []
        for split in SPLITS:
            model = px.circuit.mzi_mesh(program, splitter_coupling=split)
            S = realized(model, n)
            t0 = time.perf_counter()
            S = realized(model, n)          # timed second call: excludes one-off JAX compilation
            dt = time.perf_counter() - t0
            if split == 0.50:
                timing[n] = dt
                err = np.max(np.abs(S - U))
                print(f"{n:2d}x{n:<2d} mesh: {program.n_mzis:3d} MZIs, evaluated in {dt:.2f} s, "
                      f"max |S - U| = {err:.1e}")
                assert err < 1e-10
            rows.append(max(1.0 - fidelity(U, S), 1e-16))
        infidelity[n] = rows

    # Reverse-mode gradient through the 32x32 mesh (sparse LU adjoint).
    n = 32
    U = unitary_group.rvs(n, random_state=n)
    model = px.circuit.mzi_mesh(px.circuit.clements_decompose(U), splitter_coupling=0.48)

    Uh = px.xp.asarray(U.conj())

    def overlap(theta):
        """Re Tr(U^H S) / N as a function of one internal heater phase."""
        s = model(wl=1.55, int100={"phase0": theta})
        return sum(px.xp.real(Uh[j, i] * s[(f"o{i + 1}", f"o{2 * n - j}")]) for i in range(n) for j in range(n)) / n

    theta0 = float(px.circuit.clements_decompose(U).mzis[100][1])
    if px.HAS_JAX:
        g = float(px.grad(overlap)(theta0))
        h = 1e-6
        fd = float((overlap(theta0 + h) - overlap(theta0 - h)) / (2 * h))
        print(f"d Re Tr(U^H S)/N / d theta_100 through 6016 terminals: adjoint {g:.9f}, finite difference {fd:.9f}")
        assert abs(g - fd) < 1e-6 and abs(g) > 1e-4

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    imbalance = 0.5 - np.asarray(SPLITS)
    for n_, rows in infidelity.items():
        axes[0].plot(imbalance[1:], rows[1:], "o-", label=f"{n_}x{n_}")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("coupler imbalance 0.5 - kappa")
    axes[0].set_ylabel("1 - fidelity")
    axes[0].set_title("Haar-random unitary, uncorrected mesh")
    axes[0].legend()
    sizes = np.asarray(list(timing))
    axes[1].plot(sizes * (sizes - 1) // 2, list(timing.values()), "s-")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("number of MZIs")
    axes[1].set_ylabel("model + solve time [s], one wavelength")
    axes[1].set_title("Component-level mesh evaluation")
    fig.tight_layout()
    print("Saved", save(fig, "large_mesh_engine.png"))


if __name__ == "__main__":
    main()
