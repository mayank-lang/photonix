"""Example: inverse-design a directional coupler to a target split ratio.

Uses end-to-end gradients (JAX) through the component model and an Adam loop to
find the coupling that yields a target cross-port power. Demonstrates the
differentiable-design workflow that sets photonix apart.

Run:  python examples/inverse_design_coupler.py
Saves: examples/outputs/inverse_design_coupler.png
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _output import save

import photonix as px
import photonix.components as comp
import photonix.optim as opt


def main() -> None:
    if not px.HAS_JAX:
        print("This example needs the JAX backend: pip install 'photonix[jax]'.")
        return

    target = 0.25  # want 25% cross-coupled power at o3
    loss = opt.make_loss(comp.directional_coupler, opt.target_transmission,
                         wl=1.55, port=("o1", "o3"), target=target)
    res = opt.adam(loss, {"coupling": 0.5}, steps=300, lr=0.02)

    achieved = float(px.power(comp.directional_coupler(coupling=float(res.params["coupling"]))[("o1", "o3")]))
    print(f"target cross power = {target:.3f}")
    print(f"optimized coupling = {float(res.params['coupling']):.4f}")
    print(f"achieved cross power = {achieved:.4f}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(res.history)
    ax.set_xlabel("Adam iteration")
    ax.set_ylabel("loss (MSE to target)")
    ax.set_yscale("log")
    ax.set_title("Inverse design of a directional coupler")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    print("Saved", save(fig, "inverse_design_coupler.png"))


if __name__ == "__main__":
    main()
