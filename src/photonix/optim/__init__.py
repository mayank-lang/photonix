"""photonix inverse-design / optimization subpackage.

End-to-end differentiable optimization of photonic circuits: define a model,
choose a differentiable objective, and optimize its parameters with gradients
that flow through the whole simulation.
"""
from __future__ import annotations

from .adjoint import loss_and_grad, make_loss
from .objectives import (
    extinction_ratio,
    flatness,
    insertion_loss,
    match_spectrum,
    target_transmission,
)
from .optimizers import OptResult, adam, minimize_scipy

__all__ = [
    "make_loss", "loss_and_grad",
    "target_transmission", "match_spectrum", "insertion_loss",
    "extinction_ratio", "flatness",
    "adam", "minimize_scipy", "OptResult",
]
