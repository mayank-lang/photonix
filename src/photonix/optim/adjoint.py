"""Helpers to build differentiable losses from a model + objective.

These wrap :func:`photonix.core.backend.value_and_grad` so that inverse design
reads as: pick a model, pick an objective, get gradients w.r.t. its parameters.
"""
from __future__ import annotations

from collections.abc import Callable

from photonix.core.backend import HAS_JAX, value_and_grad

__all__ = ["make_loss", "loss_and_grad"]


def make_loss(model: Callable, objective: Callable, *, wl=1.55, **objective_kwargs) -> Callable:
    """Compose ``params -> objective(model(wl=wl, **params))`` into a scalar loss.

    Parameters
    ----------
    model
        A callable ``model(*, wl, **params) -> SDict``.
    objective
        A callable ``objective(sdict, **objective_kwargs) -> scalar``.
    wl
        Wavelength(s) passed to the model.

    Returns
    -------
    callable
        ``loss(params: dict) -> scalar``.

    Examples
    --------
    >>> import photonix as px
    >>> from photonix.optim import make_loss, target_transmission
    >>> loss = make_loss(px.components.directional_coupler,
    ...                   target_transmission, port=("o1", "o3"), target=0.5)
    >>> float(loss({"coupling": 0.4})) > 0
    True
    """

    def loss(params: dict):
        s = model(wl=wl, **params)
        return objective(s, **objective_kwargs)

    return loss


def loss_and_grad(loss: Callable):
    """Return ``value_and_grad(loss)``; requires JAX.

    Raises
    ------
    RuntimeError
        If JAX is not installed (autodiff unavailable).
    """
    if not HAS_JAX:
        raise RuntimeError("loss_and_grad requires JAX. Install photonix[jax].")
    return value_and_grad(loss)
