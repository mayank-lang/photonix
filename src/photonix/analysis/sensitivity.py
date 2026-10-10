"""Parametric sensitivity, moment propagation, corners and worst-case distance.

These tools answer the questions that follow a nominal simulation: *which
parameters matter* (:func:`sensitivity`), *how much does the response spread*
(:func:`fosm`), *which deterministic cases bracket it* (:func:`corners`), and
*how far is the nominal design from failing a spec* (:func:`worst_case_distance`,
:func:`importance_sampling_yield`).

Gradients come from JAX automatic differentiation when it is available and the
function is traceable; otherwise (NumPy backend, or a function containing
non-differentiable code) central finite differences with Richardson
extrapolation are used.

References
----------
* Hasofer & Lind, J. Eng. Mech. Div. ASCE 100(1), 111 (1974) -- FOSM / reliability index.
* Antreich, Graeb & Wieser, IEEE TCAD 13(1), 57 (1994) -- worst-case distance.
* Plackett & Burman, Biometrika 33(4), 305 (1946) -- screening designs.
* Hesterberg, Technometrics 37(2), 185 (1995) -- defensive importance sampling.
"""
from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "SensitivityTable", "sensitivity", "MomentEstimate", "fosm", "corners",
    "WorstCase", "worst_case_distance", "ISYield", "importance_sampling_yield",
]


# --------------------------------------------------------------------------- #
# Gradients and Hessians
# --------------------------------------------------------------------------- #
def _as_vector_fn(fn: Callable, names: Sequence[str]):
    def vec(x):
        return fn({k: x[i] for i, k in enumerate(names)})

    return vec


def _fd_gradient(f: Callable, x0: np.ndarray, step: float) -> np.ndarray:
    """Central differences with one Richardson step (O(h^4))."""
    g = np.zeros_like(x0)
    for i in range(x0.size):
        h = step * max(1.0, abs(x0[i]))
        e = np.zeros_like(x0)
        e[i] = 1.0

        def d(hh, e=e):
            return (float(f(x0 + hh * e)) - float(f(x0 - hh * e))) / (2 * hh)

        g[i] = (4 * d(h / 2) - d(h)) / 3
    return g


def _fd_hessian(f: Callable, x0: np.ndarray, step: float) -> np.ndarray:
    n = x0.size
    hess = np.zeros((n, n))
    hs = [step * max(1.0, abs(v)) for v in x0]
    f0 = float(f(x0))
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = hs[i]
        hess[i, i] = (float(f(x0 + ei)) - 2 * f0 + float(f(x0 - ei))) / hs[i] ** 2
        for j in range(i + 1, n):
            ej = np.zeros(n)
            ej[j] = hs[j]
            v = (float(f(x0 + ei + ej)) - float(f(x0 + ei - ej)) - float(f(x0 - ei + ej))
                 + float(f(x0 - ei - ej))) / (4 * hs[i] * hs[j])
            hess[i, j] = hess[j, i] = v
    return hess


def _gradient(f: Callable, x0: np.ndarray, method: str, step: float) -> tuple[np.ndarray, str]:
    from photonix.core.backend import HAS_JAX

    if method in ("auto", "ad") and HAS_JAX:
        import jax
        import jax.numpy as jnp

        try:
            g = np.asarray(jax.grad(lambda x: jnp.real(f(x)))(jnp.asarray(x0, dtype=float)), dtype=float)
            if np.all(np.isfinite(g)):
                return g, "ad"
        except Exception:  # noqa: BLE001 - untraceable function: fall back
            if method == "ad":
                raise
    elif method == "ad":
        raise RuntimeError("method='ad' requires JAX")
    return _fd_gradient(f, x0, step), "fd"


def _hessian(f: Callable, x0: np.ndarray, method: str, step: float) -> np.ndarray:
    from photonix.core.backend import HAS_JAX

    if method in ("auto", "ad") and HAS_JAX:
        import jax
        import jax.numpy as jnp

        try:
            h = np.asarray(jax.hessian(lambda x: jnp.real(f(x)))(jnp.asarray(x0, dtype=float)), dtype=float)
            if np.all(np.isfinite(h)):
                return h
        except Exception:  # noqa: BLE001
            if method == "ad":
                raise
    return _fd_hessian(f, x0, step)


def _covariance(names: Sequence[str], covariance) -> np.ndarray:
    from photonix.pdk.process import MonteCarloSpec

    if isinstance(covariance, MonteCarloSpec):
        if tuple(covariance.parameters) != tuple(names):
            order = [covariance.parameters.index(n) for n in names]
            return np.asarray(covariance.covariance)[np.ix_(order, order)]
        return np.asarray(covariance.covariance, dtype=float)
    cov = np.asarray(covariance, dtype=float)
    if cov.ndim == 1:
        cov = np.diag(cov)
    if cov.shape != (len(names), len(names)):
        raise ValueError(f"covariance must be ({len(names)}, {len(names)})")
    return cov


# --------------------------------------------------------------------------- #
# Sensitivity table
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SensitivityTable:
    """First-order sensitivities of a scalar metric.

    Attributes
    ----------
    names, values
        Parameter names and nominal values.
    value
        Metric at the nominal point.
    gradient
        ``d metric / d parameter``.
    normalized
        Dimensionless elasticities ``(p / f) d f / d p`` (``nan`` where ``f`` or
        ``p`` is zero).
    method
        ``"ad"`` or ``"fd"``.
    """

    names: tuple[str, ...]
    values: np.ndarray
    value: float
    gradient: np.ndarray
    normalized: np.ndarray
    method: str

    def ranked(self, *, by: str = "normalized") -> list[tuple[str, float]]:
        """Parameters sorted by decreasing ``|normalized|`` (or ``|gradient|``)."""
        data = self.normalized if by == "normalized" else self.gradient
        key = np.where(np.isfinite(data), np.abs(data), -1.0)
        return [(self.names[i], float(data[i])) for i in np.argsort(-key)]

    def as_rows(self) -> list[dict]:
        return [{"parameter": n, "value": float(v), "gradient": float(g), "normalized": float(e)}
                for n, v, g, e in zip(self.names, self.values, self.gradient, self.normalized, strict=True)]


def sensitivity(fn: Callable[[dict], float], params: Mapping[str, float], *, method: str = "auto",
                fd_step: float = 1e-5) -> SensitivityTable:
    """Gradient of a scalar ``fn(params_dict)`` at ``params``.

    Examples
    --------
    >>> import photonix as px
    >>> def resonance_power(p):
    ...     s = px.components.all_pass_ring(wl=1.55, neff=p["neff"], coupling=p["coupling"])
    ...     return px.power(s[("o1", "o2")])
    >>> t = sensitivity(resonance_power, {"neff": 2.4, "coupling": 0.1})
    >>> t.names, t.method in ("ad", "fd")
    (('neff', 'coupling'), True)
    """
    if method not in ("auto", "ad", "fd"):
        raise ValueError("method must be 'auto', 'ad' or 'fd'")
    names = tuple(params)
    x0 = np.array([float(params[k]) for k in names])
    f = _as_vector_fn(fn, names)
    value = float(np.real(f(x0)))
    g, used = _gradient(f, x0, method, fd_step) if method != "fd" else (_fd_gradient(f, x0, fd_step), "fd")
    with np.errstate(divide="ignore", invalid="ignore"):
        norm = np.where((x0 != 0) & (value != 0), g * x0 / value, np.nan)
    return SensitivityTable(names, x0, value, g, norm, used)


# --------------------------------------------------------------------------- #
# First-order second-moment propagation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MomentEstimate:
    """Moments of a metric under Gaussian parameter variation (FOSM)."""

    mean: float
    std: float
    mean_shift_2nd: float
    gradient: np.ndarray

    @property
    def mean_2nd(self) -> float:
        """Mean including the second-order shift ``tr(H Sigma) / 2``."""
        return self.mean + self.mean_shift_2nd


def fosm(fn: Callable[[dict], float], nominal: Mapping[str, float], covariance, *, second_order: bool = False,
         method: str = "auto", fd_step: float = 1e-5) -> MomentEstimate:
    """First-order second-moment estimate ``(f(p0), sqrt(g^T Sigma g))``.

    ``covariance`` is a matrix, a vector of variances, or a
    :class:`~photonix.pdk.MonteCarloSpec`. With ``second_order=True`` the mean
    shift ``tr(H Sigma) / 2`` from the Hessian is also returned.

    Examples
    --------
    >>> m = fosm(lambda p: 2 * p["a"] - 3 * p["b"], {"a": 1.0, "b": 0.5}, [0.01, 0.04])
    >>> round(m.mean, 12), round(m.std, 12)
    (0.5, 0.632455532034)
    """
    names = tuple(nominal)
    x0 = np.array([float(nominal[k]) for k in names])
    cov = _covariance(names, covariance)
    f = _as_vector_fn(fn, names)
    g, _ = _gradient(f, x0, method, fd_step) if method != "fd" else (_fd_gradient(f, x0, fd_step), "fd")
    shift = 0.0
    if second_order:
        h = _hessian(f, x0, method, fd_step)
        shift = 0.5 * float(np.trace(h @ cov))
    return MomentEstimate(float(np.real(f(x0))), float(math.sqrt(max(g @ cov @ g, 0.0))), shift, g)


# --------------------------------------------------------------------------- #
# Corners
# --------------------------------------------------------------------------- #
def _plackett_burman(k: int) -> np.ndarray:
    """Two-level Plackett-Burman screening design with ``n >= k + 1`` runs (n % 4 == 0).

    Built from a Sylvester-Hadamard matrix (n a power of two), whose columns
    beyond the first are mutually orthogonal and balanced.
    """
    n = 4
    while n < k + 1:
        n *= 2
    h = np.array([[1]])
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h[:, 1:k + 1]


def corners(nominal: Mapping[str, float], deltas: Mapping[str, float], *, design: str = "full_factorial",
            fn: Callable[[dict], float] | None = None):
    """Deterministic process cases around ``nominal``.

    Parameters
    ----------
    deltas
        Half-ranges ``+/- delta`` per parameter (a subset of ``nominal``).
    design
        ``"full_factorial"`` (all ``2^k`` sign patterns, ``k <= 12``),
        ``"plackett_burman"`` (orthogonal screening design), ``"axial"``
        (one parameter at a time, ``2k`` cases) or ``"gradient_sign"`` (the
        two linear worst corners ``p0 +/- sign(grad) delta``; needs ``fn``).

    Returns
    -------
    tuple of :class:`~photonix.pdk.ProcessCorner`
        Directly usable as ``yield_analysis(..., cases=...)`` (the nominal is
        not included).

    Examples
    --------
    >>> cs = corners({"w": 0.5, "t": 0.22}, {"w": 0.01, "t": 0.005})
    >>> len(cs), sorted(cs[0].parameters)
    (4, ['t', 'w'])
    """
    from photonix.pdk.process import ProcessCorner

    names = [k for k in nominal if k in deltas]
    if set(deltas) - set(nominal):
        raise KeyError(f"deltas for unknown parameters: {sorted(set(deltas) - set(nominal))}")
    base = {k: float(v) for k, v in nominal.items()}

    def case(label, signs):
        p = dict(base)
        for n, sgn in zip(names, signs, strict=True):
            p[n] = base[n] + sgn * float(deltas[n])
        return ProcessCorner(label, p, {"kind": "corner", "design": design})

    if design == "full_factorial":
        if len(names) > 12:
            raise ValueError("full_factorial is limited to 12 parameters; use plackett_burman")
        return tuple(case("ff_" + "".join("+" if s > 0 else "-" for s in signs), signs)
                     for signs in itertools.product((-1, 1), repeat=len(names)))
    if design == "plackett_burman":
        m = _plackett_burman(len(names))
        return tuple(case(f"pb_{i:03d}", row) for i, row in enumerate(m))
    if design == "axial":
        out = []
        for i, n in enumerate(names):
            for sgn in (-1, 1):
                signs = [0] * len(names)
                signs[i] = sgn
                out.append(case(f"ax_{n}{'+' if sgn > 0 else '-'}", signs))
        return tuple(out)
    if design == "gradient_sign":
        if fn is None:
            raise ValueError("design='gradient_sign' needs fn")
        g = sensitivity(fn, base).gradient
        idx = [list(base).index(n) for n in names]
        signs = np.sign(g[idx])
        signs[signs == 0] = 1
        return (case("worst_high", signs), case("worst_low", -signs))
    raise ValueError("design must be full_factorial, plackett_burman, axial or gradient_sign")


# --------------------------------------------------------------------------- #
# Worst-case distance
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class WorstCase:
    """Nearest spec-boundary point in whitened parameter space.

    ``beta`` is the worst-case distance in standard deviations (positive when
    the nominal design meets the spec, negative when it fails), ``point`` the
    parameter values on the boundary, and ``yield_linear = Phi(beta)`` the
    yield of a linearized one-sided spec.
    """

    beta: float
    point: dict
    metric_at_point: float
    yield_linear: float
    converged: bool
    bound: float
    side: str = field(default="")


def worst_case_distance(fn: Callable[[dict], object], spec, nominal: Mapping[str, float], covariance, *,
                        n_starts: int = 4, seed: int = 0, method: str = "auto") -> WorstCase:
    """Antreich-Graeb worst-case distance of ``spec`` (one bound) from ``nominal``.

    As in :func:`~photonix.analysis.yield_analysis`, ``fn(params)`` returns a
    result and ``spec.metric(result)`` the number the spec bounds. In
    whitened coordinates ``z = L^-1 (p - p0)`` with ``Sigma = L L^T`` the
    worst-case point minimizes ``|z|`` subject to ``f(p) = bound``; ``beta`` is
    that distance, signed by whether the nominal passes. For a spec with both
    bounds, the bound the nominal is closer to (in ``f``) is used.

    Examples
    --------
    >>> from photonix.analysis import Spec
    >>> wc = worst_case_distance(lambda p: p["a"] + p["b"], Spec("s", lambda v: v, maximum=3.0),
    ...                          {"a": 1.0, "b": 0.0}, [1.0, 1.0])
    >>> round(wc.beta, 6), round(wc.yield_linear, 4)          # (3 - 1) / sqrt(2)
    (1.414214, 0.9214)
    """
    from scipy.optimize import minimize
    from scipy.stats import norm

    names = tuple(nominal)
    x0 = np.array([float(nominal[k]) for k in names])
    cov = _covariance(names, covariance)
    chol = np.linalg.cholesky(cov + 1e-15 * np.eye(len(names)) * max(1.0, float(np.max(np.abs(cov)))))
    f = _as_vector_fn(lambda p: spec.metric(fn(p)), names)
    f0 = float(np.real(f(x0)))
    bounds = [b for b in (spec.minimum, spec.maximum) if b is not None]
    bound = min(bounds, key=lambda b: abs(b - f0))
    side = "max" if bound == spec.maximum else "min"
    passes = spec.passes(f0)

    def g(z):
        return float(np.real(f(x0 + chol @ z))) - bound

    def grad_g(z):
        gp, _ = _gradient(f, x0 + chol @ z, method, 1e-6)
        return chol.T @ gp

    rng = np.random.default_rng(seed)
    gz0 = grad_g(np.zeros(len(names)))
    best = None
    starts = [np.zeros(len(names))]
    if np.linalg.norm(gz0) > 0:
        starts[0] = -g(np.zeros(len(names))) * gz0 / float(gz0 @ gz0)   # linear guess
    starts += [rng.standard_normal(len(names)) for _ in range(max(0, n_starts - 1))]
    for z_start in starts:
        res = minimize(lambda z: 0.5 * float(z @ z), z_start, jac=lambda z: z, method="SLSQP",
                       constraints=[{"type": "eq", "fun": g, "jac": grad_g}],
                       options={"maxiter": 200, "ftol": 1e-14})
        if res.success and abs(g(res.x)) < 1e-6 * max(1.0, abs(bound)):
            if best is None or res.fun < best.fun:
                best = res
    converged = best is not None
    z = best.x if best is not None else starts[0]
    beta = float(np.linalg.norm(z)) * (1.0 if passes else -1.0)
    point = {k: float(v) for k, v in zip(names, x0 + chol @ z, strict=True)}
    return WorstCase(beta, point, float(np.real(f(x0 + chol @ z))), float(norm.cdf(beta)), converged, bound, side)


# --------------------------------------------------------------------------- #
# Importance sampling
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ISYield:
    """Importance-sampling yield estimate."""

    fraction: float
    failure_probability: float
    ci: tuple[float, float]
    ess: float
    samples: int


def importance_sampling_yield(fn: Callable[[dict], float], specs, nominal: Mapping[str, float], covariance, *,
                              samples: int = 2000, seed: int = 0, shift: Mapping[str, float] | str = "worst_case",
                              defensive: float = 0.1, confidence: float = 0.95) -> ISYield:
    """Yield of rare-failure designs by defensive mixture importance sampling.

    Samples are drawn from ``(1 - defensive) N(p_shift, Sigma) +
    defensive N(p0, Sigma)``; with ``shift="worst_case"`` the shifted centre is
    the worst-case point of the most critical spec. Weights are the ratio of
    the nominal density to the mixture density; the failure probability is
    the self-normalized weighted failure fraction, with a delta-method normal
    confidence interval and Kish's effective sample size.

    Examples
    --------
    >>> from photonix.analysis import Spec
    >>> r = importance_sampling_yield(lambda p: p["x"], [Spec("x", lambda v: v, maximum=3.0)],
    ...                               {"x": 0.0}, [1.0], samples=4000, seed=1)
    >>> abs(r.failure_probability - 1.3499e-3) / 1.3499e-3 < 0.15     # 1 - Phi(3)
    True
    """
    from scipy.stats import multivariate_normal, norm

    names = tuple(nominal)
    x0 = np.array([float(nominal[k]) for k in names])
    cov = _covariance(names, covariance)
    specs = list(specs)
    if not specs:
        raise ValueError("at least one spec is required")
    if not 0 <= defensive < 1:
        raise ValueError("defensive must lie in [0, 1)")
    if shift == "worst_case":
        wcs = [worst_case_distance(fn, sp, nominal, cov) for sp in specs]
        centre = np.array([wcs[int(np.argmin([w.beta for w in wcs]))].point[k] for k in names])
    elif isinstance(shift, Mapping):
        centre = np.array([float(shift.get(k, nominal[k])) for k in names])
    else:
        raise ValueError("shift must be 'worst_case' or a mapping of parameter values")
    rng = np.random.default_rng(seed)
    pick = rng.random(samples) < defensive
    z = rng.multivariate_normal(np.zeros(len(names)), cov, size=samples)
    pts = np.where(pick[:, None], x0, centre) + z
    p_nom = multivariate_normal(x0, cov, allow_singular=True).pdf(pts)
    q = defensive * p_nom + (1 - defensive) * multivariate_normal(centre, cov, allow_singular=True).pdf(pts)
    w = np.atleast_1d(p_nom / q)
    fail = np.array([not all(sp.passes(float(sp.metric(fn(dict(zip(names, p, strict=True)))))) for sp in specs)
                     for p in pts], dtype=float)
    wn = w / w.sum()
    pf = float(np.sum(wn * fail))
    var = float(np.sum(wn**2 * (fail - pf) ** 2))
    zq = float(norm.ppf(0.5 + confidence / 2))
    half = zq * math.sqrt(var)
    ess = float(w.sum() ** 2 / np.sum(w**2))
    return ISYield(1.0 - pf, pf, (max(0.0, 1.0 - pf - half), min(1.0, 1.0 - pf + half)), ess, samples)
