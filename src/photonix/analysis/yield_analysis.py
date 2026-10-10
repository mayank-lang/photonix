"""Parametric yield over process corners and Monte Carlo samples.

The process description stays in :class:`photonix.pdk.ProcessStudy` (named
corners, correlated Gaussian Monte Carlo); this module adds specifications and
statistics on top. The user's ``evaluate`` callback owns the mapping from
process parameters to a simulated response, so no hidden assumption is made
about what a parameter means.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["Spec", "YieldResult", "yield_analysis", "wilson_interval"]


@dataclass(frozen=True)
class Spec:
    """A pass/fail specification ``minimum <= metric(result) <= maximum``."""

    name: str
    metric: Callable[[Any], float]
    minimum: float | None = None
    maximum: float | None = None
    unit: str = ""

    def __post_init__(self) -> None:
        if self.minimum is None and self.maximum is None:
            raise ValueError(f"spec {self.name!r} needs a minimum and/or a maximum")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError(f"spec {self.name!r}: minimum exceeds maximum")

    def passes(self, value: float) -> bool:
        if not np.isfinite(value):
            return False
        return (self.minimum is None or value >= self.minimum) and (self.maximum is None or value <= self.maximum)


def wilson_interval(passed: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial pass fraction.

    Examples
    --------
    >>> lo, hi = wilson_interval(90, 100)
    >>> round(lo, 3), round(hi, 3)
    (0.826, 0.945)
    """
    if total <= 0:
        raise ValueError("total must be positive")
    from scipy.stats import norm

    z = float(norm.ppf(0.5 + confidence / 2.0))
    p = passed / total
    denom = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


@dataclass
class YieldResult:
    """Metric values and pass flags per case, with yield statistics.

    ``values[spec][case]`` and ``passed[spec][case]`` are keyed by case name.
    Yield statistics use the Monte Carlo cases only; deterministic corners are
    reported separately in :attr:`corner_pass`.
    """

    specs: tuple[Spec, ...]
    values: dict[str, dict[str, float]] = field(default_factory=dict)
    passed: dict[str, dict[str, bool]] = field(default_factory=dict)
    monte_carlo_cases: tuple[str, ...] = ()
    corner_cases: tuple[str, ...] = ()
    confidence: float = 0.95

    def _all_pass(self, case: str) -> bool:
        return all(self.passed[s.name][case] for s in self.specs)

    @property
    def n_samples(self) -> int:
        return len(self.monte_carlo_cases)

    @property
    def yield_fraction(self) -> float:
        """Fraction of Monte Carlo samples meeting every spec."""
        if not self.monte_carlo_cases:
            raise ValueError("no Monte Carlo samples were evaluated")
        return sum(self._all_pass(c) for c in self.monte_carlo_cases) / self.n_samples

    @property
    def yield_interval(self) -> tuple[float, float]:
        n_pass = sum(self._all_pass(c) for c in self.monte_carlo_cases)
        return wilson_interval(n_pass, self.n_samples, self.confidence)

    def spec_yield(self) -> dict[str, float]:
        """Monte Carlo pass fraction of each spec on its own."""
        return {s.name: sum(self.passed[s.name][c] for c in self.monte_carlo_cases) / self.n_samples
                for s in self.specs}

    @property
    def corner_pass(self) -> dict[str, bool]:
        """Whether each deterministic case (nominal and corners) meets every spec."""
        return {c: self._all_pass(c) for c in self.corner_cases}

    def statistics(self, spec: str) -> dict[str, float]:
        """Mean, standard deviation, min and max of one metric over Monte Carlo cases."""
        v = np.array([self.values[spec][c] for c in self.monte_carlo_cases], dtype=float)
        return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if v.size > 1 else 0.0,
                "min": float(v.min()), "max": float(v.max())}

    def __str__(self) -> str:
        lines = []
        if self.monte_carlo_cases:
            lo, hi = self.yield_interval
            lines.append(f"yield {self.yield_fraction:.1%} over {self.n_samples} samples "
                         f"({self.confidence:.0%} CI {lo:.1%}-{hi:.1%})")
            for name, frac in self.spec_yield().items():
                lines.append(f"  {name}: {frac:.1%}")
        for case, ok in self.corner_pass.items():
            lines.append(f"  corner {case}: {'pass' if ok else 'FAIL'}")
        return "\n".join(lines)


def yield_analysis(
    evaluate: Callable[[Any], Any],
    specs: Sequence[Spec],
    *,
    study=None,
    samples: int = 0,
    seed: int | None = None,
    cases: Sequence | None = None,
    confidence: float = 0.95,
) -> YieldResult:
    """Evaluate ``specs`` over process cases and estimate parametric yield.

    Parameters
    ----------
    evaluate
        ``evaluate(case) -> result`` where ``case`` is a
        :class:`~photonix.pdk.ProcessCorner`; each spec's ``metric`` maps the
        result to a number. Typically ``evaluate`` simulates a model with the
        case's parameters.
    specs
        Specifications that every passing case must meet.
    study, samples, seed
        A :class:`~photonix.pdk.ProcessStudy`: its nominal and corners are
        evaluated as deterministic cases, plus ``samples`` reproducible Monte
        Carlo draws.
    cases
        Alternatively, an explicit sequence of ``ProcessCorner`` objects whose
        metadata ``kind == "monte_carlo"`` marks the statistical ones.

    Examples
    --------
    >>> import photonix as px
    >>> from photonix.pdk import MonteCarloSpec, ProcessCorner, ProcessStudy
    >>> study = ProcessStudy(ProcessCorner("tt", {"coupling": 0.5}), units={"coupling": "1"},
    ...                      monte_carlo=MonteCarloSpec.independent({"coupling": 0.5}, {"coupling": 0.02}))
    >>> def evaluate(case):
    ...     return px.components.directional_coupler(coupling=case.parameters["coupling"])
    >>> spec = Spec("split", lambda s: float(px.power(s[("o1", "o3")])), minimum=0.47, maximum=0.53)
    >>> res = yield_analysis(evaluate, [spec], study=study, samples=200, seed=1)
    >>> 0.8 < res.yield_fraction < 0.95       # +/-1.5 sigma window
    True
    """
    specs = tuple(specs)
    if not specs:
        raise ValueError("at least one spec is required")
    if len({s.name for s in specs}) != len(specs):
        raise ValueError("spec names must be unique")
    if (study is None) == (cases is None):
        raise ValueError("pass exactly one of study= or cases=")
    if study is not None:
        all_cases = tuple(study.cases(monte_carlo_samples=samples, seed=seed))
    else:
        assert cases is not None
        all_cases = tuple(cases)
    result = YieldResult(specs=specs, confidence=confidence,
                         values={s.name: {} for s in specs}, passed={s.name: {} for s in specs})
    mc: list[str] = []
    det: list[str] = []
    for case in all_cases:
        out = evaluate(case)
        for s in specs:
            value = float(s.metric(out))
            result.values[s.name][case.name] = value
            result.passed[s.name][case.name] = s.passes(value)
        (mc if case.metadata.get("kind") == "monte_carlo" else det).append(case.name)
    result.monte_carlo_cases = tuple(mc)
    result.corner_cases = tuple(det)
    return result
