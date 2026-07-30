"""Validated process-corner and Monte Carlo study definitions.

These objects carry process parameters; they do not assign physical meaning to
those parameters or claim a foundry distribution.  A PDK or user supplies the
nominal values, named corners, covariance, and units from an authoritative deck.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

import numpy as np

__all__ = ["ProcessCorner", "MonteCarloSpec", "ProcessStudy"]

_T = TypeVar("_T")


def _parameter_values(values, *, label: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for raw_name, raw_value in dict(values).items():
        name = str(raw_name)
        if not name or name != name.strip():
            raise ValueError(f"{label} parameter names must be non-empty and trimmed")
        if isinstance(raw_value, (bool, np.bool_)):
            raise ValueError(f"{label} parameter {name!r} must be a real number, not bool")
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} parameter {name!r} must be a real number") from exc
        if not np.isfinite(value):
            raise ValueError(f"{label} parameter {name!r} must be finite")
        out[name] = value
    if not out:
        raise ValueError(f"{label} parameters must not be empty")
    return out


@dataclass(frozen=True)
class ProcessCorner:
    """One named, absolute process-parameter set.

    Values are absolute rather than implicit offsets, avoiding ambiguity when a
    corner is handed to geometry or compact-model code.  Units belong in
    :class:`ProcessStudy`; Photonix performs no hidden unit conversion.
    """

    name: str
    parameters: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("corner name must be a non-empty string")
        parameters = _parameter_values(self.parameters, label=f"corner {self.name!r}")
        metadata = dict(self.metadata)
        try:
            json.dumps(metadata)
        except (TypeError, ValueError) as exc:
            raise ValueError("corner metadata must be JSON-serializable") from exc
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "metadata", metadata)


@dataclass(frozen=True)
class MonteCarloSpec:
    """Correlated Gaussian process-variation specification.

    ``covariance`` is in the squared units of ``parameters`` and must be finite,
    symmetric, and positive semidefinite.  Sampling is local and deterministic
    for a supplied seed; no global NumPy random state is modified.
    """

    parameters: tuple[str, ...]
    mean: np.ndarray
    covariance: np.ndarray

    def __post_init__(self) -> None:
        parameters = tuple(str(name) for name in self.parameters)
        if not parameters or any(not name or name != name.strip() for name in parameters):
            raise ValueError("Monte Carlo parameter names must be non-empty and trimmed")
        if len(parameters) != len(set(parameters)):
            raise ValueError("Monte Carlo parameter names must be unique")
        mean = np.asarray(self.mean, dtype=float)
        covariance = np.asarray(self.covariance, dtype=float)
        n = len(parameters)
        if mean.shape != (n,):
            raise ValueError(f"mean must have shape ({n},), got {mean.shape}")
        if covariance.shape != (n, n):
            raise ValueError(f"covariance must have shape ({n}, {n}), got {covariance.shape}")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(covariance)):
            raise ValueError("mean and covariance must contain only finite values")
        if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-14):
            raise ValueError("covariance must be symmetric")
        eigenvalues = np.linalg.eigvalsh(covariance)
        scale = max(1.0, float(np.max(np.abs(covariance))))
        if float(eigenvalues.min()) < -1e-12 * scale:
            raise ValueError("covariance must be positive semidefinite")
        mean = mean.copy()
        covariance = covariance.copy()
        mean.setflags(write=False)
        covariance.setflags(write=False)
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "covariance", covariance)

    @classmethod
    def independent(cls, nominal, sigma) -> MonteCarloSpec:
        """Build independent Gaussian variations from named nominal values and sigmas."""
        nominal_values = _parameter_values(nominal, label="nominal")
        sigma_values = _parameter_values(sigma, label="sigma")
        if set(sigma_values) != set(nominal_values):
            raise ValueError("sigma keys must exactly match nominal keys")
        if any(value < 0 for value in sigma_values.values()):
            raise ValueError("sigmas must be non-negative")
        parameters = tuple(nominal_values)
        mean = np.asarray([nominal_values[name] for name in parameters])
        covariance = np.diag([sigma_values[name] ** 2 for name in parameters])
        return cls(parameters, mean, covariance)

    def sample(self, count: int, *, seed: int | None = None) -> tuple[ProcessCorner, ...]:
        """Draw ``count`` parameter sets from this specification."""
        if (not isinstance(count, (int, np.integer))
                or isinstance(count, (bool, np.bool_)) or count <= 0):
            raise ValueError("count must be a positive integer")
        if seed is not None and (not isinstance(seed, (int, np.integer))
                                 or isinstance(seed, (bool, np.bool_)) or seed < 0):
            raise ValueError("seed must be a non-negative integer or None")
        eigenvalues, eigenvectors = np.linalg.eigh(self.covariance)
        factor = eigenvectors @ np.diag(np.sqrt(np.clip(eigenvalues, 0.0, None)))
        rng = np.random.default_rng(None if seed is None else int(seed))
        values = self.mean + rng.standard_normal((int(count), len(self.parameters))) @ factor.T
        return tuple(
            ProcessCorner(
                f"mc_{index:06d}",
                dict(zip(self.parameters, row, strict=True)),
                {
                    "kind": "monte_carlo",
                    "sample_index": index,
                    "seed": None if seed is None else int(seed),
                },
            )
            for index, row in enumerate(values)
        )


@dataclass(frozen=True)
class ProcessStudy:
    """Nominal/corner cases with an optional Monte Carlo distribution."""

    nominal: ProcessCorner
    corners: tuple[ProcessCorner, ...] = ()
    monte_carlo: MonteCarloSpec | None = None
    units: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        corners = tuple(self.corners)
        names = [self.nominal.name, *(corner.name for corner in corners)]
        if len(names) != len(set(names)):
            raise ValueError("nominal and corner names must be unique")
        expected = set(self.nominal.parameters)
        for corner in corners:
            if set(corner.parameters) != expected:
                raise ValueError(f"corner {corner.name!r} parameter keys must match nominal")
        if self.monte_carlo is not None and set(self.monte_carlo.parameters) != expected:
            raise ValueError("Monte Carlo parameter keys must match nominal")
        units = {str(name): str(unit) for name, unit in dict(self.units).items()}
        if (set(units) != expected
                or any(not unit or unit != unit.strip() for unit in units.values())):
            raise ValueError(
                "units must provide exactly one non-empty entry per parameter; "
                "use '1' for dimensionless parameters"
            )
        object.__setattr__(self, "corners", corners)
        object.__setattr__(self, "units", units)

    def cases(
        self,
        *,
        monte_carlo_samples: int = 0,
        seed: int | None = None,
        include_nominal: bool = True,
    ) -> tuple[ProcessCorner, ...]:
        """Return deterministic named cases followed by optional random samples."""
        if (not isinstance(monte_carlo_samples, (int, np.integer))
                or isinstance(monte_carlo_samples, (bool, np.bool_))
                or monte_carlo_samples < 0):
            raise ValueError("monte_carlo_samples must be a non-negative integer")
        cases = ((self.nominal,) if include_nominal else ()) + self.corners
        if monte_carlo_samples:
            if self.monte_carlo is None:
                raise ValueError("this study has no Monte Carlo specification")
            cases += self.monte_carlo.sample(int(monte_carlo_samples), seed=seed)
        return cases

    def evaluate(
        self,
        callback: Callable[[ProcessCorner], _T],
        *,
        monte_carlo_samples: int = 0,
        seed: int | None = None,
        include_nominal: bool = True,
    ) -> dict[str, _T]:
        """Apply ``callback(case)`` in case order and preserve every case name.

        The callback owns the interpretation of parameters, for example
        ``study.evaluate(lambda case: model(**case.parameters))``. Photonix does
        not silently translate absolute corner values into offsets.
        """
        if not callable(callback):
            raise ValueError("callback must be callable")
        return {
            case.name: callback(case)
            for case in self.cases(
                monte_carlo_samples=monte_carlo_samples,
                seed=seed,
                include_nominal=include_nominal,
            )
        }

    def map(
        self,
        callback: Callable[[ProcessCorner], _T],
        *,
        monte_carlo_samples: int = 0,
        seed: int | None = None,
        include_nominal: bool = True,
    ) -> dict[str, _T]:
        """Alias of :meth:`evaluate` for mapping arbitrary workflows over cases."""
        return self.evaluate(
            callback,
            monte_carlo_samples=monte_carlo_samples,
            seed=seed,
            include_nominal=include_nominal,
        )
