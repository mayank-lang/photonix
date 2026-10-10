"""Material data model: provenance, validity ranges, dispersion and anisotropy.

The cross-package **material protocol** (:class:`MaterialLike`) is::

    index(wl, temperature=None)    -> complex N = n + i k   (k >= 0 is loss)
    epsilon(wl, temperature=None)  -> complex N**2           (Im >= 0 is loss)
    thermal_conductivity           W/(m K)  or None
    heat_capacity                  J/(kg K) or None (specific, constant pressure)
    density                        kg/m^3   or None
    dn_dt(wl, temperature=None)    1/K  (raises MissingPropertyError without a T model)
    relative_permittivity_static   float, (eps_x, eps_y, eps_z) crystal frame, or None

Wavelengths are vacuum wavelengths in micrometres and temperatures are in
kelvin; ``temperature=None`` means "the reference temperature of the data".

Sign convention
---------------
``index``/``epsilon`` follow the materials-science convention of
refractiveindex.info, Palik and Meep: ``N = n + i k`` with ``k >= 0`` for an
absorbing medium (time dependence ``exp(-i w t)``). The ``photonix.em`` field
solvers propagate as ``exp(-j beta z)`` with ``exp(+j w t)``, where loss is
``Im(eps) < 0``. Pass ``convention="engineering"`` (or call
:func:`to_engineering_sign`) to get ``n - j k`` for those solvers; the two differ
only by complex conjugation.

Validity ranges
---------------
Every :class:`OpticalModel` carries the wavelength and temperature window of
its source. Evaluating outside it emits :class:`MaterialRangeWarning` (or raises
:class:`MaterialRangeError` with ``strict=True``) -- extrapolating a Sellmeier
fit past its data is a silent accuracy loss otherwise. Turn the warnings into
errors globally with ``warnings.simplefilter("error", MaterialRangeWarning)``.
Inside ``jax.jit`` the inputs are abstract and the check is skipped; under
``jax.grad``/``jax.vmap`` it still runs on the primal values.
"""
from __future__ import annotations

import dataclasses
import math
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np

from photonix.core.backend import xp

__all__ = [
    "Reference",
    "OpticalModel",
    "Uniaxial",
    "MaterialSpec",
    "MaterialLike",
    "MaterialRangeWarning",
    "MaterialRangeError",
    "MissingPropertyError",
    "ROOM_TEMPERATURE_K",
    "to_engineering_sign",
    "as_material",
    "constant_model",
    "callable_model",
    "tabulated_model",
    "with_linear_thermo_optic",
    "with_temperature_shift",
]

#: Window (K) assumed for data a source describes only as "room temperature" (15-30 degC).
ROOM_TEMPERATURE_K: tuple[float, float] = (288.15, 303.15)

Convention = Literal["physics", "engineering"]
_CRYSTAL_AXES = ("x", "y", "z")


class MaterialRangeWarning(UserWarning):
    """A material model was evaluated outside the range its source supports."""


class MaterialRangeError(ValueError):
    """Raised instead of :class:`MaterialRangeWarning` when ``strict=True``."""


class MissingPropertyError(LookupError):
    """The requested property has no model or literature value for this material."""


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Reference:
    """Provenance of one set of material data.

    Parameters
    ----------
    citation
        Human-readable citation (authors, title, journal, volume, pages, year).
    doi
        DOI without the ``https://doi.org/`` prefix, if the source has one.
    wl_range_um
        Wavelength range the source states its data or fit for (informational;
        the enforced range is :attr:`OpticalModel.wl_range_um`).
    temperature_range_k
        Temperature range the source covers (informational).
    notes
        Measurement conditions, transcription route, known limitations.

    Examples
    --------
    >>> r = Reference("I. H. Malitson, J. Opt. Soc. Am. 55, 1205 (1965)", doi="10.1364/JOSA.55.001205")
    >>> r.url
    'https://doi.org/10.1364/JOSA.55.001205'
    """

    citation: str
    doi: str | None = None
    wl_range_um: tuple[float, float] | None = None
    temperature_range_k: tuple[float, float] | None = None
    notes: str = ""

    @property
    def url(self) -> str | None:
        """``https://doi.org/<doi>`` or ``None``."""
        return None if self.doi is None else f"https://doi.org/{self.doi}"

    def __str__(self) -> str:
        s = self.citation
        if self.doi:
            s += f", doi:{self.doi}"
        return s


_USER = Reference("user supplied", notes="no provenance or validity range recorded")


# --------------------------------------------------------------------------- #
# Range guard
# --------------------------------------------------------------------------- #
def _concrete(x: Any) -> np.ndarray | None:
    """Concrete NumPy values of ``x``, unwrapping JAX AD/batching tracers; ``None`` under jit."""
    for _ in range(16):
        try:
            return np.asarray(x, dtype=float)
        except Exception:  # noqa: BLE001 - abstract tracer
            for attr in ("primal", "val"):
                inner = getattr(x, attr, None)
                if inner is not None:
                    x = inner
                    break
            else:
                return None
    return None


def _check_range(label: str, value: Any, bounds: tuple[float, float] | None, unit: str,
                 source: str, strict: bool) -> None:
    if bounds is None or value is None:
        return
    arr = _concrete(value)
    if arr is None or arr.size == 0:
        return
    lo, hi = float(np.min(arr)), float(np.max(arr))
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return
    a, b = bounds
    tol = 1e-9 * max(abs(a), abs(b), 1.0)
    if lo >= a - tol and hi <= b + tol:
        return
    shown = f"{lo:.6g}" if lo == hi else f"{lo:.6g}-{hi:.6g}"
    msg = (f"{source}: {label} {shown} {unit} is outside the supported range {a:.6g}-{b:.6g} {unit}; "
           "the result is an extrapolation")
    if strict:
        raise MaterialRangeError(msg)
    warnings.warn(msg, MaterialRangeWarning, stacklevel=5)


def to_engineering_sign(value: Any) -> Any:
    """Convert ``n + i k`` / ``eps' + i eps''`` to the ``exp(+j w t)`` form ``n - j k``.

    The :mod:`photonix.em` solvers propagate fields as ``exp(-j beta z)`` and treat
    ``Im(eps) < 0`` as loss; the material protocol returns the physics sign.
    The conversion is complex conjugation and is its own inverse.

    Examples
    --------
    >>> complex(to_engineering_sign(2.0 + 0.1j))
    (2-0.1j)
    """
    return xp.conj(value)


def _apply_convention(value: Any, convention: str) -> Any:
    if convention == "physics":
        return value
    if convention == "engineering":
        return xp.conj(value)
    raise ValueError(f"convention must be 'physics' or 'engineering', got {convention!r}")


# --------------------------------------------------------------------------- #
# Optical models
# --------------------------------------------------------------------------- #
Evaluator = Callable[[Any, Any], Any]


@dataclass(frozen=True)
class OpticalModel:
    """Complex refractive index ``N(wl, T) = n + i k`` with provenance and validity.

    Parameters
    ----------
    kind
        Model family (``"sellmeier"``, ``"cauchy"``, ``"tabulated"``,
        ``"lorentz_drude"``, ``"li1980"``, ``"constant"``, ``"callable"``,
        ``"refractiveindex.info formula N"``, ...; combinations such as
        ``"sellmeier+linear_dn_dT"``). Informational.
    evaluate
        ``evaluate(wl, T) -> N``; ``T`` is a temperature in kelvin, or ``None``
        for the reference temperature. Must use :data:`photonix.core.backend.xp`
        to stay differentiable. A real return value means ``k = 0``.
    references
        Sources of the dispersion data (first) and of any temperature model.
    wl_range_um
        Enforced wavelength window, or ``None`` for unbounded.
    temperature_range_k
        Enforced temperature window, or ``None`` for unchecked.
    reference_temperature_k
        Temperature the data refer to; what ``temperature=None`` means.
    temperature_dependent
        Whether ``evaluate`` actually varies with ``T``.
    coeffs
        Model coefficients for introspection.

    Examples
    --------
    >>> m = constant_model(1.5)
    >>> complex(m.n(1.55))
    (1.5+0j)
    """

    kind: str
    evaluate: Evaluator
    references: tuple[Reference, ...] = (_USER,)
    wl_range_um: tuple[float, float] | None = None
    temperature_range_k: tuple[float, float] | None = None
    reference_temperature_k: float | None = None
    temperature_dependent: bool = False
    coeffs: Mapping[str, Any] = field(default_factory=dict)
    label: str = ""

    @property
    def reference(self) -> Reference:
        """The primary (dispersion-data) reference."""
        return self.references[0]

    def _source(self) -> str:
        return self.label or self.reference.citation.split(",")[0]

    def check(self, wl: Any, temperature: Any = None, *, strict: bool = False) -> None:
        """Warn (or raise with ``strict``) if ``wl``/``temperature`` leave the validity window."""
        _check_range("wavelength", wl, self.wl_range_um, "um", self._source(), strict)
        _check_range("temperature", temperature, self.temperature_range_k, "K", self._source(), strict)

    def n(self, wl: Any, temperature: Any = None, *, strict: bool = False) -> Any:
        """Complex index ``n + i k`` at ``wl`` [um] and ``temperature`` [K] (physics sign).

        Examples
        --------
        >>> m = constant_model(2.0, k=0.01)
        >>> complex(m.n(1.31))
        (2+0.01j)
        """
        self.check(wl, temperature, strict=strict)
        return self.evaluate(xp.asarray(wl, dtype=float), temperature) + 0j

    def dn_dt(self, wl: Any, temperature: Any = None, *, strict: bool = False, step_k: float = 0.05) -> Any:
        """Thermo-optic coefficient ``d Re(n) / dT`` [1/K] by a central difference.

        The step ``step_k`` (default 0.05 K) keeps the truncation error of the
        smooth library models below 1e-10 relative; the difference is
        differentiable under JAX. Raises :class:`MissingPropertyError` for a
        model with no temperature dependence -- a missing thermo-optic
        coefficient is unknown, not zero.

        Examples
        --------
        >>> from photonix.materials import get_material
        >>> 2.4e-5 < float(get_material("Si3N4").dn_dt(1.55)) < 2.5e-5
        True
        """
        if not self.temperature_dependent:
            raise MissingPropertyError(f"{self._source()}: no temperature dependence is modelled")
        t0 = self.reference_temperature_k if temperature is None else temperature
        self.check(wl, t0, strict=strict)
        lam = xp.asarray(wl, dtype=float)
        t = xp.asarray(t0, dtype=float)
        up = xp.real(self.evaluate(lam, t + step_k) + 0j)
        dn = xp.real(self.evaluate(lam, t - step_k) + 0j)
        return (up - dn) / (2.0 * step_k)


def constant_model(n: float, k: float = 0.0, *, reference: Reference | None = None) -> OpticalModel:
    """Non-dispersive, temperature-independent model ``N = n + i k``.

    Examples
    --------
    >>> complex(constant_model(1.0).n(10.0))
    (1+0j)
    """
    value = complex(n, k)

    def evaluate(wl: Any, T: Any) -> Any:
        return value * xp.ones_like(xp.asarray(wl, dtype=float))

    return OpticalModel("constant", evaluate, (reference or _USER,), coeffs={"n": n, "k": k})


def callable_model(fn: Callable[[Any], Any], *, reference: Reference | None = None,
                   wl_range_um: tuple[float, float] | None = None,
                   temperature_range_k: tuple[float, float] | None = None,
                   reference_temperature_k: float | None = None, kind: str = "callable",
                   label: str = "") -> OpticalModel:
    """Wrap a temperature-independent dispersion callable ``fn(wl) -> N``.

    The callable is stored and called as-is (``model.coeffs["fn"] is fn``), so a
    library entry built on :func:`photonix.em.silicon` returns bit-identical values.

    Examples
    --------
    >>> import photonix.em as em
    >>> m = callable_model(em.silica, wl_range_um=(0.21, 6.7))
    >>> float(m.n(1.55).real) == float(em.silica(1.55))
    True
    """

    def evaluate(wl: Any, T: Any) -> Any:
        return fn(wl)

    return OpticalModel(kind, evaluate, (reference or _USER,), wl_range_um, temperature_range_k,
                        reference_temperature_k, False, {"fn": fn}, label)


def _table_column(wl: Any, values: Any, what: str) -> tuple[np.ndarray, np.ndarray]:
    """Sort a (wavelength, value) table and average rows that share a wavelength."""
    x = np.asarray(wl, dtype=float)
    y = np.asarray(values, dtype=float)
    if x.ndim != 1 or x.shape != y.shape:
        raise ValueError(f"{what} and its wavelength grid must be 1-D arrays of equal length")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))) or np.any(x <= 0):
        raise ValueError(f"{what} table must be finite with positive wavelengths")
    ux, inverse = np.unique(x, return_inverse=True)
    if ux.size < 2:
        raise ValueError(f"{what} table needs at least two distinct wavelengths")
    uy = np.bincount(inverse, weights=y) / np.bincount(inverse)
    return ux, uy


def tabulated_model(wl: Any, n: Any, k: Any = None, *, wl_k: Any = None,
                    reference: Reference | None = None,
                    temperature_range_k: tuple[float, float] | None = None,
                    reference_temperature_k: float | None = None, label: str = "") -> OpticalModel:
    """Linear interpolation of tabulated ``n`` (and optionally ``k``) versus wavelength.

    ``k`` may be sampled on its own grid ``wl_k``; the enforced wavelength range
    is the overlap of the ``n`` and ``k`` grids. Rows may come in any order;
    repeated wavelengths (stitched spectrometer ranges, common in published
    tables) are merged by averaging their values. Linear interpolation in
    wavelength is the convention of refractiveindex.info; it is continuous but
    its derivative is piecewise constant, so a group index computed from a
    sparse table is only as good as the sampling.

    Examples
    --------
    >>> m = tabulated_model([1.0, 2.0], [3.0, 2.0], [0.2, 0.0])
    >>> complex(m.n(1.5))
    (2.5+0.1j)
    """
    wl_n, n_arr = _table_column(wl, n, "n")
    lo, hi = float(wl_n[0]), float(wl_n[-1])
    k_grid = k_arr = None
    if k is not None:
        k_grid, k_arr = _table_column(wl if wl_k is None else wl_k, k, "k")
        if np.any(k_arr < 0):
            raise ValueError("tabulated k must be >= 0 (physics sign: k >= 0 is loss)")
        lo, hi = max(lo, float(k_grid[0])), min(hi, float(k_grid[-1]))
        if lo >= hi:
            raise ValueError("the n and k tables do not overlap in wavelength")
    xn, yn = xp.asarray(wl_n), xp.asarray(n_arr)
    xk = yk = None
    if k_grid is not None:
        xk, yk = xp.asarray(k_grid), xp.asarray(k_arr)

    def evaluate(w: Any, T: Any) -> Any:
        w = xp.asarray(w, dtype=float)
        out = xp.interp(w, xn, yn)
        if xk is not None:
            out = out + 1j * xp.interp(w, xk, yk)
        return out

    coeffs = {"wl": wl_n, "n": n_arr, "wl_k": k_grid, "k": k_arr}
    return OpticalModel("tabulated", evaluate, (reference or _USER,), (lo, hi), temperature_range_k,
                        reference_temperature_k, False, coeffs, label)


def with_linear_thermo_optic(base: OpticalModel, dn_dT: float, *, reference_temperature_k: float,
                             temperature_range_k: tuple[float, float], reference: Reference,
                             label: str | None = None) -> OpticalModel:
    """Add a constant thermo-optic coefficient: ``N(wl, T) = N_base(wl) + dn_dT (T - T_ref)``.

    ``T = None`` returns the base model unchanged (bit-identical).

    Examples
    --------
    >>> m = with_linear_thermo_optic(constant_model(2.0), 1e-4, reference_temperature_k=300.0,
    ...                              temperature_range_k=(250.0, 400.0), reference=Reference("demo"))
    >>> round(float(m.n(1.55, 310.0).real), 6)
    2.001
    """
    t_ref = float(reference_temperature_k)

    def evaluate(wl: Any, T: Any) -> Any:
        n0 = base.evaluate(wl, None)
        if T is None:
            return n0
        return n0 + dn_dT * (xp.asarray(T, dtype=float) - t_ref)

    return OpticalModel(base.kind + "+linear_dn_dT", evaluate, base.references + (reference,), base.wl_range_um,
                        temperature_range_k, t_ref, True, {**dict(base.coeffs), "dn_dT": dn_dT},
                        base.label if label is None else label)


def _intersect(a: tuple[float, float] | None, b: tuple[float, float] | None) -> tuple[float, float] | None:
    if a is None:
        return b
    if b is None:
        return a
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    if lo >= hi:
        raise ValueError(f"wavelength ranges {a} and {b} do not overlap")
    return (lo, hi)


def with_temperature_shift(base: OpticalModel, shift_model: Callable[[Any, Any], Any], *,
                           reference_temperature_k: float, temperature_range_k: tuple[float, float],
                           shift_wl_range_um: tuple[float, float] | None, reference: Reference,
                           label: str | None = None) -> OpticalModel:
    """Temperature dependence borrowed from a second model ``n_T(wl, T)``.

    ``N(wl, T) = N_base(wl) + [n_T(wl, T) - n_T(wl, T_ref)]``: the base data set
    keeps its dispersion at ``T_ref`` and the shift model contributes only the
    temperature change. ``T = None`` returns the base model unchanged. The
    enforced wavelength range is the overlap of both models.

    Examples
    --------
    >>> from photonix.materials.formulas import li1980_silicon
    >>> m = with_temperature_shift(constant_model(3.48), li1980_silicon, reference_temperature_k=300.0,
    ...                            temperature_range_k=(100.0, 750.0), shift_wl_range_um=(1.2, 14.0),
    ...                            reference=Reference("Li 1980"))
    >>> 1.8e-4 < float(m.dn_dt(1.55, 300.0)) < 1.9e-4
    True
    """
    t_ref = float(reference_temperature_k)

    def evaluate(wl: Any, T: Any) -> Any:
        n0 = base.evaluate(wl, None)
        if T is None:
            return n0
        return n0 + (shift_model(wl, T) - shift_model(wl, t_ref))

    return OpticalModel(base.kind + "+thermal_shift", evaluate, base.references + (reference,),
                        _intersect(base.wl_range_um, shift_wl_range_um), temperature_range_k, t_ref, True,
                        dict(base.coeffs), base.label if label is None else label)


@dataclass(frozen=True)
class Uniaxial:
    """Uniaxial crystal: ordinary (``no``) and extraordinary (``ne``) index models.

    The optic axis is the crystal ``z`` (``c``) axis; light polarised along it
    sees ``ne``, light polarised perpendicular to it sees ``no``.
    """

    ordinary: OpticalModel
    extraordinary: OpticalModel

    def axis_model(self, axis: str) -> OpticalModel:
        """``"o"`` -> ordinary model, ``"e"`` -> extraordinary model."""
        if axis in ("o", "ordinary"):
            return self.ordinary
        if axis in ("e", "extraordinary"):
            return self.extraordinary
        raise ValueError(f"axis must be 'o' or 'e' for a uniaxial material, got {axis!r}")


# --------------------------------------------------------------------------- #
# Material
# --------------------------------------------------------------------------- #
def _crystal_to_waveguide(cut: str, propagation_axis: str) -> tuple[str, str, str]:
    """Crystal axes lying along the waveguide (x lateral, y vertical, z propagation) axes."""
    if cut not in _CRYSTAL_AXES or propagation_axis not in _CRYSTAL_AXES:
        raise ValueError("cut and propagation_axis must each be one of 'x', 'y', 'z'")
    if cut == propagation_axis:
        raise ValueError(f"propagation_axis {propagation_axis!r} cannot be the wafer normal (cut {cut!r})")
    lateral = next(a for a in _CRYSTAL_AXES if a not in (cut, propagation_axis))
    return lateral, cut, propagation_axis


@runtime_checkable
class MaterialLike(Protocol):
    """Structural protocol every photonix material provides (see module docstring)."""

    def index(self, wl: Any, temperature: Any = None) -> Any: ...

    def epsilon(self, wl: Any, temperature: Any = None) -> Any: ...


@dataclass(frozen=True)
class MaterialSpec:
    """A material: optical model plus thermal, electrical and electro-optic data.

    Parameters
    ----------
    name
        Registry name (e.g. ``"Si"``).
    optical
        :class:`OpticalModel` (isotropic) or :class:`Uniaxial`.
    thermal_conductivity
        W/(m K) at :attr:`property_temperature_k`, or ``None``.
    heat_capacity
        Specific heat at constant pressure, J/(kg K), or ``None``.
    density
        kg/m^3, or ``None``.
    relative_permittivity_static
        Low-frequency relative permittivity: a float, a ``(eps_x, eps_y, eps_z)``
        tuple in the crystal frame for anisotropic media, or ``None``.
    resistivity
        Electrical resistivity, ohm m, at :attr:`resistivity_temperature_k`.
    resistivity_tcr
        Linear temperature coefficient of resistivity, 1/K
        (``rho(T) = rho0 (1 + tcr (T - T0))``).
    resistivity_temperature_k
        Temperature ``T0`` of :attr:`resistivity` (defaults to
        :attr:`property_temperature_k`).
    pockels_pm_per_v
        Linear electro-optic coefficients in pm/V, keyed ``"r13"``, ``"r33"``, ...
        (crystal frame, contracted notation).
    references
        Provenance of the non-optical data keyed by property group
        (``"thermal"``, ``"electrical"``, ``"static_permittivity"``,
        ``"electro_optic"``).
    property_temperature_k
        Temperature the scalar non-optical properties refer to.
    tags, description
        Free-form classification and notes.

    Examples
    --------
    >>> from photonix.materials import get_material
    >>> si = get_material("Si")
    >>> 3.47 < float(si.index(1.55).real) < 3.48
    True
    >>> bool(si(1.55) == si.index(1.55).real)      # plain callable: drops into a LayerStack
    True
    """

    name: str
    optical: OpticalModel | Uniaxial
    thermal_conductivity: float | None = None
    heat_capacity: float | None = None
    density: float | None = None
    relative_permittivity_static: float | tuple[float, float, float] | None = None
    resistivity: float | None = None
    resistivity_tcr: float | None = None
    pockels_pm_per_v: Mapping[str, float] | None = None
    references: Mapping[str, Reference] = field(default_factory=dict)
    property_temperature_k: float = 300.0
    resistivity_temperature_k: float | None = None
    tags: tuple[str, ...] = ()
    description: str = ""

    # -- structure ------------------------------------------------------------
    @property
    def isotropic(self) -> bool:
        """``True`` unless the optical model is :class:`Uniaxial`."""
        return not isinstance(self.optical, Uniaxial)

    @property
    def reference_temperature_k(self) -> float | None:
        """Temperature the optical data refer to (what ``temperature=None`` means)."""
        model = self.optical.ordinary if isinstance(self.optical, Uniaxial) else self.optical
        return model.reference_temperature_k

    @property
    def has_temperature_model(self) -> bool:
        """Whether the optical model varies with temperature."""
        if isinstance(self.optical, Uniaxial):
            return self.optical.ordinary.temperature_dependent and self.optical.extraordinary.temperature_dependent
        return self.optical.temperature_dependent

    def _model(self, axis: str | None) -> OpticalModel:
        if isinstance(self.optical, Uniaxial):
            if axis is None:
                raise ValueError(f"{self.name} is uniaxial: pass axis='o' or axis='e', use eps_tensor(), "
                                 "or take an isotropic view with .component('o'/'e')")
            return self.optical.axis_model(axis)
        if axis not in (None, "o", "e", "ordinary", "extraordinary"):
            raise ValueError(f"unknown axis {axis!r}")
        return self.optical

    # -- optical protocol -------------------------------------------------------
    def index(self, wl: Any, temperature: Any = None, *, axis: str | None = None, strict: bool = False,
              convention: Convention = "physics") -> Any:
        """Complex refractive index ``n + i k`` (``k >= 0`` is loss) at ``wl`` [um], ``temperature`` [K].

        ``axis`` (``"o"``/``"e"``) is required for a uniaxial material. With
        ``convention="engineering"`` the conjugate ``n - j k`` is returned, the
        sign the :mod:`photonix.em` solvers use.

        Examples
        --------
        >>> from photonix.materials import get_material
        >>> au = get_material("Au")
        >>> N = complex(au.index(1.55))
        >>> N.imag > 9.0                                   # strongly absorbing
        True
        >>> complex(au.index(1.55, convention="engineering")) == N.conjugate()
        True
        """
        return _apply_convention(self._model(axis).n(wl, temperature, strict=strict), convention)

    def epsilon(self, wl: Any, temperature: Any = None, *, axis: str | None = None, strict: bool = False,
                convention: Convention = "physics") -> Any:
        """Relative permittivity ``(n + i k)^2`` (``Im >= 0`` is loss).

        Examples
        --------
        >>> from photonix.materials import get_material
        >>> eps = get_material("Al").epsilon(1.55)
        >>> bool(eps.real < -100 and eps.imag > 0)
        True
        """
        n = self._model(axis).n(wl, temperature, strict=strict)
        return _apply_convention(n * n, convention)

    def dn_dt(self, wl: Any, temperature: Any = None, *, axis: str | None = None, strict: bool = False) -> Any:
        """Thermo-optic coefficient ``d Re(n) / dT`` [1/K] (see :meth:`OpticalModel.dn_dt`).

        Examples
        --------
        >>> from photonix.materials import get_material
        >>> round(float(get_material("Si").dn_dt(1.55, 300.0)), 6)
        0.000184
        """
        return self._model(axis).dn_dt(wl, temperature, strict=strict)

    def eps_tensor(self, wl: Any, temperature: Any = None, *, cut: str | None = None,
                   propagation_axis: str | None = None, strict: bool = False,
                   convention: Convention = "physics") -> tuple[Any, Any, Any]:
        """Diagonal permittivity ``(eps_xx, eps_yy, eps_zz)`` in the waveguide frame.

        Waveguide frame: ``x`` lateral (in the wafer plane), ``y`` vertical
        (wafer normal), ``z`` propagation. For a uniaxial crystal, ``cut`` names
        the crystal axis along the wafer normal and ``propagation_axis`` the
        crystal axis along the waveguide; e.g. a y-propagating x-cut LiNbO3
        guide has the optic axis lateral, so its TE mode sees ``ne``. Only
        orientations with crystal axes along the waveguide axes are supported
        (the tensor then stays diagonal). Isotropic materials ignore the
        orientation.

        Examples
        --------
        >>> from photonix.materials import get_material
        >>> ln = get_material("LiNbO3")
        >>> exx, eyy, ezz = ln.eps_tensor(1.55, cut="x", propagation_axis="y")
        >>> bool(abs(exx - ln.epsilon(1.55, axis="e")) < 1e-12 and abs(eyy - ln.epsilon(1.55, axis="o")) < 1e-12)
        True
        """
        if isinstance(self.optical, Uniaxial):
            if cut is None or propagation_axis is None:
                raise ValueError(f"{self.name} is uniaxial: pass cut= and propagation_axis= (crystal axes)")
            eps_o = self.epsilon(wl, temperature, axis="o", strict=strict, convention=convention)
            eps_e = self.epsilon(wl, temperature, axis="e", strict=strict, convention=convention)
            axes = _crystal_to_waveguide(cut, propagation_axis)
            return tuple(eps_e if a == "z" else eps_o for a in axes)  # type: ignore[return-value]
        eps = self.epsilon(wl, temperature, strict=strict, convention=convention)
        return (eps, eps, eps)

    def static_permittivity_tensor(self, *, cut: str | None = None,
                                   propagation_axis: str | None = None) -> tuple[float, float, float]:
        """Static relative permittivity rotated into the waveguide frame (see :meth:`eps_tensor`).

        Examples
        --------
        >>> from photonix.materials import get_material
        >>> get_material("SiO2").static_permittivity_tensor()
        (3.9, 3.9, 3.9)
        """
        value = self.relative_permittivity_static
        if value is None:
            raise MissingPropertyError(f"{self.name}: no static permittivity recorded")
        if isinstance(value, tuple):
            if cut is None or propagation_axis is None:
                raise ValueError(f"{self.name} is anisotropic: pass cut= and propagation_axis=")
            crystal = dict(zip(_CRYSTAL_AXES, value, strict=True))
            return tuple(float(crystal[a]) for a in _crystal_to_waveguide(cut, propagation_axis))  # type: ignore
        v = float(value)
        return (v, v, v)

    def component(self, axis: str) -> MaterialSpec:
        """Isotropic view of one principal index of a uniaxial material (``"o"``/``"e"``).

        Useful for scalar consumers such as a :class:`~photonix.pdk.LayerStack`
        level, when the mode of interest is polarised along one crystal axis.

        Examples
        --------
        >>> from photonix.materials import get_material
        >>> ln_e = get_material("LiNbO3").component("e")
        >>> ln_e.isotropic, ln_e.name
        (True, 'LiNbO3 (e)')
        """
        if not isinstance(self.optical, Uniaxial):
            return self
        model = self.optical.axis_model(axis)
        tag = "o" if axis in ("o", "ordinary") else "e"
        return dataclasses.replace(self, name=f"{self.name} ({tag})", optical=model)

    def resistivity_at(self, temperature: Any) -> Any:
        """Resistivity [ohm m] from the linear model ``rho0 (1 + tcr (T - T0))``.

        Examples
        --------
        >>> from photonix.materials import get_material
        >>> cu = get_material("Cu")
        >>> round(float(cu.resistivity_at(293.15 + 10.0)) / cu.resistivity, 5)     # IACS: +0.393 %/K
        1.0393
        """
        if self.resistivity is None:
            raise MissingPropertyError(f"{self.name}: no resistivity recorded")
        if self.resistivity_tcr is None:
            raise MissingPropertyError(f"{self.name}: no temperature coefficient of resistivity recorded")
        t0 = self.property_temperature_k if self.resistivity_temperature_k is None else self.resistivity_temperature_k
        t = xp.asarray(temperature, dtype=float)
        return self.resistivity * (1.0 + self.resistivity_tcr * (t - t0))

    def __call__(self, wl: Any) -> Any:
        """Real part of the index at the reference temperature (``LayerLevel`` compatibility)."""
        if isinstance(self.optical, Uniaxial):
            raise TypeError(f"{self.name} is uniaxial and has no single scalar index; use "
                            ".component('o'/'e') or eps_tensor()")
        return xp.real(self.optical.n(wl))

    # -- misc -------------------------------------------------------------------
    def replace(self, **changes: Any) -> MaterialSpec:
        """Copy with fields replaced (e.g. a measured film thermal conductivity).

        Examples
        --------
        >>> from photonix.materials import get_material
        >>> film = get_material("Si").replace(name="Si_film", thermal_conductivity=60.0)
        >>> film.thermal_conductivity, get_material("Si").thermal_conductivity
        (60.0, 148.0)
        """
        return dataclasses.replace(self, **changes)

    def citations(self) -> str:
        """Every reference attached to this material, one per line.

        Examples
        --------
        >>> from photonix.materials import get_material
        >>> "Zelmon" in get_material("LiNbO3").citations()
        True
        """
        lines: list[str] = []
        models = ([self.optical.ordinary, self.optical.extraordinary] if isinstance(self.optical, Uniaxial)
                  else [self.optical])
        for m in models:
            for r in m.references:
                lines.append(f"optical: {r}")
        for key, r in self.references.items():
            lines.append(f"{key}: {r}")
        return "\n".join(dict.fromkeys(lines))

    def __repr__(self) -> str:
        return f"MaterialSpec({self.name!r})"


def as_material(obj: Any, *, name: str | None = None) -> MaterialSpec:
    """Coerce a number, an ``n(wl)`` callable or a material object to :class:`MaterialSpec`.

    Lets consumers accept any of the material forms used in photonix (a bare
    index, :class:`photonix.em.materials.Material`, a dispersion function, or a
    :class:`MaterialSpec`). Coerced inputs carry no provenance or validity range.

    Examples
    --------
    >>> complex(as_material(1.444).index(1.55))
    (1.444+0j)
    >>> import photonix.em as em
    >>> float(as_material(em.silicon).index(1.55).real) == float(em.silicon(1.55))
    True
    """
    if isinstance(obj, MaterialSpec):
        return obj
    if isinstance(obj, (int, float, complex)) and not isinstance(obj, bool):
        z = complex(obj)
        return MaterialSpec(name or f"n={z.real:g}", constant_model(z.real, z.imag))
    if callable(obj):
        fn = obj.index if hasattr(obj, "index") and callable(obj.index) else obj
        label = name or getattr(obj, "name", None) or getattr(obj, "__name__", "callable")
        return MaterialSpec(str(label), callable_model(fn))
    raise TypeError(f"cannot interpret {type(obj).__name__} as a material")
