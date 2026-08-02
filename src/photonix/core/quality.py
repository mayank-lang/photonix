"""Quantitative physicality checks for power-normalized S-parameters.

Photonic circuit models use modal amplitudes whose squared magnitudes are
powers.  With that normalization, a linear network is passive exactly when its
scattering matrix is a contraction: ``S.conj().T @ S <= I``, or equivalently
``sigma_max(S) <= 1``.  Reciprocity is ``S == S.T`` when all ports use the same
modal phase convention, and a lossless network is unitary.

This module turns those statements into numerical diagnostics and provides the
orthogonal (minimum Frobenius-distance) projection onto the set of passive
matrices.  The projection is useful for removing small numerical passivity
violations from fitted or interpolated data.  It is deliberately pointwise in
frequency: it does *not* establish causality, broadband realizability, or
reciprocity.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .sparams import as_sdense
from .types import SDense, SType

__all__ = ["SParameterDiagnostics", "analyze_sparameters", "project_passive"]


@dataclass(frozen=True)
class SParameterDiagnostics:
    """Physicality diagnostics for one or a batch of square S-matrices.

    Array-valued fields have ``batch_shape``.  ``singular_values`` has an extra
    final port axis.  Errors are absolute amplitude errors rather than relative
    errors, which keeps the tolerances meaningful for nearly dark networks.

    Parameters
    ----------
    maximum_singular_value
        Worst coherent power-amplitude gain for each sample.  Its square is the
        maximum output/input power ratio.
    passivity_margin
        ``1 - maximum_singular_value``.  Positive values are passive margin;
        negative values quantify the amplitude-domain violation.
    minimum_dissipation_eigenvalue
        Smallest eigenvalue of ``I - S**H S``.  It is non-negative exactly for
        a passive sample and measures the worst coherent power deficit.
    reciprocity_error
        Maximum elementwise magnitude of ``S - S.T``.
    unitarity_error
        Spectral norm of ``S**H S - I``.
    """

    ports: tuple[str, ...]
    batch_shape: tuple[int, ...]
    singular_values: np.ndarray
    maximum_singular_value: np.ndarray
    passivity_margin: np.ndarray
    minimum_dissipation_eigenvalue: np.ndarray
    reciprocity_error: np.ndarray
    unitarity_error: np.ndarray
    passive: bool
    reciprocal: bool
    lossless: bool
    passivity_atol: float
    reciprocity_atol: float
    lossless_atol: float

    @property
    def sample_count(self) -> int:
        """Number of matrices represented by this report."""
        return int(np.prod(self.batch_shape, dtype=int)) if self.batch_shape else 1

    @property
    def worst_passivity_violation(self) -> float:
        """Largest excess singular value above one, or zero if passive."""
        if self.maximum_singular_value.size == 0:
            return 0.0
        return max(0.0, float(np.max(self.maximum_singular_value)) - 1.0)

    @property
    def worst_reciprocity_error(self) -> float:
        """Largest ``|S_ij - S_ji|`` over every sample and port pair."""
        return float(np.max(self.reciprocity_error)) if self.reciprocity_error.size else 0.0

    @property
    def worst_unitarity_error(self) -> float:
        """Largest spectral-norm departure from a unitary matrix."""
        return float(np.max(self.unitarity_error)) if self.unitarity_error.size else 0.0


def _validated_dense(x: SType) -> tuple[np.ndarray, dict[str, int]]:
    matrix, port_map = as_sdense(x)
    matrix = np.asarray(matrix, dtype=complex)
    n_ports = len(port_map)
    indices = list(port_map.values())
    if any(not isinstance(index, int) for index in indices) or set(indices) != set(range(n_ports)):
        raise ValueError("port_map indices must be the unique contiguous integers 0..N-1.")
    if matrix.ndim < 2 or matrix.shape[-2:] != (n_ports, n_ports):
        raise ValueError(
            "S-parameter physicality requires square matrices ending in "
            f"shape ({n_ports}, {n_ports}); got {matrix.shape}."
        )
    if not np.all(np.isfinite(matrix.real)) or not np.all(np.isfinite(matrix.imag)):
        raise ValueError("S-parameters must be finite for physicality analysis.")
    return matrix, dict(port_map)


def _validate_tolerance(name: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return value


def analyze_sparameters(
    x: SType,
    *,
    passivity_atol: float = 1e-9,
    reciprocity_atol: float = 1e-9,
    lossless_atol: float = 1e-9,
) -> SParameterDiagnostics:
    """Measure passivity, reciprocity, and losslessness of S-parameters.

    ``x`` can be any Photonix scattering representation and may contain leading
    sweep/batch dimensions.  Diagnostics are evaluated independently at every
    sample using complex128 NumPy linear algebra; this is an analysis routine,
    not a JIT/autodiff primitive.

    The passivity result assumes power-normalized modal waves.  If data uses
    voltage waves, unequal/complex reference impedances, or inconsistent modal
    normalization, renormalize it before applying this criterion.
    """
    passivity_atol = _validate_tolerance("passivity_atol", passivity_atol)
    reciprocity_atol = _validate_tolerance("reciprocity_atol", reciprocity_atol)
    lossless_atol = _validate_tolerance("lossless_atol", lossless_atol)
    matrix, port_map = _validated_dense(x)
    n_ports = matrix.shape[-1]
    batch_shape = matrix.shape[:-2]
    ports = tuple(name for name, _ in sorted(port_map.items(), key=lambda item: item[1]))

    if n_ports == 0:
        empty_sv = np.empty((*batch_shape, 0), dtype=float)
        zeros = np.zeros(batch_shape, dtype=float)
        ones = np.ones(batch_shape, dtype=float)
        return SParameterDiagnostics(
            ports=ports,
            batch_shape=batch_shape,
            singular_values=empty_sv,
            maximum_singular_value=zeros,
            passivity_margin=ones,
            minimum_dissipation_eigenvalue=ones,
            reciprocity_error=zeros,
            unitarity_error=zeros,
            passive=True,
            reciprocal=True,
            lossless=True,
            passivity_atol=passivity_atol,
            reciprocity_atol=reciprocity_atol,
            lossless_atol=lossless_atol,
        )

    singular_values = np.linalg.svd(matrix, compute_uv=False)
    maximum_sv = singular_values[..., 0]
    passivity_margin = 1.0 - maximum_sv
    # Eigenvalues(S^H S) are squared singular values.  Expressing the
    # dissipation bound this way is both more accurate and less expensive than
    # forming a potentially ill-conditioned Gram matrix a second time.
    minimum_dissipation = 1.0 - maximum_sv**2

    transpose = np.swapaxes(matrix, -1, -2)
    reciprocity_error = np.max(np.abs(matrix - transpose), axis=(-2, -1))

    gram = np.swapaxes(matrix.conj(), -1, -2) @ matrix
    identity = np.eye(n_ports, dtype=complex)
    # The residual is Hermitian, so abs(eigvalsh) gives its spectral norm with
    # less noise than a general SVD.
    gram_residual_eigenvalues = np.linalg.eigvalsh(gram - identity)
    unitarity_error = np.max(np.abs(gram_residual_eigenvalues), axis=-1)

    return SParameterDiagnostics(
        ports=ports,
        batch_shape=batch_shape,
        singular_values=singular_values,
        maximum_singular_value=maximum_sv,
        passivity_margin=passivity_margin,
        minimum_dissipation_eigenvalue=minimum_dissipation,
        reciprocity_error=reciprocity_error,
        unitarity_error=unitarity_error,
        passive=bool(np.all(maximum_sv <= 1.0 + passivity_atol)),
        reciprocal=bool(np.all(reciprocity_error <= reciprocity_atol)),
        lossless=bool(np.all(unitarity_error <= lossless_atol)),
        passivity_atol=passivity_atol,
        reciprocity_atol=reciprocity_atol,
        lossless_atol=lossless_atol,
    )


def project_passive(x: SType, *, limit: float = 1.0) -> SDense:
    """Return the nearest pointwise passive S-matrix in Frobenius norm.

    Singular values above ``limit`` are clipped while their singular vectors
    are retained.  For ``limit=1`` this is the orthogonal projection onto the
    closed convex set of contractive matrices, and therefore makes the smallest
    possible Frobenius-norm change that guarantees passivity at each sample.

    The returned value is always dense and preserves the input port ordering.
    Projection can fill structural zeros and can break exact reciprocity by
    floating-point roundoff; test or impose those properties separately.  Like
    any independent-frequency repair, it does not guarantee causality.
    """
    limit = float(limit)
    if not np.isfinite(limit) or limit < 0.0 or limit > 1.0:
        raise ValueError("limit must be finite and lie in [0, 1].")
    matrix, port_map = _validated_dense(x)
    if matrix.shape[-1] == 0:
        return matrix.copy(), port_map
    left, singular_values, right_h = np.linalg.svd(matrix, full_matrices=False)
    clipped = np.minimum(singular_values, limit)
    projected = (left * clipped[..., np.newaxis, :]) @ right_h
    return projected, port_map
