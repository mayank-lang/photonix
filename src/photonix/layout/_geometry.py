"""Centre-line geometry shared by layout generators and the router.

Every curve generator returns ``(points, headings)``: ``(n, 2)`` centre-line
points starting at the origin with heading +x, and the tangent angle (radians)
at each point. Turning is counter-clockwise for positive angles. Normals taken
from the exact headings give clean constant-width polygons for curved paths.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "arc_curve", "euler_curve", "sbend_curve", "path_polygon", "polyline_length",
    "corner_offset", "place_curve",
]


def _integrate(step_lengths: np.ndarray, headings: np.ndarray) -> np.ndarray:
    """Integrate a heading profile; exact for piecewise-constant curvature."""
    dphi = np.diff(headings)
    mid = 0.5 * (headings[1:] + headings[:-1])
    chord = step_lengths * np.sinc(dphi / (2.0 * np.pi))  # sin(x/2)/(x/2), numpy sinc is sin(pi t)/(pi t)
    pts = np.zeros((len(headings), 2))
    pts[1:, 0] = np.cumsum(chord * np.cos(mid))
    pts[1:, 1] = np.cumsum(chord * np.sin(mid))
    return pts


def arc_curve(radius: float, angle: float, npts: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Circular arc of ``radius`` turning ``angle`` degrees (sign = direction)."""
    if radius <= 0:
        raise ValueError("bend radius must be positive")
    th = np.deg2rad(angle) * np.linspace(0.0, 1.0, max(int(npts), 2))
    sgn = 1.0 if angle >= 0 else -1.0
    pts = np.stack([radius * np.sin(np.abs(th)), sgn * radius * (1.0 - np.cos(th))], axis=1)
    return pts, th


def euler_curve(radius: float, angle: float, p: float = 0.5, npts: int = 96,
                oversample: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Euler (clothoid) bend with minimum radius ``radius``; see ``bend_euler``.

    A fraction ``p`` of the turn is made in two linear-curvature ramps.
    ``p = 0`` reproduces :func:`arc_curve`.
    """
    if radius <= 0:
        raise ValueError("bend radius must be positive")
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must lie in [0, 1]")
    theta = np.deg2rad(abs(angle))
    sgn = 1.0 if angle >= 0 else -1.0
    if p == 0.0 or theta == 0.0:
        return arc_curve(radius, angle, npts)
    lc = p * theta * radius                  # each clothoid ramp
    la = (1.0 - p) * theta * radius          # circular part
    total = 2.0 * lc + la
    n_fine = max(int(npts), 2) * oversample
    s = np.linspace(0.0, total, n_fine)
    phi = np.where(
        s < lc, s * s / (2.0 * radius * lc),
        np.where(s <= lc + la, p * theta / 2.0 + (s - lc) / radius,
                 theta - (total - s) ** 2 / (2.0 * radius * lc)),
    )
    pts = _integrate(np.diff(s), phi)
    idx = np.unique(np.linspace(0, n_fine - 1, max(int(npts), 2)).round().astype(int))
    pts, phi = pts[idx], phi[idx]
    pts[:, 1] *= sgn
    return pts, sgn * phi


def sbend_curve(length: float, offset: float, npts: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Cosine S-bend ``y = offset/2 (1 - cos(pi x / length))`` from (0,0) to (length, offset)."""
    if length <= 0:
        raise ValueError("sbend length must be positive")
    x = np.linspace(0.0, length, max(int(npts), 2))
    y = 0.5 * offset * (1.0 - np.cos(np.pi * x / length))
    dy = 0.5 * offset * np.pi / length * np.sin(np.pi * x / length)
    return np.stack([x, y], axis=1), np.arctan(dy)


def corner_offset(curve: tuple[np.ndarray, np.ndarray]) -> float:
    """Distance from a 90-degree bend's start to its tangent intersection."""
    pts, _ = curve
    return float(pts[-1, 0])


def place_curve(curve: tuple[np.ndarray, np.ndarray], start, heading: float,
                mirror: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Mirror (y -> -y), rotate to ``heading`` (radians) and translate to ``start``."""
    pts, phi = curve
    pts = pts.copy()
    phi = phi.copy()
    if mirror:
        pts[:, 1] = -pts[:, 1]
        phi = -phi
    c, s = np.cos(heading), np.sin(heading)
    rot = np.array([[c, -s], [s, c]])
    return pts @ rot.T + np.asarray(start, dtype=float), phi + heading


def polyline_length(points) -> float:
    pts = np.asarray(points, dtype=float)
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def path_polygon(points, width: float, headings=None) -> np.ndarray:
    """Constant-width polygon around a centre line.

    With ``headings`` the offset normals are exact; otherwise they are taken
    from neighbouring points (fine for straight segments).
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] != 2:
        raise ValueError("A centre line needs at least two 2-D points.")
    if not np.isfinite(width) or width <= 0:
        raise ValueError(f"Path width must be positive and finite, got {width!r}.")
    if np.any(np.linalg.norm(np.diff(pts, axis=0), axis=1) == 0):
        raise ValueError("Consecutive centre-line points must be distinct.")
    if headings is None:
        d = np.empty_like(pts)
        d[0] = pts[1] - pts[0]
        d[-1] = pts[-1] - pts[-2]
        d[1:-1] = pts[2:] - pts[:-2]
        d /= np.linalg.norm(d, axis=1, keepdims=True)
    else:
        h = np.asarray(headings, dtype=float)
        d = np.stack([np.cos(h), np.sin(h)], axis=1)
    nrm = np.stack([-d[:, 1], d[:, 0]], axis=1)
    half = 0.5 * width
    return np.concatenate([pts + half * nrm, (pts - half * nrm)[::-1]], axis=0)
