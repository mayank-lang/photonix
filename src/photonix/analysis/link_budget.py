"""Optical link (power) budgets.

A :class:`LinkBudget` chains a source power through lossy stages to a
receiver and reports the received power and margin against a sensitivity. A
stage can be entered by hand or measured from a simulated S-parameter path with
:func:`stage_from_sparameters`.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

__all__ = ["LinkStage", "LinkBudget", "stage_from_sparameters"]


@dataclass(frozen=True)
class LinkStage:
    """One element of the link: ``count`` copies of a ``loss_db`` loss each."""

    name: str
    loss_db: float
    count: int = 1
    note: str = ""

    def __post_init__(self) -> None:
        if not np.isfinite(self.loss_db):
            raise ValueError(f"stage {self.name!r}: loss_db must be finite")
        if not isinstance(self.count, (int, np.integer)) or self.count < 0:
            raise ValueError(f"stage {self.name!r}: count must be a non-negative integer")

    @property
    def total_db(self) -> float:
        return float(self.loss_db * self.count)


@dataclass
class LinkBudget:
    """Power budget from ``source_dbm`` through ``stages`` to a receiver.

    Examples
    --------
    >>> lb = LinkBudget(source_dbm=10.0, sensitivity_dbm=-15.0)
    >>> _ = lb.add("grating coupler", 3.5, count=2).add("waveguide", 0.2, count=10).add("modulator", 6.0)
    >>> round(lb.total_loss_db, 6), round(lb.received_dbm, 6), round(lb.margin_db, 6)
    (15.0, -5.0, 10.0)
    """

    source_dbm: float
    stages: list[LinkStage] = field(default_factory=list)
    sensitivity_dbm: float | None = None

    def add(self, name: str, loss_db: float, *, count: int = 1, note: str = "") -> LinkBudget:
        self.stages.append(LinkStage(name, float(loss_db), int(count), note))
        return self

    def extend(self, stages: Sequence[LinkStage]) -> LinkBudget:
        self.stages.extend(stages)
        return self

    @property
    def total_loss_db(self) -> float:
        return float(sum(s.total_db for s in self.stages))

    @property
    def received_dbm(self) -> float:
        return float(self.source_dbm - self.total_loss_db)

    @property
    def received_mw(self) -> float:
        return float(10.0 ** (self.received_dbm / 10.0))

    @property
    def margin_db(self) -> float | None:
        """Received power above the sensitivity (negative means the link fails)."""
        return None if self.sensitivity_dbm is None else float(self.received_dbm - self.sensitivity_dbm)

    def table(self) -> list[dict]:
        """Rows with the power after every stage."""
        rows, p = [], self.source_dbm
        for s in self.stages:
            p -= s.total_db
            rows.append({"stage": s.name, "count": s.count, "loss_db": s.loss_db,
                         "subtotal_db": s.total_db, "power_dbm": p, "note": s.note})
        return rows

    def __str__(self) -> str:
        lines = [f"{'stage':<24s}{'count':>6s}{'loss/ea dB':>12s}{'total dB':>10s}{'power dBm':>11s}",
                 f"{'source':<24s}{'':>6s}{'':>12s}{'':>10s}{self.source_dbm:>11.2f}"]
        for r in self.table():
            lines.append(f"{r['stage']:<24s}{r['count']:>6d}{r['loss_db']:>12.3f}"
                         f"{r['subtotal_db']:>10.3f}{r['power_dbm']:>11.2f}")
        if self.margin_db is not None:
            lines.append(f"margin vs {self.sensitivity_dbm:.2f} dBm sensitivity: {self.margin_db:+.2f} dB")
        return "\n".join(lines)


def stage_from_sparameters(name: str, sdict, path: tuple[str, str], *, worst_case: bool = True,
                           count: int = 1) -> LinkStage:
    """A stage whose loss is ``-10 log10 |S[path]|^2`` of a simulated component.

    Over a wavelength sweep the worst (``worst_case=True``) or mean loss is
    used and recorded in the stage note.

    Examples
    --------
    >>> import photonix as px
    >>> s = px.components.grating_coupler(wl=px.linspace(1.53, 1.57, 41))
    >>> st = stage_from_sparameters("GC", s, ("o1", "o2"))
    >>> st.loss_db > 3.0                     # band edges are worse than the 3 dB peak
    True
    """
    p = np.asarray(np.abs(np.asarray(sdict[path])) ** 2, dtype=float)
    loss = -10.0 * np.log10(np.maximum(p, 1e-30))
    value = float(loss.max() if worst_case else loss.mean())
    note = f"{'worst' if worst_case else 'mean'} of {loss.size} sample(s)"
    return LinkStage(name, value, count, note)
