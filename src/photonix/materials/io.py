"""Read material data files: refractiveindex.info YAML and plain n,k tables.

:func:`load_refractiveindex_yaml` understands the database's file format
(``DATA`` blocks of type ``formula 1`` ... ``formula 9``, ``tabulated n``,
``tabulated k`` and ``tabulated nk``, plus ``REFERENCES``/``COMMENTS``/``SPECS``)
and returns an :class:`~photonix.materials.OpticalModel` whose validity range is
the file's ``wavelength_range`` (formulas) or the span of its tables. The
database is public domain (CC0 1.0): https://refractiveindex.info.

The YAML is read with a small built-in parser for the subset the database
uses (mappings, a sequence of mappings, plain/quoted scalars and ``|`` literal
blocks), so no YAML package is required. It is checked against PyYAML's
``safe_load`` in the test suite when PyYAML is installed.
"""
from __future__ import annotations

import html
import os
import re
from pathlib import Path
from typing import Any

import numpy as np

from photonix.core.backend import xp

from .formulas import HC_EV_UM, ciddor1996_air, refractiveindex_formula
from .model import ROOM_TEMPERATURE_K, OpticalModel, Reference, _table_column, tabulated_model

__all__ = [
    "parse_refractiveindex_yaml",
    "load_refractiveindex_yaml",
    "load_nk_csv",
    "DATA_DIR",
]

#: Directory holding the data files shipped with :mod:`photonix.materials` (CC0 refractiveindex.info entries).
DATA_DIR: Path = Path(__file__).resolve().parent / "data"

#: Half-width (K) of the window in which single-temperature data are used without a warning.
_SINGLE_T_HALF_WIDTH_K = 5.0


# --------------------------------------------------------------------------- #
# Minimal YAML subset parser
# --------------------------------------------------------------------------- #
class _Lines:
    def __init__(self, text: str) -> None:
        self.raw = text.replace("\t", "    ").splitlines()
        self.i = 0

    def skip_blank(self) -> None:
        while self.i < len(self.raw):
            s = self.raw[self.i].strip()
            if s and not s.startswith("#"):
                return
            self.i += 1

    def peek(self) -> tuple[int, str] | None:
        self.skip_blank()
        if self.i >= len(self.raw):
            return None
        line = self.raw[self.i]
        return len(line) - len(line.lstrip(" ")), line.strip()


def _strip_comment(s: str) -> str:
    m = re.search(r"\s#", s)
    return s[: m.start()].rstrip() if m else s.rstrip()


def _double_quoted(lines: _Lines, first: str) -> str:
    """Read a double-quoted scalar starting at ``first`` (which begins with '"'), maybe spanning lines."""
    buf = first[1:]
    parts: list[str] = []
    while True:
        out, j, closed = [], 0, False
        while j < len(buf):
            ch = buf[j]
            if ch == "\\" and j + 1 < len(buf):
                nxt = buf[j + 1]
                out.append({"n": "\n", "t": "\t", '"': '"', "\\": "\\", "/": "/"}.get(nxt, nxt))
                j += 2
                continue
            if ch == '"':
                closed = True
                break
            out.append(ch)
            j += 1
        parts.append("".join(out))
        if closed:
            break
        lines.i += 1
        if lines.i >= len(lines.raw):
            raise ValueError("unterminated double-quoted string")
        buf = lines.raw[lines.i].strip()
    return " ".join(p.strip() if k else p.rstrip() for k, p in enumerate(parts))


def _scalar(lines: _Lines, rest: str) -> str:
    if rest.startswith('"'):
        return _double_quoted(lines, rest)
    if rest.startswith("'"):
        body = rest[1:]
        end = body.rfind("'")
        return body[:end].replace("''", "'") if end >= 0 else body
    return _strip_comment(rest)


def _block_scalar(lines: _Lines, indent: int) -> str:
    """Collect a ``|`` literal block: every following line indented deeper than ``indent``."""
    lines.i += 1
    body: list[str] = []
    while lines.i < len(lines.raw):
        line = lines.raw[lines.i]
        if line.strip() and len(line) - len(line.lstrip(" ")) <= indent:
            break
        body.append(line)
        lines.i += 1
    lines.i -= 1
    nonblank = [ln for ln in body if ln.strip()]
    cut = min((len(ln) - len(ln.lstrip(" ")) for ln in nonblank), default=0)
    return "\n".join(ln[cut:] for ln in body).strip("\n") + "\n"


def _parse_node(lines: _Lines, indent: int) -> Any:
    nxt = lines.peek()
    if nxt is None:
        return None
    ind, content = nxt
    if content.startswith("- ") or content == "-":
        return _parse_sequence(lines, ind)
    return _parse_mapping(lines, ind)


def _parse_sequence(lines: _Lines, indent: int) -> list[Any]:
    out: list[Any] = []
    while True:
        nxt = lines.peek()
        if nxt is None or nxt[0] != indent or not (nxt[1].startswith("- ") or nxt[1] == "-"):
            return out
        line = lines.raw[lines.i]
        item_indent = indent + 2
        lines.raw[lines.i] = " " * item_indent + line.strip()[2:]
        if not lines.raw[lines.i].strip():
            lines.i += 1
            out.append(_parse_node(lines, item_indent))
            continue
        if re.match(r"^[^\"'#][^:]*:(\s|$)", lines.raw[lines.i].strip()):
            out.append(_parse_mapping(lines, item_indent))
        else:
            out.append(_scalar(lines, lines.raw[lines.i].strip()))
            lines.i += 1


def _parse_mapping(lines: _Lines, indent: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    while True:
        nxt = lines.peek()
        if nxt is None or nxt[0] < indent:
            return out
        ind, content = nxt
        if ind > indent:
            raise ValueError(f"unexpected indentation on line {lines.i + 1}: {content!r}")
        if content.startswith("- "):
            return out
        m = re.match(r"^([^:]+?):(?:\s+(.*)|)$", content)
        if not m:
            raise ValueError(f"cannot parse line {lines.i + 1}: {content!r}")
        key, rest = m.group(1).strip().strip('"').strip("'"), (m.group(2) or "").strip()
        if rest in ("|", "|-", "|+", ">", ">-"):
            out[key] = _block_scalar(lines, ind)
            lines.i += 1
        elif rest == "" or rest.startswith("#"):
            lines.i += 1
            sub = lines.peek()
            if sub is not None and (sub[0] > ind or (sub[0] == ind and sub[1].startswith("- "))):
                out[key] = _parse_node(lines, sub[0])
            else:
                out[key] = None
        else:
            out[key] = _scalar(lines, rest)
            lines.i += 1


def parse_refractiveindex_yaml(text: str) -> dict[str, Any]:
    """Parse a refractiveindex.info YAML document into nested dicts/lists of strings.

    Scalars are returned as strings (numbers are converted by the caller), the
    ``DATA`` entry is a list of block mappings, and ``data: |`` tables are
    returned verbatim.

    Examples
    --------
    >>> doc = parse_refractiveindex_yaml('''
    ... REFERENCES: "A. Author, J. Opt. 1, 2 (2000)"
    ... DATA:
    ...   - type: formula 1
    ...     wavelength_range: 0.5 2.0
    ...     coefficients: 0 1.0 0.1
    ... ''')
    >>> doc["DATA"][0]["type"], doc["DATA"][0]["coefficients"]
    ('formula 1', '0 1.0 0.1')
    """
    lines = _Lines(text)
    root = _parse_node(lines, 0)
    if lines.peek() is not None:
        raise ValueError(f"trailing content at line {lines.i + 1}: {lines.raw[lines.i]!r}")
    if not isinstance(root, dict):
        raise ValueError("a refractiveindex.info file must be a mapping at the top level")
    return root


# --------------------------------------------------------------------------- #
# refractiveindex.info -> OpticalModel
# --------------------------------------------------------------------------- #
def _floats(s: Any) -> list[float]:
    return [float(v) for v in str(s).replace(",", " ").split()]


def _table(s: str) -> np.ndarray:
    rows = [_floats(r) for r in str(s).strip().splitlines() if r.strip() and not r.strip().startswith("#")]
    width = {len(r) for r in rows}
    if len(width) != 1:
        raise ValueError("tabulated data rows have inconsistent column counts")
    return np.asarray(rows, dtype=float)


def _reference(doc: dict[str, Any], wl_range: tuple[float, float] | None, t_ref: float | None) -> Reference:
    raw = str(doc.get("REFERENCES") or "unknown source")
    doi = None
    m = re.search(r"doi\.org/([^\"'\s<>]+)", raw)
    if m:
        doi = html.unescape(m.group(1)).rstrip(".")
    citation = html.unescape(re.sub(r"<[^>]+>", "", raw))
    citation = re.sub(r"\s+", " ", citation).strip()
    comments = html.unescape(re.sub(r"<[^>]+>", "", str(doc.get("COMMENTS") or ""))).strip()
    t_range = None if t_ref is None else (t_ref, t_ref)
    return Reference(citation, doi=doi, wl_range_um=wl_range, temperature_range_k=t_range, notes=comments)


def _temperature(doc: dict[str, Any]) -> tuple[float | None, tuple[float, float] | None]:
    """Reference temperature [K] and the window in which the data are used without warning."""
    specs = doc.get("SPECS") or {}
    texts = [str(specs.get("temperature", "")) if isinstance(specs, dict) else "", str(doc.get("COMMENTS") or "")]
    # Only a temperature that *leads* the field counts: comments also mention deposition or
    # annealing temperatures ("deposited by ALD at 200 °C"), which are not measurement temperatures.
    for text in texts:
        m = re.match(r"^\s*(-?\d+(?:\.\d+)?)\s*(°C|ºC|K)(?![A-Za-z])", text)
        if m:
            value = float(m.group(1))
            t = value + 273.15 if "C" in m.group(2) else value
            return t, (t - _SINGLE_T_HALF_WIDTH_K, t + _SINGLE_T_HALF_WIDTH_K)
    if any("room temperature" in t.lower() for t in texts):
        return 0.5 * (ROOM_TEMPERATURE_K[0] + ROOM_TEMPERATURE_K[1]), ROOM_TEMPERATURE_K
    return None, None


def _spec_flag(doc: dict[str, Any], name: str) -> bool:
    """A boolean ``SPECS`` entry; absent means ``True`` (the database default: absolute, vacuum)."""
    specs = doc.get("SPECS")
    value = specs.get(name) if isinstance(specs, dict) else None
    if value is None:
        return True
    text = str(value).strip().lower()
    if text in ("true", "yes", "1"):
        return True
    if text in ("false", "no", "0"):
        return False
    raise ValueError(f"SPECS {name}: expected true or false, got {value!r}")


def _air_wavelength_to_vacuum(wl_air: Any, t_air: float | None) -> np.ndarray:
    """Vacuum wavelength whose in-air wavelength is ``wl_air``: ``l_vac = n_air(l_vac) l_air``."""
    lam_air = np.asarray(wl_air, dtype=float)
    lam = lam_air.copy()
    for _ in range(4):                     # contraction factor ~1e-6: converged to rounding
        lam = lam_air * np.asarray(ciddor1996_air(lam, t_air), dtype=float)
    return lam


def _to_absolute_vacuum(evaluate: Any, wl_range: tuple[float, float], *, n_absolute: bool,
                        wavelength_vacuum: bool, t_air: float | None) -> tuple[Any, tuple[float, float]]:
    """Wrap ``evaluate(wl_file, T)`` (file conventions) as a function of the vacuum wavelength.

    Data relative to air (``n_is_absolute: false``) give ``N_abs = N_rel n_air``,
    and data on an air-wavelength axis (``wavelength_is_vacuum: false``) are
    read at ``l_air = l_vac / n_air(l_vac)``; ``n_air`` is
    :func:`~photonix.materials.formulas.ciddor1996_air` at the data temperature.
    """
    def converted(wl: Any, T: Any) -> Any:
        lam = xp.asarray(wl, dtype=float)
        n_air = ciddor1996_air(lam, t_air)
        n = evaluate(lam if wavelength_vacuum else lam / n_air, T)
        return n if n_absolute else n * n_air

    if not wavelength_vacuum:
        lo, hi = _air_wavelength_to_vacuum(np.asarray(wl_range), t_air).tolist()
        wl_range = (float(lo), float(hi))
    return converted, wl_range


def load_refractiveindex_yaml(path: str | os.PathLike[str], *, label: str = "",
                              reference: Reference | None = None,
                              reference_temperature_k: float | None = None,
                              temperature_range_k: tuple[float, float] | None = None,
                              to_vacuum: bool = True) -> OpticalModel:
    """Load a refractiveindex.info YAML file as an :class:`~photonix.materials.OpticalModel`.

    Supports ``formula 1``-``formula 9`` and ``tabulated n``/``k``/``nk`` blocks,
    including the common "formula for n + tabulated k" combination. The
    enforced wavelength range is the file's ``wavelength_range`` (or table span),
    intersected over the n and k blocks. The data temperature is read from
    ``SPECS: temperature`` or the comments ("20 °C", "300 K", "room
    temperature"); passing ``reference_temperature_k``/``temperature_range_k``
    overrides it. The model is temperature-independent; evaluating it more than
    5 K away from a stated data temperature warns.

    The returned model always takes vacuum wavelengths and gives the absolute
    index, the photonix convention. Files whose ``SPECS`` say
    ``n_is_absolute: false`` (index relative to air, usual for optical-glass
    and Malitson-type Sellmeier fits) or ``wavelength_is_vacuum: false``
    (wavelengths in air) are converted with Ciddor's standard-air model
    (:func:`~photonix.materials.formulas.ciddor1996_air`, at the data
    temperature when one is stated): ``N_abs(l) = N_rel(l / n_air(l)) n_air(l)``,
    and the validity range is mapped to vacuum wavelengths. This changes ``n``
    by about ``(n_air - 1) n``, i.e. 3e-4 to 1e-3. Pass ``to_vacuum=False`` to
    evaluate the stored formula or table as written, for example to compare
    coefficients with a source.

    Parameters
    ----------
    path
        File to read.
    label
        Short name used in range warnings.
    reference
        Replaces the citation parsed from ``REFERENCES``.
    reference_temperature_k, temperature_range_k
        Override the temperature metadata.
    to_vacuum
        Convert air-relative and air-wavelength data (``SPECS``) to the
        absolute index versus vacuum wavelength (default). ``False`` uses the
        file's numbers as stored.

    Examples
    --------
    >>> from photonix.materials.io import DATA_DIR, load_refractiveindex_yaml
    >>> ge = load_refractiveindex_yaml(DATA_DIR / "Ge_Nunley2016.yml")
    >>> ge.wl_range_um, ge.reference_temperature_k
    ((0.1879, 2.48), 300.0)
    >>> complex(ge.n(1.55))
    (4.25+0.02519j)
    """
    text = Path(path).read_text(encoding="utf-8")
    doc = parse_refractiveindex_yaml(text)
    blocks = doc.get("DATA")
    if not isinstance(blocks, list) or not blocks:
        raise ValueError(f"{path}: no DATA blocks")

    n_fn = None
    n_range: tuple[float, float] | None = None
    k_tab: tuple[np.ndarray, np.ndarray] | None = None
    n_tab: tuple[np.ndarray, np.ndarray] | None = None
    kinds: list[str] = []
    coeffs: dict[str, Any] = {}
    for blk in blocks:
        btype = str(blk.get("type", "")).strip()
        kinds.append(btype)
        if btype.startswith("formula"):
            if n_fn is not None or n_tab is not None:
                raise ValueError(f"{path}: more than one block defines n")
            number = int(btype.split()[1])
            c = _floats(blk["coefficients"])
            r = _floats(blk["wavelength_range"])
            n_range = (r[0], r[1])
            coeffs.update(formula=number, coefficients=c)

            def n_fn(wl: Any, _num: int = number, _c: list[float] = c) -> Any:
                return refractiveindex_formula(_num, _c, wl)

        elif btype in ("tabulated n", "tabulated nk", "tabulated k"):
            tab = _table(blk["data"])
            if btype in ("tabulated n", "tabulated nk"):
                if n_fn is not None or n_tab is not None:
                    raise ValueError(f"{path}: more than one block defines n")
                n_tab = (tab[:, 0], tab[:, 1])
            if btype in ("tabulated k", "tabulated nk"):
                if k_tab is not None:
                    raise ValueError(f"{path}: more than one block defines k")
                kcol = 1 if btype == "tabulated k" else 2
                k_tab = (tab[:, 0], tab[:, kcol])
        else:
            raise ValueError(f"{path}: unsupported DATA type {btype!r}")
    if n_fn is None and n_tab is None:
        raise ValueError(f"{path}: no block defines the refractive index n")

    t_ref, t_range = _temperature(doc)
    if reference_temperature_k is not None:
        t_ref = reference_temperature_k
    if temperature_range_k is not None:
        t_range = temperature_range_k
    kind = " + ".join(kinds)

    n_absolute = _spec_flag(doc, "n_is_absolute") or not to_vacuum
    wavelength_vacuum = _spec_flag(doc, "wavelength_is_vacuum") or not to_vacuum

    def finish(evaluate: Any, wl_range: tuple[float, float], model_coeffs: dict[str, Any]) -> OpticalModel:
        name = kind
        if not (n_absolute and wavelength_vacuum):
            evaluate, wl_range = _to_absolute_vacuum(evaluate, wl_range, n_absolute=n_absolute,
                                                     wavelength_vacuum=wavelength_vacuum, t_air=t_ref)
            model_coeffs = {**model_coeffs, "air_conversion": {
                "n_is_absolute": n_absolute, "wavelength_is_vacuum": wavelength_vacuum,
                "air_model": "ciddor1996", "air_temperature_k": t_ref}}
            name = f"{kind} (air -> vacuum)"
        ref = reference or _reference(doc, wl_range, t_ref)
        return OpticalModel(name, evaluate, (ref,), wl_range, t_range, t_ref, False, model_coeffs, label)

    if n_tab is not None:
        model = tabulated_model(n_tab[0], n_tab[1], None if k_tab is None else k_tab[1],
                                wl_k=None if k_tab is None else k_tab[0], reference_temperature_k=t_ref,
                                temperature_range_k=t_range, label=label)
        assert model.wl_range_um is not None
        return finish(model.evaluate, model.wl_range_um, dict(model.coeffs))

    assert n_fn is not None and n_range is not None
    wl_range = n_range
    k_x = k_y = None
    if k_tab is not None:
        kx, ky = _table_column(k_tab[0], k_tab[1], "k")
        if np.any(ky < 0):
            raise ValueError(f"{path}: tabulated k must be >= 0")
        wl_range = (max(n_range[0], float(kx[0])), min(n_range[1], float(kx[-1])))
        if wl_range[0] >= wl_range[1]:
            raise ValueError(f"{path}: the n formula and the k table do not overlap")
        k_x, k_y = xp.asarray(kx), xp.asarray(ky)
        coeffs.update(wl_k=kx, k=ky)
    formula_fn = n_fn

    def evaluate(wl: Any, T: Any) -> Any:
        w = xp.asarray(wl, dtype=float)
        n = formula_fn(w)
        if k_x is not None:
            n = n + 1j * xp.interp(w, k_x, k_y)
        return n

    return finish(evaluate, wl_range, coeffs)


# --------------------------------------------------------------------------- #
# Plain n,k tables
# --------------------------------------------------------------------------- #
_WL_SCALE = {"um": 1.0, "µm": 1.0, "micron": 1.0, "nm": 1e-3, "m": 1e6, "mm": 1e3, "a": 1e-4, "angstrom": 1e-4}


def load_nk_csv(path: str | os.PathLike[str], *, wl_unit: str = "um", reference: Reference | None = None,
                reference_temperature_k: float | None = None,
                temperature_range_k: tuple[float, float] | None = None, label: str = "") -> OpticalModel:
    """Load a two- or three-column ``wavelength, n[, k]`` table as a tabulated model.

    Columns may be separated by commas, semicolons, tabs or spaces; lines that
    start with ``#`` and non-numeric header lines are skipped. ``wl_unit`` is
    one of ``"um"``, ``"nm"``, ``"m"``, ``"mm"``, ``"angstrom"`` or ``"eV"``
    (photon energy). ``k`` must be non-negative (``k >= 0`` is loss).

    Examples
    --------
    >>> import tempfile, os
    >>> with tempfile.TemporaryDirectory() as d:
    ...     p = os.path.join(d, "film.csv")
    ...     _ = open(p, "w").write("wavelength_nm,n,k\\n1500,2.00,0.001\\n1600,1.98,0.000\\n")
    ...     m = load_nk_csv(p, wl_unit="nm")
    >>> m.wl_range_um
    (1.5, 1.6)
    >>> complex(m.n(1.55))
    (1.99+0.0005j)
    """
    rows: list[list[float]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        try:
            vals = [float(v) for v in re.split(r"[,;\s]+", s) if v]
        except ValueError:
            if rows:
                raise ValueError(f"{path}: non-numeric line after data: {s!r}") from None
            continue
        rows.append(vals)
    if len(rows) < 2:
        raise ValueError(f"{path}: need at least two data rows")
    tab = np.asarray(rows, dtype=float)
    if tab.ndim != 2 or tab.shape[1] not in (2, 3):
        raise ValueError(f"{path}: expected 2 or 3 numeric columns (wavelength, n[, k])")
    unit = wl_unit.lower()
    if unit == "ev":
        wl = HC_EV_UM / tab[:, 0]
    elif unit in _WL_SCALE:
        wl = tab[:, 0] * _WL_SCALE[unit]
    else:
        raise ValueError(f"unknown wl_unit {wl_unit!r}")
    k = tab[:, 2] if tab.shape[1] == 3 else None
    ref = reference or Reference(f"table {Path(path).name}", notes="loaded with load_nk_csv")
    return tabulated_model(wl, tab[:, 1], k, reference=ref, reference_temperature_k=reference_temperature_k,
                           temperature_range_k=temperature_range_k, label=label or Path(path).stem)
