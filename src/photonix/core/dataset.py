"""Versioned sampled S-parameter datasets for simulation and measurement data."""
from __future__ import annotations

import importlib.util
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .constants import C0_UM_S
from .sparams import sdense_to_sdict, sdict_to_sdense
from .types import SDict

__all__ = ["SParameterDataset", "touchstone_capabilities"]

_FORMAT_VERSION = 1
_TOUCHSTONE_SUFFIX = re.compile(r"\.s([1-9][0-9]*)p$", re.IGNORECASE)
_FREQUENCY_SCALES = {
    "hz": 1.0,
    "khz": 1e3,
    "mhz": 1e6,
    "ghz": 1e9,
}


def _has_skrf() -> bool:
    try:
        return importlib.util.find_spec("skrf") is not None
    except (ImportError, ValueError):
        return False


def touchstone_capabilities() -> dict[str, Any]:
    """Report dependency-free and optional Touchstone interchange support.

    The built-in reader/writer intentionally implements the interoperable
    Touchstone 1.0, single-ended S-parameter, real/imaginary subset.  scikit-rf
    is only needed to bridge to richer Touchstone variants and RF workflows.
    """
    return {
        "internal_reader": True,
        "internal_writer": True,
        "internal_version": "1.0",
        "internal_parameter": "S",
        "internal_data_format": "RI",
        "scikit_rf": _has_skrf(),
    }


def _touchstone_port_count(path: Path) -> int:
    match = _TOUCHSTONE_SUFFIX.search(path.name)
    if match is None:
        raise ValueError("Touchstone 1.0 paths must end in .sNp, for example .s2p")
    return int(match.group(1))


def _parse_touchstone_option(line: str) -> tuple[str, str, str, float]:
    unit, parameter, data_format, reference = "ghz", "s", "ma", 50.0
    tokens = line[1:].split()
    index = 0
    while index < len(tokens):
        token = tokens[index].lower()
        if token in _FREQUENCY_SCALES:
            unit = token
        elif token in {"s", "y", "z", "g", "h"}:
            parameter = token
        elif token in {"ri", "ma", "db"}:
            data_format = token
        elif token == "r":
            index += 1
            if index >= len(tokens):
                raise ValueError("Touchstone option R must be followed by a reference resistance")
            try:
                reference = float(tokens[index])
            except ValueError as exc:
                raise ValueError("Touchstone reference resistance must be numeric") from exc
        else:
            raise ValueError(f"unsupported Touchstone option token {tokens[index]!r}")
        index += 1
    if parameter != "s" or data_format != "ri":
        raise ValueError(
            "the internal Touchstone reader supports only single-ended S RI data; "
            "use the optional scikit-rf bridge for other parameter/data formats"
        )
    if not np.isfinite(reference) or reference <= 0:
        raise ValueError("Touchstone reference resistance must be positive and finite")
    return unit, parameter, data_format, reference


def _import_skrf():
    try:
        import skrf  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "scikit-rf is required for this bridge; install it with "
            "`pip install 'photonix[rf]'`"
        ) from exc
    return skrf


def _reference_impedance(value: float | None, metadata: dict[str, Any]) -> float:
    if value is None:
        touchstone = metadata.get("touchstone")
        if isinstance(touchstone, dict):
            value = touchstone.get("reference_impedance_ohm")
        if value is None:
            value = metadata.get("reference_impedance_ohm", 50.0)
    reference = float(value)
    if not np.isfinite(reference) or reference <= 0:
        raise ValueError("reference_impedance must be positive and finite")
    return reference


@dataclass(frozen=True)
class SParameterDataset:
    """Sampled complex scattering matrix over strictly increasing wavelengths.

    ``s`` has shape ``(n_wavelengths, n_ports, n_ports)`` and follows the
    standard matrix convention ``s[:, out, in]``. SDict conversion preserves
    Photonix's human-readable ``(in_port, out_port)`` key convention.
    """

    wavelengths: np.ndarray
    ports: tuple[str, ...]
    s: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        wavelengths = np.asarray(self.wavelengths, dtype=float)
        ports = tuple(str(port) for port in self.ports)
        s = np.asarray(self.s, dtype=complex)
        if wavelengths.ndim != 1 or wavelengths.size == 0:
            raise ValueError("wavelengths must be a non-empty one-dimensional array")
        if not np.all(np.isfinite(wavelengths)) or np.any(wavelengths <= 0):
            raise ValueError("wavelengths must be positive and finite")
        if np.any(np.diff(wavelengths) <= 0):
            raise ValueError("wavelengths must be strictly increasing")
        if not ports or len(ports) != len(set(ports)) or any(not port for port in ports):
            raise ValueError("ports must contain unique, non-empty names")
        expected = (wavelengths.size, len(ports), len(ports))
        if s.shape != expected:
            raise ValueError(f"s must have shape {expected}, got {s.shape}")
        if not np.all(np.isfinite(s.real)) or not np.all(np.isfinite(s.imag)):
            raise ValueError("s must contain finite complex values")
        metadata = dict(self.metadata)
        try:
            json.dumps(metadata)
        except (TypeError, ValueError) as exc:
            raise ValueError("metadata must be JSON-serializable") from exc
        object.__setattr__(self, "wavelengths", wavelengths)
        object.__setattr__(self, "ports", ports)
        object.__setattr__(self, "s", s)
        object.__setattr__(self, "metadata", metadata)

    @classmethod
    def from_sdict(
        cls,
        wavelengths,
        sdict: SDict,
        *,
        ports: tuple[str, ...] | list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SParameterDataset:
        """Build from an array-valued SDict, broadcasting scalar entries."""
        wavelengths = np.asarray(wavelengths, dtype=float)
        if wavelengths.ndim != 1:
            raise ValueError("wavelengths must be one-dimensional")
        selected = None if ports is None else list(ports)
        dense, port_map = sdict_to_sdense(sdict, selected)
        dense = np.asarray(dense, dtype=complex)
        if dense.ndim == 2:
            dense = np.broadcast_to(dense, (wavelengths.size, *dense.shape)).copy()
        elif dense.shape[:-2] != wavelengths.shape:
            try:
                dense = np.broadcast_to(dense, (*wavelengths.shape, *dense.shape[-2:])).copy()
            except ValueError as exc:
                raise ValueError("SDict values do not broadcast over wavelengths") from exc
        ordered_ports = tuple(sorted(port_map, key=port_map.__getitem__))
        return cls(wavelengths.reshape(-1), ordered_ports, dense.reshape(wavelengths.size, *dense.shape[-2:]),
                   dict(metadata or {}))

    def to_sdict(self, *, index: int | None = None, drop_zeros: bool = False) -> SDict:
        """Return an array-valued SDict, or one wavelength sample by ``index``."""
        matrix = self.s if index is None else self.s[index]
        port_map = {port: i for i, port in enumerate(self.ports)}
        return sdense_to_sdict((matrix, port_map), drop_zeros=drop_zeros)

    def sdict_at(self, wavelengths, *, extrapolate: bool = False) -> SDict:
        """Evaluate the sampled data as a circuit-compatible SDict model."""
        query = np.asarray(wavelengths, dtype=float)
        if not np.all(np.isfinite(query)) or np.any(query <= 0):
            raise ValueError("query wavelengths must be positive and finite")
        flat = query.reshape(-1)
        order = np.argsort(flat)
        sorted_query = flat[order]
        if sorted_query.size == 0:
            raise ValueError("query wavelengths must not be empty")
        unique, inverse = np.unique(sorted_query, return_inverse=True)
        sampled = self.interpolate(unique, extrapolate=extrapolate).to_sdict()
        undo_sort = np.empty_like(order)
        undo_sort[order] = np.arange(order.size)
        out: SDict = {}
        for key, values in sampled.items():
            sorted_values = np.asarray(values)[inverse]
            restored = sorted_values[undo_sort].reshape(query.shape)
            out[key] = complex(restored) if query.ndim == 0 else restored
        return out

    def __call__(self, *, wl=1.55, **_kwargs) -> SDict:
        """Use the dataset directly as a Photonix component model."""
        return self.sdict_at(wl)

    def interpolate(self, wavelengths, *, extrapolate: bool = False) -> SParameterDataset:
        """Linearly interpolate real/imaginary parts onto a new wavelength grid."""
        target = np.asarray(wavelengths, dtype=float)
        if target.ndim != 1 or target.size == 0 or np.any(np.diff(target) <= 0):
            raise ValueError("target wavelengths must be non-empty and strictly increasing")
        if not np.all(np.isfinite(target)) or np.any(target <= 0):
            raise ValueError("target wavelengths must be positive and finite")
        if not extrapolate and (target[0] < self.wavelengths[0] or target[-1] > self.wavelengths[-1]):
            raise ValueError("target wavelengths extend outside the sampled range")
        out = np.empty((target.size, len(self.ports), len(self.ports)), dtype=complex)
        for i in range(len(self.ports)):
            for j in range(len(self.ports)):
                values = self.s[:, i, j]
                real = np.interp(target, self.wavelengths, values.real)
                imag = np.interp(target, self.wavelengths, values.imag)
                if extrapolate and self.wavelengths.size > 1:
                    left = target < self.wavelengths[0]
                    right = target > self.wavelengths[-1]
                    for part, sampled in ((real, values.real), (imag, values.imag)):
                        slope_left = (sampled[1] - sampled[0]) / (self.wavelengths[1] - self.wavelengths[0])
                        slope_right = (sampled[-1] - sampled[-2]) / (
                            self.wavelengths[-1] - self.wavelengths[-2]
                        )
                        part[left] = sampled[0] + slope_left * (target[left] - self.wavelengths[0])
                        part[right] = sampled[-1] + slope_right * (target[right] - self.wavelengths[-1])
                out[:, i, j] = real + 1j * imag
        metadata = {**self.metadata, "interpolated_from_samples": int(self.wavelengths.size)}
        return SParameterDataset(target, self.ports, out, metadata)

    def save_npz(self, path: str | Path) -> None:
        """Write a portable, non-pickle NPZ data contract."""
        np.savez_compressed(
            Path(path),
            format_version=np.asarray(_FORMAT_VERSION, dtype=np.int64),
            wavelengths_um=self.wavelengths,
            ports=np.asarray(self.ports, dtype=str),
            s=self.s,
            metadata_json=np.asarray(json.dumps(self.metadata, sort_keys=True)),
        )

    @classmethod
    def load_npz(cls, path: str | Path) -> SParameterDataset:
        """Load :meth:`save_npz` output with pickle disabled and version checks."""
        with np.load(Path(path), allow_pickle=False) as data:
            version = int(np.asarray(data["format_version"]).item())
            if version != _FORMAT_VERSION:
                raise ValueError(f"unsupported S-parameter dataset version {version}")
            metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
            return cls(
                np.asarray(data["wavelengths_um"], dtype=float),
                tuple(str(port) for port in np.asarray(data["ports"])),
                np.asarray(data["s"], dtype=complex),
                metadata,
            )

    def save_touchstone(
        self,
        path: str | Path,
        *,
        reference_impedance: float | None = None,
        frequency_unit: str = "ghz",
    ) -> None:
        """Write standards-compatible Touchstone 1.0 single-ended ``S RI``.

        Photonix wavelengths are converted from micrometres to increasing
        frequencies.  Two-port data uses the required legacy order
        ``S11, S21, S12, S22``; larger matrices are written row-wise.  Port
        names and JSON metadata are preserved in ignorable comments.
        """
        target = Path(path)
        n_ports = _touchstone_port_count(target)
        if n_ports != len(self.ports):
            raise ValueError(
                f"Touchstone suffix declares {n_ports} ports, dataset has {len(self.ports)}"
            )
        reference = _reference_impedance(reference_impedance, self.metadata)
        unit = str(frequency_unit).lower()
        if unit not in _FREQUENCY_SCALES:
            raise ValueError("frequency_unit must be one of Hz, kHz, MHz, or GHz")

        frequencies = C0_UM_S / self.wavelengths
        order = np.argsort(frequencies)
        frequencies = frequencies[order] / _FREQUENCY_SCALES[unit]
        matrices = self.s[order]
        lines = [
            "! Touchstone 1.0 S-parameters written by Photonix",
            f"! Photonix ports: {json.dumps(self.ports, ensure_ascii=True, separators=(',', ':'))}",
            "! Photonix metadata: "
            + json.dumps(self.metadata, ensure_ascii=True, sort_keys=True, separators=(",", ":")),
            f"# {unit.upper()} S RI R {reference:.17g}",
        ]
        for frequency, matrix in zip(frequencies, matrices, strict=True):
            if n_ports <= 2:
                values = matrix.reshape(-1, order="F" if n_ports == 2 else "C")
                pairs = " ".join(f"{value.real:.17g} {value.imag:.17g}" for value in values)
                lines.append(f"{frequency:.17g} {pairs}")
                continue
            for row_index, row in enumerate(matrix):
                for start in range(0, n_ports, 4):
                    pairs = " ".join(
                        f"{value.real:.17g} {value.imag:.17g}" for value in row[start:start + 4]
                    )
                    prefix = f"{frequency:.17g} " if row_index == 0 and start == 0 else ""
                    lines.append(prefix + pairs)
        target.write_text("\n".join(lines) + "\n", encoding="ascii", newline="\n")

    @classmethod
    def load_touchstone(cls, path: str | Path) -> SParameterDataset:
        """Read the internal Touchstone 1.0 single-ended ``S RI`` subset.

        The ``.sNp`` suffix supplies the port count.  Richer Touchstone 2.x,
        mixed-mode, noise, and non-RI files are deliberately left to scikit-rf.
        """
        source = Path(path)
        n_ports = _touchstone_port_count(source)
        try:
            text = source.read_text(encoding="ascii")
        except UnicodeDecodeError as exc:
            raise ValueError("Touchstone files must contain ASCII text") from exc

        option_line: str | None = None
        data_tokens: list[str] = []
        embedded_ports: tuple[str, ...] | None = None
        embedded_metadata: dict[str, Any] = {}
        saw_data = False
        for line_number, raw_line in enumerate(text.splitlines(), start=1):
            content, separator, comment = raw_line.partition("!")
            if separator:
                stripped_comment = comment.strip()
                lower_comment = stripped_comment.lower()
                if lower_comment.startswith("photonix ports:"):
                    payload = stripped_comment.split(":", 1)[1].strip()
                    try:
                        decoded = json.loads(payload)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"invalid Photonix port comment on line {line_number}") from exc
                    if not isinstance(decoded, list) or not all(isinstance(item, str) for item in decoded):
                        raise ValueError("Photonix port comment must contain a JSON string list")
                    embedded_ports = tuple(decoded)
                elif lower_comment.startswith("photonix metadata:"):
                    payload = stripped_comment.split(":", 1)[1].strip()
                    try:
                        decoded = json.loads(payload)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"invalid Photonix metadata comment on line {line_number}") from exc
                    if not isinstance(decoded, dict):
                        raise ValueError("Photonix metadata comment must contain a JSON object")
                    embedded_metadata = decoded
            content = content.strip()
            if not content:
                continue
            if content.startswith("["):
                raise ValueError(
                    "the internal reader supports Touchstone 1.0 RI files only; "
                    "use scikit-rf for Touchstone 2.x"
                )
            if content.startswith("#"):
                if option_line is not None or saw_data:
                    raise ValueError("Touchstone must contain one option line before network data")
                option_line = content
                continue
            if option_line is None:
                raise ValueError("Touchstone option line must precede network data")
            saw_data = True
            data_tokens.extend(content.split())

        if option_line is None:
            raise ValueError("Touchstone option line is missing")
        if not data_tokens:
            raise ValueError("Touchstone network data is missing")
        unit, _, data_format, reference = _parse_touchstone_option(option_line)
        record_size = 1 + 2 * n_ports * n_ports
        if len(data_tokens) % record_size:
            raise ValueError(
                "Touchstone network data is truncated or contains unsupported noise/extra data"
            )
        try:
            numeric = np.asarray([float(token) for token in data_tokens], dtype=float)
        except ValueError as exc:
            raise ValueError("Touchstone network data must be numeric") from exc
        if not np.all(np.isfinite(numeric)):
            raise ValueError("Touchstone network data must be finite")
        records = numeric.reshape(-1, record_size)
        frequencies = records[:, 0] * _FREQUENCY_SCALES[unit]
        if np.any(frequencies <= 0) or np.any(np.diff(frequencies) <= 0):
            raise ValueError("Touchstone frequencies must be positive and strictly increasing")

        matrices = np.empty((records.shape[0], n_ports, n_ports), dtype=complex)
        for sample_index, record in enumerate(records):
            pairs = record[1:].reshape(-1, 2)
            values = pairs[:, 0] + 1j * pairs[:, 1]
            matrices[sample_index] = values.reshape(
                n_ports, n_ports, order="F" if n_ports == 2 else "C"
            )
        wavelengths = C0_UM_S / frequencies
        order = np.argsort(wavelengths)
        ports = embedded_ports or tuple(f"o{index + 1}" for index in range(n_ports))
        if len(ports) != n_ports:
            raise ValueError(
                f"Photonix port comment defines {len(ports)} ports, expected {n_ports}"
            )
        metadata = dict(embedded_metadata)
        metadata["touchstone"] = {
            "version": "1.0",
            "parameter": "S",
            "data_format": data_format.upper(),
            "frequency_unit": unit.upper(),
            "reference_impedance_ohm": reference,
        }
        return cls(wavelengths[order], ports, matrices[order], metadata)

    def to_skrf(self, *, reference_impedance: float | None = None, name: str | None = None):
        """Convert to an optional :class:`skrf.Network` with increasing frequency."""
        reference = _reference_impedance(reference_impedance, self.metadata)
        skrf = _import_skrf()
        frequencies = C0_UM_S / self.wavelengths
        order = np.argsort(frequencies)
        frequency = skrf.Frequency.from_f(frequencies[order], unit="hz")
        network = skrf.Network(
            frequency=frequency,
            s=self.s[order],
            z0=reference,
            name=name,
        )
        network.port_names = list(self.ports)
        return network

    @classmethod
    def from_skrf(
        cls,
        network,
        *,
        ports: tuple[str, ...] | list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SParameterDataset:
        """Build from a scikit-rf-like network without importing scikit-rf."""
        frequencies = np.asarray(network.f, dtype=float)
        matrices = np.asarray(network.s, dtype=complex)
        if frequencies.ndim != 1 or frequencies.size == 0:
            raise ValueError("scikit-rf network frequencies must be one-dimensional and non-empty")
        if not np.all(np.isfinite(frequencies)) or np.any(frequencies <= 0):
            raise ValueError("scikit-rf network frequencies must be positive and finite")
        if matrices.ndim != 3 or matrices.shape[0] != frequencies.size or matrices.shape[1] != matrices.shape[2]:
            raise ValueError("scikit-rf network s must have shape (frequencies, ports, ports)")
        n_ports = matrices.shape[1]
        if ports is None:
            network_ports = getattr(network, "port_names", None)
            if network_ports is not None and len(network_ports) == n_ports:
                selected_ports = tuple(str(port) for port in network_ports)
            else:
                selected_ports = tuple(f"o{index + 1}" for index in range(n_ports))
        else:
            selected_ports = tuple(ports)
        provenance = {"source_format": "scikit-rf", **dict(metadata or {})}
        z0 = np.asarray(getattr(network, "z0", np.asarray([])))
        if z0.size and np.all(np.isfinite(z0)) and np.allclose(z0.imag, 0.0):
            real_z0 = z0.real
            if np.allclose(real_z0, real_z0.reshape(-1)[0]):
                provenance.setdefault("reference_impedance_ohm", float(real_z0.reshape(-1)[0]))
        network_name = getattr(network, "name", None)
        if network_name and "network_name" not in provenance:
            provenance["network_name"] = str(network_name)
        wavelengths = C0_UM_S / frequencies
        order = np.argsort(wavelengths)
        return cls(wavelengths[order], selected_ports, matrices[order], provenance)
