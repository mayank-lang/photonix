"""Validated solver-neutral scalar-field exchange and calibrated response maps."""
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "MeshCellBlock",
    "FieldDataset",
    "LinearResponseTerm",
    "LinearIndexModel",
]

_FIELD_FORMAT_VERSION = 1


@dataclass(frozen=True)
class MeshCellBlock:
    """One homogeneous unstructured-mesh cell block."""

    cell_type: str
    data: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.cell_type, str) or not self.cell_type.strip():
            raise ValueError("cell_type must be a non-empty string")
        data = np.asarray(self.data)
        if data.ndim != 2 or data.shape[1] == 0:
            raise ValueError("cell data must have shape (n_cells, nodes_per_cell)")
        if not np.issubdtype(data.dtype, np.integer) or np.issubdtype(data.dtype, np.bool_):
            raise ValueError("cell data must contain integer point indices")
        data = np.asarray(data, dtype=np.int64).copy()
        data.setflags(write=False)
        object.__setattr__(self, "cell_type", self.cell_type.strip())
        object.__setattr__(self, "data", data)


@dataclass(frozen=True)
class FieldDataset:
    """Mesh points with named scalar point fields and explicit units.

    Fields may be real or complex but must contain exactly one scalar per point.
    Units are opaque, required strings; Photonix checks equality and performs no
    implicit unit conversion. Use ``"1"`` for dimensionless quantities.
    """

    points: np.ndarray
    coordinate_unit: str
    fields: dict[str, np.ndarray]
    units: dict[str, str]
    cells: tuple[MeshCellBlock, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        points = np.asarray(self.points, dtype=float)
        if points.ndim != 2 or points.shape[0] == 0 or not 1 <= points.shape[1] <= 3:
            raise ValueError("points must have shape (n_points, dimension) with dimension 1, 2, or 3")
        if not np.all(np.isfinite(points)):
            raise ValueError("points must contain only finite coordinates")
        points = points.copy()
        points.setflags(write=False)
        if (not isinstance(self.coordinate_unit, str) or not self.coordinate_unit
                or self.coordinate_unit != self.coordinate_unit.strip()):
            raise ValueError("coordinate_unit must be a non-empty, trimmed string")

        fields: dict[str, np.ndarray] = {}
        for raw_name, raw_values in dict(self.fields).items():
            name = str(raw_name)
            if not name or name != name.strip():
                raise ValueError("field names must be non-empty and trimmed")
            values = np.asarray(raw_values)
            if values.shape != (len(points),):
                raise ValueError(f"field {name!r} must have shape ({len(points)},), got {values.shape}")
            if not np.issubdtype(values.dtype, np.number) or np.issubdtype(values.dtype, np.bool_):
                raise ValueError(f"field {name!r} must contain numeric scalar values")
            values = np.asarray(values, dtype=complex if np.iscomplexobj(values) else float)
            if not np.all(np.isfinite(values.real)) or not np.all(np.isfinite(values.imag)):
                raise ValueError(f"field {name!r} must contain only finite values")
            values = values.copy()
            values.setflags(write=False)
            fields[name] = values
        if not fields:
            raise ValueError("fields must not be empty")

        units = {str(name): str(unit) for name, unit in dict(self.units).items()}
        if set(units) != set(fields):
            raise ValueError("units must provide exactly one entry for every field")
        if any(not unit or unit != unit.strip() for unit in units.values()):
            raise ValueError(
                "field units must be non-empty, trimmed strings; use '1' for dimensionless fields"
            )

        cells = tuple(self.cells)
        for block in cells:
            if not isinstance(block, MeshCellBlock):
                raise ValueError("cells must contain MeshCellBlock objects")
            if block.data.size and (block.data.min() < 0 or block.data.max() >= len(points)):
                raise ValueError(f"cell block {block.cell_type!r} references a point outside the mesh")

        metadata = dict(self.metadata)
        try:
            json.dumps(metadata)
        except (TypeError, ValueError) as exc:
            raise ValueError("field metadata must be JSON-serializable") from exc
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "coordinate_unit", self.coordinate_unit)
        object.__setattr__(self, "fields", fields)
        object.__setattr__(self, "units", units)
        object.__setattr__(self, "cells", cells)
        object.__setattr__(self, "metadata", metadata)

    def save_npz(self, path: str | Path) -> None:
        """Write a versioned, compressed, pickle-free field exchange file."""
        field_names = tuple(self.fields)
        payload: dict[str, Any] = {
            "format_version": np.asarray(_FIELD_FORMAT_VERSION, dtype=np.int64),
            "points": self.points,
            "coordinate_unit": np.asarray(self.coordinate_unit),
            "field_names": np.asarray(field_names, dtype=str),
            "field_units": np.asarray([self.units[name] for name in field_names], dtype=str),
            "cell_types": np.asarray([block.cell_type for block in self.cells], dtype=str),
            "metadata_json": np.asarray(json.dumps(self.metadata, sort_keys=True)),
        }
        payload.update({f"field_{index}": self.fields[name] for index, name in enumerate(field_names)})
        payload.update({f"cell_{index}": block.data for index, block in enumerate(self.cells)})
        np.savez_compressed(Path(path), **payload)

    @classmethod
    def load_npz(cls, path: str | Path) -> FieldDataset:
        """Load :meth:`save_npz` output with pickle disabled and version checks."""
        with np.load(Path(path), allow_pickle=False) as data:
            version = int(np.asarray(data["format_version"]).item())
            if version != _FIELD_FORMAT_VERSION:
                raise ValueError(f"unsupported field dataset version {version}")
            names = tuple(str(value) for value in np.asarray(data["field_names"]))
            unit_values = tuple(str(value) for value in np.asarray(data["field_units"]))
            if len(names) != len(unit_values):
                raise ValueError("field_names and field_units lengths do not match")
            fields = {name: np.asarray(data[f"field_{index}"]) for index, name in enumerate(names)}
            cells = tuple(
                MeshCellBlock(str(cell_type), np.asarray(data[f"cell_{index}"]))
                for index, cell_type in enumerate(np.asarray(data["cell_types"]))
            )
            metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
            return cls(
                np.asarray(data["points"], dtype=float),
                str(np.asarray(data["coordinate_unit"]).item()),
                fields,
                dict(zip(names, unit_values, strict=True)),
                cells,
                metadata,
            )

    def sample(
        self,
        points,
        *,
        coordinate_unit: str,
        fields: tuple[str, ...] | list[str] | None = None,
        method: str = "linear",
        outside: str = "raise",
        fill_value: complex | None = None,
    ) -> dict[str, np.ndarray]:
        """Interpolate fields onto caller-supplied optical/grid points.

        ``method`` is passed to :func:`scipy.interpolate.griddata` and must be
        ``"linear"``, ``"nearest"``, or ``"cubic"``. Out-of-convex-hull
        handling is explicit: ``outside="raise"`` rejects it, ``"fill"`` uses
        ``fill_value``, and ``"nearest"`` fills only outside points from the
        nearest mesh point. Because nearest-neighbor interpolation extrapolates
        everywhere by definition, ``method="nearest"`` requires
        ``outside="nearest"``.
        """
        from scipy.interpolate import griddata

        if coordinate_unit != self.coordinate_unit:
            raise ValueError(
                f"query coordinates have unit {coordinate_unit!r}; "
                f"dataset coordinates use {self.coordinate_unit!r}"
            )
        query = np.asarray(points, dtype=float)
        expected_dimension = self.points.shape[1]
        if query.ndim != 2 or query.shape[1] != expected_dimension:
            raise ValueError(f"query points must have shape (n, {expected_dimension})")
        if not np.all(np.isfinite(query)):
            raise ValueError("query points must contain only finite coordinates")
        if method not in {"linear", "nearest", "cubic"}:
            raise ValueError("method must be 'linear', 'nearest', or 'cubic'")
        if outside not in {"raise", "fill", "nearest"}:
            raise ValueError("outside must be 'raise', 'fill', or 'nearest'")
        if method == "nearest" and outside != "nearest":
            raise ValueError("method='nearest' requires outside='nearest'")
        selected = tuple(self.fields) if fields is None else tuple(fields)
        if not selected:
            raise ValueError("fields selection must not be empty")
        missing = set(selected) - set(self.fields)
        if missing:
            raise ValueError(f"field dataset does not contain fields: {sorted(missing)}")
        if outside == "fill":
            if fill_value is None:
                raise ValueError("outside='fill' requires an explicit fill_value")
            fill = complex(fill_value)
            if np.isinf(fill.real) or np.isinf(fill.imag):
                raise ValueError("fill_value must not contain infinity; pass NaN explicitly if intended")
        elif fill_value is not None:
            raise ValueError("fill_value is only used when outside='fill'")

        # Meshio commonly represents a planar mesh with three coordinates and a
        # constant z column. Drop every source-constant axis after requiring the
        # query to lie in that same plane, avoiding a degenerate Qhull problem.
        scale = max(1.0, float(np.max(np.abs(self.points))))
        tolerance = 1e-12 * scale
        span = np.ptp(self.points, axis=0)
        constant_axes = span <= tolerance
        if np.any(constant_axes):
            plane = self.points[0, constant_axes]
            if np.any(np.abs(query[:, constant_axes] - plane) > tolerance):
                raise ValueError("query points do not lie on the source mesh's constant-coordinate plane")
        interpolation_points = self.points[:, ~constant_axes]
        interpolation_query = query[:, ~constant_axes]
        if interpolation_points.shape[1] == 0:
            raise ValueError("cannot interpolate a mesh with no varying coordinate axis")
        if method == "cubic" and interpolation_points.shape[1] > 2:
            raise ValueError("method='cubic' supports only one- or two-dimensional active coordinates")

        sampled: dict[str, np.ndarray] = {}
        for name in selected:
            values = self.fields[name]
            if np.iscomplexobj(values):
                real = griddata(
                    interpolation_points, values.real, interpolation_query,
                    method=method, fill_value=np.nan,
                )
                imag = griddata(
                    interpolation_points, values.imag, interpolation_query,
                    method=method, fill_value=np.nan,
                )
                result = np.asarray(real) + 1j * np.asarray(imag)
            else:
                result = np.asarray(
                    griddata(
                        interpolation_points, values, interpolation_query,
                        method=method, fill_value=np.nan,
                    )
                )
            invalid = ~np.isfinite(result.real) | ~np.isfinite(result.imag)
            if np.any(invalid):
                if outside == "raise":
                    raise ValueError(
                        f"{int(np.sum(invalid))} query point(s) lie outside the interpolation domain"
                    )
                if outside == "nearest":
                    nearest = np.asarray(
                        griddata(
                            interpolation_points, values, interpolation_query[invalid], method="nearest"
                        )
                    )
                    result[invalid] = nearest
                else:
                    assert fill_value is not None
                    if np.iscomplexobj(fill_value) and fill_value.imag and not np.iscomplexobj(result):
                        result = result.astype(complex)
                    result[invalid] = fill_value
            sampled[name] = result
        return sampled

    @classmethod
    def from_meshio(
        cls,
        path: str | Path,
        *,
        coordinate_unit: str,
        units: Mapping[str, str],
        fields: tuple[str, ...] | list[str] | None = None,
        file_format: str | None = None,
    ) -> FieldDataset:
        """Read scalar point data through optional ``meshio`` (e.g. VTU output).

        Mesh formats generally do not define trustworthy physical field units,
        so the caller must provide them explicitly rather than accepting guesses.
        """
        try:
            import meshio
        except ImportError as exc:  # pragma: no cover - depends on optional package
            raise ImportError("FieldDataset.from_meshio requires meshio; install photonix[multiphysics]") from exc
        mesh = meshio.read(Path(path), file_format=file_format)
        selected = tuple(mesh.point_data) if fields is None else tuple(fields)
        missing = set(selected) - set(mesh.point_data)
        if missing:
            raise ValueError(f"mesh point data does not contain fields: {sorted(missing)}")
        point_fields: dict[str, np.ndarray] = {}
        for name in selected:
            values = np.asarray(mesh.point_data[name])
            if values.ndim == 2 and values.shape[1] == 1:
                values = values[:, 0]
            point_fields[name] = values
        cells = tuple(MeshCellBlock(block.type, np.asarray(block.data)) for block in mesh.cells)
        return cls(
            np.asarray(mesh.points, dtype=float),
            coordinate_unit,
            point_fields,
            dict(units),
            cells,
            {"source": str(Path(path)), "source_format": file_format},
        )

    def to_meshio(self, path: str | Path, *, file_format: str | None = None) -> None:
        """Write geometry and point data through optional ``meshio``.

        The companion ``units`` mapping is not embedded because mesh formats do
        not share one portable unit convention; retain the NPZ file when units
        and metadata must round-trip exactly.
        """
        try:
            import meshio
        except ImportError as exc:  # pragma: no cover - depends on optional package
            raise ImportError("FieldDataset.to_meshio requires meshio; install photonix[multiphysics]") from exc
        mesh = meshio.Mesh(
            points=self.points,
            cells=[(block.cell_type, block.data) for block in self.cells],
            point_data=self.fields,
        )
        mesh.write(Path(path), file_format=file_format)


@dataclass(frozen=True)
class LinearResponseTerm:
    """One calibrated linear contribution ``coefficient * (field-reference)``."""

    field: str
    reference: float
    coefficient: complex
    field_unit: str
    coefficient_unit: str

    def __post_init__(self) -> None:
        if not self.field or self.field != self.field.strip():
            raise ValueError("response field name must be non-empty and trimmed")
        if isinstance(self.reference, (bool, np.bool_)) or not np.isfinite(self.reference):
            raise ValueError("response reference must be finite")
        if isinstance(self.coefficient, (bool, np.bool_)):
            raise ValueError("response coefficient must be numeric, not bool")
        coefficient = complex(self.coefficient)
        if not np.isfinite(coefficient.real) or not np.isfinite(coefficient.imag):
            raise ValueError("response coefficient must be finite")
        if (not self.field_unit or not self.coefficient_unit
                or self.field_unit != self.field_unit.strip()
                or self.coefficient_unit != self.coefficient_unit.strip()):
            raise ValueError("response field and coefficient units must be explicit and trimmed")
        object.__setattr__(self, "coefficient", coefficient)


@dataclass(frozen=True)
class LinearIndexModel:
    """User-calibrated linear field-to-index/permittivity mapping.

    No material coefficients are bundled. ``provenance`` is required so an
    output cannot silently look foundry-qualified without naming its calibration
    source. Relative permittivity is computed as ``n**2`` for this scalar,
    isotropic contract.
    """

    reference_index: complex
    terms: tuple[LinearResponseTerm, ...]
    provenance: str

    def __post_init__(self) -> None:
        reference_index = complex(self.reference_index)
        if not np.isfinite(reference_index.real) or not np.isfinite(reference_index.imag):
            raise ValueError("reference_index must be finite")
        terms = tuple(self.terms)
        if not terms or len({term.field for term in terms}) != len(terms):
            raise ValueError("terms must contain unique input fields")
        if not isinstance(self.provenance, str) or not self.provenance.strip():
            raise ValueError("calibration provenance must be a non-empty string")
        object.__setattr__(self, "reference_index", reference_index)
        object.__setattr__(self, "terms", terms)
        object.__setattr__(self, "provenance", self.provenance.strip())

    def evaluate(self, dataset: FieldDataset) -> np.ndarray:
        """Evaluate calibrated complex refractive index at every mesh point."""
        index = np.full(len(dataset.points), self.reference_index, dtype=complex)
        for term in self.terms:
            if term.field not in dataset.fields:
                raise ValueError(f"field dataset does not contain response input {term.field!r}")
            if dataset.units[term.field] != term.field_unit:
                raise ValueError(
                    f"field {term.field!r} has unit {dataset.units[term.field]!r}; "
                    f"model requires {term.field_unit!r}"
                )
            index += term.coefficient * (dataset.fields[term.field] - term.reference)
        return index

    def apply(
        self,
        dataset: FieldDataset,
        *,
        index_field: str = "refractive_index",
        permittivity_field: str = "relative_permittivity",
    ) -> FieldDataset:
        """Return a dataset augmented with scalar index and relative permittivity."""
        if not index_field or not permittivity_field or index_field == permittivity_field:
            raise ValueError("output field names must be distinct and non-empty")
        collisions = {index_field, permittivity_field} & set(dataset.fields)
        if collisions:
            raise ValueError(f"response output fields already exist: {sorted(collisions)}")
        index = self.evaluate(dataset)
        terms = [
            {
                "field": term.field,
                "reference": term.reference,
                "coefficient_real": term.coefficient.real,
                "coefficient_imag": term.coefficient.imag,
                "field_unit": term.field_unit,
                "coefficient_unit": term.coefficient_unit,
            }
            for term in self.terms
        ]
        metadata = {
            **dataset.metadata,
            "linear_index_model": {"provenance": self.provenance, "terms": terms},
        }
        return FieldDataset(
            dataset.points,
            dataset.coordinate_unit,
            {**dataset.fields, index_field: index, permittivity_field: index ** 2},
            {**dataset.units, index_field: "1", permittivity_field: "1"},
            dataset.cells,
            metadata,
        )
