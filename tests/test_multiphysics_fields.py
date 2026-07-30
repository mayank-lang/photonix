"""Physics and unit contracts for external scalar-field exchange."""
from __future__ import annotations

import numpy as np
import pytest

from photonix.multiphysics import (
    FieldDataset,
    LinearIndexModel,
    LinearResponseTerm,
    MeshCellBlock,
)


def _planar_fields() -> FieldDataset:
    points = np.array(
        [
            [0.0, 0.0, 2.0],
            [1.0, 0.0, 2.0],
            [0.0, 1.0, 2.0],
            [1.0, 1.0, 2.0],
        ]
    )
    return FieldDataset(
        points,
        "um",
        {
            "temperature": 300.0 + points[:, 0] + 2 * points[:, 1],
            "carrier": np.array([0.0, 1.0, 2.0, 3.0]) * 1e16,
        },
        {"temperature": "K", "carrier": "1/cm^3"},
        (MeshCellBlock("quad", np.array([[0, 1, 3, 2]])),),
        {"solver": "test"},
    )


def test_field_dataset_npz_roundtrip_preserves_units_topology_and_complex_data(tmp_path):
    source = _planar_fields()
    model = LinearIndexModel(
        3.4 + 1e-5j,
        (
            LinearResponseTerm("temperature", 300.0, 1e-4 + 2e-7j, "K", "1/K"),
            LinearResponseTerm("carrier", 0.0, -1e-18 + 1e-20j, "1/cm^3", "cm^3"),
        ),
        "test calibration",
    )
    augmented = model.apply(source)
    path = tmp_path / "fields.npz"
    augmented.save_npz(path)
    restored = FieldDataset.load_npz(path)
    assert restored.coordinate_unit == "um"
    assert restored.units == augmented.units
    assert restored.cells[0].cell_type == "quad"
    assert np.array_equal(restored.cells[0].data, augmented.cells[0].data)
    assert np.allclose(restored.fields["refractive_index"], augmented.fields["refractive_index"])
    assert np.allclose(
        restored.fields["relative_permittivity"],
        restored.fields["refractive_index"] ** 2,
    )
    assert restored.metadata["linear_index_model"]["provenance"] == "test calibration"


def test_linear_response_matches_declared_equation_and_rejects_unit_mismatch():
    fields = _planar_fields()
    term = LinearResponseTerm("temperature", 300.0, 2e-4, "K", "1/K")
    model = LinearIndexModel(3.4, (term,), "calibration report 42")
    expected = 3.4 + 2e-4 * (fields.fields["temperature"] - 300.0)
    assert np.allclose(model.evaluate(fields), expected)

    bad_units = FieldDataset(
        fields.points,
        "um",
        fields.fields,
        {"temperature": "degC", "carrier": "1/cm^3"},
        fields.cells,
    )
    with pytest.raises(ValueError, match="model requires 'K'"):
        model.evaluate(bad_units)


def test_planar_meshio_style_coordinates_interpolate_without_degenerate_qhull():
    fields = _planar_fields()
    query = np.array([[0.25, 0.25, 2.0], [0.75, 0.5, 2.0]])
    sampled = fields.sample(
        query,
        coordinate_unit="um",
        fields=("temperature",),
        method="linear",
    )
    assert np.allclose(sampled["temperature"], 300.0 + query[:, 0] + 2 * query[:, 1])

    outside = np.array([[2.0, 2.0, 2.0]])
    with pytest.raises(ValueError, match="outside"):
        fields.sample(outside, coordinate_unit="um", fields=("temperature",))
    nearest = fields.sample(
        outside,
        coordinate_unit="um",
        fields=("temperature",),
        method="linear",
        outside="nearest",
    )
    assert nearest["temperature"] == pytest.approx([303.0])
    with pytest.raises(ValueError, match="constant-coordinate plane"):
        fields.sample(
            np.array([[0.5, 0.5, 3.0]]),
            coordinate_unit="um",
            fields=("temperature",),
        )


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"coordinate_unit": " "}, "coordinate_unit"),
        ({"units": {"temperature": " K", "carrier": "1/cm^3"}}, "trimmed"),
    ],
)
def test_field_units_are_explicit_and_trimmed(kwargs, match):
    source = _planar_fields()
    values = {
        "points": source.points,
        "coordinate_unit": source.coordinate_unit,
        "fields": source.fields,
        "units": source.units,
        "cells": source.cells,
    }
    values.update(kwargs)
    with pytest.raises(ValueError, match=match):
        FieldDataset(**values)


def test_mesh_connectivity_and_calibration_provenance_are_not_optional():
    with pytest.raises(ValueError, match="outside the mesh"):
        FieldDataset(
            np.array([[0.0], [1.0]]),
            "um",
            {"temperature": np.array([300.0, 301.0])},
            {"temperature": "K"},
            (MeshCellBlock("line", np.array([[0, 2]])),),
        )
    with pytest.raises(ValueError, match="provenance"):
        LinearIndexModel(
            3.4,
            (LinearResponseTerm("temperature", 300.0, 1e-4, "K", "1/K"),),
            "",
        )
