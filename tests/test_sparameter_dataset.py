"""Tests for the versioned sampled S-parameter data contract."""
from __future__ import annotations

import numpy as np
import pytest

import photonix.core.dataset as dataset_module
from photonix.core import SParameterDataset, touchstone_capabilities


def test_dataset_sdict_matrix_convention_and_interpolation():
    wavelengths = np.array([1.5, 1.6])
    sdict = {
        ("o1", "o1"): np.array([0.1, 0.2]),
        ("o1", "o2"): np.array([0.8 + 0.1j, 0.6 + 0.3j]),
        ("o2", "o1"): np.array([0.7, 0.5]),
        ("o2", "o2"): 0.05,
    }
    dataset = SParameterDataset.from_sdict(wavelengths, sdict, metadata={"solver": "test"})
    assert dataset.s.shape == (2, 2, 2)
    assert np.allclose(dataset.s[:, 1, 0], sdict[("o1", "o2")])
    assert np.allclose(dataset.to_sdict()[("o2", "o1")], sdict[("o2", "o1")])
    mid = dataset.interpolate([1.55])
    assert mid.s[0, 1, 0] == pytest.approx(0.7 + 0.2j)
    assert dataset(wl=1.55)[("o1", "o2")] == pytest.approx(0.7 + 0.2j)
    unordered = dataset.sdict_at(np.array([1.6, 1.5]))[("o1", "o2")]
    assert np.allclose(unordered, np.array([0.6 + 0.3j, 0.8 + 0.1j]))
    with pytest.raises(ValueError, match="outside"):
        dataset.interpolate([1.4])


def test_dataset_npz_roundtrip_without_pickle(tmp_path):
    source = SParameterDataset(
        np.array([1.5, 1.6]),
        ("in", "out"),
        np.arange(8).reshape(2, 2, 2).astype(complex),
        {"backend": "meep", "version": 1},
    )
    path = tmp_path / "device.npz"
    source.save_npz(path)
    loaded = SParameterDataset.load_npz(path)
    assert loaded.ports == source.ports
    assert loaded.metadata == source.metadata
    assert np.array_equal(loaded.wavelengths, source.wavelengths)
    assert np.array_equal(loaded.s, source.s)


def test_dataset_rejects_nonmonotonic_wavelengths_and_bad_shape():
    with pytest.raises(ValueError, match="strictly increasing"):
        SParameterDataset(np.array([1.6, 1.5]), ("o1",), np.zeros((2, 1, 1)))
    with pytest.raises(ValueError, match="shape"):
        SParameterDataset(np.array([1.5]), ("o1", "o2"), np.zeros((1, 2, 1)))


def test_touchstone_ri_two_port_order_and_roundtrip(tmp_path):
    matrices = np.array([
        [[0.1 + 0.01j, 0.2 + 0.02j], [0.3 + 0.03j, 0.4 + 0.04j]],
        [[0.5 + 0.05j, 0.6 + 0.06j], [0.7 + 0.07j, 0.8 + 0.08j]],
    ])
    source = SParameterDataset(
        np.array([1.5, 1.6]),
        ("input", "output"),
        matrices,
        {"solver": "unit-test", "run": 3},
    )
    path = tmp_path / "asymmetric.s2p"
    source.save_touchstone(path, reference_impedance=75.0)

    data_line = next(
        line for line in path.read_text(encoding="ascii").splitlines()
        if line and not line.startswith(("!", "#"))
    )
    tokens = [float(token) for token in data_line.split()]
    values = np.asarray(tokens[1:]).reshape(-1, 2)
    values = values[:, 0] + 1j * values[:, 1]
    # Increasing frequency reverses the wavelength samples. Touchstone 1.0's
    # special two-port order is S11, S21, S12, S22.
    assert np.allclose(values, matrices[1][[0, 1, 0, 1], [0, 0, 1, 1]])

    loaded = SParameterDataset.load_touchstone(path)
    assert loaded.ports == source.ports
    assert np.allclose(loaded.wavelengths, source.wavelengths)
    assert np.allclose(loaded.s, source.s)
    assert loaded.metadata["solver"] == "unit-test"
    assert loaded.metadata["touchstone"]["reference_impedance_ohm"] == 75.0


def test_touchstone_ri_three_port_row_order_and_validation(tmp_path):
    path = tmp_path / "known.s3p"
    path.write_text(
        "! wrapped row-wise 3-port fixture\n"
        "# MHz S RI R 50\n"
        "100 1 1 2 2 3 3\n"
        "4 4 5 5 6 6\n"
        "7 7 8 8 9 9\n"
        "200 11 11 12 12 13 13\n"
        "14 14 15 15 16 16\n"
        "17 17 18 18 19 19\n",
        encoding="ascii",
    )
    loaded = SParameterDataset.load_touchstone(path)
    assert loaded.ports == ("o1", "o2", "o3")
    # The higher-frequency record is first after conversion to increasing wavelength.
    expected = np.arange(11, 20).reshape(3, 3) * (1 + 1j)
    assert np.array_equal(loaded.s[0], expected)

    bad_format = tmp_path / "bad.s1p"
    bad_format.write_text("# GHz S MA R 50\n1 1 0\n", encoding="ascii")
    with pytest.raises(ValueError, match="only single-ended S RI"):
        SParameterDataset.load_touchstone(bad_format)
    with pytest.raises(ValueError, match="declares 2 ports"):
        loaded.save_touchstone(tmp_path / "wrong.s2p")


def test_touchstone_capabilities_and_optional_skrf_bridge(monkeypatch):
    capabilities = touchstone_capabilities()
    assert capabilities["internal_version"] == "1.0"
    assert capabilities["internal_data_format"] == "RI"
    assert isinstance(capabilities["scikit_rf"], bool)

    class FakeFrequency:
        def __init__(self, frequencies):
            self.f = np.asarray(frequencies)

        @classmethod
        def from_f(cls, frequencies, unit):
            assert unit == "hz"
            return cls(frequencies)

    class FakeNetwork:
        def __init__(self, *, frequency, s, z0, name):
            self.frequency = frequency
            self.f = frequency.f
            self.s = np.asarray(s)
            self.z0 = z0
            self.name = name
            self.port_names = None

    fake_skrf = type("FakeSkrf", (), {"Frequency": FakeFrequency, "Network": FakeNetwork})
    monkeypatch.setattr(dataset_module, "_import_skrf", lambda: fake_skrf)
    source = SParameterDataset(
        np.array([1.5, 1.6]),
        ("in", "out"),
        np.arange(8).reshape(2, 2, 2).astype(complex),
    )
    network = source.to_skrf(reference_impedance=60.0, name="bridge")
    assert network.z0 == 60.0
    assert network.port_names == ["in", "out"]
    assert np.array_equal(network.s, source.s[::-1])
    restored = SParameterDataset.from_skrf(network)
    assert restored.ports == source.ports
    assert np.allclose(restored.wavelengths, source.wavelengths)
    assert np.array_equal(restored.s, source.s)
    assert restored.metadata["network_name"] == "bridge"
    assert restored.metadata["reference_impedance_ohm"] == 60.0
