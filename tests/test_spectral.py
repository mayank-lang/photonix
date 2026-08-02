"""Accuracy and safeguards for sampled phase/delay analysis."""
from __future__ import annotations

import numpy as np
import pytest

import photonix as px
from photonix.core.constants import C0_UM_S


def test_nonuniform_derivative_is_polynomial_exact_and_vectorized():
    x = np.array([-1.2, -0.7, -0.1, 0.2, 0.9, 1.7, 2.4])
    polynomial = x**4 - 2.0 * x**3 + 0.5 * x**2 + 3.0 * x - 7.0
    values = np.stack([polynomial, (2.0 - 1.0j) * polynomial], axis=0)

    first = px.differentiate_samples(x, values, axis=1, stencil=5)
    second = px.differentiate_samples(x, values, axis=1, derivative=2, stencil=5)
    expected_first = 4.0 * x**3 - 6.0 * x**2 + x + 3.0
    expected_second = 12.0 * x**2 - 12.0 * x + 1.0

    assert np.allclose(first[0], expected_first, rtol=2e-13, atol=2e-13)
    assert np.allclose(first[1], (2.0 - 1.0j) * expected_first, rtol=2e-13, atol=2e-13)
    assert np.allclose(second[0], expected_second, rtol=2e-12, atol=2e-12)


def test_waveguide_group_delay_sign_units_and_nonuniform_frequency_grid():
    # Uniform wavelength sampling is deliberately nonuniform in omega.
    wavelengths = np.linspace(1.54, 1.56, 1001) ** 1.0003
    length_um, group_index = 1234.0, 4.2
    transfer = px.to_numpy(
        px.components.straight(wl=wavelengths, length=length_um, neff=2.4, ng=group_index)[("o1", "o2")]
    )
    delay = px.group_delay(wavelengths, transfer)
    expected = group_index * length_um / C0_UM_S

    assert np.all(delay > 0.0)
    assert np.allclose(delay, expected, rtol=2e-10, atol=1e-22)
    # Residual numerical curvature stays below 0.2 fs^2, six orders below the
    # deliberately dispersive fixture tested next.
    assert np.max(np.abs(px.group_delay_dispersion(wavelengths, transfer))) < 2e-31


def test_quadratic_spectral_phase_recovers_delay_and_gdd():
    wavelengths = np.linspace(1.545, 1.555, 301)
    omega = 2.0 * np.pi * C0_UM_S / wavelengths
    omega0 = float(np.mean(omega))
    delay0 = 3.0e-12
    gdd = 1.7e-25
    phase = -(delay0 * (omega - omega0) + 0.5 * gdd * (omega - omega0) ** 2)
    transfer = 0.7 * np.exp(1j * phase)

    delay = px.group_delay(wavelengths, transfer)
    recovered_gdd = px.group_delay_dispersion(wavelengths, transfer)
    assert np.allclose(delay, delay0 + gdd * (omega - omega0), rtol=2e-9, atol=2e-21)
    assert np.allclose(recovered_gdd, gdd, rtol=2e-6, atol=1e-31)


def test_dataset_delay_uses_sdict_input_output_port_order():
    wavelengths = np.linspace(1.54, 1.56, 401)
    delay_s = 2.5e-12
    omega = 2.0 * np.pi * C0_UM_S / wavelengths
    s = np.zeros((wavelengths.size, 2, 2), dtype=complex)
    s[:, 1, 0] = np.exp(-1j * omega * delay_s)
    dataset = px.SParameterDataset(wavelengths, ("in", "out"), s)

    assert np.allclose(dataset.group_delay("in", "out"), delay_s, rtol=2e-10)
    with pytest.raises(KeyError, match="unknown dataset port"):
        dataset.group_delay("missing", "out")


def test_delay_rejects_phase_zeros_and_near_nyquist_sampling():
    wavelengths = np.linspace(1.54, 1.56, 9)
    response = np.ones(wavelengths.size, dtype=complex)
    response[4] = 0.0
    with pytest.raises(ValueError, match="phase is undefined"):
        px.group_delay(wavelengths, response)

    # An alternating 0/near-pi phase is too close to the unwrap ambiguity for
    # accuracy-first delay extraction, even though np.unwrap would return data.
    response = np.exp(1j * np.arange(wavelengths.size) * 0.95 * np.pi)
    with pytest.raises(ValueError, match="refine the wavelength grid"):
        px.group_delay(wavelengths, response)


def test_derivative_rejects_nonmonotonic_and_undersized_stencils():
    with pytest.raises(ValueError, match="strictly monotonic"):
        px.differentiate_samples([0.0, 1.0, 0.5], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="requires at least"):
        px.differentiate_samples([0.0, 1.0], [1.0, 2.0], derivative=2)
