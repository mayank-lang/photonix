"""Tests for the core foundation: types, sparams, units, backend."""
from __future__ import annotations

import numpy as np
import pytest

import photonix as px
from photonix.core import sparams, units
from photonix.core.constants import C0_UM_S


def test_backend_loads():
    assert px.backend_name() in {"jax:cpu", "jax:gpu", "jax:tpu", "numpy"}
    assert px.xp.asarray([1.0, 2.0]).sum() == 3.0


def test_sdict_roundtrip(four_port):
    S, pm = sparams.sdict_to_sdense(four_port)
    assert S.shape[-1] == len(pm)
    back = sparams.sdense_to_sdict((S, pm))
    for k in four_port:
        assert np.allclose(np.asarray(back[k]), np.asarray(four_port[k]))


def test_power_and_il():
    assert abs(float(px.power(0.5 + 0j)) - 0.25) < 1e-12
    assert abs(float(px.insertion_loss_db(np.sqrt(0.5))) - 3.0103) < 1e-3


def test_reciprocity_and_passivity(four_port):
    assert sparams.is_reciprocal(four_port)
    assert sparams.is_passive(four_port)


def test_unit_conversions():
    # 1.55 um <-> ~193.4 THz
    assert abs(float(units.wl_to_freq(1.55)) - C0_UM_S * 1e-12 / 1.55) < 1e-6
    assert abs(float(units.freq_to_wl(units.wl_to_freq(1.55))) - 1.55) < 1e-9
    # 3 dB ~ half power
    assert abs(float(units.db_to_lin(-3.0)) ** 2 - 0.5012) < 1e-3


def test_validate_sdict_rejects_bad():
    with pytest.raises(ValueError):
        sparams.validate_sdict({("o1",): 1.0})  # malformed key
