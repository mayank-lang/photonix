"""Higher-level layout cells built from :mod:`photonix.layout.components`.

:mod:`~photonix.layout.library.packaging` holds the die-level pieces:
fibre-array grating rows with an alignment loopback, bond-pad arrays, die
frames and automated-test labels.
"""
from __future__ import annotations

from .packaging import (
    FIBER_ARRAY_PITCH,
    TEST_LABEL_LAYER,
    add_test_labels,
    die_frame,
    fiber_array,
    loopback_length,
    opt_in_label,
    pad_array,
)

__all__ = [
    "fiber_array", "loopback_length", "pad_array", "die_frame", "add_test_labels", "opt_in_label",
    "FIBER_ARRAY_PITCH", "TEST_LABEL_LAYER",
]
