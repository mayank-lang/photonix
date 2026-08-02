"""photonix core: backend, constants, units, type system, and S-parameter utils.

This subpackage is the stable foundation every other photonix module builds on.
It has no dependencies on the rest of photonix and must remain import-light.
"""
from __future__ import annotations

from . import constants, units
from .backend import (
    HAS_JAX,
    asarray,
    backend_name,
    device_count,
    grad,
    jit,
    to_numpy,
    use_x64,
    value_and_grad,
    vmap,
    xp,
)
from .dataset import SParameterDataset, touchstone_capabilities
from .quality import SParameterDiagnostics, analyze_sparameters, project_passive
from .sparams import (
    as_sdense,
    as_sdict,
    insertion_loss_db,
    is_passive,
    is_reciprocal,
    power,
    reciprocal,
    sdense_to_sdict,
    sdict_to_sdense,
    validate_sdict,
)
from .spectral import differentiate_samples, group_delay, group_delay_dispersion
from .types import (
    AliasedSDict,
    Array,
    Complex,
    Float,
    Model,
    ModelFactory,
    PortName,
    PortPair,
    SCoo,
    SDense,
    SDict,
    Settings,
    SType,
    is_scoo,
    is_sdense,
    is_sdict,
    ports_of,
)

__all__ = [
    # backend
    "HAS_JAX", "xp", "jit", "grad", "value_and_grad", "vmap",
    "asarray", "to_numpy", "use_x64", "device_count", "backend_name",
    # submodules
    "constants", "units",
    # types
    "Array", "Complex", "Float", "Model", "ModelFactory", "PortName", "PortPair",
    "SCoo", "SDense", "SDict", "AliasedSDict", "SType", "Settings",
    "SParameterDataset", "touchstone_capabilities",
    "SParameterDiagnostics", "analyze_sparameters", "project_passive",
    "is_sdict", "is_sdense", "is_scoo", "ports_of",
    # sparams
    "as_sdict", "as_sdense", "sdict_to_sdense", "sdense_to_sdict",
    "reciprocal", "is_reciprocal", "is_passive", "power",
    "insertion_loss_db", "validate_sdict",
    "differentiate_samples", "group_delay", "group_delay_dispersion",
]
