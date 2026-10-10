"""photonix circuit subpackage: netlists, the differentiable S-param solver, and builders."""
from __future__ import annotations

from . import engine, reduction, sparse_linalg
from .builders import (
    MeshProgram,
    clements_decompose,
    crow,
    lattice_filter,
    mesh_matrix,
    mzi_mesh,
    mzi_unit_matrix,
    sagnac_mirror,
)
from .engine import (
    CircuitSingularError,
    CompiledTopology,
    FloatingSubcircuitWarning,
    compile_circuit,
    compile_topology,
)
from .netlist import Circuit, InstancePort, Netlist
from .reduction import innerconnect, reduce_network
from .solver import circuit_from_netlist, evaluate_circuit, mzi, ring

__all__ = [
    "Netlist",
    "Circuit",
    "InstancePort",
    "evaluate_circuit",
    "circuit_from_netlist",
    "mzi",
    "ring",
    "lattice_filter",
    "crow",
    "sagnac_mirror",
    "MeshProgram",
    "clements_decompose",
    "mzi_unit_matrix",
    "mesh_matrix",
    "mzi_mesh",
    "engine",
    "reduction",
    "sparse_linalg",
    "compile_circuit",
    "compile_topology",
    "CompiledTopology",
    "CircuitSingularError",
    "FloatingSubcircuitWarning",
    "innerconnect",
    "reduce_network",
]
