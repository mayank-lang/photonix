"""photonix circuit subpackage: netlists, the differentiable S-param solver, and builders."""
from __future__ import annotations

from . import engine, hierarchy, modal, netlist_io, reduction, sparse_linalg
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
from .hierarchy import (
    HierarchicalNetlist,
    Subcircuit,
    compile_hierarchical,
    evaluate_expression,
    is_expression,
    load_netlist,
)
from .modal import (
    MODE_SEP,
    ModalNetlist,
    ModeMismatchWarning,
    eme_to_modal,
    is_modal,
    lift_model,
    meep_dataset_to_modal,
    modal_port,
    split_modal_port,
)
from .netlist import Circuit, InstancePort, Netlist
from .netlist_io import format_instance_port, from_sax_netlist, parse_instance_port, to_sax_netlist
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
    "hierarchy",
    "Subcircuit",
    "HierarchicalNetlist",
    "compile_hierarchical",
    "evaluate_expression",
    "is_expression",
    "load_netlist",
    "modal",
    "MODE_SEP",
    "modal_port",
    "split_modal_port",
    "is_modal",
    "lift_model",
    "ModalNetlist",
    "ModeMismatchWarning",
    "eme_to_modal",
    "meep_dataset_to_modal",
    "netlist_io",
    "to_sax_netlist",
    "from_sax_netlist",
    "parse_instance_port",
    "format_instance_port",
]
