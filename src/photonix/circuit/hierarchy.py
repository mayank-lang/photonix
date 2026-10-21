"""Hierarchical netlists: parameterised subcircuits, expressions and per-call caching.

A :class:`HierarchicalNetlist` is a top-level :class:`~photonix.circuit.Netlist`
plus a library of named :class:`Subcircuit` definitions. An instance whose
model name is a subcircuit name is a *subcircuit instance*; its ports are the
subcircuit's external port names and its settings bind the subcircuit's
parameters. Subcircuits may contain further subcircuit instances to any depth
(cycles are rejected).

Parameters and expressions
--------------------------
Scoping follows SPICE ``.subckt`` semantics (Nagel, SPICE2, UCB/ERL M520,
1975): instance settings override a subcircuit's parameter defaults, which in
turn override the leaf models' own defaults. A setting (or a parameter
default) that is a string starting with ``"="`` is an *expression*, evaluated
by :func:`evaluate_expression` in the scope of the enclosing subcircuit
(``{"length": "=pi*radius"}``). Instance settings written in a parent are
evaluated in the parent's scope before they are bound in the child. The
evaluator is a whitelist walk of the Python AST -- numbers, ``+ - * / ** %``,
unary signs, parentheses, scope names and the functions ``pi e sqrt sin cos
tan exp log abs min max`` -- and never calls :func:`eval`. Arithmetic stays in
the backend, so parameters may be JAX tracers and gradients flow through
expressions.

Evaluation
----------
:func:`compile_hierarchical` offers two modes that solve the same linear
equations:

``"hierarchical"`` (default)
    Recursive block elimination: every subcircuit instance is replaced by the
    exposed-port S-matrix the circuit engine computes for it. Because the
    interconnection equations are linear, eliminating a subcircuit's internal
    waves first is exactly a Schur complement of the flat system, so the
    result equals the flat one up to round-off.
``"flatten"``
    One flat solve of :meth:`HierarchicalNetlist.flatten`'s topology.

Within one call, identical work is done once: a leaf model is evaluated once
per distinct ``(model, settings)`` and a subcircuit once per distinct
``(subcircuit, resolved parameters, call-time overrides inside it)``. Values
are compared by value for Python/NumPy numbers and strings and by identity for
JAX arrays and tracers, so under ``jax.jit`` only instances that share the
same setting objects are deduplicated and gradients still reach every
instance. The cache lives for one call only (wavelength arrays are not hashed).

Flattening names leaf instances ``"<inst>.<subinst>..."``, the naming of
:func:`photonix.layout.extract_netlist`, so
``photonix.layout.extract_hierarchical_netlist(cell).flatten()`` equals
``photonix.layout.extract_netlist(cell)``.

Serialization
-------------
:meth:`HierarchicalNetlist.to_dict` writes ``{"format": "photonix.netlist",
"version": 2, ...}``: the version-1 fields of the top netlist plus
``"parameters"`` and ``"subcircuits"``. Version-1 files load as a hierarchy
with no subcircuits. References: SAX recursive netlists
(github.com/flaport/sax), gdsfactory ``get_netlist_recursive``, Ansys
Lumerical INTERCONNECT compound elements.

Evidence
--------
Level 3 (independent computation): hierarchical mode, flatten mode and
hand-flattened netlists solved by the engine agree to ``<= 1e-12`` on MZI
lattices, CROW chains, three-level nesting and a Clements mesh rebuilt from an
``mzi`` subcircuit; gradients with respect to subcircuit parameters match
flatten mode and central finite differences. See
``tests/test_circuit_hierarchy.py`` and ``docs/HIERARCHY_AND_INTEROP.md``.
"""
from __future__ import annotations

import ast
import json
import math
import os
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import numpy as np

from photonix.core.backend import xp
from photonix.core.types import Model, SDict

from . import engine as _engine
from .netlist import InstancePort, Netlist, _json_value

__all__ = [
    "Subcircuit",
    "HierarchicalNetlist",
    "compile_hierarchical",
    "evaluate_expression",
    "is_expression",
    "load_netlist",
]

EXPRESSION_PREFIX = "="
_MAX_EXPRESSION_LENGTH = 2000
ParamValue = Any  # float | int | str (expression or literal) | backend array


# --------------------------------------------------------------------------- #
# Safe expression evaluation
# --------------------------------------------------------------------------- #
def _is_concrete(*values: Any) -> bool:
    return all(isinstance(v, (int, float, np.number)) and not isinstance(v, bool) for v in values)


def _unary(np_fn: Callable, xp_name: str) -> Callable:
    def fn(x):
        if _is_concrete(x):
            return float(np_fn(float(x)))
        return getattr(xp, xp_name)(x)

    return fn


def _reduce(name: str) -> Callable:
    pick = min if name == "min" else max
    xp_name = "minimum" if name == "min" else "maximum"

    def fn(*args):
        if not args:
            raise ValueError(f"{name}() needs at least one argument")
        if _is_concrete(*args):
            return pick(args)
        out = args[0]
        for a in args[1:]:
            out = getattr(xp, xp_name)(out, a)
        return out

    return fn


def _abs(x):
    return abs(x)


_FUNCTIONS: dict[str, Callable] = {
    "sqrt": _unary(math.sqrt, "sqrt"),
    "sin": _unary(math.sin, "sin"),
    "cos": _unary(math.cos, "cos"),
    "tan": _unary(math.tan, "tan"),
    "exp": _unary(math.exp, "exp"),
    "log": _unary(math.log, "log"),
    "abs": _abs,
    "min": _reduce("min"),
    "max": _reduce("max"),
}
_CONSTANTS: dict[str, float] = {"pi": math.pi, "e": math.e}
_BINOPS: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.Mod: lambda a, b: a % b,
}


def _pow(a, b):
    # Integer powers such as 10**10**10 would allocate unbounded memory.
    if isinstance(a, int) and not isinstance(a, bool):
        a = float(a)
    if isinstance(b, int) and not isinstance(b, bool):
        b = float(b)
    return a ** b


def is_expression(value: Any) -> bool:
    """True if ``value`` is a parameter expression string (starts with ``"="``).

    Examples
    --------
    >>> is_expression("=2*pi*radius"), is_expression("te"), is_expression(3.0)
    (True, False, False)
    """
    return isinstance(value, str) and value.startswith(EXPRESSION_PREFIX)


def evaluate_expression(expr: str, scope: Mapping[str, Any]) -> Any:
    """Evaluate an arithmetic parameter expression safely (no :func:`eval`).

    Parameters
    ----------
    expr
        The expression, with or without the leading ``"="``.
    scope
        Names visible to the expression (parameter name -> value). Values may
        be Python/NumPy numbers or backend arrays (JAX tracers included).

    Returns
    -------
    float or array
        A Python ``float``/``int`` when every operand is concrete, otherwise a
        backend array (so gradients flow through the expression).

    Raises
    ------
    ValueError
        For syntax errors, unknown names, and anything outside the whitelist:
        attribute access, subscripts, comprehensions, keyword arguments, calls
        to functions other than ``sqrt sin cos tan exp log abs min max``,
        dunder names, string/boolean literals.

    Examples
    --------
    >>> evaluate_expression("=2*pi*radius", {"radius": 10})
    62.83185307179586
    >>> evaluate_expression("max(1, w/2) + 0.5", {"w": 3.0})
    2.0
    >>> evaluate_expression("open('secrets')", {})
    Traceback (most recent call last):
    ...
    ValueError: expression "open('secrets')": name 'open' is not allowed
    """
    if not isinstance(expr, str):
        raise ValueError(f"expression must be a string, got {expr!r}")
    text = expr[len(EXPRESSION_PREFIX):] if expr.startswith(EXPRESSION_PREFIX) else expr
    if len(text) > _MAX_EXPRESSION_LENGTH:
        raise ValueError(f"expression longer than {_MAX_EXPRESSION_LENGTH} characters")
    try:
        tree = ast.parse(text.strip(), mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"expression {text!r}: invalid syntax ({exc.msg})") from None

    def fail(why: str):
        raise ValueError(f"expression {text!r}: {why}")

    def check_name(name: str) -> None:
        if name.startswith("_"):
            fail(f"name {name!r} is not allowed")

    def walk(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant):
            v = node.value
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                fail(f"literal {v!r} is not a number")
            return v
        if isinstance(node, ast.Name):
            check_name(node.id)
            if node.id in scope:
                return scope[node.id]
            if node.id in _CONSTANTS:
                return _CONSTANTS[node.id]
            if node.id in _FUNCTIONS:
                fail(f"function {node.id!r} must be called")
            fail(f"unknown name {node.id!r} (known: {sorted(set(scope) | set(_CONSTANTS))})")
        if isinstance(node, ast.BinOp):
            left, right = walk(node.left), walk(node.right)
            try:
                if isinstance(node.op, ast.Pow):
                    return _pow(left, right)
                op = _BINOPS.get(type(node.op))
                if op is None:
                    fail(f"operator {type(node.op).__name__} is not allowed")
                return cast(Callable, op)(left, right)
            except (ZeroDivisionError, OverflowError) as exc:
                fail(str(exc))
        if isinstance(node, ast.UnaryOp):
            operand = walk(node.operand)
            if isinstance(node.op, ast.USub):
                return -operand
            if isinstance(node.op, ast.UAdd):
                return +operand
            fail(f"operator {type(node.op).__name__} is not allowed")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                fail("only direct calls of whitelisted functions are allowed")
            name = cast(ast.Name, node.func).id
            check_name(name)
            if name not in _FUNCTIONS:
                fail(f"name {name!r} is not allowed")
            if node.keywords or any(isinstance(a, ast.Starred) for a in node.args):
                fail("keyword and star arguments are not allowed")
            try:
                return _FUNCTIONS[name](*[walk(a) for a in node.args])
            except (ValueError, OverflowError, TypeError) as exc:
                fail(f"{name}(): {exc}")
        fail(f"{type(node).__name__} is not allowed")

    return walk(tree)


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #
@dataclass
class Subcircuit:
    """A reusable, parameterised circuit block (a SPICE ``.subckt``).

    Parameters
    ----------
    netlist
        The block's topology. Its external ports (``netlist.ports``) are the
        ports of every instance of the subcircuit. Instance settings may be
        expressions (``"=pi*radius"``) over :attr:`parameters`.
    parameters
        Declared parameters and their defaults. A default may itself be an
        expression over the other parameters. Instances of the subcircuit may
        set declared parameters only.
    doc
        Free-text description.

    Examples
    --------
    >>> nl = Netlist()
    >>> _ = nl.add("wg", "straight", length="=pi*radius")
    >>> _ = nl.expose("o1", ("wg", "o1")).expose("o2", ("wg", "o2"))
    >>> half_ring = Subcircuit(nl, {"radius": 10.0})
    >>> half_ring.port_names
    ['o1', 'o2']
    """

    netlist: Netlist
    parameters: dict[str, ParamValue] = field(default_factory=dict)
    doc: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.netlist, Netlist):
            raise TypeError(f"Subcircuit.netlist must be a Netlist, got {type(self.netlist).__name__}")
        self.parameters = _check_parameters(self.parameters)
        if not isinstance(self.doc, str):
            raise TypeError("Subcircuit.doc must be a string")

    @property
    def port_names(self) -> list[str]:
        """Sorted external port names (the ports of an instance of this block)."""
        return sorted(self.netlist.ports)

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible description (used inside the version-2 netlist format)."""
        body = _netlist_body(self.netlist)
        out: dict[str, Any] = {}
        if self.doc:
            out["doc"] = self.doc
        if self.parameters:
            out["parameters"] = {k: _json_value(v, "<parameters>", k) for k, v in self.parameters.items()}
        out.update(body)
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Subcircuit:
        """Inverse of :meth:`to_dict`."""
        body = {k: v for k, v in data.items() if k not in ("doc", "parameters")}
        return cls(_netlist_from_body(body), dict(data.get("parameters", {})), str(data.get("doc", "")))


def _check_parameters(parameters: Mapping[str, Any]) -> dict[str, Any]:
    params = dict(parameters)
    for k in params:
        if not isinstance(k, str) or not k.isidentifier() or k.startswith("_"):
            raise ValueError(f"parameter names must be identifiers without a leading '_', got {k!r}")
    return params


def _netlist_body(nl: Netlist) -> dict[str, Any]:
    d = nl.to_dict()
    d.pop("format", None)
    d.pop("version", None)
    return d


def _netlist_from_body(body: Mapping[str, Any]) -> Netlist:
    return Netlist.from_dict({**body, "format": "photonix.netlist", "version": 1})


@dataclass
class HierarchicalNetlist:
    """A top-level netlist plus a library of named :class:`Subcircuit` s.

    Parameters
    ----------
    top
        The top-level circuit. Instances whose model name is a key of
        :attr:`subcircuits` are subcircuit instances; all others are leaf
        models resolved against the model registry. Subcircuit names take
        precedence over registry models of the same name.
    subcircuits
        ``name -> Subcircuit``.
    parameters
        Top-level parameters (with defaults) visible to expressions in the top
        netlist and overridable at call time (``model(params={...})``).
    name
        Optional circuit name; defaults to ``top.name``.

    Examples
    --------
    >>> import numpy as np
    >>> arm = Netlist()
    >>> _ = arm.add("wg", "straight", length="=2*pi*radius")
    >>> _ = arm.expose("o1", ("wg", "o1")).expose("o2", ("wg", "o2"))
    >>> top = Netlist(name="chain")
    >>> _ = top.add("a", "arm", radius=5.0).add("b", "arm")
    >>> _ = top.connect(("a", "o2"), ("b", "o1"))
    >>> _ = top.expose("in", ("a", "o1")).expose("out", ("b", "o2"))
    >>> hnl = HierarchicalNetlist(top, {"arm": Subcircuit(arm, {"radius": 10.0})})
    >>> flat = hnl.flatten()
    >>> sorted(flat.instances), flat.settings["b.wg"]["length"]
    (['a.wg', 'b.wg'], 62.83185307179586)
    >>> f = compile_hierarchical(hnl)
    >>> bool(np.isclose(abs(complex(f(wl=1.55)[("in", "out")])), 1.0))
    True
    """

    top: Netlist
    subcircuits: dict[str, Subcircuit] = field(default_factory=dict)
    parameters: dict[str, ParamValue] = field(default_factory=dict)
    name: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.top, Netlist):
            raise TypeError(f"top must be a Netlist, got {type(self.top).__name__}")
        subs: dict[str, Subcircuit] = {}
        for k, v in dict(self.subcircuits).items():
            if not isinstance(k, str) or not k:
                raise TypeError(f"subcircuit names must be non-empty strings, got {k!r}")
            if not isinstance(v, Subcircuit):
                raise TypeError(f"subcircuit {k!r} must be a Subcircuit, got {type(v).__name__}")
            subs[k] = v
        self.subcircuits = subs
        self.parameters = _check_parameters(self.parameters)
        if self.name is None:
            self.name = self.top.name

    # -- structure ----------------------------------------------------------- #
    def _netlists(self) -> list[tuple[str | None, Netlist]]:
        return [(None, self.top), *((k, s.netlist) for k, s in self.subcircuits.items())]

    def validate(self, models: Mapping[str, Any] | None = None) -> None:
        """Raise ``ValueError`` if the hierarchy is inconsistent.

        Checks every netlist (:meth:`Netlist.validate`); that the subcircuit
        graph is acyclic (a coloured depth-first search reporting the cycle
        path); that every port used on a subcircuit instance is one of the
        subcircuit's external ports; that subcircuit instances only set
        declared parameters; and, when ``models`` is given, that every
        non-subcircuit model name is in it.

        Examples
        --------
        >>> inner = Netlist()
        >>> _ = inner.add("x", "B")
        >>> _ = inner.expose("o1", ("x", "o1"))
        >>> loop = Netlist()
        >>> _ = loop.add("y", "A")
        >>> _ = loop.expose("o1", ("y", "o1"))
        >>> top = Netlist()
        >>> _ = top.add("t", "A")
        >>> _ = top.expose("o1", ("t", "o1"))
        >>> HierarchicalNetlist(top, {"A": Subcircuit(inner), "B": Subcircuit(loop)}).validate()
        Traceback (most recent call last):
        ...
        ValueError: subcircuit cycle: A -> B -> A
        """
        for owner, nl in self._netlists():
            where = "top netlist" if owner is None else f"subcircuit {owner!r}"
            try:
                nl.validate()
            except ValueError as exc:
                raise ValueError(f"{where}: {exc}") from None
            for inst, model_name in nl.instances.items():
                sub = self.subcircuits.get(model_name)
                if sub is None:
                    if models is not None and model_name not in models:
                        raise ValueError(
                            f"{where}: instance {inst!r} uses unknown model {model_name!r} "
                            f"(not a subcircuit and not in the model registry)"
                        )
                    continue
                unknown = sorted(set(nl.settings.get(inst, {})) - set(sub.parameters))
                if unknown:
                    raise ValueError(
                        f"{where}: instance {inst!r} of subcircuit {model_name!r} sets undeclared "
                        f"parameter(s) {unknown}; declared: {sorted(sub.parameters)}"
                    )
            terminals = [*nl.connections.keys(), *nl.connections.values(), *nl.ports.values()]
            for inst, port in terminals:
                sub = self.subcircuits.get(nl.instances[inst])
                if sub is not None and port not in sub.netlist.ports:
                    raise ValueError(
                        f"{where}: port {port!r} of instance {inst!r} is not an exposed port of subcircuit "
                        f"{nl.instances[inst]!r}; available ports: {sub.port_names}"
                    )
        # Coloured DFS for cycles.
        white, grey, black = 0, 1, 2
        colour = dict.fromkeys(self.subcircuits, white)

        def children(name: str) -> list[str]:
            insts = self.subcircuits[name].netlist.instances.values()
            return [m for m in dict.fromkeys(insts) if m in self.subcircuits]

        def visit(name: str, path: list[str]) -> None:
            colour[name] = grey
            for child in children(name):
                if colour[child] == grey:
                    cycle = path[path.index(child):] + [child]
                    raise ValueError("subcircuit cycle: " + " -> ".join(cycle))
                if colour[child] == white:
                    visit(child, [*path, child])
            colour[name] = black

        for name in self.subcircuits:
            if colour[name] == white:
                visit(name, [name])

    def used_subcircuits(self) -> list[str]:
        """Names of the subcircuits reachable from the top netlist."""
        seen: dict[str, None] = {}
        stack = [m for m in self.top.instances.values() if m in self.subcircuits]
        while stack:
            name = stack.pop(0)
            if name in seen:
                continue
            seen[name] = None
            stack.extend(m for m in self.subcircuits[name].netlist.instances.values() if m in self.subcircuits)
        return list(seen)

    def instance_paths(self) -> dict[str, str]:
        """Every instance path in the expanded hierarchy -> its model or subcircuit name.

        Examples
        --------
        >>> inner = Netlist()
        >>> _ = inner.add("wg", "straight")
        >>> _ = inner.expose("o1", ("wg", "o1"))
        >>> top = Netlist()
        >>> _ = top.add("a", "blk")
        >>> _ = top.expose("o1", ("a", "o1"))
        >>> HierarchicalNetlist(top, {"blk": Subcircuit(inner)}).instance_paths()
        {'a': 'blk', 'a.wg': 'straight'}
        """
        self.validate()
        out: dict[str, str] = {}

        def walk(nl: Netlist, prefix: str) -> None:
            for inst, model_name in nl.instances.items():
                path = f"{prefix}{inst}"
                out[path] = model_name
                if model_name in self.subcircuits:
                    walk(self.subcircuits[model_name].netlist, f"{path}.")

        walk(self.top, "")
        return out

    # -- flattening ---------------------------------------------------------- #
    def _flat_structure(self) -> tuple[dict[str, str], dict[InstancePort, InstancePort], dict[str, InstancePort]]:
        """Leaf instances, connections and ports of the flattened circuit (no settings)."""
        instances: dict[str, str] = {}
        connections: dict[InstancePort, InstancePort] = {}

        def leaf(prefix: str, nl: Netlist, terminal: InstancePort) -> InstancePort:
            inst, port = terminal
            path = f"{prefix}{inst}"
            sub = self.subcircuits.get(nl.instances[inst])
            if sub is None:
                return (path, port)
            return leaf(f"{path}.", sub.netlist, sub.netlist.ports[port])

        def walk(nl: Netlist, prefix: str) -> None:
            for inst, model_name in nl.instances.items():
                sub = self.subcircuits.get(model_name)
                if sub is None:
                    instances[f"{prefix}{inst}"] = model_name
                else:
                    walk(sub.netlist, f"{prefix}{inst}.")
            for a, b in nl.connections.items():
                connections[leaf(prefix, nl, a)] = leaf(prefix, nl, b)

        walk(self.top, "")
        ports = {ext: leaf("", self.top, ip) for ext, ip in self.top.ports.items()}
        return instances, connections, ports

    def flatten(self, params: Mapping[str, Any] | None = None) -> Netlist:
        """The equivalent flat :class:`~photonix.circuit.Netlist`.

        Leaf instances are named by their dotted path (``"mesh.mzi3.int"``);
        settings are fully resolved (expressions evaluated with the default
        parameters, updated by ``params`` for the top level). External port
        names are those of the top netlist.
        """
        self.validate()
        instances, connections, ports = self._flat_structure()
        leaf_settings = _resolve_leaf_settings(self, params or {}, {})
        settings = {k: v for k, v in leaf_settings.items() if v}
        return Netlist(instances=instances, connections=connections, ports=ports, settings=settings,
                       name=self.name)

    # -- serialization ------------------------------------------------------- #
    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible version-2 netlist description.

        Examples
        --------
        >>> inner = Netlist()
        >>> _ = inner.add("wg", "straight", length="=2*radius")
        >>> _ = inner.expose("o1", ("wg", "o1")).expose("o2", ("wg", "o2"))
        >>> top = Netlist(name="t")
        >>> _ = top.add("a", "blk", radius=3.0)
        >>> _ = top.expose("o1", ("a", "o1")).expose("o2", ("a", "o2"))
        >>> hnl = HierarchicalNetlist(top, {"blk": Subcircuit(inner, {"radius": 1.0})})
        >>> d = hnl.to_dict()
        >>> d["version"], sorted(d["subcircuits"])
        (2, ['blk'])
        >>> HierarchicalNetlist.from_json(hnl.to_json()) == hnl
        True
        """
        self.validate()
        out: dict[str, Any] = {"format": "photonix.netlist", "version": 2}
        body = _netlist_body(self.top)
        if self.name != self.top.name:
            body["name"] = self.name
        out.update(body)
        if self.top.name != self.name:
            out["top_name"] = self.top.name
        if self.parameters:
            out["parameters"] = {k: _json_value(v, "<parameters>", k) for k, v in self.parameters.items()}
        out["subcircuits"] = {k: s.to_dict() for k, s in self.subcircuits.items()}
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> HierarchicalNetlist:
        """Inverse of :meth:`to_dict`; a version-1 netlist loads with no subcircuits."""
        if data.get("format", "photonix.netlist") != "photonix.netlist":
            raise ValueError(f"not a photonix netlist: format={data.get('format')!r}")
        version = int(data.get("version", 1))
        if version == 1:
            return cls(Netlist.from_dict(data))
        if version != 2:
            raise ValueError(f"unsupported netlist version {data.get('version')!r}")
        body = {k: v for k, v in data.items()
                if k not in ("format", "version", "subcircuits", "parameters", "top_name")}
        name = body.get("name")
        if "top_name" in data:
            body["name"] = data["top_name"]
        top = _netlist_from_body(body)
        subs = data.get("subcircuits", {})
        if not isinstance(subs, Mapping):
            raise ValueError("'subcircuits' must be a mapping of name -> subcircuit")
        hnl = cls(top, {k: Subcircuit.from_dict(v) for k, v in subs.items()},
                  dict(data.get("parameters", {})), name)
        hnl.validate()
        return hnl

    def to_json(self, path: str | os.PathLike[str] | None = None, *, indent: int = 2) -> str:
        """Serialize to JSON text; also write it to ``path`` if given."""
        text = json.dumps(self.to_dict(), indent=indent, allow_nan=False)
        if path is not None:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text + "\n")
        return text

    @classmethod
    def from_json(cls, source: str | os.PathLike[str]) -> HierarchicalNetlist:
        """Load from JSON text or from a path (version 1 or 2)."""
        return cls.from_dict(_read_json(source))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        nm = f" {self.name!r}" if self.name else ""
        return (f"<HierarchicalNetlist{nm}: {len(self.top.instances)} top instances, "
                f"{len(self.subcircuits)} subcircuits>")


def _read_json(source: str | os.PathLike[str]) -> dict[str, Any]:
    text = os.fspath(source)
    if not text.lstrip().startswith("{"):
        with open(text, encoding="utf-8") as fh:
            text = fh.read()
    return json.loads(text)


def load_netlist(source: str | os.PathLike[str] | Mapping[str, Any]) -> Netlist | HierarchicalNetlist:
    """Load a photonix netlist of either version.

    Returns a :class:`~photonix.circuit.Netlist` for a version-1 document and
    a :class:`HierarchicalNetlist` for version 2.

    Examples
    --------
    >>> nl = Netlist(name="one")
    >>> _ = nl.add("a", "straight")
    >>> _ = nl.expose("o1", ("a", "o1")).expose("o2", ("a", "o2"))
    >>> type(load_netlist(nl.to_json())).__name__
    'Netlist'
    """
    data = dict(source) if isinstance(source, Mapping) else _read_json(source)
    if int(data.get("version", 1)) == 1:
        return Netlist.from_dict(data)
    return HierarchicalNetlist.from_dict(data)


# --------------------------------------------------------------------------- #
# Parameter resolution
# --------------------------------------------------------------------------- #
def _resolve_value(value: Any, scope: Mapping[str, Any]) -> Any:
    return evaluate_expression(value, scope) if is_expression(value) else value


def _bind_scope(declared: Mapping[str, Any], bound: Mapping[str, Any], where: str) -> dict[str, Any]:
    """Bound values over declared defaults; defaults may be expressions over the scope."""
    unknown = sorted(set(bound) - set(declared))
    if unknown:
        raise ValueError(f"{where}: undeclared parameter(s) {unknown}; declared: {sorted(declared)}")
    raw = {**declared, **bound}
    done: dict[str, Any] = {k: v for k, v in raw.items() if not is_expression(v)}
    pending = {k: v for k, v in raw.items() if k not in done}
    resolving: list[str] = []

    class _Scope(dict):
        def __contains__(self, key: object) -> bool:
            return key in done or key in pending

        def __getitem__(self, key: str) -> Any:
            if key in done:
                return done[key]
            if key in resolving:
                raise ValueError(f"{where}: parameter defaults form a cycle: {' -> '.join([*resolving, key])}")
            resolving.append(key)
            done[key] = evaluate_expression(pending[key], self)
            resolving.pop()
            return done[key]

    lazy = _Scope()
    for k in list(pending):
        if k not in done:
            lazy[k]  # noqa: B018 - resolves and stores
    return {k: done[k] for k in raw}


def _split_overrides(
    hnl: HierarchicalNetlist, overrides: Mapping[str, Any], paths: Mapping[str, str]
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for path, values in overrides.items():
        if path not in paths:
            raise KeyError(f"override targets unknown instance path {path!r}; known paths include "
                           f"{sorted(paths)[:12]}")
        if not isinstance(values, Mapping):
            raise TypeError(f"override for {path!r} must be a mapping of settings, got {values!r}")
        model_name = paths[path]
        sub = hnl.subcircuits.get(model_name)
        if sub is not None:
            unknown = sorted(set(values) - set(sub.parameters))
            if unknown:
                raise ValueError(f"override for {path!r} sets undeclared parameter(s) {unknown} of subcircuit "
                                 f"{model_name!r}; declared: {sorted(sub.parameters)}")
        out[path] = dict(values)
    return out


def _instance_settings(nl: Netlist, inst: str, path: str, overrides: Mapping[str, Mapping[str, Any]],
                       scope: Mapping[str, Any]) -> dict[str, Any]:
    extra = overrides.get(path)
    raw = nl.merged_settings(inst, dict(extra) if extra is not None else None)
    try:
        return {k: _resolve_value(v, scope) for k, v in raw.items()}
    except ValueError as exc:
        raise ValueError(f"instance {path!r}: {exc}") from None


def _resolve_leaf_settings(hnl: HierarchicalNetlist, params: Mapping[str, Any],
                           overrides: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Fully resolved settings of every leaf instance path."""
    out: dict[str, dict[str, Any]] = {}

    def walk(nl: Netlist, prefix: str, scope: Mapping[str, Any]) -> None:
        for inst, model_name in nl.instances.items():
            path = f"{prefix}{inst}"
            settings = _instance_settings(nl, inst, path, overrides, scope)
            sub = hnl.subcircuits.get(model_name)
            if sub is None:
                out[path] = settings
            else:
                walk(sub.netlist, f"{path}.", _bind_scope(sub.parameters, settings, f"instance {path!r}"))

    walk(hnl.top, "", _bind_scope(hnl.parameters, params, "top-level params"))
    return out


# --------------------------------------------------------------------------- #
# Compilation
# --------------------------------------------------------------------------- #
def compile_hierarchical(
    hnl: HierarchicalNetlist | Netlist,
    models: Mapping[str, Callable[..., SDict]] | None = None,
    *,
    mode: Literal["hierarchical", "flatten"] = "hierarchical",
    backend: _engine.Backend | str = "auto",
    cache: bool = True,
) -> Model:
    """Compile a hierarchical netlist into a differentiable model.

    Parameters
    ----------
    hnl
        The hierarchy (a plain :class:`~photonix.circuit.Netlist` is accepted
        as a hierarchy without subcircuits).
    models
        Leaf ``model_name -> callable`` registry; defaults to
        :data:`photonix.components.MODELS`.
    mode
        ``"hierarchical"`` (recursive block elimination; default) or
        ``"flatten"`` (one flat solve). Both give the same S-matrix up to
        round-off.
    backend
        Circuit-engine backend for every solve (see
        :mod:`photonix.circuit.engine`).
    cache
        Evaluate identical leaf models and identical subcircuit instances once
        per call (module docs). Requires pure models; pass ``False`` for a
        model that draws fresh random numbers on every call.

    Returns
    -------
    Model
        ``f(*, wl=1.55, params=None, **overrides) -> SDict``. ``params`` sets
        top-level parameters; each other keyword is an instance path
        (``"a"`` or ``"mesh.mzi3.int"``) mapping to settings. For a leaf path
        they override model settings; for a subcircuit path they override the
        subcircuit's parameter bindings. Because paths contain dots, pass
        nested ones with ``**{"mesh.mzi3.int": {...}}``. An instance named
        ``params`` cannot be overridden.

    Examples
    --------
    >>> import numpy as np
    >>> import photonix as px
    >>> cell = Netlist()
    >>> _ = cell.add("ps", "phase_shifter", length=0.0, phase0="=phase")
    >>> _ = cell.expose("o1", ("ps", "o1")).expose("o2", ("ps", "o2"))
    >>> top = Netlist()
    >>> _ = top.add("a", "shift").add("b", "shift", phase="=2*base")
    >>> _ = top.connect(("a", "o2"), ("b", "o1"))
    >>> _ = top.expose("in", ("a", "o1")).expose("out", ("b", "o2"))
    >>> hnl = HierarchicalNetlist(top, {"shift": Subcircuit(cell, {"phase": 0.0})}, parameters={"base": 0.1})
    >>> f = compile_hierarchical(hnl)
    >>> s = f(wl=1.55, params={"base": 0.25}, a={"phase": 0.5})
    >>> bool(np.isclose(complex(s[("in", "out")]), np.exp(-1j * (0.5 + 0.5))))
    True
    """
    if isinstance(hnl, Netlist):
        hnl = HierarchicalNetlist(hnl)
    if mode not in ("hierarchical", "flatten"):
        raise ValueError(f"mode must be 'hierarchical' or 'flatten', got {mode!r}")
    if backend not in _engine.BACKENDS:
        raise ValueError(f"unknown circuit backend {backend!r}; choose one of {_engine.BACKENDS}")
    if models is None:
        from photonix.components import MODELS

        registry = cast(Mapping[str, Callable[..., SDict]], MODELS)
    else:
        registry = models
    hnl.validate(registry)
    for sub_name in hnl.used_subcircuits():
        if not hnl.subcircuits[sub_name].netlist.ports:
            raise ValueError(f"subcircuit {sub_name!r} exposes no ports")
    caches: dict[str | None, _engine._TopologyCache] = {}
    paths = hnl.instance_paths()
    flat = hnl._flat_structure() if mode == "flatten" else None

    def leaf_eval(memo: dict[Hashable, SDict], model_name: str, settings: Mapping[str, Any], wl: Any) -> SDict:
        fn = registry[model_name]
        if not cache:
            return fn(wl=wl, **settings)
        key = ("leaf", id(fn), tuple(sorted((k, _engine._freeze(v)) for k, v in settings.items())))
        sd = memo.get(key)
        if sd is None:
            sd = memo[key] = fn(wl=wl, **settings)
        return sd

    def solve_level(owner: str | None, nl: Netlist, inst_sdicts: dict[str, SDict]) -> SDict:
        tc = caches.setdefault(owner, _engine._TopologyCache(maxsize=4))
        topo = tc.get(inst_sdicts, nl.connections, nl.ports)
        if topo.m == 0:
            return {}
        return _engine.solve(topo, _engine.gather_values(topo, inst_sdicts), backend=backend)

    def model(*, wl=1.55, params: Mapping[str, Any] | None = None, **overrides) -> SDict:
        ov = _split_overrides(hnl, overrides, paths) if overrides else {}
        memo: dict[Hashable, SDict] = {}
        if flat is not None:
            instances, connections, ports = flat
            leaf_settings = _resolve_leaf_settings(hnl, params or {}, ov)
            inst_sdicts = {path: leaf_eval(memo, instances[path], leaf_settings[path], wl) for path in instances}
            tc = caches.setdefault(None, _engine._TopologyCache(maxsize=4))
            topo = tc.get(inst_sdicts, connections, ports)
            if topo.m == 0:
                return {}
            return _engine.solve(topo, _engine.gather_values(topo, inst_sdicts), backend=backend)

        def evaluate(owner: str | None, nl: Netlist, prefix: str, scope: Mapping[str, Any]) -> SDict:
            inst_sdicts: dict[str, SDict] = {}
            for inst, model_name in nl.instances.items():
                path = f"{prefix}{inst}"
                settings = _instance_settings(nl, inst, path, ov, scope)
                sub = hnl.subcircuits.get(model_name)
                if sub is None:
                    inst_sdicts[inst] = leaf_eval(memo, model_name, settings, wl)
                    continue
                child_scope = _bind_scope(sub.parameters, settings, f"instance {path!r}")
                inner = f"{path}."
                if cache:
                    rel = tuple(sorted(
                        (p[len(inner):], tuple(sorted((k, _engine._freeze(v)) for k, v in vals.items())))
                        for p, vals in ov.items() if p.startswith(inner)
                    ))
                    key = ("sub", model_name, tuple(sorted((k, _engine._freeze(v)) for k, v in child_scope.items())),
                           rel)
                    sd = memo.get(key)
                    if sd is None:
                        sd = memo[key] = evaluate(model_name, sub.netlist, inner, child_scope)
                else:
                    sd = evaluate(model_name, sub.netlist, inner, child_scope)
                inst_sdicts[inst] = sd
            return solve_level(owner, nl, inst_sdicts)

        return evaluate(None, hnl.top, "", _bind_scope(hnl.parameters, params or {}, "top-level params"))

    model.__name__ = hnl.name or "hierarchical_circuit"
    return model
