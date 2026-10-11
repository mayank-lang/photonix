"""Plan and write the API reference pages from every module's ``__all__``.

The plan documents every public object exactly once, on the page of the
photonix module that defines it (its *home*), and lists every exported name on
the page of each module that exports it, linking to that home. That avoids
duplicate object descriptions for names re-exported by several packages and
makes coverage checkable: :func:`plan` returns, for every ``(module, name)``
pair in a public ``__all__``, the fully qualified name under which it is
documented, and ``tests/test_docs_coverage.py`` checks every one of them
against the built site's ``objects.inv``.

Pages are the ``photonix`` package, every photonix module reachable from it
through module-valued ``__all__`` entries, and every public module that is the
home of an exported object. Objects defined in private modules (a path
component starting with ``_``) or outside photonix (``xp`` is ``jax.numpy`` or
``numpy``) are documented on the most specific public page that exports them.

This module needs only the standard library and photonix (no Sphinx).
"""
from __future__ import annotations

import ast
import importlib
import inspect
import pkgutil
import types
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["Entry", "Page", "plan", "write_pages", "render_page"]

ROOT_PACKAGE = "photonix"


@dataclass
class Entry:
    """One exported name on one page."""

    name: str
    kind: str          # "module", "class", "function", "data", "module-alias"
    target: str        # fully qualified name it is documented under


@dataclass
class Page:
    """One API page and what it documents and lists."""

    module: str
    exports: list[Entry] = field(default_factory=list)       # every name in __all__ (or exported from here)
    documented: list[Entry] = field(default_factory=list)    # objects whose home is this page
    submodules: list[str] = field(default_factory=list)      # child pages


def _private(modname: str) -> bool:
    return any(part.startswith("_") for part in modname.split("."))


def _import(name: str) -> types.ModuleType | None:
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


def _all_modules() -> dict[str, types.ModuleType]:
    root = importlib.import_module(ROOT_PACKAGE)
    mods = {ROOT_PACKAGE: root}
    for info in pkgutil.walk_packages(root.__path__, prefix=f"{ROOT_PACKAGE}."):
        if info.name.endswith("__main__"):
            continue
        mod = _import(info.name)
        if mod is not None:
            mods[info.name] = mod
    return mods


def _assigned_names(mod: types.ModuleType) -> set[str]:
    """Top-level names a module assigns (not imports)."""
    try:
        source = inspect.getsource(mod)
    except (OSError, TypeError):
        return set()
    names: set[str] = set()
    for node in ast.parse(source).body:
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets = [node.target]
        for t in targets:
            for sub in ast.walk(t):
                if isinstance(sub, ast.Name):
                    names.add(sub.id)
    return names


def _exports(mod: types.ModuleType) -> list[str]:
    return [str(n) for n in getattr(mod, "__all__", [])]


def _value(mod: types.ModuleType, name: str):
    try:
        return getattr(mod, name)
    except AttributeError:
        sub = _import(f"{mod.__name__}.{name}")
        if sub is None:
            raise
        return sub


def _kind(value) -> str:
    if isinstance(value, types.ModuleType):
        return "module"
    if inspect.isclass(value):
        return "class"
    if callable(value) and (inspect.isroutine(value) or hasattr(value, "__wrapped__")
                            or type(value).__name__ in ("PjitFunction", "custom_vjp", "custom_jvp")):
        return "function"
    return "data"


def plan() -> dict[str, Page]:
    """The API pages, keyed by module name, in a stable order."""
    mods = _all_modules()
    assigned = {name: _assigned_names(mod) for name, mod in mods.items()}

    # 1. Page modules reachable through module-valued __all__ entries.
    pages: dict[str, Page] = {}
    queue = [ROOT_PACKAGE]
    while queue:
        name = queue.pop(0)
        if name in pages:
            continue
        pages[name] = Page(name)
        for export in _exports(mods[name]):
            value = _value(mods[name], export)
            if isinstance(value, types.ModuleType) and value.__name__ == f"{name}.{export}":
                pages[name].submodules.append(value.__name__)
                queue.append(value.__name__)

    # 2. Homes of exported objects.
    def defining_module(value, export: str) -> tuple[str, str] | None:
        """``(module, name)`` an exported object is defined under, or None.

        The name differs from ``export`` for an alias such as
        ``from photonix.em.fde import ModeData as ModeResult``.
        """
        if inspect.isclass(value) or inspect.isroutine(value) or hasattr(value, "__wrapped__"):
            modname = getattr(value, "__module__", None)
            if isinstance(modname, str) and modname in mods and getattr(mods[modname], export, None) is value:
                return modname, export
            if isinstance(modname, str) and modname in mods:
                qual = getattr(value, "__qualname__", "")
                if qual and getattr(mods[modname], qual, None) is value:
                    return modname, qual
            return None
        for modname in sorted(mods, key=len, reverse=True):
            if export in assigned[modname] and getattr(mods[modname], export, object()) is value:
                return modname, export
        return None

    exporters: dict[int, list[str]] = {}
    homes: dict[tuple[str, str], tuple[str, str, object]] = {}
    work = list(pages)
    seen_pages = set()
    while work:
        page_name = work.pop(0)
        if page_name in seen_pages:
            continue
        seen_pages.add(page_name)
        mod = mods[page_name]
        for export in _exports(mod):
            value = _value(mod, export)
            if isinstance(value, types.ModuleType):
                continue
            exporters.setdefault(id(value), []).append(page_name)
            found = defining_module(value, export)
            home, home_name = found if found is not None else (None, export)
            if home is not None and not _private(home) and (home != page_name or home_name != export):
                homes[(page_name, export)] = (home, home_name, value)
                if home not in pages:
                    pages[home] = Page(home)
                    parent = home.rsplit(".", 1)[0]
                    owner = parent if parent in pages else page_name
                    if home not in pages[owner].submodules:
                        pages[owner].submodules.append(home)
                    work.append(home)
            else:
                homes[(page_name, export)] = (page_name, export, value)

    # 3. Objects with no public photonix home live on the most specific page exporting them.
    final: dict[tuple[str, str], str] = {}
    for (page_name, export), (home, home_name, value) in homes.items():
        if home == page_name and home_name == export:
            candidates = [p for p in exporters.get(id(value), [page_name]) if (p, export) in homes]
            # Prefer a page where it has no other home: the longest (most specific) module name.
            local = [p for p in candidates if homes[(p, export)][0] == p]
            home = max(local, key=lambda p: (len(p), p)) if local else page_name
        final[(page_name, export)] = f"{home}.{home_name}"

    # 4. Fill the pages.
    documented: set[str] = set()
    for page_name, page in pages.items():
        mod = mods[page_name]
        names = _exports(mod)
        # A home page also documents the names others export from it, even if its own __all__ omits them.
        prefix = f"{page_name}."
        for target in final.values():
            local_name = target[len(prefix):] if target.startswith(prefix) else ""
            if local_name and "." not in local_name and local_name not in names:
                names.append(local_name)
        for export in names:
            try:
                value = _value(mod, export)
            except AttributeError:
                continue
            if isinstance(value, types.ModuleType):
                own = value.__name__ == f"{page_name}.{export}"
                target = value.__name__ if own or value.__name__ in pages else f"{page_name}.{export}"
                kind = "module" if own or value.__name__ in pages else "module-alias"
                entry = Entry(export, kind, target)
                page.exports.append(entry)
                if kind == "module-alias":
                    page.documented.append(entry)
                continue
            target = final.get((page_name, export), f"{page_name}.{export}")
            entry = Entry(export, _kind(value), target)
            if export in _exports(mod):
                page.exports.append(entry)
            if target == f"{page_name}.{export}" and target not in documented:
                documented.add(target)
                page.documented.append(entry)
    return pages


def _relative(target: str, module: str) -> str:
    """``target`` relative to the page's module (autosummary resolves it there).

    Sphinx warns when an autosummary entry repeats the current module's name.
    """
    prefix = f"{module}."
    return target[len(prefix):] if target.startswith(prefix) else target


_DIRECTIVE = {"class": "autoclass", "function": "autofunction", "data": "autodata"}


def render_page(page: Page, pages: dict[str, Page]) -> str:
    """reStructuredText of one API page."""
    title = page.module
    lines = [title, "=" * len(title), "", f".. automodule:: {page.module}", "   :no-members:", ""]
    listed = [e for e in page.exports if e.kind not in ("module", "module-alias")]
    if listed:
        lines += [".. rubric:: Public names", "", ".. autosummary::", "   :nosignatures:", ""]
        lines += [f"   ~{_relative(e.target, page.module)}" for e in listed]
        lines.append("")
    alias = [e for e in page.documented if e.kind == "module-alias"]
    for e in alias:
        value = _value(importlib.import_module(page.module), e.name)
        lines += [f".. py:data:: {e.name}", f"   :module: {page.module}", "",
                  f"   The module ``{value.__name__}``, re-exported as ``{page.module}.{e.name}``"
                  + (" (the active array backend: ``jax.numpy`` under JAX, ``numpy`` otherwise)."
                     if e.name == "xp" else "."), ""]
    objects = [e for e in page.documented if e.kind in _DIRECTIVE]
    if objects:
        lines += [".. rubric:: Reference", ""]
        for e in objects:
            lines.append(f".. {_DIRECTIVE[e.kind]}:: {e.target}")
            if e.kind == "class":
                lines += ["   :members:", "   :show-inheritance:"]
            lines.append("")
    if page.submodules:
        lines += [".. rubric:: Modules", "", ".. autosummary::", "   :nosignatures:", ""]
        lines += [f"   ~{_relative(m, page.module)}" for m in page.submodules]
        lines += ["", ".. toctree::", "   :hidden:", ""]
        lines += [f"   {m}" for m in page.submodules]
        lines.append("")
    return "\n".join(lines)


def write_pages(outdir: str | Path) -> dict[str, Page]:
    """Write one ``<module>.rst`` per page into ``outdir``; returns the plan."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    pages = plan()
    wanted = {f"{name}.rst" for name in pages}
    for stale in out.glob("*.rst"):
        if stale.name not in wanted:
            stale.unlink()
    for name, page in pages.items():
        text = render_page(page, pages)
        path = out / f"{name}.rst"
        if not path.exists() or path.read_text(encoding="utf-8") != text:
            path.write_text(text, encoding="utf-8")
    return pages
