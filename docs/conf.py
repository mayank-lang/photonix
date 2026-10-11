"""Sphinx configuration for the photonix documentation site.

Build (warnings are errors, as in CI)::

    pip install -e ".[docs]"
    python -m sphinx -W --keep-going -b html docs docs/_build/html

The API reference under ``api/generated`` is written when the build starts by
``_ext/photonix_apigen.py`` from every module's ``__all__``, so a public name
cannot be left out by hand (``tests/test_docs_coverage.py`` checks the built
inventory). The narrative pages are the Markdown files in this directory,
parsed with MyST. Links in them that point outside ``docs/``
(``../examples/x.py``, or ``RELEASING.md`` from the included changelog) are
rewritten to the GitHub repository during the build, so the same files render
correctly both on GitHub and on the site.

Set ``PHOTONIX_DOCS_OFFLINE=1`` to skip the intersphinx inventories (which need
network access).
"""
from __future__ import annotations

import importlib.util
import os
import re
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parent
ROOT = DOCS.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(DOCS / "_ext"))

import photonix_apigen  # noqa: E402

import photonix  # noqa: E402

REPOSITORY = "https://github.com/mayank-lang/photonix"

project = "photonix"
author = "photonix contributors"
copyright = "photonix contributors, MIT License"
version = release = photonix.__version__

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]
if os.environ.get("PHOTONIX_DOCS_OFFLINE") != "1":
    extensions.append("sphinx.ext.intersphinx")
if importlib.util.find_spec("sphinx_copybutton") is not None:
    extensions.append("sphinx_copybutton")

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
root_doc = "index"
exclude_patterns = ["_build", "_ext", "Thumbs.db", ".DS_Store"]

# -- MyST ------------------------------------------------------------------- #
myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence", "deflist"]
myst_heading_anchors = 3

# -- API reference ---------------------------------------------------------- #
# The pages are written by photonix_apigen; autosummary only renders their
# summary tables, so it must not generate stubs of its own.
autosummary_generate = False
autodoc_member_order = "bysource"
autodoc_typehints = "signature"
autodoc_preserve_defaults = True
autodoc_class_signature = "mixed"

# NumPy-style docstrings. Besides the standard sections, photonix docstrings use
# descriptive headings ("Ports", "Physics", "Sign convention", ...). They are
# collected from the sources and rendered as generic sections, never dropped.
_SECTION = re.compile(r"^[ \t]*([A-Z][A-Za-z0-9 ,\-/()':.]*?)[ \t]*\n[ \t]*(-{3,})[ \t]*$", re.MULTILINE)
_STANDARD = {"parameters", "returns", "examples", "notes", "raises", "see also", "references", "attributes",
             "yields", "warns", "warnings", "other parameters", "methods", "example", "note", "warning", "todo"}


def _custom_sections() -> list[str]:
    found: set[str] = set()
    for path in (ROOT / "src" / "photonix").rglob("*.py"):
        for match in _SECTION.finditer(path.read_text(encoding="utf-8")):
            name = match.group(1).strip()
            if len(match.group(2)) >= len(name) - 1 and name.lower() not in _STANDARD:
                found.add(name)
    return sorted(found)


napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False
napoleon_custom_sections = _custom_sections()

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "jax": ("https://docs.jax.dev/en/latest", None),
}
intersphinx_timeout = 20

# -- HTML ------------------------------------------------------------------- #
if importlib.util.find_spec("pydata_sphinx_theme") is not None:
    html_theme = "pydata_sphinx_theme"
    html_theme_options = {
        "github_url": REPOSITORY,
        "navigation_with_keys": False,
        "show_toc_level": 2,
    }
else:  # the docs extra installs the theme; fall back to Sphinx's default
    html_theme = "alabaster"
html_title = f"photonix {release}"
html_static_path: list[str] = []
html_show_sourcelink = False

linkcheck_ignore = [r"https://github\.com/mayank-lang/photonix/(blob|tree|raw|compare|releases)/.*"]
linkcheck_timeout = 20

# --------------------------------------------------------------------------- #
# Markdown links that leave docs/ point at the repository
# --------------------------------------------------------------------------- #
_LINK = re.compile(r"(!?)\[([^\]\n]*)\]\(([^)\s]+)\)")
_SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")


def rewrite_links(text: str, source: Path, document_dir: Path) -> str:
    """Point relative links that leave ``docs/`` at the GitHub repository.

    ``source`` is the Markdown file the text came from (its links are relative
    to it) and ``document_dir`` the directory of the document it is rendered in
    (they differ for an included file). Links into ``docs/`` stay relative, so
    Sphinx resolves and checks them.
    """

    def replace(match: re.Match[str]) -> str:
        bang, label, target = match.groups()
        if _SCHEME.match(target) or target.startswith(("#", "/")):
            return match.group(0)
        path, _, fragment = target.partition("#")
        anchor = f"#{fragment}" if fragment else ""
        resolved = (source.parent / path).resolve()
        if resolved.is_relative_to(DOCS):
            if source.parent == document_dir:
                return match.group(0)
            rel = os.path.relpath(resolved, document_dir).replace(os.sep, "/")
            return f"{bang}[{label}]({rel}{anchor})"
        rel = resolved.relative_to(ROOT).as_posix() if resolved.is_relative_to(ROOT) else path
        kind = "raw/main" if bang else ("tree/main" if resolved.is_dir() else "blob/main")
        return f"{bang}[{label}]({REPOSITORY}/{kind}/{rel}{anchor})"

    return _LINK.sub(replace, text)


def _source_read(app, docname: str, source: list[str]) -> None:
    path = Path(app.env.doc2path(docname))
    if path.suffix == ".md":
        source[0] = rewrite_links(source[0], path, path.parent)


def _include_read(app, relative_path, parent_docname: str, content: list[str]) -> None:
    included = (Path(app.srcdir) / relative_path).resolve()
    if included.suffix == ".md":
        parent = Path(app.env.doc2path(parent_docname)).parent
        content[0] = rewrite_links(content[0], included, parent)


# --------------------------------------------------------------------------- #
# API pages and docstring hygiene
# --------------------------------------------------------------------------- #
# Fully qualified name each documented photonix object is described under,
# keyed by id(object). Filled when the pages are written.
_DOCUMENTED: dict[int, str] = {}
_PRIMITIVE = (int, float, complex, str, bytes, bool, type(None))


def _write_api_pages(app) -> None:
    pages = photonix_apigen.write_pages(DOCS / "api" / "generated")
    _DOCUMENTED.clear()
    for page in pages.values():
        for entry in page.documented:
            obj = _resolve_name(entry.target)
            if obj is not None and not isinstance(obj, _PRIMITIVE):
                _DOCUMENTED.setdefault(id(obj), entry.target)


def _resolve_name(name: str) -> object | None:
    """The object a dotted photonix name refers to, or None."""
    parts = name.split(".")
    for split in range(len(parts), 0, -1):
        modname = ".".join(parts[:split])
        if not modname.startswith("photonix"):
            return None
        try:
            obj: object = importlib.import_module(modname)
        except ImportError:
            continue
        try:
            for attr in parts[split:]:
                obj = getattr(obj, attr)
        except AttributeError:
            return None
        return obj
    return None


def _missing_reference(app, env, node, contnode):
    """Resolve a reference to a re-exported name (``photonix.layout.Cell``, or
    ``SDict`` inside a module that imports it) to the page that documents it."""
    if node.get("refdomain") != "py":
        return None
    target = node.get("reftarget", "")
    module = node.get("py:module") or ""
    candidates = [target] if target.startswith("photonix") else []
    if module:
        candidates.append(f"{module}.{target}")
    for name in candidates:
        obj = _resolve_name(name)
        home = _DOCUMENTED.get(id(obj)) if obj is not None and not isinstance(obj, _PRIMITIVE) else None
        if home is not None and home != target:
            domain = env.domains["py"] if hasattr(env, "domains") else env.get_domain("py")
            return domain.resolve_xref(env, node["refdoc"], app.builder, "obj", home, node, contnode)
    return None


# ":class:`Cell`s" is not valid reStructuredText (a role must end at a word
# boundary); render it as ":class:`Cell`\ s" instead of failing the build.
_ROLE_SUFFIX = re.compile(r"(:[\w:.+-]+:`[^`]+`)(?=[A-Za-z0-9])")


def _process_docstring(app, what, name, obj, options, lines) -> None:
    module = getattr(obj, "__module__", None)
    if what in ("function", "class", "method", "exception") and isinstance(module, str) \
            and not module.startswith("photonix"):
        qualname = getattr(obj, "__qualname__", getattr(obj, "__name__", ""))
        lines[:] = [f"Re-export of ``{module}.{qualname}`` from the active backend (JAX or NumPy); "
                    "see that project's documentation.", ""]
        return
    if what in ("data", "attribute") and not callable(obj) and type(obj).__module__ == "builtins" \
            and getattr(obj, "__doc__", None) == type(obj).__doc__:
        # A constant such as a dict or tuple has no docstring of its own; autodoc
        # would show the builtin type's documentation instead.
        lines[:] = [f"Module constant of type :class:`{type(obj).__name__}`.", ""]
        return
    lines[:] = [_ROLE_SUFFIX.sub(r"\1\\ ", line) for line in lines]


def setup(app):
    app.connect("builder-inited", _write_api_pages)
    app.connect("source-read", _source_read)
    app.connect("include-read", _include_read)
    app.connect("missing-reference", _missing_reference)
    # Before napoleon (priority 500), so third-party docstrings are never parsed.
    app.connect("autodoc-process-docstring", _process_docstring, priority=400)
    return {"parallel_read_safe": True}
