"""Release helpers used by ``.github/workflows/release.yml`` (and by hand).

The script has no dependencies beyond the standard library and never imports
photonix, so it runs before anything is installed::

    python tools/release.py check --tag v0.2.0      # tag/version/changelog/citation agree?
    python tools/release.py notes --tag v0.2.0      # release notes from CHANGELOG.md
    python tools/release.py target --tag v0.2.1rc1  # "pypi" or "testpypi"

``check`` exits 1 with one line per problem. The rules are:

* the tag is ``v`` + ``photonix.__version__``, and the version is PEP 440
  (``X.Y.Z`` or ``X.Y.ZaN`` / ``bN`` / ``rcN``);
* a final release has a dated ``## [X.Y.Z] - YYYY-MM-DD`` section in
  ``CHANGELOG.md``, and ``CITATION.cff`` carries the same version;
* a pre-release (``a``/``b``/``rc``) only needs the ``X.Y.Z`` section or
  ``[Unreleased]`` to exist; it is published to TestPyPI.
"""
from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_VERSION_RE = re.compile(r'^__version__\s*=\s*["\']([^"\']+)["\']', re.MULTILINE)
_PEP440_RE = re.compile(r"^(?P<release>\d+\.\d+\.\d+)(?P<pre>(?:a|b|rc)\d+)?$")
_HEADING_RE = re.compile(r"^## \[(?P<name>[^\]]+)\](?:\s+-\s+(?P<date>.+?))?\s*$", re.MULTILINE)
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_CFF_VERSION_RE = re.compile(r"^version:\s*['\"]?([^'\"\s]+)['\"]?\s*$", re.MULTILINE)

__all__ = [
    "package_version", "parse_version", "changelog_sections", "changelog_section",
    "release_notes", "release_target", "check_release", "main",
]


def package_version(root: Path = ROOT) -> str:
    """``__version__`` from ``src/photonix/__init__.py`` (the hatch version source)."""
    text = (root / "src" / "photonix" / "__init__.py").read_text(encoding="utf-8")
    match = _VERSION_RE.search(text)
    if match is None:
        raise ValueError("no __version__ assignment in src/photonix/__init__.py")
    return match.group(1)


def parse_version(version: str) -> tuple[str, bool]:
    """Split a supported PEP 440 version into ``(release, is_prerelease)``.

    >>> parse_version("0.2.0"), parse_version("1.0.1rc2")
    (('0.2.0', False), ('1.0.1', True))
    """
    match = _PEP440_RE.match(version)
    if match is None:
        raise ValueError(f"version {version!r} is not X.Y.Z or X.Y.Z(a|b|rc)N")
    return match.group("release"), match.group("pre") is not None


def _tag_version(tag: str) -> str:
    tag = tag.removeprefix("refs/tags/")
    if not tag.startswith("v"):
        raise ValueError(f"release tags look like v1.2.3, got {tag!r}")
    return tag[1:]


def changelog_sections(text: str) -> dict[str, tuple[str | None, str]]:
    """Map each ``## [name] - date`` heading to ``(date_or_None, body)``.

    >>> s = changelog_sections("# C\\n## [Unreleased]\\n\\n## [1.0.0] - 2026-01-02\\n### Added\\n- x\\n")
    >>> s["1.0.0"]
    ('2026-01-02', '### Added\\n- x')
    """
    matches = list(_HEADING_RE.finditer(text))
    out: dict[str, tuple[str | None, str]] = {}
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[match.end():end]
        # Link reference definitions ("[1.0.0]: https://...") close the last section.
        body = re.split(r"^\[[^\]]+\]:\s", body, maxsplit=1, flags=re.MULTILINE)[0]
        out[match.group("name")] = (match.group("date"), body.strip())
    return out


def changelog_section(text: str, name: str) -> tuple[str | None, str]:
    """``(date, body)`` of one changelog section; ``KeyError`` if absent."""
    sections = changelog_sections(text)
    if name not in sections:
        raise KeyError(f"CHANGELOG.md has no '## [{name}]' section")
    return sections[name]


def release_notes(tag: str, root: Path = ROOT) -> str:
    """Release-notes Markdown for ``tag`` (the changelog section body)."""
    release, pre = parse_version(_tag_version(tag))
    text = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    sections = changelog_sections(text)
    for name in ([release, "Unreleased"] if pre else [release]):
        if name in sections and sections[name][1]:
            return sections[name][1] + "\n"
    raise KeyError(f"CHANGELOG.md has no non-empty section for {tag}")


def release_target(tag: str) -> str:
    """``"testpypi"`` for a pre-release tag, ``"pypi"`` for a final release."""
    return "testpypi" if parse_version(_tag_version(tag))[1] else "pypi"


def check_release(tag: str, root: Path = ROOT) -> list[str]:
    """Every inconsistency between ``tag`` and the repository metadata."""
    problems: list[str] = []
    try:
        version = _tag_version(tag)
        release, pre = parse_version(version)
    except ValueError as exc:
        return [str(exc)]
    actual = package_version(root)
    if actual != version:
        problems.append(f"tag {tag} does not match photonix.__version__ = {actual!r}")
    text = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    sections = changelog_sections(text)
    if "Unreleased" not in sections:
        problems.append("CHANGELOG.md has no '## [Unreleased]' section")
    if pre:
        if release not in sections and "Unreleased" not in sections:
            problems.append(f"CHANGELOG.md has neither a [{release}] nor an [Unreleased] section")
    elif release not in sections:
        problems.append(f"CHANGELOG.md has no '## [{release}]' section")
    else:
        date, body = sections[release]
        if date is None or not _DATE_RE.match(date):
            problems.append(f"CHANGELOG.md section [{release}] must be dated '- YYYY-MM-DD' (found {date!r})")
        if not body:
            problems.append(f"CHANGELOG.md section [{release}] is empty")
    cff = (root / "CITATION.cff").read_text(encoding="utf-8")
    match = _CFF_VERSION_RE.search(cff)
    cff_version = match.group(1) if match else None
    if not pre and cff_version != release:
        problems.append(f"CITATION.cff version is {cff_version!r}, expected {release!r}")
    return problems


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point; see the module docstring."""
    parser = argparse.ArgumentParser(prog="tools/release.py", description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("check", "notes", "target", "version"))
    parser.add_argument("--tag", help="release tag, e.g. v0.2.0 (default: v + __version__)")
    args = parser.parse_args(argv)
    tag = args.tag or f"v{package_version()}"
    if args.command == "version":
        print(package_version())
        return 0
    if args.command == "target":
        print(release_target(tag))
        return 0
    if args.command == "notes":
        sys.stdout.write(release_notes(tag))
        return 0
    problems = check_release(tag)
    for problem in problems:
        print(f"release check: {problem}", file=sys.stderr)
    if not problems:
        print(f"release check: {tag} is consistent ({release_target(tag)})")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
