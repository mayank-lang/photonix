"""Release metadata stays consistent: URLs, single-sourced version, changelog, citation.

These checks read repository files, so they skip when the tests run from an
installed sdist/wheel without the repository around them.
"""
from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import pytest

import photonix as px

ROOT = Path(__file__).resolve().parents[1]
REPO = "github.com/mayank-lang/photonix"

pytestmark = pytest.mark.skipif(not (ROOT / "pyproject.toml").is_file(), reason="needs the source checkout")


def _toml():
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10
        tomllib = pytest.importorskip("tomli")
    return tomllib


@pytest.fixture(scope="module")
def pyproject() -> dict:
    with open(ROOT / "pyproject.toml", "rb") as fh:
        return _toml().load(fh)


def _release_tool():
    spec = importlib.util.spec_from_file_location("photonix_release_tool", ROOT / "tools" / "release.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
# pyproject.toml
# --------------------------------------------------------------------------- #
def test_every_project_url_points_at_the_real_repository(pyproject):
    urls = pyproject["project"]["urls"]
    assert {"Homepage", "Repository", "Issues", "Changelog", "Documentation"} <= set(urls)
    for name, url in urls.items():
        assert url.startswith(f"https://{REPO}"), (name, url)


def test_version_is_single_sourced_from_the_package(pyproject):
    project = pyproject["project"]
    assert "version" not in project and "version" in project["dynamic"]
    assert pyproject["tool"]["hatch"]["version"]["path"] == "src/photonix/__init__.py"
    assert _release_tool().package_version(ROOT) == px.__version__
    assert re.fullmatch(r"\d+\.\d+\.\d+((a|b|rc)\d+)?", px.__version__)


def test_license_and_classifiers(pyproject):
    project = pyproject["project"]
    assert project["license"] == "MIT" and project["license-files"] == ["LICENSE"]
    assert (ROOT / "LICENSE").read_text(encoding="utf-8").startswith("MIT License")
    classifiers = set(project["classifiers"])
    assert "Operating System :: OS Independent" in classifiers
    assert "Typing :: Typed" in classifiers and (Path(px.__file__).parent / "py.typed").is_file()
    floor = tuple(int(v) for v in project["requires-python"].removeprefix(">=").split("."))
    declared = {tuple(int(v) for v in c.rsplit(" ", 1)[1].split("."))
                for c in classifiers if re.fullmatch(r"Programming Language :: Python :: 3\.\d+", c)}
    assert min(declared) == floor and (3, 13) in declared
    # Every declared interpreter is tested in CI.
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    for major, minor in declared:
        assert f'"{major}.{minor}"' in ci, f"Python {major}.{minor} is declared but not in the CI matrix"


def test_installed_distribution_metadata_matches_the_package():
    try:
        dist = metadata.distribution("photonix")
    except metadata.PackageNotFoundError:
        pytest.skip("photonix is imported from a path, not an installed distribution")
    direct = json.loads(dist.read_text("direct_url.json") or "{}")
    url = direct.get("url", "")
    if url.startswith("file://") and Path(url[7:]).resolve() != ROOT:
        pytest.skip(f"the installed distribution belongs to another checkout ({url})")
    package_root = Path(px.__file__).resolve().parents[2]
    if package_root != ROOT and url.startswith("file://"):
        pytest.skip("photonix is imported from outside this checkout")
    assert dist.version == px.__version__, (
        "installed metadata is stale: re-run `pip install -e .` after changing __version__"
    )


# --------------------------------------------------------------------------- #
# CHANGELOG, CITATION, Zenodo
# --------------------------------------------------------------------------- #
def test_changelog_has_unreleased_and_current_version_sections():
    tool = _release_tool()
    sections = tool.changelog_sections((ROOT / "CHANGELOG.md").read_text(encoding="utf-8"))
    assert "Unreleased" in sections
    release, _ = tool.parse_version(px.__version__)
    assert release in sections and sections[release][1], f"CHANGELOG.md needs a non-empty [{release}] section"
    # Every dated section uses ISO dates, newest first.
    dates = [d for d, _ in sections.values() if d and re.fullmatch(r"\d{4}-\d{2}-\d{2}", d)]
    assert dates == sorted(dates, reverse=True)


def test_citation_and_zenodo_metadata():
    cff = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert re.search(r"^cff-version: 1\.2\.0$", cff, re.MULTILINE)
    assert re.search(rf"^version: {re.escape(px.__version__)}$", cff, re.MULTILINE)
    assert re.search(r"^license: MIT$", cff, re.MULTILINE)
    assert f"https://{REPO}" in cff
    if importlib.util.find_spec("yaml") is not None:  # optional: full YAML parse
        import yaml

        data = yaml.safe_load(cff)
        assert str(data["version"]) == px.__version__ and data["authors"]
        assert {"cff-version", "message", "title", "authors"} <= set(data)
    zenodo = json.loads((ROOT / ".zenodo.json").read_text(encoding="utf-8"))
    assert zenodo["license"] == "MIT" and zenodo["upload_type"] == "software" and zenodo["creators"]
    assert zenodo.get("version", px.__version__) == px.__version__


def test_no_stale_placeholder_urls_or_setup_files():
    assert not (ROOT / "GITHUB_SETUP.md").exists()
    for name in ("README.md", "CONTRIBUTING.md", "RELEASING.md", "SECURITY.md", "pyproject.toml", "CITATION.cff"):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert "github.com/photonix/photonix" not in text and "readthedocs" not in text, name
    assert f"git clone https://{REPO}" in (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "## Citing photonix" in readme and f"v{px.__version__}" in readme


def test_community_files_exist():
    for name in ("SECURITY.md", "CODE_OF_CONDUCT.md", "RELEASING.md", ".pre-commit-config.yaml",
                 ".github/pull_request_template.md", ".github/ISSUE_TEMPLATE/bug_report.md",
                 ".github/ISSUE_TEMPLATE/feature_request.md", ".github/dependabot.yml",
                 ".github/workflows/release.yml", ".github/workflows/docs.yml", ".github/workflows/meep.yml"):
        assert (ROOT / name).is_file(), name


# --------------------------------------------------------------------------- #
# tools/release.py (used by the release workflow)
# --------------------------------------------------------------------------- #
def _fake_repo(tmp_path: Path, *, version: str, changelog: str, cff_version: str) -> Path:
    (tmp_path / "src" / "photonix").mkdir(parents=True)
    (tmp_path / "src" / "photonix" / "__init__.py").write_text(f'__version__ = "{version}"\n')
    (tmp_path / "CHANGELOG.md").write_text(changelog)
    (tmp_path / "CITATION.cff").write_text(f"cff-version: 1.2.0\nversion: {cff_version}\n")
    return tmp_path


_DATED = "# Changelog\n\n## [Unreleased]\n\n## [1.2.0] - 2026-05-01\n\n### Added\n\n- thing\n\n" \
         "## [1.1.0] - 2026-01-01\n\n- old\n\n[1.2.0]: https://example.invalid\n"


def test_release_check_accepts_a_consistent_release(tmp_path):
    tool = _release_tool()
    root = _fake_repo(tmp_path, version="1.2.0", changelog=_DATED, cff_version="1.2.0")
    assert tool.check_release("v1.2.0", root) == []
    assert tool.check_release("refs/tags/v1.2.0", root) == []
    assert tool.release_notes("v1.2.0", root) == "### Added\n\n- thing\n"
    assert tool.release_target("v1.2.0") == "pypi"


def test_release_check_rejects_each_inconsistency(tmp_path):
    tool = _release_tool()
    undated = _DATED.replace("## [1.2.0] - 2026-05-01", "## [1.2.0] - Unreleased")
    root = _fake_repo(tmp_path, version="1.2.0", changelog=undated, cff_version="1.1.0")
    problems = tool.check_release("v1.3.0", root)
    assert any("does not match photonix.__version__" in p for p in problems)
    assert any("no '## [1.3.0]' section" in p for p in problems)
    problems = tool.check_release("v1.2.0", root)
    assert any("must be dated" in p for p in problems)
    assert any("CITATION.cff version is '1.1.0'" in p for p in problems)
    assert tool.check_release("1.2.0", root) and tool.check_release("v1.2", root)


def test_prereleases_go_to_testpypi_with_unreleased_notes(tmp_path):
    tool = _release_tool()
    changelog = "# C\n\n## [Unreleased]\n\n### Fixed\n\n- pending fix\n\n## [1.1.0] - 2026-01-01\n\n- old\n"
    root = _fake_repo(tmp_path, version="1.2.0rc1", changelog=changelog, cff_version="1.1.0")
    assert tool.check_release("v1.2.0rc1", root) == []
    assert tool.release_target("v1.2.0rc1") == "testpypi"
    assert tool.release_target("v1.2.0b2") == "testpypi"
    assert "pending fix" in tool.release_notes("v1.2.0rc1", root)
    with pytest.raises(ValueError):
        tool.parse_version("1.2.0.dev1")


def test_release_check_on_this_repository_is_clean_except_the_release_date():
    tool = _release_tool()
    problems = tool.check_release(f"v{px.__version__}", ROOT)
    assert [p for p in problems if "must be dated" not in p] == []
    notes = tool.release_notes(f"v{px.__version__}", ROOT)
    assert "### Added" in notes and "## [" not in notes


def test_release_tool_runs_as_a_script():
    out = subprocess.run([sys.executable, str(ROOT / "tools" / "release.py"), "version"],
                         capture_output=True, text=True, check=True, timeout=60)
    assert out.stdout.strip() == px.__version__
