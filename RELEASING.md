# Releasing photonix

Releases are **tag-driven**. Pushing a tag `vX.Y.Z` to GitHub runs
[`.github/workflows/release.yml`](.github/workflows/release.yml), which:

1. checks that the tag, `photonix.__version__`, `CHANGELOG.md` and
   `CITATION.cff` agree (`python tools/release.py check --tag vX.Y.Z`);
2. builds the sdist and wheel (`python -m build`) and checks them with
   `twine check --strict` and `check-wheel-contents`. It also asserts that the
   wheel ships `py.typed` and no `tests/` or `benchmarks/`;
3. installs the wheel into a clean virtual environment and runs
   `photonix info --json` and `photonix validate --quick`;
4. publishes to **PyPI** through
   [Trusted Publishing](https://docs.pypi.org/trusted-publishers/) (OIDC; no
   token is stored anywhere). Pre-release tags (`vX.Y.ZaN`, `bN`, `rcN`) go to
   **TestPyPI** instead;
5. creates the GitHub Release. Its body is the version's `CHANGELOG.md` section
   (`python tools/release.py notes --tag vX.Y.Z`), and the distributions are
   attached. If the Zenodo GitHub integration is enabled, Zenodo archives the
   release and mints a DOI (metadata from [`.zenodo.json`](.zenodo.json)).

Versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html),
written in PEP 440 form for pre-releases (`0.3.0rc1`, not `0.3.0-rc.1`).
`photonix.__version__` in `src/photonix/__init__.py` is the **only** place the
version is written. `pyproject.toml` declares `dynamic = ["version"]`, and
hatchling reads it at build time.

## Cutting a release

1. On a branch, prepare the release commit:
   - set `__version__ = "X.Y.Z"` in `src/photonix/__init__.py`;
   - in `CHANGELOG.md`, rename `## [Unreleased]` content into
     `## [X.Y.Z] - YYYY-MM-DD` (today's date), leave an empty `## [Unreleased]`
     above it, and update the link references at the bottom;
   - in `CITATION.cff`, set `version: X.Y.Z` and `date-released: YYYY-MM-DD`;
   - run `python tools/release.py check --tag vX.Y.Z` (it must print
     "consistent") and the full test suite.
2. Open a pull request, let CI pass, and merge it into `main`.
3. Tag the merge commit and push the tag:

   ```bash
   git switch main && git pull
   git tag -a vX.Y.Z -m "photonix X.Y.Z"
   git push origin vX.Y.Z
   ```

4. Watch the *Release* workflow. If the `pypi` environment requires approval,
   approve the publish job.
5. Check that the result installs from a clean environment:

   ```bash
   python -m venv /tmp/px && /tmp/px/bin/pip install "photonix==X.Y.Z"
   /tmp/px/bin/photonix info --json && /tmp/px/bin/photonix validate --quick
   ```

### Dry run through TestPyPI

Tag a release candidate first (`vX.Y.Zrc1`, with `__version__ = "X.Y.Zrc1"`). It
needs only an `[Unreleased]` or `[X.Y.Z]` changelog section, and it publishes to
TestPyPI:

```bash
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple "photonix==X.Y.Zrc1"
photonix info --json
```

## One-time repository setup (owner)

These settings live on GitHub, PyPI and Zenodo, not in the repository:

- **PyPI Trusted Publisher.** On pypi.org (project *photonix*, or a "pending
  publisher" before the first upload), add a GitHub publisher: owner
  `mayank-lang`, repository `photonix`, workflow `release.yml`, environment
  `pypi`. Do the same on test.pypi.org with environment `testpypi`.
- **GitHub environments.** Create `pypi` and `testpypi` under *Settings ->
  Environments*. Adding required reviewers to `pypi` is recommended.
- **GitHub Pages.** Under *Settings -> Pages*, set the source to *GitHub
  Actions* so that [`docs.yml`](.github/workflows/docs.yml) can deploy the
  documentation site. Once it is live, change `Documentation` in
  `pyproject.toml` from the `tree/main/docs` URL to the Pages URL.
- **Private vulnerability reporting.** Enable it under *Settings -> Code
  security* (see [`SECURITY.md`](SECURITY.md)).
- **Zenodo.** Sign in to zenodo.org with GitHub and switch the repository on, so
  every GitHub Release gets a DOI. Then add the concept DOI badge to the README
  and an `identifiers` entry to `CITATION.cff`.
- **Authors.** `pyproject.toml`, `CITATION.cff` and `.zenodo.json` list the
  placeholder "photonix contributors". Replace it with the real author list
  when the owner chooses one.

## What CI checks on every push

`ci.yml` runs on every branch push, on pull requests, weekly (a canary against
new upstream releases) and on demand. It runs:

- lint: ruff, mypy, actionlint;
- the test matrix: Python 3.10 to 3.13 on Linux, plus macOS and Windows;
- `numpy-backend`: `PHOTONIX_BACKEND=numpy` with JAX uninstalled;
- `minimal`: only NumPy, SciPy and pytest installed;
- `oldest-deps`: the declared floors (numpy 1.24, scipy 1.10, jax 0.4.20 on
  Python 3.10);
- coverage with a `fail_under` ratchet;
- the package build and a clean-venv wheel smoke test;
- the examples and the benchmark smoke run.

`meep.yml` runs the Meep backend tests nightly in a conda environment and fails
if any of them skip. `docs.yml` builds the Sphinx site with warnings treated as
errors.
