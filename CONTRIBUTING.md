# Contributing to photonix

Thanks for your interest in contributing. photonix is an accuracy-first,
differentiable photonic-design library. A change is ready to merge when it is
clear, comes with tests and validation evidence, and passes lint and type checks.
Everyone taking part follows the [Code of Conduct](CODE_OF_CONDUCT.md). Report
security issues privately as described in [SECURITY.md](SECURITY.md), not in a
public issue.

## Development setup

```bash
git clone https://github.com/mayank-lang/photonix
cd photonix
python -m pip install -e ".[dev]"     # numpy, scipy, jax, gdstk, matplotlib, pytest, ruff, mypy, pre-commit
pre-commit install                     # optional: run ruff and the file hygiene hooks on every commit
```

Optional FDTD backend (`photonix.em.meep`): MEEP is conda-only and is not a pip
dependency. On Linux/macOS (or Windows WSL2):

```bash
conda install -c conda-forge pymeep
```

## Running the checks

```bash
ruff check src tests benchmarks examples tools   # lint (must be clean)
python -m mypy src/photonix                      # type check
python -m pytest --doctest-modules src/photonix --ignore=src/photonix/em/meep
python -m pytest                                 # full suite (~600 tests)
PHOTONIX_BACKEND=numpy python -m pytest          # the NumPy fallback backend
python -m pytest tests/test_em_meep.py -v        # the MEEP backend, specifically
photonix validate --quick                        # packaged analytic self-checks
```

Tests that need an optional dependency skip when it is absent. The Meep tests
skip without MEEP, and the JAX-gradient tests are marked `requires_jax`. The
suite is green on a minimal `pip install -e ".[test]"` (NumPy, SciPy and pytest
only). CI runs that minimal job, a NumPy-backend job, the oldest supported
dependency versions, Python 3.10 to 3.13 on Linux (plus macOS and Windows), a
coverage gate, the examples and the docs build. A Meep-enabled job runs nightly.
See [`.github/workflows/`](.github/workflows/).

When you add a test that needs an optional package, never import it at module
level. Use `requires_gdstk` / `requires_matplotlib` / `requires_jax` from
`tests/conftest.py`, or `pytest.importorskip` inside the test.

## Conventions

- **Style:** `ruff` with the repo config (line length 120; `E,F,I,UP,B`).
- **Backend-agnostic:** numerical code should work under both the JAX and NumPy
  backends. Import the array module via `photonix.core.backend.xp`.
- **Differentiability:** new solvers should expose a differentiable entry point
  (a `jax.custom_vjp` adjoint where the forward pass is non-trivial).
- **Tests:** every change ships with a test. Validate against an analytic limit,
  the literature, or an independent solver where possible. See the existing
  `tests/test_em_*` and `benchmarks/` for the pattern, and
  [`docs/VALIDATION.md`](docs/VALIDATION.md) for the evidence levels.
- **Docs:** public functions get a NumPy-style docstring with a runnable
  `Examples` block, and every public name is listed in its module's `__all__`.
  The API reference is generated from `__all__`.
- **Changelog:** add a line under `## [Unreleased]` in
  [`CHANGELOG.md`](CHANGELOG.md) for any user-visible change.

## Pull requests

1. Branch from `main`.
2. Keep PRs focused, one logical change each.
3. Make sure `ruff check`, `mypy` and `pytest` pass.
4. Fill in the pull-request template: what changed, how you validated it, and
   the evidence level.

Releases are cut by the maintainers; see [`RELEASING.md`](RELEASING.md).

By contributing you agree your contributions are licensed under the MIT License.
