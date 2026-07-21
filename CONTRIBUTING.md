# Contributing to photonix

Thanks for your interest in contributing! photonix is an accuracy-first,
differentiable photonic-design library; the bar for merging is a clear change
with tests and a passing lint.

## Development setup

```bash
git clone https://github.com/photonix/photonix
cd photonix
python -m pip install -e ".[dev]"     # numpy, scipy, jax, gdstk, matplotlib, pytest, ruff, mypy
```

Optional FDTD backend (`photonix.em.meep`) — MEEP is conda-only and not a pip
dependency. On Linux/macOS (or Windows WSL2):

```bash
conda install -c conda-forge pymeep
```

## Running the checks

```bash
ruff check src tests        # lint (must be clean)
pytest -q                   # full suite (~105 tests)
pytest tests/test_em_meep.py -v   # the MEEP backend, specifically
```

The MEEP-requiring tests `skip` automatically when MEEP is absent, so the suite
is green with or without it. CI runs `ruff` + `pytest` on Python 3.10–3.12
without MEEP; if you change the MEEP backend, please also run the backend tests
locally on a MEEP-equipped machine and note the result in your PR.

## Conventions

- **Style:** `ruff` with the repo config (line length 120; `E,F,I,UP,B`).
- **Backend-agnostic:** numerical code should work under both the JAX and NumPy
  backends — import the array module via `photonix.core.backend.xp`.
- **Differentiability:** new solvers should expose a differentiable entry point
  (a `jax.custom_vjp` adjoint where the forward pass is non-trivial).
- **Tests:** every change ships with a test. Validate against an analytic limit,
  the literature, or an independent solver where possible — see the existing
  `tests/test_em_*` and `benchmarks/` for the pattern.
- **Docs:** public functions get a docstring with a runnable `Examples` block.

## Pull requests

1. Branch from `main`.
2. Keep PRs focused; one logical change each.
3. Ensure `ruff check` and `pytest` pass.
4. Describe what you changed and how you validated it.

By contributing you agree your contributions are licensed under the MIT License.
