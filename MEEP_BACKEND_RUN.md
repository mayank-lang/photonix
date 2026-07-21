# Running the Meep backend + full suite on a real machine

Everything below assumes you're at the repo root: `photonix/`.

## 1. Environment

The Meep backend needs **Meep + MPB** (conda-only). The rest of photonix needs
`numpy`, `scipy`, and (optionally) `jax`.

> **Windows note:** conda-forge has **no Windows build of `pymeep`** (Linux/macOS
> only). On Windows you must use **WSL2** (Ubuntu) — native Windows conda will fail
> the solve. There is also **no `pymeep-extras` package**; MPB is bundled in
> `pymeep`.

```powershell
# Windows only: install WSL2 + Ubuntu (PowerShell as admin), then reboot if asked
wsl --install -d Ubuntu
```

```bash
# Linux / macOS / inside WSL Ubuntu: install Miniconda first, then:
conda create -n photonix -c conda-forge python=3.11 pymeep
conda activate photonix
# faster parallel build instead of plain pymeep (optional):
#   conda install -c conda-forge "pymeep=*=mpi_mpich_*"

# photonix runtime deps
pip install numpy scipy
pip install "jax[cpu]"          # optional: enables the differentiable backend
pip install pytest ruff         # dev: tests + lint

# put the package on the path. From WSL, the repo is under /mnt/c/...
#   cd "/mnt/c/Users/My PC/Desktop/Photonics/photonix"
pip install -e .                # if there's a pyproject/setup
export PYTHONPATH="$PWD/src"    # or just point at src/
```

Verify Meep imported:
```bash
python -c "import meep, meep.mpb; print('meep', meep.__version__)"
```

## 2. The Meep backend tests

```bash
pytest tests/test_em_meep.py -v
```

Expected **with Meep installed**: the 2 contract tests pass and all 13 backend
tests run (MaterialGrid weights, coordinate mapping, the unit bridge, MPB `n_eff`
vs the in-house FDE, and a straight-waveguide FDTD transmission ≈ 1).

Expected **without Meep**: the 2 contract tests pass, the 13 backend tests skip.

Two backend tests do real solves and are the ones to watch:
- `test_mpb_neff_matches_fde` — MPB vs `n_eff_fullvector`, asserts agreement < 0.03.
- `test_meep_fdtd_straight_waveguide_transmits` — straight guide, asserts 0.9 < T ≤ 1.02.

If either fails on numbers (not import), it's almost certainly a tolerance/parity/
resolution tune, not a structural bug — see notes in §5.

## 3. The full existing suite (~89 tests)

```bash
pytest -q
```

My changes were additive (a new `photonix.em.meep` subpackage + lazy hook in
`em/__init__.py`), so the prior suite should remain green. The import contract is
covered by `tests/test_em_meep.py::test_em_imports_without_meep`.

## 4. The external benchmark (real Meep reference numbers)

```bash
python benchmarks/run.py --external
```

Without Meep it prints `[external] meep: skipped (ImportError: ...)`. With Meep it
fills the SOI-strip TE0 `n_eff` (MPB) and width-step TE transmission (Meep FDTD)
columns and writes `benchmarks/RESULTS.md`.

## 5. Lint

```bash
ruff check src/photonix/em/meep/ tests/test_em_meep.py benchmarks/external/meep_adapter.py
```

## What to send back if something's off

- Full `pytest tests/test_em_meep.py -v` output (esp. the two solver tests).
- `meep.__version__`.
- If a number is just outside tolerance: the printed value. The likely knobs are
  the eigenmode `parity` mapping in `fdtd.parity_for`, the FDTD `decay_by`
  threshold / monitor placement, and MPB `resolution` — all are parameters, not
  hard-coded assumptions.
```
