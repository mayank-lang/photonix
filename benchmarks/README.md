# photonix benchmarks

Reproducible cross-solver benchmarks — the "validation as a product" track. Each
case is a small structure with one number (n_eff, transmission, loss) compared
against a reference (literature/analytic, an internal independent solver, or an
external solver) with an explicit tolerance and pass/fail.

## Run

```bash
python benchmarks/run.py             # photonix vs literature + internal references
python benchmarks/run.py --external  # additionally invoke installed external solvers
```

Writes `benchmarks/RESULTS.md` (a Markdown table) and prints it.

## Layout

- `cases.py` — the case registry. Each `Case` computes one photonix value.
- `references.json` — reference values keyed by case: `literature` anchors
  (published/analytic) and `external` (filled by adapters).
- `run.py` — runs all cases, compares to references, emits the table.
- `external/` — solver adapters (`meep_adapter.py`, `tidy3d_adapter.py`). Each
  exposes `run_all() -> {case_key: value}` and raises `ImportError` if its solver
  isn't installed, so the runner skips it cleanly.

## Adding a case

1. Append a `Case(key, description, quantity, compute)` to `CASES` in `cases.py`.
2. Optionally add a reference under `literature` in `references.json` (same `key`,
   with `value`, `tol`, `source`). No reference → the row reports the value only.

## Plugging in an external solver

The adapters are stubs today (they raise `ImportError`/`NotImplementedError`).
To make an external comparison real, install the solver (e.g.
`conda install -c conda-forge pymeep`) and fill in the per-case bodies in
`external/meep_adapter.py` so they build the same structure and return the same
quantity. `run.py --external` then records the external numbers alongside
photonix in the table.

## Status

Today the table compares photonix full-vector/EME results to **literature
anchors** and to **internal independent solvers** (EME ↔ FDFD full-wave). The
external solvers (Meep / Tidy3D) are scaffolded but not yet wired — that is the
next credibility rung, and the reason this directory exists as an append-only
artifact rather than a one-off script.
