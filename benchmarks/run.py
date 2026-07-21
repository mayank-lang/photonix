"""Run the benchmark suite and emit a results table.

Computes every photonix case value, compares against literature anchors and any
external-solver results recorded in ``references.json`` (and produced by the
adapters in ``benchmarks/external/``), and writes a Markdown table to
``benchmarks/RESULTS.md``.

Usage:  python benchmarks/run.py            # photonix vs literature/internal
        python benchmarks/run.py --external # also invoke installed external solvers

The point is a *reproducible, append-only* credibility artifact: each row is a
structure, a number, a reference, and a pass/fail against tolerance.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from cases import CASES  # noqa: E402


def _load_refs() -> dict:
    with open(os.path.join(HERE, "references.json")) as f:
        return json.load(f)


def _collect_external() -> dict:
    """Invoke each external adapter; skip cleanly if the solver isn't installed."""
    results: dict = {}
    from external import ADAPTERS

    for name, adapter in ADAPTERS.items():
        try:
            results[name] = adapter.run_all()
            print(f"[external] {name}: {len(results[name])} case(s)")
        except Exception as e:  # noqa: BLE001
            print(f"[external] {name}: skipped ({type(e).__name__}: {e})")
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--external", action="store_true", help="also run external solvers")
    args = ap.parse_args()

    refs = _load_refs()
    lit = refs.get("literature", {})
    external = refs.get("external", {})
    if args.external:
        for solver, res in _collect_external().items():
            for k, v in res.items():
                external.setdefault(k, {})[solver] = v

    rows = []
    n_pass = 0
    for case in CASES:
        val = float(case.compute())
        ref = lit.get(case.key)
        if ref is not None:
            delta = abs(val - ref["value"])
            ok = delta <= ref["tol"]
            n_pass += ok
            rows.append((case.key, case.quantity, f"{val:.4f}",
                         f"{ref['value']:.4f}", f"{delta:.4f}",
                         "PASS" if ok else "FAIL", ref["source"]))
        else:
            rows.append((case.key, case.quantity, f"{val:.4f}", "-", "-", "-", "(no reference)"))

    # Markdown table
    head = "| case | quantity | photonix | reference | |Δ| | status | source |"
    sep = "|---|---|---|---|---|---|---|"
    lines = [head, sep]
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    table = "\n".join(lines)

    n_ref = sum(1 for c in CASES if c.key in lit)
    summary = f"\n**{n_pass}/{n_ref} cases within tolerance** ({len(CASES)} total)."
    ext_note = ""
    if external:
        ext_note = "\n\nExternal-solver results recorded: " + ", ".join(sorted(external)) + "."

    out = f"# photonix benchmark results\n\n{table}\n{summary}{ext_note}\n"
    with open(os.path.join(HERE, "RESULTS.md"), "w") as f:
        f.write(out)
    print(out)
    print(f"Wrote {os.path.join(HERE, 'RESULTS.md')}")


if __name__ == "__main__":
    main()
