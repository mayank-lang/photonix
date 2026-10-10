"""Circuit-engine benchmark: Clements MZI meshes from 4x4 to 32x32.

Each case programs a Haar-random unitary onto a rectangular mesh with
``photonix.circuit.mzi_mesh(clements_decompose(U))`` and times

* ``model``   -- evaluating every instance model (outside the engine);
* ``legacy``  -- the original solver: one functional ``.at[i, j].set`` per
  ``SDict`` entry and a solve for all ``n`` right-hand sides (re-implemented
  here verbatim for comparison; skipped where it would take minutes);
* ``dense``   -- the engine's vectorized dense backend;
* ``sparse``  -- the engine's SuperLU backend (the ``auto`` choice for large n);
* ``reduce``  -- the engine's subnetwork-growth backend;

and checks every backend against the target unitary (the mesh is wavelength
independent: ideal couplers and zero-length phase shifters) and the sparse
result against the dense one.

Usage::

    python benchmarks/bench_circuit.py            # print the table
    python benchmarks/bench_circuit.py --write    # also write CIRCUIT_RESULTS.md
    python benchmarks/bench_circuit.py --quick    # small cases only

Timings are wall-clock on whatever machine runs the script and include JAX
dispatch; compare backends within one run rather than across machines.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import numpy as np  # noqa: E402
from scipy.stats import unitary_group  # noqa: E402

import photonix as px  # noqa: E402
from photonix.circuit import engine  # noqa: E402
from photonix.core.backend import xp  # noqa: E402
from photonix.core.types import ports_of  # noqa: E402


def legacy_evaluate(instances, connections, ports):
    """The pre-engine solver (photonix <= ed47c6c), kept for timing only."""
    terminals = [(inst, p) for inst, sd in instances.items() for p in ports_of(sd)]
    index = {t: i for i, t in enumerate(terminals)}
    n = len(terminals)
    batch: tuple[int, ...] = ()
    for sd in instances.values():
        for v in sd.values():
            batch = xp.broadcast_shapes(batch, xp.asarray(v).shape)
    S = xp.zeros((*batch, n, n), dtype=complex)
    for inst, sd in instances.items():
        for (p_in, p_out), val in sd.items():
            v = xp.broadcast_to(xp.asarray(val, dtype=complex), batch)
            if hasattr(S, "at"):
                S = S.at[..., index[(inst, p_out)], index[(inst, p_in)]].set(v)
            else:
                S = S.copy()
                S[..., index[(inst, p_out)], index[(inst, p_in)]] = v
    G = np.zeros((n, n), dtype=complex)
    for a, b in connections.items():
        G[index[a], index[b]] = G[index[b], index[a]] = 1.0
    M = xp.linalg.solve(xp.eye(n, dtype=complex) - S @ xp.asarray(G), S)
    return {(ei, eo): M[..., index[to], index[ti]] for ei, ti in ports.items() for eo, to in ports.items()}


def capture(model, wl):
    """Evaluate ``model`` and return the (instances, connections, ports) the engine saw."""
    seen = {}
    original = engine._TopologyCache.get

    def spy(self, instances, connections, ports):
        seen.update(i=dict(instances), c=dict(connections), p=dict(ports))
        return original(self, instances, connections, ports)

    engine._TopologyCache.get = spy
    try:
        t0 = time.perf_counter()
        model(wl=wl)
        elapsed = time.perf_counter() - t0
    finally:
        engine._TopologyCache.get = original
    return seen["i"], seen["c"], seen["p"], elapsed


def mesh_error(s, U):
    n = U.shape[0]
    S = np.array([[np.asarray(s[(f"o{i + 1}", f"o{2 * n - j}")]) for i in range(n)] for j in range(n)])
    return float(np.max(np.abs(S - (U[..., None] if S.ndim == 3 else U))))


def timed(fn, repeat=2):
    """Best of ``repeat`` runs: the first one also pays per-shape XLA compilation."""
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        for v in out.values():
            np.asarray(v)
        best = min(best, time.perf_counter() - t0)
    return out, best


def run_case(n, n_wl, *, legacy, dense, reduce):
    U = unitary_group.rvs(n, random_state=n)
    model = px.circuit.mzi_mesh(px.circuit.clements_decompose(U))
    wl = np.linspace(1.50, 1.60, n_wl)
    inst, conn, ports, t_model = capture(model, wl)
    row = {"case": f"{n}x{n}", "wavelengths": n_wl, "terminals": None, "model_s": t_model}
    t0 = time.perf_counter()
    topo = engine.compile_topology(inst, conn, ports)
    values = engine.gather_values(topo, inst)
    row["compile_gather_s"] = time.perf_counter() - t0
    row["terminals"] = topo.n
    s_sparse, row["sparse_s"] = timed(lambda: engine.solve(topo, values, backend="sparse"))
    row["sparse_err"] = mesh_error(s_sparse, U)
    if dense:
        # Dense memory grows as batch * n^2: time one wavelength for large n.
        one = n_wl if topo.n <= 2000 else 1
        v1 = values if one == n_wl else values[:1]
        s_dense, row["dense_s"] = timed(lambda: engine.solve(topo, v1, backend="dense"))
        row["dense_wavelengths"] = one
        row["dense_err"] = mesh_error(s_dense, U)
        sl = slice(None) if one == n_wl else slice(0, 1)
        row["sparse_vs_dense"] = max(float(np.max(np.abs(np.asarray(s_sparse[k])[sl] - np.asarray(s_dense[k]))))
                                     for k in s_sparse)
    if reduce:
        s_red, row["reduce_s"] = timed(lambda: engine.solve(topo, values, backend="reduce"))
        row["reduce_err"] = mesh_error(s_red, U)
    if legacy:
        s_leg, row["legacy_s"] = timed(lambda: legacy_evaluate(inst, conn, ports), repeat=1)
        row["legacy_err"] = mesh_error(s_leg, U)
    return row


def memory_probe(n=16, n_wl=1000):
    code = (
        "import resource, time, numpy as np, photonix as px\n"
        "from scipy.stats import unitary_group\n"
        f"U = unitary_group.rvs({n}, random_state=1)\n"
        "f = px.circuit.mzi_mesh(px.circuit.clements_decompose(U))\n"
        "f(wl=np.linspace(1.5, 1.6, 3))\n"
        "base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024\n"
        f"t = time.perf_counter(); s = f(wl=np.linspace(1.5, 1.6, {n_wl})); t = time.perf_counter() - t\n"
        "peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024\n"
        "import json; print(json.dumps({'base_mb': base, 'peak_mb': peak, 'seconds': t}))\n"
    )
    env = {**os.environ, "PYTHONPATH": os.path.join(HERE, "..", "src") + os.pathsep + os.environ.get("PYTHONPATH", "")}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=1800)
    if out.returncode:
        raise RuntimeError(out.stderr)
    return json.loads(out.stdout.strip().splitlines()[-1])


def fmt(x, spec=".3f"):
    return "-" if x is None else format(x, spec)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="write benchmarks/CIRCUIT_RESULTS.md")
    ap.add_argument("--quick", action="store_true", help="small cases only")
    args = ap.parse_args()

    cases = [(4, 5, True, True, True), (8, 5, True, True, True), (12, 5, True, True, True),
             (16, 5, False, True, True), (32, 101, False, True, False)]
    if args.quick:
        cases = cases[:2]
    rows = []
    for n, n_wl, legacy, dense, reduce in cases:
        print(f"running {n}x{n} mesh at {n_wl} wavelengths ...", flush=True)
        rows.append(run_case(n, n_wl, legacy=legacy, dense=dense, reduce=reduce))

    lines = [
        "# Circuit engine benchmark",
        "",
        f"Machine: {platform.processor() or platform.machine()}, {os.cpu_count()} CPUs, "
        f"backend {px.backend_name()}, Python {platform.python_version()}. Times in seconds (wall clock, "
        "best of two runs so per-shape JAX compilation is excluded; `legacy` is a single run).",
        "",
        "| mesh | wl | terminals | model eval | compile+gather | sparse | dense | reduce | legacy | "
        "max err vs U (sparse / dense / reduce / legacy) | sparse vs dense |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        dense_label = fmt(r.get("dense_s"))
        if r.get("dense_wavelengths", r["wavelengths"]) != r["wavelengths"]:
            dense_label += f" ({r['dense_wavelengths']} wl)"
        errs = " / ".join(fmt(r.get(k), ".1e") for k in ("sparse_err", "dense_err", "reduce_err", "legacy_err"))
        lines.append(
            f"| {r['case']} | {r['wavelengths']} | {r['terminals']} | {fmt(r['model_s'])} | "
            f"{fmt(r['compile_gather_s'])} | {fmt(r['sparse_s'])} | {dense_label} | {fmt(r.get('reduce_s'))} | "
            f"{fmt(r.get('legacy_s'))} | {errs} | {fmt(r.get('sparse_vs_dense'), '.1e')} |"
        )
    if not args.quick:
        print("running 1000-wavelength 16x16 memory probe ...", flush=True)
        mem = memory_probe()
        lines += [
            "",
            f"1000-wavelength sweep of the 16x16 mesh (model + engine, fresh process): "
            f"{mem['seconds']:.2f} s, peak RSS {mem['peak_mb']:.0f} MB "
            f"({mem['peak_mb'] - mem['base_mb']:.0f} MB above the post-import baseline of {mem['base_mb']:.0f} MB).",
        ]
    lines += [
        "",
        "`legacy` is the pre-engine algorithm (per-entry functional scatter, all-n right-hand sides); it is skipped "
        "where it takes minutes (the gap analysis measured 142 s for the 16x16 mesh at 5 wavelengths). The 32x32 "
        "dense solve uses one wavelength because its batched matrix needs ~0.6 GB per wavelength.",
    ]
    text = "\n".join(lines) + "\n"
    print(text)
    if args.write:
        with open(os.path.join(HERE, "CIRCUIT_RESULTS.md"), "w", encoding="utf-8") as fh:
            fh.write(text)


if __name__ == "__main__":
    main()
