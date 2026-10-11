# Circuit engine (`photonix.circuit.engine`)

`photonix.circuit.evaluate_circuit` and `circuit_from_netlist` run on a
scalable engine. It solves the same linear model as the original solver: all
instance terminals stacked into one wave vector, `b = S a`, with the
connections giving `a = Gamma b + u`, so that `(I - S Gamma) b = S u`. The
work is split into three stages.

1. **`compile_topology`** handles the port structure only, no values:
   terminal numbering, the `(row, col)` of every `SDict` entry, the connection
   permutation and the exposed terminals. The result is cached per distinct
   structure (`TOPOLOGY_CACHE`), so repeated evaluations of one circuit skip
   it.
2. **`gather_values`** builds one `(*batch, nnz)` stack of entry values. There
   is no per-entry scatter into an `n x n` array.
3. **`solve`** solves only for the `m` exposed right-hand sides and returns
   the `m x m` exposed block.

## Backends

| Backend | Method | Autodiff | Cost |
|---|---|---|---|
| `dense` | batched dense LU of `I - S Gamma` | all JAX transforms, forward and reverse | `O(n^3)` per sample, `O(batch n^2)` memory |
| `sparse` | SuperLU on `A = I - S Gamma` with a hand-written transpose adjoint (`sparse_linalg.py`) | `jit` / `vmap` / `grad` via a host callback; **no forward mode** (`jvp` / `jacfwd`) | `O(batch (nnz + m^2))` memory |
| `reduce` | subnetwork growth: pairwise `innerconnect` on small dense blocks in greedy minimum-size order (`reduction.py`) | all JAX transforms; stays on device | set by the largest intermediate block |
| `auto` (default) | `dense` up to `DENSE_MAX_TERMINALS = 300` terminals while the batched system fits in `DENSE_MAX_BYTES`, else `sparse` | as chosen | |

Choose `backend="dense"` or `"reduce"` explicitly when you need forward-mode
AD on a large circuit:

```python
f = px.circuit.circuit_from_netlist(nl, backend="reduce")
px.circuit.compile_circuit(nl, backend="sparse", dedupe=True)   # also evaluates identical instances once
```

`compile_circuit(..., dedupe=True)` evaluates instances that share a model and
identical settings only once per call, for example the hundreds of identical
couplers of an MZI mesh. Settings are compared by value for numbers, tuples and
NumPy arrays, and by identity for JAX arrays and tracers, so gradients still
reach every instance that has its own traced setting.

## Behaviour changes versus the original solver

* **Floating subcircuits.** Instances with no path to any exposed port cannot
  affect the result. They are pruned with a `FloatingSubcircuitWarning`,
  which also removes spurious singularities from decoupled lossless loops.
* **Singular systems.** A genuinely singular system, for example a lossless
  ring with `coupling=0` exactly on resonance, raises `CircuitSingularError`
  (a `ValueError`) on concrete inputs. Inside a JAX trace the affected samples
  come back as NaN with a warning, because exceptions cannot cross a trace.

## Validation

`tests/test_circuit_engine.py` (Level 3, cross-solver evidence):

* the three backends agree with each other, with an explicit
  `(I - S Gamma)^{-1} S` reference, and with closed forms to about 1e-12. The
  circuits covered are the circuit library, imperfect meshes, extracted
  layouts and random non-reciprocal networks;
* Clements meshes up to 32x32 reproduce their Haar-random target unitaries;
* gradients through every backend agree with JAX autodiff of the dense path
  and with central finite differences, including under `jit` and `vmap`;
* lossless reciprocal circuits stay unitary and reciprocal.

Benchmark: `python benchmarks/bench_circuit.py` times the model evaluation,
the original solver and each backend on 4x4 to 32x32 meshes. The example
`examples/large_mesh_engine.py` studies splitter-error tolerance of meshes up
to 32x32.

## Limits

* The sparse backend factorizes on the host (SciPy), so on an accelerator each
  call moves `O(batch nnz)` values to the host and `O(batch m^2)` back.
* Forward-mode AD is not available on the sparse backend.
* The reduction backend's cost grows with the largest intermediate block, so
  it suits cascades and meshes better than densely interconnected networks.
