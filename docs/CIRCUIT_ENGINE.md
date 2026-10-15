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
| `sparse` | SuperLU on `A = I - S Gamma` through a host callback, with implicit-function derivatives (`sparse_linalg.py`) | all JAX transforms, forward and reverse, higher order (`jax.hessian`) | `O(batch (nnz + m^2))` memory; `O(batch n m)` more under differentiation |
| `reduce` | subnetwork growth: pairwise `innerconnect` on small dense blocks in greedy minimum-size order (`reduction.py`) | all JAX transforms; stays on device | set by the largest intermediate block |
| `auto` (default) | `dense` up to `DENSE_MAX_TERMINALS = 300` terminals while the batched system fits in `DENSE_MAX_BYTES`, else `sparse` | as chosen | |

`auto` decides from the sizes alone, never from whether the values are JAX
tracers. Every backend supports every transformation, so code that uses
`jax.jacfwd`, `jax.jvp` or `jax.hessian` (for example
`photonix.analysis.fosm(..., second_order=True)`) works on any circuit size.

The sparse derivative is exact. The primal solve is a host callback returning
only the `m x m` block. Its `custom_jvp` tangent is
`dX = A^{-1} (dS P_E + dS Gamma X)`, with `A^{-1}` applied by
`jax.lax.custom_linear_solve` (SuperLU callbacks for the solve and the
transposed solve). JAX obtains reverse mode by transposing this linear
tangent. The solve, the tangent solve and the transposed solve of one
differentiated call share one LU factorization per sample.

The `reduce` backend is an alternative to `auto` when nothing may leave the
device. Under `jit` it unrolls the whole elimination into XLA, so a 16x16
mesh takes minutes to compile there, against seconds for `sparse`. Choose a
backend explicitly with:

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
  (a `ValueError`) on concrete inputs. Every backend applies this, including
  `dense` under JAX: `jnp.linalg.solve` would return a finite but meaningless
  value there. Instead the engine factors explicitly and applies the same
  relative-pivot test as the sparse backend (`PIVOT_RTOL = 1e-14`). Inside a
  JAX trace no exception can be raised: the sparse and reduce backends return
  NaN for the affected samples (sparse also warns), and `dense` returns
  whatever the LU solve gives.

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
  call moves `O(batch nnz)` values to the host and `O(batch m^2)` back. Under
  differentiation, the full `n x m` solution and the tangent or cotangent
  solves (`O(batch n m)`) also cross the host boundary.
* The reduction backend's cost grows with the largest intermediate block, so
  it suits cascades and meshes better than densely interconnected networks.
* The reduction backend assumes passive blocks. When a partial sub-network is
  lossless or active and resonant, a pairwise `innerconnect` denominator can
  vanish even though the whole circuit is regular. On concrete inputs that
  call is then re-solved with the global sparse solve, which raises
  `CircuitSingularError` only if `I - S Gamma` itself is singular. Inside a JAX
  trace the affected samples come back as NaN.
