---
name: Validation discrepancy
about: A photonix result disagrees with an analytic value, a publication, measurement or another solver
title: "[validation] "
labels: validation
---

## Quantity

What disagrees (for example: TE0 n_eff of a 500 x 220 nm SOI strip at 1.55 um,
or the drop-port extinction of `ring_double`).

## photonix value

The call that produced it (all non-default arguments, including grid resolution
and domain size) and its output:

```python
```

## Reference value and source

- Value:
- Source: paper and DOI / analytic formula / measurement / solver and version
- Reference uncertainty, if known:

## Tolerance and evidence level

- Discrepancy (absolute and relative):
- Tolerance you expected and why:
- Evidence level of the photonix model (see `docs/COMPONENT_LIBRARY.md` /
  `docs/VALIDATION.md`):
- Convergence evidence (for example `photonix mode ... --converge` output, or a
  resolution sweep):

## Environment

Output of `photonix info --json` and `photonix validate --json`:

```json
```
