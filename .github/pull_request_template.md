## What and why

<!-- One logical change. Link the issue it closes, if any: "Closes #123". -->

## Validation

<!-- How do you know it is right? An analytic limit, a published value (cite it),
     an independent solver, or a convergence study, with the tolerance achieved.
     State the evidence level per docs/VALIDATION.md. -->

## Checklist

- [ ] Tests added or updated; `python -m pytest` passes
- [ ] Physics or algorithm changes are validated against an analytic limit, a publication or an independent computation (evidence above)
- [ ] Safe on the NumPy backend: `PHOTONIX_BACKEND=numpy python -m pytest` passes, and JAX-only tests use `requires_jax`
- [ ] No new hard dependency; optional packages are imported lazily at the call boundary
- [ ] `ruff check src tests benchmarks examples tools` and `python -m mypy src/photonix` are clean
- [ ] Public API: docstring with a runnable `Examples` block, name listed in `__all__`
- [ ] Docs updated (README / `docs/`) where behaviour changes
- [ ] `CHANGELOG.md` entry under `## [Unreleased]`
