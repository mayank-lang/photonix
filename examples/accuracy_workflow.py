"""A lightweight, evidence-oriented Photonix simulation workflow."""
from __future__ import annotations

import json

import photonix as px
import photonix.em as em

# Anchor a discretized solver to an independent analytic solution.  Three
# aligned grids make this a clean second-order example; adaptive_convergence
# reports a safety-factored Grid Convergence Index rather than only comparing
# the final pair of meshes.
study = em.adaptive_convergence(
    lambda resolution: em.slab_neff(
        thickness=0.20,
        polarization="te",
        resolution=resolution,
        richardson=False,
    ),
    initial_resolution=30,
    refinement=2.0,
    max_levels=3,
    rtol=5e-4,
)
analytic = em.slab_neff_analytic(thickness=0.20, polarization="te")
extrapolated = study.extrapolated.item()
relative_error = abs(extrapolated - analytic) / analytic

print(f"slab TE0 analytic n_eff: {analytic:.8f}")
print("resolutions:", study.resolutions)
print("numerical n_eff:", study.values.tolist())
print(f"Richardson n_eff: {extrapolated:.8f}")
print(f"GCI / analytic error: {study.grid_convergence_index:.3e} / {relative_error:.3e}")
print(f"observed order: {study.observed_order:.3f}; converged: {study.converged}")
if not study.converged:
    raise RuntimeError("slab effective index did not reach the requested grid accuracy")

# Physicality is a coherent matrix property; checking every coefficient alone
# cannot establish passivity for a multiport network.
s_parameters = px.components.directional_coupler(coupling=0.5)
quality = px.analyze_sparameters(s_parameters)
print(
    "coupler physicality:",
    {"passive": quality.passive, "reciprocal": quality.reciprocal, "lossless": quality.lossless},
)

# Store this alongside geometry, material provenance, solver settings, and the
# complete convergence data in a real project.
print("runtime manifest:")
print(json.dumps(px.runtime_info().as_dict(), indent=2, sort_keys=True))
