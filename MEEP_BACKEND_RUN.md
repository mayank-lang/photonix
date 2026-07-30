# Running the Meep backend and full suite

Run commands from the Photonix repository root.

## 1. Environment

The runtime backend needs Meep with bundled MPB. Meep is distributed through
conda-forge for Linux/macOS; use WSL2 on Windows.

```bash
conda create -n photonix -c conda-forge python=3.11 pymeep
conda activate photonix
pip install -e ".[dev]"
python -c "import meep, meep.mpb; print(meep.__version__)"
```

For the MPI build, select the appropriate `pymeep=*=mpi_mpich_*` conda package.

## 2. Backend tests

```bash
pytest tests/test_em_meep.py tests/test_meep_multiport.py -v
```

Without Meep, all pure unit, layout, coordinate, and full-S assembly tests pass;
only the three runtime tests skip. With Meep installed, also verify:

- `test_mpb_neff_matches_fde`: MPB and native full-vector FDE agree within the
  documented discretization tolerance;
- `test_meep_material_grid_constructs`: density-grid realization succeeds;
- `test_meep_fdtd_straight_waveguide_transmits`: a straight guide has near-unit
  modal transmission.

The FDTD integration test requires MPB because `EigenModeSource` and eigenmode
decomposition invoke it.

## 3. Native-layout multimode and multiport matrices

`simulate_multiport_sparameters` expands each physical layout port into modal
terminals and performs every incident run needed for the full square matrix:

```python
from photonix.em import meep

prepared = meep.prepare_layout(
    top,
    [meep.LayerSpec((1, 0), epsilon=3.48**2)],
    margin=1.5,
)
dataset = meep.simulate_multiport_sparameters(
    prepared,
    wavelengths=[1.50, 1.55, 1.60],
    resolution=30,
    port_modes={
        "o1": meep.PortModeSpec((1, 2), monitor_offset=0.2),
        "o2": meep.PortModeSpec((1, 2), monitor_offset=0.2),
        "drop": (1,),
    },
    pml=1.0,
)
assert dataset.ports == ("o1:m1", "o1:m2", "o2:m1", "o2:m2", "drop:m1")
```

The matrix convention is `dataset.s[wavelength, outgoing, incoming]`. Meep band
numbers are one-based. Each mode plane is decomposed with a wave-vector guess
along its local outward normal, so coefficient direction 0 is outgoing and
direction 1 is incident for ports on every side of the device. No reciprocity,
mirror symmetry, or mode-isolation assumption is used.

For quantitative work, pass a separately prepared straight `reference=` layout
with identical port cross-sections and settings. Its incoming coefficient is the
normalization denominator; its complete backward modal vector at the incident
physical port is subtracted as launch background. Without that reference, the
incoming coefficient measured in the device run is used, which can be biased by
strong reflections or a source colocated with its monitor.

## 4. Full verification

```bash
pytest -q
ruff check src tests benchmarks
python -m mypy src/photonix
python -m doctest README.md
python -m build
```

The import/runtime boundary is covered by
`test_meep_backend_is_import_safe_but_runtime_is_guarded`: specifications remain
available without Meep, while object realization and solver calls fail locally
with the installation hint.

## 5. External benchmark

```bash
python benchmarks/run.py --external
```

With Meep installed this fills the MPB SOI-strip mode and Meep width-step columns.
Without it the benchmark reports the external backend as skipped.

## 6. Numerical checks for a real Meep run

Do not treat one passing number as convergence. Repeat relevant runs while varying:

- spatial resolution and PML thickness;
- source/monitor distance and decay threshold;
- reference-plane position (S-parameter phase changes with it);
- modal band/parity and transverse span;
- the optional straight reference used by `normalization_eps`.

For native-layout modal matrices also vary the source and monitor inward offsets,
the modal basis size at every physical port, and the reference geometry. MPB
mode launch/decomposition ignores magnetic, conductive, nonlinear, and dispersive
material terms; propagating power-normalized modes are required before
interpreting `abs(S)**2` as a power fraction. Diagonal port planes are rejected
because the current native-layout adapter constructs axis-aligned Meep DFT planes.

Report the Meep version, backend test output, and the values at each convergence
setting when diagnosing a discrepancy.
