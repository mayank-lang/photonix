# Lossless layout I/O and die-level assembly

Two pieces sit above the device-level layout tools:

* **Lossless GDSII/OASIS I/O** (`photonix.layout.gds_io`): `write_layout` /
  `read_layout` round-trip a whole `Cell` tree, ports and circuit semantics
  through a mask file, and read foreign files (foundry black boxes, KLayout,
  gdsfactory or SiEPIC output, earlier tape-outs) with their hierarchy,
  arrays, units and ports intact.
* **Die assembly**: drawn text and GDS labels (`layout.labels`),
  placement and floorplanning (`layout.placement`), and packaging cells
  (`layout.library.packaging`): fibre-array grating rows with an alignment
  loopback, pad arrays, die frames, and automated-test labels.

The older `read_gds` / `read_oas` (flattened top cell, ports from label
positions only) and `write_gds` / `write_oas` are unchanged. A file written by
`write_gds` reads back with `read_layout` too (tree and geometry intact, ports
inferred from its labels).

## Round trip

```python
import photonix.layout as lay
from photonix.layout import components as lc

art = lay.write_layout(lc.mzi(delta_length=20.0), "mzi.gds")   # + mzi.gds.photonix.json
back = lay.read_layout("mzi.gds")
nl = lay.extract_netlist(back)            # same instances, connections and drawn settings
```

`write_layout` returns a `LayoutArtifact`: file paths, the SHA-256 of the
layout bytes, the Photonix fingerprint of every structure written, the top
structure's name and `snap_error`, the largest distance anything moved when
snapped to the database grid.

### What is preserved, and how

| Item | Mechanism |
|---|---|
| Units | Files are read with `unit=1e-6`, so a file drawn in nm user units comes back in µm. `layout_units(path)` reports the file's unit and grid; a grid coarser than 1 nm warns |
| Top cell | Exactly one top structure, or `top=` (a list of candidates is raised otherwise); `list_top_cells(path)` |
| Hierarchy | One `LabeledCell` per structure, shared by all its references; origin, rotation, x-reflection kept (both formats apply mirror → rotate → translate) |
| Magnification | Photonix references cannot scale: flattened locally into a scaled, `GEOMETRY_ONLY` copy with a warning, or `magnification="error"` |
| Arrays | GDS `AREF` and OASIS repetitions expand into references `<base>_<col>_<row>` (`<base>_<k>` for explicit offsets) |
| Paths | `FlexPath`/`RobustPath` become polygons (except recognised pin markers) |
| Instance names | `SREF` property `INSTANCE_NAME_ATTR` (= 1) |
| Labels | `LabeledCell.labels` ↔ GDS `TEXT`. OASIS `TEXT` has no anchor, rotation or magnification: the sidecar records them and `read_layout` restores them (matched by text, layer and position) |
| Ports | sidecar → PinRec pins → labels on edges, per structure (below) |
| `model`, `model_settings`, original names, `DieCell` floorplan attributes, OASIS label presentation | JSON sidecar only |

### Ports

1. **Sidecar** (`<file>.photonix.json`): exact centre, orientation, width,
   layer and type. Each structure entry also stores a content hash of the
   structure as written: polygons, paths and placements on the integer grid,
   independent of element order, with `TEXT` excluded, plus the content hash
   of every structure it places (recursively). If a structure, or any
   structure below it, is edited after writing, its entry is ignored with a
   `StaleSidecarWarning`, so a model never describes changed sub-cell
   geometry. That structure falls back to markers and becomes `GEOMETRY_ONLY`
   (extraction refuses it), while untouched structures keep their semantics.
   Re-saving a file without changing its geometry, or adding text, keeps the
   sidecar valid.
2. **PinRec** markers (SiEPIC-Tools convention; L. Chrostowski and
   M. Hochberg, *Silicon Photonics Design*, CUP 2015): a two-point `PATH`
   on the pin layer (1/10 optical, 1/11 electrical), 0.1 µm long, centred
   on the port and drawn inside → outside, with the port width as its
   width, plus a `TEXT` holding the port name. Photonix adds the port
   layer (`PORT_LAYER_ATTR`) and the exact orientation
   (`PORT_ORIENTATION_ATTR`) as path properties, so non-Manhattan pins are
   exact. Foreign pins without these properties take the layer of the
   smallest shape under the port and the orientation of the path.
3. **Labels on edges** (`infer_ports`): a `TEXT` within `tol` of an edge of the
   merged geometry on the port layer. Orientation is the outward edge normal;
   its sign comes from the shoelace orientation of the polygon. Width is the
   length of the straight boundary run through the label: collinear edges are
   merged, including the parts of a face split by a hole's keyhole bridge.

### Exactness (evidence)

`tests/test_layout_gds_io.py`:

* For an MZI, a `route_manhattan` circuit, and a circuit with 30°-rotated
  and mirrored references, written as GDS and as OASIS:
  `read_layout(write_layout(c))` has the same `_fingerprint` tree as
  `snap_to_grid(c)`, and `extract_netlist` gives equal instances,
  connections, settings and ports. A JAX gradient through the read-back
  circuit matches the original's to 1e-12 and finite differences to 1e-5.
* A gdstk library drawn in nanometres (`unit=1e-9`) with a 10000 × 500
  rectangle reads back as a 10 µm × 0.5 µm box. The legacy `read_gds` returns
  10000 µm for the same file.
* Arrays (rectangular, and rotated + reflected), and OASIS polygon
  repetitions: the flattened polygons match gdstk's own `flatten()` in count,
  per-layer area and centroid set.
* A GDSII record decoder written in the test file from the stream-format
  record layout checks the bytes on disk independently of gdstk. It
  covers units, structure names, boundary counts and integer areas, `SREF`
  instance-name properties, pin paths with their layer property, and port
  text. For a rotated, reflected `AREF`, each expanded instance origin
  equals `P1 + i (P2 − P1)/cols + j (P3 − P1)/rows`, computed from the
  decoded `COLROW`/`XY` records. When the `klayout` Python package is
  installed, its reader is compared as well.
* Files written by the legacy `write_gds` (labels only, polygons fractured at
  199 vertices) reproduce the snapped cell tree. The flattened geometry
  differs by less than a tenth of a grid step along every edge (XOR area
  ≤ 0.1 × grid × perimeter).
* Two writes give identical bytes and SHA-256 (the GDS timestamp is fixed by
  default). A different timestamp changes the bytes but not the structures.

Limits: `write_layout` snaps vertices, origins, port centres and widths and
label positions to the grid (`snap_to_grid` does exactly the same) and drops
polygons left with zero area. Rotations survive to 1e-9°.

Without a sidecar, the label convention is exact for Manhattan faces. A
non-Manhattan face is limited by its snapped vertices: about
`atan(√2·grid/width)` (0.16° for a 0.5 µm face). Ports inside rotated
references are inferred in the child frame and are exact. The label
convention also cannot recover:

* a port narrower than its face: MMI ports on the body edge read the
  body width, and a corrugated Bragg grating reads the tooth width;
* a port inside a shape (a pad centre);
* a port at a corner: ambiguous, so it is skipped with a warning and kept as
  a label;
* a port on a layer without geometry (a grating coupler's fibre port).

PinRec markers or the sidecar carry all of these.

## Text and labels

* `text(string, size=, layer=(66, 0), position=, anchor="sw", min_stroke=)`
  draws polygon lettering with gdstk's font. Every glyph vertex lies on a
  `size/16` grid, and the narrowest stroke is exactly `size/16`
  (`TEXT_STROKE_RATIO`). The tests recompute this from the glyph polygons:
  the smallest interior vertex-to-edge distance over all 94 printable
  characters. `min_stroke` raises `size` to `16 × min_stroke`, so the text
  passes `check_min_width(min_stroke)`.
* `Label` (frozen) and `LabeledCell` (`Cell` + `labels`, with `add_label`,
  `get_labels`, `flatten`, `from_cell`) carry GDS `TEXT`.

## Placement and floorplanning

* `grid(cells, shape=, spacing=, align=)`: uniform pitch (largest cell plus
  spacing), reading order, and each bounding box's `align` point on its
  slot's. With `"c"` the box centres lie exactly on the lattice (tested).
* `pack(cells, max_size=, spacing=, allow_rotation=False)`: MaxRects with
  Best-Short-Side-Fit (J. Jylänki, *A Thousand Ways to Pack the Bin*,
  2010). Boxes are inflated by `spacing` and sorted by decreasing area, and
  a new bin opens when one is full. Tested: n² equal squares fill an n·s
  bin completely (n = 3, 4, 5), eight 10 × 20 dominoes tile 40 × 40, and
  40 random boxes have no pairwise overlap (spacing included), stay inside
  the bin and pack deterministically. Rotation is opt-in and quarter-turn
  only, because it rotates ports.
* `align(refs, edge)` and `distribute(refs, axis, spacing)` work on exact
  bounding boxes (`reference_bbox`), which come from transformed vertices,
  so rotated cells are correct.
* `DieCell` + `check_floorplan(top, die)` report four kinds of issue
  (`FloorplanIssue`):
  * `outside_die`: a top-level block leaves the die;
  * `keepout`: any leaf instance, or polygons drawn in a composite, enters
    the dicing lane or the keep-out band. A facet device (`facet_models`) may
    cross the band only towards the edge its `o2` faces; inside another
    side's lane it is reported like any leaf;
  * `overlap`: two top-level blocks overlap (routes excluded unless asked);
  * `facet`: an edge coupler's `o2` misses the facet by more than
    `facet_tol`, faces a non-facet side, or faces no edge.

  All checks run in die coordinates: when `top` places the frame (e.g. a
  die centred on the origin), that reference's transform is undone first,
  and issue boxes are reported in the frame's coordinates.

## Packaging cells (`photonix.layout.library`)

* `fiber_array(n, pitch=127, loopback=True, ...)` puts the fibre ports at
  exactly `(k·pitch, 0)`, with the waveguide ports `o1..` facing the device
  side, on the inner channels when there is a loopback. The loopback joins
  channels 0 and n−1. It runs around the coupler row, so it crosses no
  device waveguide, keeps ≥ `clearance` from every coupler (tested), and
  its length is closed form: `Σ segments − 12d + 6·L_bend`. The test
  derives this from the coupler geometry and checks it against the
  recorded route length (1e-12) and the drawn polygon area / width (1e-4).
  Extraction raises no warnings and connects exactly the two loopback
  joints. The simulated loopback transmission equals coupler × waveguide ×
  coupler to 1e-10.
* `pad_array(n, pitch, size)`: bond pads with electrical ports `e1..en`.
* `die_frame(size, dicing_lane=, keepout=, die_id=, corner_marks=, facet_sides=, facet_trench_layer=)`
  returns a `DieCell`. The lane area is exactly `WH − (W−2d)(H−2d)` and the
  keep-out band area exactly the analogous difference (tested). Corner marks
  and the die ID stay out of the usable area. With a facet trench, the
  facet is its inner wall.
* `add_test_labels(top, gc_refs=, device=, user=)` writes
  `opt_in_<TE|TM>_<nm>_device_<user>_<device>` on layer 10/0 at the input
  coupler's fibre port. This is the SiEPIC/edX automated-test convention.
  It checks that the other couplers sit on the 127 µm pitch line and
  validates the names. Confirm the current rules with the test facility.

## Limits and what is not here

* The marker layers (frame 99/0, keep-out 98/0, corner marks 99/1, text
  66/0, trench marker) are placeholders. Map them to the foundry layer map.
  A facet trench drawn here is a marker, not a foundry-qualified deep-etch
  rule.
* `fiber_array` does not fan the channels out to the device pitch. Route
  them with `route_manhattan`, or with the bundle router once it is merged.
* `check_floorplan` works on bounding boxes. A bent or L-shaped leaf can be
  flagged near the keep-out even where its polygons do not enter it. Use the
  DRC pre-check or the foundry deck for polygon-exact sign-off.
* The sidecar is Photonix-specific. Other tools see a normal GDS with
  SiEPIC-style pins and `TEXT` labels.

Example: [`examples/die_assembly.py`](../examples/die_assembly.py) builds a
die (frame, fibre array with loopback, MZIs, an edge-coupled waveguide, packed
rings, test label), checks the floorplan, writes GDS + sidecar, reads it back
and simulates the read-back layout.
