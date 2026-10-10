# Tutorial: from layout to yield

This walk-through follows one design -- a ring filter between two grating
couplers -- through placement, routing, rule checks, GDSII, extraction,
simulation, spectral analysis and parametric yield. Every step uses the same
port and model conventions, so nothing is re-entered by hand. A runnable
version is [`examples/layout_to_circuit.py`](../examples/layout_to_circuit.py).

## 1. Cells know what they simulate

Every layout generator records the compact model its geometry represents:

```python
import photonix as px
import photonix.layout as lay
from photonix.layout import components as lc

wg = lc.straight(25.0)
wg.model, wg.model_settings          # ('straight', {'length': 25.0})
ring = lc.ring_single(radius=12.0, gap=0.28)
ring.model, ring.model_settings      # ('ring_single', {'radius': 12.0, 'gap': 0.28, 'length_x': 0.0})
```

`extract_netlist` copies `model_settings` into the netlist, so a 25 µm straight
simulates as 25 µm, a ring as its drawn gap and radius, and a route as its drawn
centre-line length. Electrical ports (`port_type="electrical"`, e.g. heater
contacts) never become optical terminals.

## 2. Place by ports and route with real bends

```python
pdk = px.pdk.demo_pdk()
top = lay.Cell("chip")
ring = top.add_ref(pdk.get_layout("ring_single", radius=12.0, gap=0.28), name="ring")
gc_in = top.add_ref(pdk.get_layout("grating_coupler"), name="gc_in")
gc_in.connect("o1", lay.Port("p", (-63.5, -60.0), 270.0))       # mate o1 to a target port
top.add_ref(pdk.route(gc_in.port("o1"), ring.port("o1"), bend="euler"), name="r_in")
```

`Reference.connect(port, destination)` places a reference so its port sits on
the destination and faces it. `route_manhattan` (and `pdk.route`, which applies
the PDK cross-section's width, layer and radius) fillets every corner with a
circular or Euler bend, picks a straight/L/Z/U topology automatically, accepts
explicit `waypoints`, and raises `RoutingError` with the reason when the bends
do not fit.

Composite cells (references and no `model`) are flattened during extraction,
so a hierarchical layout such as `lc.mzi()` simulates every placed bend and
straight.

## 3. Check rules and write the mask

```python
print(pdk.drc_precheck(top))          # width/space/area per the PDK rules
lay.write_gds(top, "chip.gds")        # distinct cells never collide by name
```

The pre-check finds narrow features, gaps and small islands in seconds. It is
not sign-off: run the foundry deck with `lay.run_drc` for that.

## 4. Simulate the extracted layout

```python
wl = px.linspace(1.540, 1.560, 40001)
s = pdk.simulate(top, wl=wl)                       # extraction + PDK models
```

`pdk.simulate` uses `pdk.models()`, which binds the PDK cross-section's
`neff`, `ng`, `loss_db_cm` and `wl0` into every model that declares them as
waveguide parameters, so routed waveguides carry the kit's loss rather than the
lossless library default.
Call-time overrides still win: `pdk.simulate(top, wl=wl, r_in={"loss_db_cm": 0})`.

The netlist can be stored and reloaded losslessly:

```python
nl = lay.extract_netlist(top)
nl.to_json("chip.netlist.json")
px.circuit.Netlist.from_json("chip.netlist.json") == nl     # True
```

## 5. Analyse the spectrum

```python
from photonix import analysis as an
T = px.to_numpy(px.power(pdk.evaluate("ring_single", wl=wl, radius=12.0, gap=0.28)[("o1", "o2")]))
res = an.find_resonances(px.to_numpy(wl), T)
r = an.fit_lorentzian(px.to_numpy(wl), T, center=res[0].wavelength, window=5e-5)
under, over = an.ring_q_factors(r)          # Q_i / Q_c for both coupling regimes
an.free_spectral_range(res), r.q_loaded, r.extinction_db
```

`passband_metrics`, `bandwidth` and `channel_crosstalk` cover filters and
demultiplexers; `impulse_response` and `pulse_response` give delay and
dispersion-induced broadening; `LinkBudget` totals a link and its margin.

## 6. Estimate parametric yield

```python
from photonix.pdk import MonteCarloSpec, ProcessCorner, ProcessStudy

study = ProcessStudy(
    ProcessCorner("nominal", {"gap": 0.28}),
    monte_carlo=MonteCarloSpec.independent({"gap": 0.28}, {"gap": 0.01}),
    units={"gap": "um"},
)

def evaluate(case):
    s = pdk.evaluate("ring_single", wl=wl, radius=12.0, gap=case.parameters["gap"])
    return px.to_numpy(px.power(s[("o1", "o2")]))

spec = an.Spec("extinction", lambda T: an.find_resonances(px.to_numpy(wl), T)[0].extinction_db,
               minimum=15.0, unit="dB")
result = an.yield_analysis(evaluate, [spec], study=study, samples=200, seed=1)
print(result)        # yield with a Wilson confidence interval, per-spec yield, corner pass/fail
```

The process values here are illustrative; production distributions must come
from the target foundry (see [`PIC_COMPLETENESS.md`](PIC_COMPLETENESS.md)).

## Where to go next

* [`COMPONENT_LIBRARY.md`](COMPONENT_LIBRARY.md) -- every model, its physics,
  evidence and limits.
* [`examples/awg_demux.py`](../examples/awg_demux.py),
  [`examples/programmable_mesh.py`](../examples/programmable_mesh.py),
  [`examples/bragg_and_ring_modulator.py`](../examples/bragg_and_ring_modulator.py).
