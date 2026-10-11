# Tutorial: from layout to yield

This walk-through follows one design -- a ring filter between two grating
couplers -- through placement, routing, rule checks, GDSII, extraction,
simulation, spectral analysis and parametric yield, then synthesizes a
filter from a specification and ranks what limits it. Every step uses the same
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

## 7. Synthesize a filter from a specification

`photonix.synthesis` goes in the other direction: from a target response to
component parameters. `synthesize_crow` designs an N-ring coupled-resonator
band-pass filter (Butterworth or Chebyshev).

1. Little et al.'s closed-form couplings give the starting point.
2. These are refined until the simulated lossless drop response equals the
   exact periodic target over the whole FSR, to about 1e-10 dB.
3. The couplings are kept symmetric (`K_i = K_{N-i}`).
4. Wide bands are reached by continuation from a narrow one.

```python
import numpy as np
from photonix import synthesis as sy

d = sy.synthesize_crow(4, 100e9, kind="chebyshev", ripple_db=0.2, radius=10.0)
d.couplings          # (0.3453, 0.0446, 0.0299, 0.0446, 0.3453)
d.max_error_db       # ~4e-11: worst deviation from the target, netlist simulation
d.radius, d.fsr_hz   # 9.9704 um (rounded so wl0 is resonant), 1.139e12 Hz

wl = d.wavelength(np.linspace(-300e9, 300e9, 4001))
T = px.to_numpy(px.power(d.model()(wl=wl)[("o1", "o3")]))
an.passband_metrics(wl, T).ripple_db         # 0.2

gaps = [sy.coupling_to_gap(k, radius=d.radius, length=6.0) for k in d.couplings]
# [0.1398, 0.259, 0.2821, 0.259, 0.1398] um for 6 um racetrack couplers
```

`coupling_to_gap` inverts the calibrated `evanescent_coupler` model with Brent's
method, and includes the curved-gap contribution of the ring. It refuses a
target that the given coupler length cannot reach, or a coupler that is
over-coupled inside the gap bracket. `refine=False` returns the closed-form
couplings, which shows how far off the narrow-band approximation is. The
synthesis is lossless. With propagation loss, `d.model(loss_db_cm=...)`
shows the resulting insertion loss and pass-band rounding.

FIR filters (interleavers, band splitters and flat-top MZI chains) are
synthesized with `synthesize_lattice`. It peels a power-complementary
polynomial pair into coupler ratios and heater phases exactly:

```python
il = sy.interleaver(3, fsr_hz=100e9)   # 50 GHz odd/even interleaver, maximally flat
il.couplings, il.delays                # (0.9889, 0.2297, 0.8547, 0.5), (2, 2, 1) x 713.65 um
S = il.model()(wl=il.wavelength(np.linspace(-50e9, 50e9, 1001)))   # circuit.lattice_filter
```

`equiripple_lowpass` gives minimax targets, and `complementary_bar` /
`spectral_factor` supply the partner of any cross response with `|B| <= 1`.
The algorithm, validation evidence and limits of both synthesis routes are in
[`SYNTHESIS.md`](SYNTHESIS.md).

## 8. Find what matters and how far the design is from failing

Monte Carlo yield tells you *whether* a design passes. The deterministic tools
in `photonix.analysis` tell you *why*, using far fewer simulations. All of them
take a function of a parameter dictionary and use JAX gradients when the
function can be traced, or Richardson-extrapolated finite differences when it
cannot.

```python
def centre_loss_db(p):                       # drop loss at the centre of a 4-ring filter
    k = (p["k0"], p["k1"], p["k2"], p["k1"], p["k0"])
    s = px.circuit.crow(couplings=k, radius=d.radius, neff=d.neff, ng=d.ng, loss_db_cm=0.0)
    return -10 * px.xp.log10(px.power(s(wl=d.wavelength(0.0))[("o1", "o3")]))

nominal = dict(zip(["k0", "k1", "k2"], d.couplings[:3]))       # `d` from section 7
t = an.sensitivity(centre_loss_db, nominal)
t.method, t.ranked()        # 'ad', [('k0', -11.4), ('k1', 9.4), ('k2', -4.7)]  (elasticities)

spec = an.Spec("centre loss", lambda v: v, maximum=0.5, unit="dB")
cov = [(0.05 * v) ** 2 for v in nominal.values()]               # 5 % (1 sigma) coupling error
wc = an.worst_case_distance(centre_loss_db, spec, nominal, cov)
wc.beta, wc.yield_linear    # 1.50 sigma to the nearest failing point; ~93 % linearized yield
```

* `sensitivity` returns gradients and dimensionless elasticities
  `(p / f) df/dp`, ranked by magnitude.
* `fosm` propagates a covariance (a matrix, a list of variances or a
  `MonteCarloSpec`) to the metric's mean and standard deviation. With
  `second_order=True` it adds the Hessian mean shift.
* `corners` builds full-factorial, Plackett-Burman (orthogonal two-level
  screening), axial or gradient-sign process corners as `ProcessCorner`s.
  They plug straight into `yield_analysis(..., cases=...)`.
* `worst_case_distance` finds the most probable failing point (in sigmas) for
  one spec. It runs constrained optimization from several starts in whitened
  coordinates.
* `importance_sampling_yield` centres the samples on that worst-case point,
  with a defensive mixture. It resolves failure rates of 1e-4 and below,
  where crude Monte Carlo with the same budget sees no failures at all.

## Where to go next

* [`COMPONENT_LIBRARY.md`](COMPONENT_LIBRARY.md) -- every model, its physics,
  evidence and limits.
* [`examples/awg_demux.py`](../examples/awg_demux.py),
  [`examples/programmable_mesh.py`](../examples/programmable_mesh.py),
  [`examples/bragg_and_ring_modulator.py`](../examples/bragg_and_ring_modulator.py).
