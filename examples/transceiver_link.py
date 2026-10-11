"""Example: a 25 Gb/s NRZ optical link simulated in the time domain.

PRBS9 drive -> CW laser (linewidth, RIN) -> traveling-wave Mach-Zehnder
modulator -> 2 mm waveguide and an add-drop ring filter (simulated sample by
sample by ``TimeDomainCircuit``, with the ring converted to its physical
couplers and delay lines) -> PIN photodiode with shot and thermal noise ->
Bessel-Thomson receiver -> eye diagram, Q factor, Gaussian BER and an eye-mask
test.

Left: optical power after the modulator and after the ring (the ring's photon
lifetime slows the edges). Middle: the ring's drop response from the
frequency-domain circuit solver and the response actually implemented by the
time-domain network (they agree to the interpolation error). Right: the
received eye with the mask.

Run:  python examples/transceiver_link.py
Saves: examples/outputs/transceiver_link.png
"""
from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from _output import save

import photonix as px
from photonix.circuit import Netlist
from photonix.core.constants import C0_UM_S
from photonix.timedomain import (
    AnalogFilter,
    EyeMask,
    TimeDomainCircuit,
    TravelingWaveElectrode,
    cw_laser,
    drive_waveform,
    eye_diagram,
    mask_test,
    mzm,
    photodiode,
    prbs,
    push_pull,
)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"     # categorical slots 1-3 (validated)
INK, MUTED = "#0b0b0b", "#52514e"

BAUD, SPS = 25e9, 32
DT = 1.0 / (BAUD * SPS)                                     # 1.25 ps: envelope band +/- 400 GHz


def channel() -> Netlist:
    """2 mm access waveguide into the drop port of a ring resonant at 1.55 um.

    The ring (m = 420, R = 43 um, FSR ~ 263 GHz) is large enough that its
    half-ring delay (1.5 samples) exceeds the time step and its FSR lies inside
    the envelope band -- the conditions for a faithful sample-mode model.
    """
    radius = 420 * 1.55 / 2.4 / (2 * math.pi)                # neff L = m wl0: a resonance at the carrier
    nl = Netlist(name="drop_filter")
    nl.add("wg", "straight", length=2000.0, loss_db_cm=2.0)
    nl.add("ring", "add_drop_ring", radius=radius, coupling1=0.3, coupling2=0.3, loss_db_cm=2.0)
    nl.connect(("wg", "o2"), ("ring", "o1"))
    nl.expose("in", ("wg", "o1"))
    nl.expose("drop", ("ring", "o3"))
    return nl


def main() -> None:
    bits = prbs(9, 1022)
    n = bits.size * SPS

    # transmitter: laser and traveling-wave MZM biased at quadrature, 0.8 V_pi swing
    laser = cw_laser(1e-3, n, DT, linewidth_hz=1e6, rin_db_hz=-150.0, seed=1)
    drive = drive_waveform(bits, baud=BAUD, samples_per_symbol=SPS, amplitude_v=2.4, offset_v=-1.5,
                           rise_time_20_80_s=12e-12)
    electrode = TravelingWaveElectrode(length_mm=3.0, n_m=3.9, n_g_opt=4.2, alpha_db_per_cm_sqrt_ghz=0.5)
    tx = mzm(laser, *push_pull(drive), v_pi=3.0, insertion_loss_db=4.0, extinction_db=30.0, electrode=electrode)

    # optical channel, stepped sample by sample
    nl = channel()
    tdc = TimeDomainCircuit.from_netlist(nl, carrier_wl=1.55, dt=DT)
    rx_field = tdc.simulate({"in": tx})["drop"]

    # receiver
    current = photodiode(rx_field, responsivity_a_w=0.9, noise=("shot", "thermal"),
                         bandwidth=AnalogFilter.bessel(4, 0.75 * BAUD), seed=2)
    eye = eye_diagram(current, baud=BAUD, symbols=bits, skip_symbols=8)
    mask = EyeMask()
    res = mask_test(eye, mask)
    print(f"Q = {eye.q_factor:.2f}, BER_gauss = {eye.ber_gaussian:.2e}, ER = {eye.extinction_ratio_db:.2f} dB, "
          f"OMA = {eye.oma * 1e3:.3f} mA, channel delay = {eye.delay_symbols} UI, "
          f"mask hits = {res.hits}, margin = {res.margin:.2f}")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    # (a) optical power before and after the ring
    ax = axes[0]
    k = slice(40 * SPS, 64 * SPS)
    t_ns = (np.arange(n) * DT * 1e9)[k]
    lag = int(round(tdc.describe()[0].get("delay_samples", 0.0)))          # waveguide delay, for alignment
    p_tx = np.asarray(tx.power())[k] * 1e3
    p_rx = np.asarray(rx_field.power())[k.start + lag:k.stop + lag] * 1e3
    ax.plot(t_ns, p_tx, color=BLUE, lw=1.5, label="after the MZM")
    ax.plot(t_ns, p_rx, color=ORANGE, lw=1.5, label="ring drop port")
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Optical power (mW)")
    ax.set_ylim(0.0, 1.3 * max(p_tx.max(), p_rx.max()))
    ax.set_title("Transmitter and filter", color=INK)
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)

    # (b) frequency-domain model vs the discretized time-domain network
    ax = axes[1]
    nu = np.linspace(-150e9, 150e9, 601)
    wl = C0_UM_S / (C0_UM_S / 1.55 + nu)
    s_fd = np.asarray(px.circuit.circuit_from_netlist(nl)(wl=wl)[("in", "drop")])
    s_td = tdc.discrete_response(nu)[("in", "drop")]
    ax.plot(nu / 1e9, 10 * np.log10(np.abs(s_fd) ** 2), color=BLUE, lw=2.0, label="frequency domain")
    ax.plot(nu / 1e9, 10 * np.log10(np.abs(s_td) ** 2), color=AQUA, lw=1.2, ls="--", label="time-domain network")
    ax.set_xlabel("Offset from carrier (GHz)")
    ax.set_ylabel("Drop transmission (dB)")
    ax.set_title(f"Ring drop response (max |dS| = {np.max(np.abs(s_fd - s_td)):.1e})", color=INK)
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    # (c) the received eye and the mask (normalized mask coordinates mapped onto the levels, in mA)
    ax = axes[2]
    eye.plot(ax=ax, color=BLUE, alpha=0.06, scale=1e3, unit="mA")
    mu = eye.levels * 1e3
    poly = mask.polygon()                            # UI 0..1 is the eye's first UI (centred at 0.5)
    ax.fill(poly[:, 0], mu[0] + poly[:, 1] * (mu[-1] - mu[0]), facecolor="none", edgecolor=MUTED,
            hatch="///", lw=1.0)
    ax.set_title(f"Received eye: mask hits {res.hits}, margin {res.margin:.2f}", color=INK)
    for a in axes:
        a.grid(True, color="#e5e4e0", lw=0.6)
        a.tick_params(colors=MUTED)
    fig.tight_layout()
    print("saved", save(fig, "transceiver_link.png"))


if __name__ == "__main__":
    main()
