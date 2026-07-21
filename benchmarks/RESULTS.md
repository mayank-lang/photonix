# photonix benchmark results

| case | quantity | photonix | reference | |Δ| | status | source |
|---|---|---|---|---|---|---|
| soi_strip_te0_neff | n_eff | 2.4490 | 2.4400 | 0.0090 | PASS | Si 500x220 nm, oxide cladding, 1550 nm — standard SOI strip TE0 (e.g. Lumerical MODE / gdsfactory PDK references) |
| soi_strip_tm0_neff | n_eff | 1.7891 | 1.7800 | 0.0091 | PASS | same strip, TM0 — accepted range ~1.75-1.80 |
| soi_bend_loss_r1_db90 | dB/90deg | 0.0076 | - | - | - | (no reference) |
| width_step_te_T | |T00|^2 | 0.9934 | 0.9927 | 0.0007 | PASS | internal FDFD full-wave cross-check (photonix waveguide_sparams) |
| width_step_tm_T | |T00|^2 | 0.9992 | 0.9967 | 0.0025 | PASS | internal TM FDFD full-wave cross-check |

**4/4 cases within tolerance** (5 total).
