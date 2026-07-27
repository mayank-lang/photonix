# photonix benchmark results

| case | quantity | photonix | reference | |Δ| | status | source |
|---|---|---|---|---|---|---|
| soi_strip_te0_neff | n_eff | 2.4490 | 2.4400 | 0.0090 | PASS | Si 500x220 nm, oxide cladding, 1550 nm — standard SOI strip TE0 (e.g. Lumerical MODE / gdsfactory PDK references) |
| soi_strip_tm0_neff | n_eff | 1.7891 | 1.7800 | 0.0091 | PASS | same strip, TM0 — accepted range ~1.75-1.80 |
| soi_bend_loss_r1_db90 | dB/90deg | 0.0076 | - | - | - | (no reference) |
| width_step_te_T | |T00|^2 | 0.9932 | 0.9927 | 0.0005 | PASS | internal FDFD full-wave cross-check (photonix waveguide_sparams) |
| width_step_tm_T | |T00|^2 | 0.9990 | 0.9967 | 0.0023 | PASS | internal TM FDFD full-wave cross-check |

**2/2 external references within tolerance**, 2/2 internal FDFD cross-checks pass, 1 case has no reference (5 total).
