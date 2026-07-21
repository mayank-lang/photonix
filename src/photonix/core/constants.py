"""Physical constants and default simulation settings (SI units).

photonix uses SI internally. Wavelengths and lengths in public APIs are in
**micrometers (um)** by convention (the natural scale for integrated photonics),
unless a function explicitly documents otherwise. Use :mod:`photonix.core.units`
to convert.
"""
from __future__ import annotations

import math

# Fundamental constants (CODATA 2018, exact where defined) -------------------- #
C0: float = 299_792_458.0          # speed of light in vacuum [m/s]
MU0: float = 1.25663706212e-6      # vacuum permeability [H/m]
EPS0: float = 8.8541878128e-12     # vacuum permittivity [F/m]
ETA0: float = 376.730313668        # vacuum impedance [ohm]
H_PLANCK: float = 6.62607015e-34   # Planck constant [J*s]
HBAR: float = H_PLANCK / (2.0 * math.pi)
Q_E: float = 1.602176634e-19       # elementary charge [C]

# Convenience: speed of light in um/s and related photonics scales ------------ #
C0_UM_S: float = C0 * 1e6          # [um/s]

# Default wavelength references (um) ----------------------------------------- #
WL_C_BAND: float = 1.55            # telecom C-band center [um]
WL_O_BAND: float = 1.31            # telecom O-band center [um]
WL_DEFAULT: float = WL_C_BAND

# Common material refractive indices at 1.55 um (nondispersive defaults) ------ #
# These are *defaults* for convenience; dispersive models live in
# photonix.modes.materials.
N_SI: float = 3.4757               # crystalline silicon @1.55um
N_SIO2: float = 1.444             # silicon dioxide (cladding) @1.55um
N_SIN: float = 1.9963             # stoichiometric silicon nitride @1.55um
N_AIR: float = 1.0
N_GROUP_SI_STRIP: float = 4.2     # typical 220nm SOI strip waveguide n_g

__all__ = [
    "C0", "MU0", "EPS0", "ETA0", "H_PLANCK", "HBAR", "Q_E", "C0_UM_S",
    "WL_C_BAND", "WL_O_BAND", "WL_DEFAULT",
    "N_SI", "N_SIO2", "N_SIN", "N_AIR", "N_GROUP_SI_STRIP",
]
