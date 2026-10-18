"""Physical constants in SI units."""

PLANCK = 6.62607015e-34
LIGHT_SPEED = 299792458.0


def photon_energy(wavelength_m: float) -> float:
    return PLANCK * LIGHT_SPEED / wavelength_m

