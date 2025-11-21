# pytensor_orbit.py
import pytensor.tensor as at


def kepler_E(M, e, n_iter: int = 7):
    """Solve Kepler's equation E - e*sin(E) = M with Newton's method."""
    E = M
    for _ in range(n_iter):
        E = E - (E - e * at.sin(E) - M) / (1.0 - e * at.cos(E))
    return E


def true_anomaly(E, e):
    """Convert eccentric anomaly to true anomaly."""
    return 2.0 * at.arctan2(
        at.sqrt(1.0 + e) * at.sin(E / 2.0),
        at.sqrt(1.0 - e) * at.cos(E / 2.0),
    )


def orbit_radius(a, e, f):
    """Radius r = a (1 - e^2) / (1 + e cos f)."""
    return a * (1.0 - e**2) / (1.0 + e * at.cos(f))


def sky_offset(a, e, inc, Omega, omega, f):
    """
    Sky-plane offsets (x, y) in the tangent plane.
    All angles are in radians.
    """
    cosO = at.cos(Omega)
    sinO = at.sin(Omega)
    cosw = at.cos(omega)
    sinw = at.sin(omega)
    cosi = at.cos(inc)

    # Thiele-Innes-like constants; radius handled separately via r
    A =  cosO * cosw - sinO * sinw * cosi
    B =  sinO * cosw + cosO * sinw * cosi
    F = -cosO * sinw - sinO * cosw * cosi
    G = -sinO * sinw + cosO * cosw * cosi

    r = orbit_radius(a, e, f)

    x = r * (A * at.cos(f) + F * at.sin(f))
    y = r * (B * at.cos(f) + G * at.sin(f))

    return x, y


def radec_offset(a, e, inc, Omega, omega, M0, t_obs, P_day, dec0):
    """
    Compute small-angle offsets ΔRA, ΔDEC (radians) at observation times.

    Parameters
    ----------
    a : semi-major axis (same angular units as you want for RA/DEC offsets)
    e : eccentricity
    inc, Omega, omega : orbital angles [rad]
    M0 : mean anomaly at reference epoch [rad]
    t_obs : observation times [day]
    P_day : orbital period [day]
    dec0 : reference declination [rad]
    """
    # Mean anomaly, equivalent to M = 2π (t - t_peri) / P, with t_peri
    # expressed via M0 at the chosen reference epoch.
    M = 2.0 * at.pi * (t_obs - (P_day - M0 / (2.0 * at.pi) * P_day)) / P_day
    M = M % (2.0 * at.pi)

    E = kepler_E(M, e)
    f = true_anomaly(E, e)

    x, y = sky_offset(a, e, inc, Omega, omega, f)

    # ΔRA = x / cos(dec), ΔDEC = y
    dra = x / at.cos(dec0)
    ddec = y

    return dra, ddec


def al_residual_from_radec(dra, ddec, dec0, scan_angle):
    """
    Convert ΔRA/ΔDEC (rad) to Gaia AL residuals (mas).

    Parameters
    ----------
    dra, ddec : offsets in radians
    dec0 : reference declination [rad]
    scan_angle : along-scan angle φ [rad]
    """
    # project RA offset onto local Cartesian x-axis
    dra_proj = dra * at.cos(dec0)

    sin_phi = at.sin(scan_angle)
    cos_phi = at.cos(scan_angle)

    dal_rad = dra_proj * sin_phi + ddec * cos_phi

    # radians → milliarcseconds
    dal_mas = dal_rad * (180.0 / at.pi) * 3600.0 * 1000.0

    return dal_mas
