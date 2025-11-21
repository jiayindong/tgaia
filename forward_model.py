#forward model.py
import numpy as np
import copy

# main.py needed
def t_p_to_init_phase(t_p, period):
    """Convert time of periastron to initial phase"""
    init_phase = -(t_p / period) * (2 * np.pi)
    return init_phase

def true_anomaly_calc(t, period, eccentricity, t_p): 
    """Calculate true anomaly from mean anomaly using Kepler's equation"""
    mean_anomaly = (((t - t_p) / period) * 2 * np.pi) % (2 * np.pi)
    steps = 6
    eta = copy.copy(mean_anomaly)
    for ii in range(steps):
        eta -= (eta - eccentricity * np.sin(eta) - mean_anomaly) / (1 - eccentricity * np.cos(eta))
    true_anomaly = 2 * np.arctan(np.sqrt((1 + eccentricity) / (1 - eccentricity)) * np.tan(eta / 2))
    return true_anomaly

def position(semimajor, true_anomaly, eccentricity):
    """Calculate orbital radius from semi-major axis, true anomaly, and eccentricity"""
    rad = semimajor * (1 - eccentricity ** 2) / (1 + eccentricity * np.cos(true_anomaly))
    return rad

def position3d(rad, true_anomaly, inclination, Omega, omega): #2D -> 3D
    """Convert 2D orbital position to 3D Cartesian coordinates with orbital angles"""
    x = rad * np.cos(true_anomaly)
    y = rad * np.sin(true_anomaly)
    
    x_new = (x * (np.cos(Omega * np.pi / 180) * np.cos(omega * np.pi / 180) 
             - np.sin(Omega * np.pi / 180) * np.cos(inclination * np.pi / 180) * np.sin(omega * np.pi / 180))
             + y * (-np.cos(Omega * np.pi / 180) * np.sin(omega * np.pi / 180) 
                    - np.sin(Omega * np.pi / 180) * np.cos(inclination * np.pi / 180) * np.cos(omega * np.pi / 180)))
    
    y_new = (x * (np.sin(Omega * np.pi / 180) * np.cos(omega * np.pi / 180) 
             + np.cos(Omega * np.pi / 180) * np.cos(inclination * np.pi / 180) * np.sin(omega * np.pi / 180))
             + y * (-np.sin(Omega * np.pi / 180) * np.sin(omega * np.pi / 180) 
                    + np.cos(Omega * np.pi / 180) * np.cos(inclination * np.pi / 180) * np.cos(omega * np.pi / 180)))
    
    z_new = x * np.sin(inclination * np.pi / 180) * np.sin(omega * np.pi / 180) + y * np.sin(inclination * np.pi / 180) * np.cos(omega * np.pi / 180)
    return [x_new, y_new, z_new] #orbital element rotations


def projected_position(t, semimajor, period, eccentricity, inclination, Omega, omega, t_p):
    """Get 2D projected position on sky from orbital parameters"""
    true_anomaly = true_anomaly_calc(t, period, eccentricity, t_p)
    r = position(semimajor, true_anomaly, eccentricity)
    [x, y, z] = position3d(r, true_anomaly, inclination, Omega, omega)
    # Ref direction is North, so reverse x and y
    # no z since it represents the motion away/towards from us and we dont see that in astrometric measurements
    return (y, x)

def component_positions(x, y, q, l):
    """Calculate positions of primary, secondary, and photocentre relative to barycenter
    x, y: orbital position components
    q: mass ratio (m2/m1)
    l: luminosity ratio
    """
    x1 = x * q / (1 + q)
    y1 = y * q / (1 + q)
    x2 = -x / (1 + q)
    y2 = -y / (1 + q)
    
    xp = x * abs(q - l) / ((1 + l) * (1 + q)) # Photocentre position (luminosity weighted)
    yp = y * abs(q - l) / ((1 + l) * (1 + q))
    return (x1, y1, x2, y2, xp, yp)

def thiele_innes_coefficients(semimajor, inclination, Omega, omega):
    """Calculate Thiele-Innes coefficients A, B, F, G
    
    These coefficients transform orbital coordinates to equatorial coordinates
    and are independent of time, making them useful for inference.
    
    inputs:
      semimajor    : semi-major axis (mas)
      inclination  : orbital inclination (degrees)
      Omega        : longitude of ascending node (degrees)
      omega        : argument of periapsis (degrees)
    
    returns:
      A, B, F, G   : Thiele-Innes coefficients (mas)
    """
    i_rad = inclination * np.pi / 180
    Omega_rad = Omega * np.pi / 180
    omega_rad = omega * np.pi / 180
    A = semimajor * (np.cos(omega_rad) * np.cos(Omega_rad) - np.sin(omega_rad) * np.sin(Omega_rad) * np.cos(i_rad))
    B = semimajor * (np.cos(omega_rad) * np.sin(Omega_rad) + np.sin(omega_rad) * np.cos(Omega_rad) * np.cos(i_rad))
    F = semimajor * (-np.sin(omega_rad) * np.cos(Omega_rad) - np.cos(omega_rad) * np.sin(Omega_rad) * np.cos(i_rad))
    G = semimajor * (-np.sin(omega_rad) * np.sin(Omega_rad) + np.cos(omega_rad) * np.cos(Omega_rad) * np.cos(i_rad))
    return A, B, F, G


def eccentric_anomaly_from_mean(t, period, eccentricity, t_p):
    """Calculate eccentric anomaly E from mean anomaly using Kepler's equation
    
    inputs:
      t            : time(s) (years)
      period       : orbital period (years)
      eccentricity : orbital eccentricity
      t_p          : time of periastron (years)
    
    returns:
      E            : eccentric anomaly (radians)
    """
    mean_anomaly = (((t - t_p) / period) * 2 * np.pi) % (2 * np.pi)
    steps = 6
    E = copy.copy(mean_anomaly)
    for _ in range(steps):
        E -= (E - eccentricity * np.sin(E) - mean_anomaly) / (1 - eccentricity * np.cos(E))
    return E


def orbital_coordinates(t, period, eccentricity, t_p):
    """Calculate X, Y orbital coordinates (equations 7-8 from literature)
    
    inputs:
      t            : time(s) (years)
      period       : orbital period (years)
      eccentricity : orbital eccentricity
      t_p          : time of periastron (years)
    
    returns:
      X, Y         : orbital coordinates (dimensionless)
    """
    E = eccentric_anomaly_from_mean(t, period, eccentricity, t_p)
    X = np.cos(E) - eccentricity
    Y = np.sqrt(1 - eccentricity**2) * np.sin(E)
    return X, Y


def radec_diff(t, semimajor, period, q, l, eccentricity, inclination, Omega, omega, t_p):
    """Calculate difference between photocentre and barycenter on the sky using Thiele-Innes coefficients
    
    This uses the proper formalism from astrometric binary modeling (e.g., Pourbaix et al. 2022):
    ΔRA = (B*X + G*Y) * (luminosity weighting)
    ΔDec = (A*X + F*Y) * (luminosity weighting)
    
    inputs:
      t            : observation times (years)
      semimajor    : semi-major axis (mas)
      period       : orbital period (years)
      q            : mass ratio (m_planet/m_star)
      l            : luminosity ratio (planet/star)
      eccentricity : orbital eccentricity
      inclination  : orbital inclination (degrees)
      Omega        : longitude of ascending node (degrees)
      omega        : argument of periapsis (degrees)
      t_p          : time of periastron (years)
    
    returns:
      ra_diff, dec_diff : photocentre offsets (mas)
    """
    # Calculate Thiele-Innes coefficients (time-independent)
    A, B, F, G = thiele_innes_coefficients(semimajor, inclination, Omega, omega)
    
    # Calculate orbital coordinates (time-dependent)
    X, Y = orbital_coordinates(t, period, eccentricity, t_p)
    
    # Apply luminosity weighting to account for photocenter shift
    # The photocentre is offset by the mass and luminosity ratios
    luminosity_weight = abs(q - l) / ((1 + l) * (1 + q))
    
    # Calculate RA and Dec differences using Thiele-Innes formalism
    ra_diff = (B * X + G * Y) * luminosity_weight
    dec_diff = (A * X + F * Y) * luminosity_weight
    return ra_diff, dec_diff

def gaia_position(t):
    """3D position of Gaia relative to Sun, in celestial coordinates (AU)"""
    t_p = 3.0 / 365.25  # time of peri
    Omega = 0
    i = 23.4  
    e = 0.0167 
    omega = 103.33284959908069  # argument of perihelion (deg)
    p = 1  # period (years)
    a = 1.01 * 4.84814e-6  # semi-major axis (AU)
    phi_0 = t_p_to_init_phase(t_p, p)
    f = true_anomaly_calc(t, p, e, t_p)
    r = position(a, f, e)
    [x, y, z] = position3d(r, f, i, Omega, omega)
    return [x, y, z]

def single_star(
    t: np.ndarray,
    scan_angle: np.ndarray,
    ra_off: float,
    dec_off: float,
    pmra: float,
    pmdec: float,
    parallax_mas: float,
    ra_deg: float,
    dec_deg: float
) -> np.ndarray:
    """
    gaia along-scan (AL) positions for a single-star model

    inputs:
      t             : array of times (same units your code expects; appears to be "years since 2016" in your code)
      scan_angle    : array of Gaia scan angles psi(t) in radians (same length as t)
      ra_off        : RA offset at reference epoch (mas)
      dec_off       : Dec offset at reference epoch (mas)
      pmra          : proper motion in RA direction (mas / year)
      pmdec         : proper motion in Dec direction (mas / year)
      parallax_mas  : parallax in milliarcseconds (mas)
      ra_deg, dec_deg: star coordinates in degrees (used for parallax projection via gaia_position)

    gives:
      AL_star : array of AL positions (mas) at each scan angle, same length as t
    """
    # compute the parallactic + Gaia barycentric contribution as in gaia_diff_col or gaia_matrix logic
    # Using the gaia_position function to compute Gaia geometry vector [xG,yG,zG] (dimensionless small numbers)
    xG, yG, zG = gaia_position(t)  # returns arrays if t is array

    # these give the parallax factor (mas per unit parallax?):
    # 4.84814e-6 rad per mas conversion when computing RA/Dec offsets.
    conv = 4.84814e-6  # radians per mas
    # compute contributions 
    # here we have parallax directly (mas), so simpler: parallax_mas * projection_factor
    proj_ra = (xG * np.sin(np.deg2rad(ra_deg)) - yG * np.cos(np.deg2rad(ra_deg))) / conv   # dimensionless
    proj_dec = (xG * np.cos(np.deg2rad(ra_deg)) * np.sin(np.deg2rad(dec_deg))
                + yG * np.sin(np.deg2rad(ra_deg)) * np.sin(np.deg2rad(dec_deg))
                - zG * np.cos(np.deg2rad(dec_deg))) / conv

    # total RA and Dec offsets (mas) at each time t
    ra_t = ra_off + pmra * t #+ parallax_mas * proj_ra
    dec_t = dec_off + pmdec * t #+ parallax_mas * proj_dec

    # project onto AL direction: AL = ra * sin(psi) + dec * cos(psi)
    AL_star = ra_t * np.sin(scan_angle) + dec_t * np.cos(scan_angle) + parallax_mas
    return AL_star


def planet_model( #integrate wuth pymc and pytensor, test cases, (a -ve)
    t: np.ndarray,  # observation times (years since 2016)
    scan_angle: np.ndarray,  # Gaia scan angles (radians)
    ra_off: float,  # RA offset at reference epoch (mas)
    dec_off: float,  # Dec offset at reference epoch (mas)
    pmra: float,  # proper motion in RA (mas/year)
    pmdec: float,  # proper motion in Dec (mas/year)
    parallax_mas: float,  # parallax (mas)
    ra_deg: float,  # star RA (degrees)
    dec_deg: float,  # star Dec (degrees)
    semimajor_au: float,  # orbital semi-major axis (AU)
    inclination_deg: float,  # orbital inclination (degrees)
    eccentricity: float,  # orbital eccentricity (dimensionless)
    Omega_deg: float,  # longitude of ascending node (degrees)
    omega_deg: float,  # argument of periapsis (degrees)
    Tp: float,  # time of periastron (years since 2016)
    Mp: float,  # planet mass (solar masses)
    Ms: float,  # star mass (solar masses)
    l: float  # luminosity ratio planet/star (dimensionless)
) -> np.ndarray:
    """
    Gaia along-scan (AL) positions for a planet + star model
    
    The model includes the photocenter motion due to the planet's orbit around the star.
    
    inputs:
      t                    : array of times (years since 2016)
      scan_angle           : array of Gaia scan angles psi(t) in radians (same length as t)
      ra_off               : RA offset of photocenter at reference epoch (mas)
      dec_off              : Dec offset of photocenter at reference epoch (mas)
      pmra                 : proper motion in RA direction (mas / year)
      pmdec                : proper motion in Dec direction (mas / year)
      parallax_mas         : parallax in milliarcseconds (mas)
      ra_deg, dec_deg      : star coordinates in degrees
      
      Orbital parameters for the planet:
      semimajor_au         : semi-major axis of planet orbit (AU)
      inclination_deg      : orbital inclination (degrees)
      eccentricity         : orbital eccentricity
      Omega_deg            : longitude of ascending node (degrees)
      omega_deg            : argument of periapsis (degrees)
      Tp                   : time of periastron passage (years since 2016)
      Mp                   : planet mass (solar masses)
      Ms                   : star mass (solar masses)
      l                    : luminosity ratio (planet / star)
    
    gives:
      AL_planet : array of AL positions (mas) at each scan angle, same length as t
    """
    xG, yG, zG = gaia_position(t)
    conv = 4.84814e-6 #convert - radians per mas
    
    #parallax projections
    proj_ra = (xG * np.sin(np.deg2rad(ra_deg)) - yG * np.cos(np.deg2rad(ra_deg))) / conv
    proj_dec = (xG * np.cos(np.deg2rad(ra_deg)) * np.sin(np.deg2rad(dec_deg))
                + yG * np.sin(np.deg2rad(ra_deg)) * np.sin(np.deg2rad(dec_deg))
                - zG * np.cos(np.deg2rad(dec_deg))) / conv
    
    # compute photocenter motion from the planet's orbit
    q = Mp / Ms #mass ratio
    dist_pc = 1000.0 / parallax_mas 
    semimajor_mas = semimajor_au * 1000.0 / (dist_pc * conv) #AU to mas
    # Calculate planet position on the sky from orbital elements
    # ra_diff_planet and dec_diff_planet: differences due to planet's orbit (mas)
    period_years = np.sqrt(semimajor_au ** 3 / (Mp + Ms)) #kepler's law

    ra_diff_planet, dec_diff_planet = radec_diff(
        t, 
        semimajor_mas, 
        period_years,
        q, 
        l,
        eccentricity, 
        inclination_deg, 
        Omega_deg, 
        omega_deg, 
        Tp
    )
    
    ra_t = ra_off + pmra * t + ra_diff_planet #total offsets which has planet motion included
    dec_t = dec_off + pmdec * t + dec_diff_planet
    AL_planet = ra_t * np.sin(scan_angle) + dec_t * np.cos(scan_angle) + parallax_mas #project finally
    return AL_planet