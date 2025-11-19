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

def radec_diff(t, semimajor, period, q, l, eccentricity, inclination, Omega, omega, t_p):
    """Calculate difference between photocentre and barycenter on the sky"""
    (x, y) = projected_position(t, semimajor, period, eccentricity, inclination, Omega, omega, t_p)
    (x1, y1, x2, y2, xp, yp) = component_positions(x, y, q, l)
    return (xp, yp)

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
