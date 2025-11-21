#guery_gaia.py

from astroquery.gaia import Gaia
from gaiaunlimited.scanninglaw import GaiaScanningLaw
import numpy as np
from mass_estimate import getMassFromId
#maybe use gost to query from website
def query_gaia_source(source_id):
    """
    Query Gaia DR3 for a single source ID using ADQL.
    """
    query = f"""
        SELECT
            ra, dec, parallax, pmra, pmdec,
            ra_error, dec_error, parallax_error, pmra_error, pmdec_error,
            ruwe, phot_g_mean_mag
        FROM gaiadr3.gaia_source
        WHERE source_id = {source_id}
    """
    job = Gaia.launch_job_async(query)
    result = job.get_results()
    return result[0]

    """
    Query Gaia scanning law for a position (ra, dec in degrees).
    Returns:
    - t_obs: observation times (decimal years)
    - scan_angle: scan angles in radians
    - plx_factor: along-scan parallax factors
    """
"""
def find_obs(ra_deg, dec_deg, version="dr3_nominal"):

    # Initialize scanning law
    sl = GaiaScanningLaw(version=version, gaplist='dr3/Astrometry')
    # Get observation times for both fields of view
    times_pre, times_fol = sl.query(ra_deg, dec_deg)
    # Combine into a single array
    t_all = np.concatenate([times_pre, times_fol])
    # Convert JD (TCB) to decimal year for consistency with old function
    t_obs = t_all / 365.25 + 2010
    # Get scan angles and parallax factors
    scan_angle = sl.scan_angle(ra_deg, dec_deg, t_all)
    plx_factor = sl.parallax_factor_al(ra_deg, dec_deg, t_all)

    idx = np.argsort(t_obs)
    return t_obs[idx], scan_angle[idx], plx_factor[idx]
"""
def find_obs(ra_deg, dec_deg, version="dr3_nominal"):

    sl = GaiaScanningLaw(version=version, gaplist="dr3/Astrometry")

    # 1. query Gaia observation times
    times_pre, times_fol = sl.query(ra_deg, dec_deg)
    t_query = np.concatenate([times_pre, times_fol])   # JD(TCB)

    # 2. match to Gaia attitude time grid
    t_all = sl.tcb_at_gaia        # shape (8.9M,)
    idx = np.searchsorted(t_all, t_query)

    # 3. extract rotation matrices for the matched times
    R = sl.rotmat[idx]            # shape (N,3,3)
    AL = R[:, :, 0]               # AL direction unit vector

    # 4. compute scan angle
    scan_angle = np.arctan2(AL[:,1], AL[:,0])

    # 5. compute parallax factor from Gaia geometry
    xyz = sl.xyz_fov1[idx]        # shape (N,3)
    x = xyz[:,0]
    y = xyz[:,1]
    z = xyz[:,2]

    ra_rad  = np.deg2rad(ra_deg)
    dec_rad = np.deg2rad(dec_deg)

    plx_factor = (
        (x*np.sin(ra_rad) - y*np.cos(ra_rad))*np.cos(dec_rad)
        + z*np.sin(dec_rad)
    )

    # 6. convert t_query to decimal year
    t_obs = t_query/365.25 + 2010.0

    # sort by time
    order = np.argsort(t_obs)
    return t_obs[order], scan_angle[order], plx_factor[order]

def gaia_query(source_id):
    params = query_gaia_source(source_id)

    t_obs, scan_angle, plx_factor = find_obs(
        params["ra"], params["dec"]
    )

    return params, t_obs, scan_angle, plx_factor


    """
    Query Gaia DR3 for astrometry and scanning law.
    Returns:
    - params: dictionary of Gaia parameters
    - t_obs: array of observation times
    - scan_angle: array of scan angles
    - plx_factor: array of along-scan parallax factors
    """

"""
def gaia_query(source_id):
    name = 'Gaia DR3 ' + source_id
    # Query Gaia object within 10 arcseconds
    result_table = Gaia.query_object_async(name, radius='10 arcsecond')
    all_ids = result_table['designation'].tolist()
    print(all_ids)
    if name in all_ids:
        el = all_ids.index(name)
    else:
        el = 0  # fallback to first object if exact match not found
    # Keys to extract
    all_keys = ['ra', 'dec', 'parallax', 'pmra', 'pmdec']
    all_keys += [key + '_error' for key in all_keys]
    all_keys.append('ruwe')
    all_keys.append('phot_g_mean_mag')
    all_keys.extend([
        'ra_dec_corr', 'ra_parallax_corr', 'ra_pmra_corr', 'ra_pmdec_corr',
        'dec_parallax_corr', 'dec_pmra_corr', 'dec_pmdec_corr',
        'parallax_pmra_corr', 'parallax_pmdec_corr', 'pmra_pmdec_corr'
    ])
    params = {}
    for key in all_keys:
        if key in result_table.colnames:
            params[key] = float(result_table[key][el])
        else:
            params[key] = None
    # Query scanning law
    t_obs, scan_angle, plx_factor = find_obs(params['ra'], params['dec'])
    return params, t_obs, scan_angle, plx_factor

"""

"""
if __name__ == "__main__":
    source_id = "6248534171318085376"
    params, t_obs, scan_angle, plx_factor = gaia_query(source_id)
    print(f"RA, Dec: {params['ra']}, {params['dec']}")
    print(f"Number of observations: {len(t_obs)}")
    print(f"First 5 scan angles (rad): {scan_angle[:5]}")
    print(f"First 5 along-scan parallax factors: {plx_factor[:5]}")
"""

def assemble_data(source_id, ruwe_err=0.1, Gmag_err=0.01):

    params, t_obs, scan_angle, plx_factor = gaia_query(source_id)

    # Convert RA/Dec to radians
    ra_rad  = np.deg2rad(params["ra"])
    dec_rad = np.deg2rad(params["dec"])

    ra_err  = params["ra_error"] * 4.84814e-9
    dec_err = params["dec_error"] * 4.84814e-9

    # estimate mass
    m1 = getMassFromId(source_id, Gmag_err=Gmag_err)

    data = {
        "m1": m1,

        "ra_obs":  ra_rad,
        "dec_obs": dec_rad,
        "plx_obs": params["parallax"],
        "pmra_obs": params["pmra"],
        "pmdec_obs": params["pmdec"],

        "ra_err": ra_err,
        "dec_err": dec_err,
        "plx_err": params["parallax_error"],
        "pmra_err": params["pmra_error"],
        "pmdec_err": params["pmdec_error"],

        "t_obs": t_obs,
        "scan_angle": scan_angle,
        "parallax_factor": plx_factor,

        "g_mag": params["phot_g_mean_mag"],

        "ruwe_obs": params["ruwe"],
        "ruwe_err": ruwe_err,
    }

    return data
