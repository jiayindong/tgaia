from astroquery.gaia import Gaia
from gaiaunlimited.scanninglaw import GaiaScanningLaw
import numpy as np
#maybe use gost to query from website
def find_obs(ra, dec, version="dr3_nominal"):
    """
    Query Gaia scanning law for a position (ra, dec in degrees).
    Returns:
    - t_obs: observation times (decimal years)
    - scan_angle: scan angles in radians
    - plx_factor: along-scan parallax factors
    """
    # Initialize scanning law
    sl = GaiaScanningLaw(version=version, gaplist='dr3/Astrometry')
    # Get observation times for both fields of view
    times_preceding, times_following = sl.query(ra, dec)
    # Combine into a single array
    t_all = np.concatenate([times_preceding, times_following])
    # Convert JD (TCB) to decimal year for consistency with old function
    t_obs = t_all / 365.25 + 2010
    # Get scan angles and parallax factors
    angles_data = sl.get_angles(ra, dec, t_all)
    scan_angles = angles_data['scan_angle']
    plx_factors = angles_data['parallax_factor_al']  # along-scan parallax factor
    # Sort by observation time
    idx = np.argsort(t_obs)
    t_obs = t_obs[idx]
    scan_angles = scan_angles[idx]
    plx_factors = plx_factors[idx]
    return t_obs, scan_angles, plx_factors
def gaia_query(source_id):
    """
    Query Gaia DR3 for astrometry and scanning law.
    Returns:
    - params: dictionary of Gaia parameters
    - t_obs: array of observation times
    - scan_angle: array of scan angles
    - plx_factor: array of along-scan parallax factors
    """
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
if __name__ == "__main__":
    source_id = "6248534171318085376"
    params, t_obs, scan_angle, plx_factor = gaia_query(source_id)
    print(f"RA, Dec: {params['ra']}, {params['dec']}")
    print(f"Number of observations: {len(t_obs)}")
    print(f"First 5 scan angles (rad): {scan_angle[:5]}")
    print(f"First 5 along-scan parallax factors: {plx_factor[:5]}")