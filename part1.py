from astroquery.gaia import Gaia
from gaiaunlimited.scanninglaw import GaiaScanningLaw
import numpy as np

def find_obs(ra, dec, version="dr3_nominal"):
    sl = GaiaScanningLaw(version)
    result = sl.query(ra, dec, return_angles=True, return_parallax_factors=True)

    # Convert time to decimal year (match old function)
    t_obs = result['t'] / 365.25 + 2010

    scan_angle = result['scan_angle']      # radians
    plx_factor = result['parallax_factor'] # same as old manual calculation

    # Sort by time (match old)
    idx = np.argsort(t_obs)

    return t_obs[idx], scan_angle[idx], plx_factor[idx]


def gaia_query(source_id):
    name = 'Gaia DR3 ' + source_id
    result_table = Gaia.query_object_async(name)
    all_ids = result_table['designation'].tolist()
    print(all_ids)
    if name in all_ids:
        el = all_ids.index(name)
    else:
        el = 0 
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
    (t_obs, scan_angle, plx_factor) = find_obs(params['ra'], params['dec'])
    return (params, t_obs, scan_angle, plx_factor)

def find_companion(source_id, nDraws=1000, nWarmups=5000, nChains=4, 
                   ast_error='auto', dark=False, circular=False,
                   image_data=None, rv_data=None, init=True):
    # 1. Query Gaia data and scanning law
    (row, t_obs, scan_angle, plx_factor) = gaia_query(source_id)

    # 2. Extract Gaia astrometric parameters, uncertainties, and correlations
    gaia_params = np.array([row['ra'], row['dec'], row['parallax'], row['pmra'], row['pmdec']])
    gaia_errs   = np.array([row['ra_error'], row['dec_error'], row['parallax_error'], row['pmra_error'], row['pmdec_error']])
    gaia_corr   = np.array([
        row['ra_dec_corr'], row['ra_parallax_corr'], row['ra_pmra_corr'], row['ra_pmdec_corr'],
        row['dec_parallax_corr'], row['dec_pmra_corr'], row['dec_pmdec_corr'],
        row['parallax_pmra_corr'], row['parallax_pmdec_corr'], row['pmra_pmdec_corr']
    ])
    ruwe_obs = row['ruwe']
    g_mag = row['phot_g_mean_mag']

    [ra, dec, plx_obs, pmra_obs, pmdec_obs] = gaia_params
    [ra_err, dec_err, plx_err, pmra_err, pmdec_err] = gaia_errs

    # 3. Build covariance matrix (more accurate and slightly faster)
    def build_cov_matrix(errs, corr):
        """
        Build 5x5 Gaia covariance matrix from 10 correlation coefficients.
        Correlation order matches Gaia documentation:
        [ra_dec, ra_plx, ra_pmra, ra_pmdec, dec_plx, dec_pmra,
         dec_pmdec, plx_pmra, plx_pmdec, pmra_pmdec]
        """

        cov = np.diag(errs**2)
        idx_pairs = [(0,1),(0,2),(0,3),(0,4),
                     (1,2),(1,3),(1,4),
                     (2,3),(2,4),
                     (3,4)]
        for c, (i,j) in zip(corr, idx_pairs):
            cov[i,j] = c * errs[i] * errs[j]
            cov[j,i] = cov[i,j]
        return cov

    cov_matrix = build_cov_matrix(gaia_errs, gaia_corr)

    # 4. Choose the correct RUWE model
    mod = ruwe_ecc_dark

    # 5. Determine astrometric error
    if ast_error == 'auto':
        err = float(astromet.sigma_ast(g_mag))
        if np.isnan(err):
            err = 0.4  # fallback if no valid astrometric error
    else:
        err = ast_error

    # 6. Build the probabilistic model for MCMC sampling
    # NOTE: If ruwe_ecc_dark.build_model supports cov_matrix, you can uncomment the next line
    # for slightly better precision and less recomputation.
    #
    # model, data = mod.build_model(gaia_params, cov_matrix, g_mag, ruwe_obs, 0.1,
    #                               t_obs, scan_angle, plx_factor, err, image_data, rv_data)
    #
    # Otherwise keep the original input form:
    model, data = mod.build_model(gaia_params, gaia_errs, gaia_corr, g_mag, ruwe_obs, 0.1,
                                  t_obs, scan_angle, plx_factor, err, image_data, rv_data)

    # 7. Run MCMC sampling
    import multiprocessing
    parallel_chains = min(nChains, multiprocessing.cpu_count() - 1)  # safer parallelization

    samples = model.sample(
        data=data,
        chains=nChains,
        parallel_chains=parallel_chains,
        iter_warmup=nWarmups,
        iter_sampling=nDraws,
        show_console=True,
        max_treedepth=12
    )

    # 8. Convert draws to pandas DataFrame
    samples = samples.draws_pd()

    # 9. Return samples for analysis
    return samples