import pymc as pm
import numpy as np
import arviz as az
import pytensor.tensor as at
import matplotlib.pyplot as plt
from jax import random
from jaxstar import mistfit
from astroquery.gaia import Gaia
def getMassFromId(source_id, num_samples=100, rng_seed=0, Gmag_err=0.01):
    q = f"""
        SELECT
            source_id, parallax, parallax_error,
            phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag
        FROM gaiadr3.gaia_source
        WHERE source_id = {int(source_id)}
    """
    r = Gaia.launch_job_async(q).get_results()
    plx   = float(r["parallax"][0])
    e_plx = float(r["parallax_error"][0])
    Gmag  = float(r["phot_g_mean_mag"][0])
    BPmag = float(r["phot_bp_mean_mag"][0])
    RPmag = float(r["phot_rp_mean_mag"][0])
    bp_rp = BPmag - RPmag
    def color_to_teff(bp_rp_val):
        teff_est = 8800.0 - 5400.0*bp_rp_val + 1300.0*(bp_rp_val**2)
        teff_err = 150.0
        return teff_est, teff_err
    teff_obs, teff_err = color_to_teff(bp_rp)
    mf = mistfit.MistFit()
    obs_keys = ['parallax', 'gmag', 'teff']
    obs_vals = [plx,        Gmag,   teff_obs]
    obs_errs = [e_plx,      Gmag_err, teff_err]
    mf.set_data(obs_keys, obs_vals, obs_errs)
    mf.setup_hmc(
        num_warmup=num_samples,
        num_samples=num_samples,
        target_accept_prob=0.9
    )
    rng_key = random.PRNGKey(rng_seed)
    mf.run_hmc(rng_key, linear_age=True, flat_age_marginal=False, nodata=False)
    samples = mf.samples
    mass_median = float(np.median(np.array(samples['mass']).ravel()))
    return mass_median