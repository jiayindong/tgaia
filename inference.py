# inference.py
import pymc as pm
import numpy as np
import arviz as az
import pytensor.tensor as at

from pytensor_orbit import radec_offset, al_residual_from_radec
from ruwe_calculation import ruwe_from_al

YEAR0 = 2010.0
DAY_PER_YEAR = 365.25


def build_model(data, draws=1000, tune=2000, chains=4, cores=4, max_treedepth=12):

    with pm.Model() as model:
        m1 = float(data["m1"])

        ra_obs = float(data["ra_obs"])
        dec_obs = float(data["dec_obs"])
        plx_obs = float(data["plx_obs"])
        pmra_obs = float(data["pmra_obs"])
        pmdec_obs = float(data["pmdec_obs"])

        ra_err = float(data["ra_err"])
        dec_err = float(data["dec_err"])
        plx_err = float(data["plx_err"])
        pmra_err = float(data["pmra_err"])
        pmdec_err = float(data["pmdec_err"])

        t_obs_years = pm.Data("t_obs_years", np.asarray(data["t_obs"], dtype="float64"))
        scan_angle = pm.Data("scan_angle", np.asarray(data["scan_angle"], dtype="float64"))
        parallax_factor = pm.Data("parallax_factor", np.asarray(data["parallax_factor"], dtype="float64"))

        g_mag = pm.Data("g_mag", np.array(data["g_mag"], dtype="float64"))

        ruwe_obs = float(data["ruwe_obs"])
        ruwe_err = float(data["ruwe_err"])


        ra = pm.Uniform("ra", lower=0.0, upper=2.0 * np.pi)                # [rad]
        dec = pm.Uniform("dec", lower=-0.5 * np.pi, upper=0.5 * np.pi)     # [rad]
        pmra = pm.Uniform("pmra", lower=-1000.0, upper=1000.0)             # [mas/yr]
        pmdec = pm.Uniform("pmdec", lower=-1000.0, upper=1000.0)           # [mas/yr]
        plx = pm.HalfNormal("plx", sigma=10.0)                             # [mas]

        pm.Normal("ra_like",   mu=ra,   sigma=ra_err,   observed=ra_obs)
        pm.Normal("dec_like",  mu=dec,  sigma=dec_err,  observed=dec_obs)
        pm.Normal("plx_like",  mu=plx,  sigma=plx_err,  observed=plx_obs)
        pm.Normal("pmra_like", mu=pmra, sigma=pmra_err, observed=pmra_obs)
        pm.Normal("pmdec_like",mu=pmdec,sigma=pmdec_err,observed=pmdec_obs)

        log_a = pm.Uniform("log_a", lower=np.log(0.01), upper=np.log(20.0))
        a = pm.Deterministic("a", at.exp(log_a))


        cos_i = pm.Uniform("cos_i", lower=-1.0, upper=1.0)
        inclination = pm.Deterministic("inclination", at.arccos(cos_i))  # [rad]


        e = pm.Beta("e", alpha=0.867, beta=3.03)

        Omega = pm.Uniform("Omega", lower=0.0, upper=2.0 * np.pi)
        omega = pm.Uniform("omega", lower=0.0, upper=2.0 * np.pi)

        M0 = pm.Uniform("M0", lower=0.0, upper=2.0 * np.pi)

        log_m2 = pm.Uniform("log_m2", lower=np.log(1e-3), upper=np.log(30.0))
        m2 = pm.Deterministic("m2", at.exp(log_m2))

        sigma0 = pm.HalfNormal("sigma0", sigma=0.1)

        P_yr = pm.Deterministic("P_yr", at.sqrt(a**3 / (m1 + m2)))   # [yr]
        P_day = pm.Deterministic("P", P_yr * DAY_PER_YEAR)          # [day]

        t_obs_days = (t_obs_years - YEAR0) * DAY_PER_YEAR

        dra, ddec = radec_offset(
            a,
            e,
            inclination,
            Omega,
            omega,
            M0,
            t_obs_days, # [day]
            P_day,      # [day]
            dec
        )
        dal_mas = al_residual_from_radec(dra, ddec, dec, scan_angle)

        RUWE_model = pm.Deterministic(
            "RUWE_model",
            ruwe_from_al(
                dal_mas,
                t_obs_years,
                scan_angle,
                parallax_factor,
                g_mag
            )
        )

        pm.Normal(
            "ruwe_like",
            mu=RUWE_model,
            sigma=ruwe_err,
            observed=ruwe_obs,
        )

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            max_treedepth=max_treedepth,
            target_accept=0.9,
            progressbar=True
        )

    return idata


def run_inference(source_id, assemble_data_fn, **kwargs):
    data = assemble_data_fn(source_id)
    return build_model(data, **kwargs)
