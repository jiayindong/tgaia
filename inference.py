import pymc as pm
import numpy as np
import arviz as az
import pytensor.tensor as at
import matplotlib.pyplot as plt
from jax import random
from jaxstar import mistfit
from astroquery.gaia import Gaia


#mcmc to get the posterior
JD    = 2457000.0 #[day]
m1    = 1000 #[M_J]
"""
ra_obs     = ...   # [rad]
dec_obs    = ...   # [rad]
plx_obs    = ...   # [mas]
pmra_obs   = ...   # [mas/yr]
pmdec_obs  = ...   # [mas/yr]
ra_err     = ...   # [rad]
dec_err    = ...   # [rad]
plx_err    = ...   # [mas]
pmra_err   = ...   # [mas/yr]
pmdec_err  = ...   # [mas/yr]
"""

def build_model(data, draws=1000, tune=2000, chains=4, cores=4, max_treedepth=12):
    with pm.Model() as model:
        #ra, dec, pmra, pmdec, plx
        m1, ra_err, ra_obs, dec_err, dec_obs, plx_err, plx_obs, pmra_err, pmra_obs, pmdec_err, pmdec_obs, ra, dec, pmra, pmdec, plx = data
        ra = pm.Uniform("ra", lower=0, upper=2*np.pi) #[rad]
        dec = pm.Uniform("dec", lower=-0.5*np.pi, upper=0.5*np.pi) #[rad]
        pmra = pm.Uniform("pmra", lower=-1000, upper=1000) #[mas/yr]
        pmdec = pm.Uniform("pmdec", lower=-1000, upper=1000) #[mas/yr]
        plx = pm.HalfNormal("plx", sigma=10.0) #[mas]
        pm.Normal("ra_like",   mu=ra,   sigma=ra_err,   observed=ra_obs)
        pm.Normal("dec_like",  mu=dec,  sigma=dec_err,  observed=dec_obs)
        pm.Normal("plx_like",  mu=plx,  sigma=plx_err,  observed=plx_obs)
        pm.Normal("pmra_like", mu=pmra, sigma=pmra_err, observed=pmra_obs)
        pm.Normal("pmdec_like",mu=pmdec,sigma=pmdec_err,observed=pmdec_obs)

        """
        mu_vec = at.stack([ra, dec, plx, pmra, pmdec])
        obs_vec = np.array([ra_obs, dec_obs, plx_obs, pmra_obs, pmdec_obs])
        cov_mat = cov_mat
        pm.MvNormal("astrometry_like", mu=mu_vec, cov=cov_mat,observed=obs_vec)
        """


        #a
        log_a = pm.Uniform("log_a", lower=np.log(0.01), upper=np.log(20.0)) #[au]
        a = pm.Deterministic("a", at.exp(log_a))

        #i
        cos_i = pm.Uniform("cos_i", lower = -1, upper = 1) #[]
        inclination = pm.Deterministic("inclination", at.arccos(cos_i)) #[rad]

        #e
        e = pm.Beta("e", alpha=0.867, beta=3.03) #[]

        #Omega omega
        Omega = pm.Uniform("Omega", lower = 0, upper = 2 * np.pi) #[]
        omega = pm.Uniform("omega", lower = 0, upper = 2 * np.pi) #[]


        M0 = pm.Uniform("M0", lower=0.0, upper=2*np.pi)

        #m2
        log_m2 = pm.Uniform("log_m2", lower=np.log(1e-3), upper=np.log(30))  # [log(M_J)]
        m2 = pm.Deterministic("m2", at.exp(log_m2)) # [M_J]

        #sigma0
        sigma0 = pm.HalfNormal("sigma0", sigma=0.1) #problem in it

        #t_peri
        P = pm.Deterministic("P_yr", at.sqrt(a**3 / (m1 + m2))) #[yr]
        P_day = pm.Deterministic("P", P * 365.25) #[day]
        t_peri = pm.Deterministic("t_peri", JD - (M0 / (2*np.pi)) * P) #[day]

        RUWE_model = pm.Deterministic("RUWE_model", RUWE(t_obs, scan_angle, parallax_factor, g_mag, ra_diffs, dec_diffs))


        x = dict(Omega = Omega, omega = omega)

        pm.Normal("ruwe_obs_like", mu=RUWE_model, sigma=data["ruwe_err"], observed=data["ruwe_obs"])

        idata = pm.sample(draws=draws, tune=tune, chains=chains, cores=cores,max_treedepth=max_treedepth)

        return idata