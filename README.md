# Team Gaia: RUWE-based Planetary Companion Inference Pipeline

This workflow models RUWE (Renormalised Unit Weight Error) using Gaia DR3 data to infer the parameters of planetary companions. The pipeline is modular, allowing for clear separation of concerns and easy collaboration.

---

## 1. **Companion Query (Led by Tanay and Naavya)**

**Goal:** Given a Gaia DR3 ID, extract all relevant system info for downstream modeling.

### **Input**
- Gaia DR3 Source ID

### **Output**
- Time series: `t` (observation times)
- Parallax factor: `f(t)` (from Gaia scanning law)
- Scan angle: `ψ(t)` (from Gaia scanning law)
- RUWE: `ruwe`
- Gmag: `g_mag`
- Astrometric quantities:  
    - Position offsets: `Δα`, `Δδ`
    - Proper motions: `μ_α`, `μ_δ`
    - Parallax: `ϖ`
    - Covariance matrix for astrometric solution: 5×5 for (Δα, Δδ, ϖ, μ_α, μ_δ)

### **Implementation Notes**
- Query Gaia Archive using Astroquery.
- Compute `f` and `ψ` using [GaiaUnlimited scanning law tools](https://gaiaunlimited.readthedocs.io/en/latest/notebooks/scanninglaw.html).
- Retrieve the covariance matrix (see Gaia DR3 columns: `astrometric_params_solved`, `astrometric_matched_observations`, `ra_dec_corr`, etc.) and output it along with the main parameters.
- Package results in a pandas DataFrame or dictionary.

---

## 2. **Forward Modeling Gaia Along-Scan Positions (Led by Tanmay and Ashi)**

### **Single-star Model**

#### **Inputs**
- Offsets: `Δα`, `Δδ`
- Proper motions: `μ_α`, `μ_δ`
- Parallax: `ϖ`
- Time series: `t`, `f(t)`, `ψ(t)` (from Step 1)

#### **Output**
- **Gaia along-scan (AL) positions**: `AL_star(t)` at scan angles `ψ(t)`

#### **Model**
- Predict sky position at time series `t` (using Δα, Δδ, μ_α, μ_δ, ϖ).
- Project each position onto Gaia’s along-scan direction:
    - `AL_star(t) = Δα(t) * sin(ψ(t)) + Δδ(t) * cos(ψ(t))`

### **Planet Model**

#### **Inputs**
- Semi-major axis: `a`
- Inclination: `i`
- Eccentricity: `e`
- Longitude of ascending node: `Ω`
- Argument of periastron: `ω`
- Time of periastron: `Tp`
- Planet mass: `Mp`
- Star mass: `Ms`
- Time series: `t`, `f(t)`, `ψ(t)` (from Step 1)

#### **Output**
- **Gaia along-scan (AL) positions**: `AL_planet(t)` at scan angles `ψ(t)`

#### **Model**
- Compute photocenter motion including the companion in the plane of the sky.
- Project total position onto Gaia’s along-scan direction as above.

---

## 3. **RUWE Calculation (Led by Noori and Grace)**

### **Input**
- Published RUWE from Gaia DR3 (`ruwe_obs`)
- Model-predicted RUWE (`ruwe_model`), calculated as follows:

### **Process**
- For a given set of planet/system parameters, simulate the astrometric signal (forward model).
- Generate a synthetic along-scan (AL) time series for Gaia observations (using the scanning law and your planet/star model).
- Fit the standard 5-parameter (no planet) astrometric model to the simulated AL series.
- Compute the residuals and calculate the model RUWE exactly as Gaia does:
    - `RUWE_model = sqrt(Σ (residuals / σ_i)^2 / (N - 5))`
- **Note:** For real Gaia stars, AL time series are not available; you use the published RUWE value directly.

### **Output**
- Model-predicted RUWE for the current parameter set.

---

## 4. **Bayesian Inference with PyMC (Led by Jay and Johnny)**

### **Priors**
- Astrometric: `Δα`, `Δδ`, `μ_α`, `μ_δ`, `ϖ`
- Orbital: `a`, `i`, `e`, `Ω`, `ω`, `Tp`, `Mp`
- Jitter term: `σ₀` (to account for additional noise)

### **Likelihood**
- For each parameter set, compare the model-predicted RUWE (`RUWE_model`) to the observed RUWE (`RUWE_obs`):
    - `RUWE_obs ~ Normal(RUWE_model, err_RUWE)`

### **Output**
- Posterior samples of planet and system parameters

### **Implementation**
- Use **PyMC** for all probabilistic modeling and inference (replacing previous Stan-based implementations).
- Forward models and RUWE calculation should be implemented as Python/PyMC functions for easy integration.
- Modularize to allow plugging in fast forward models.
- Validate the new PyMC implementation against the reference Stan model in `stan_codes/ruwe_ecc_dark.py`.

---

## **Pipeline Overview**

1. **Companion Query**: Retrieve Gaia astrometric and photometric data for a target star.
2. **Forward Modeling**: Predict Gaia along-scan (AL) positions for single-star and planet-companion models.
3. **RUWE Calculation**: Compute or simulate RUWE from AL model residuals (for synthetic data), or use Gaia's published value (for real data).
4. **Bayesian Inference (PyMC)**: Fit model parameters to match observed RUWE, sampling the posterior.

---

## **Flowchart**

```
Gaia DR3 ID
    ↓
[Companion Query]
    ↓
[Forward Modeling (Gaia AL)]
    ↓
[RUWE Calculation]
    ↓
[PyMC Inference]
    ↓
Posterior on Companion Parameters
```

---

## **Implementation Tips**

- Use version control and modular code (one Python module per step).
- Validate each module independently with test cases.
- Document interfaces (inputs/outputs) clearly for team handoffs.
- Use simulated test data before applying to real Gaia sources.
- Use PyMC’s vectorization and custom likelihoods for efficient inference.

---

## **References**

- [Gaia documentation](https://gea.esac.esa.int/archive/documentation/GDR3/)
- [GaiaUnlimited: Scanning Law notebook](https://gaiaunlimited.readthedocs.io/en/latest/notebooks/scanninglaw.html)
- [RUWE explanation](https://www.cosmos.esa.int/web/gaia/dr3-astrometry)
- [Astrometric modeling (Perryman+ 1997)](https://www.aanda.org/articles/aa/pdf/1997/08/ds1309.pdf)
- [PyMC documentation](https://www.pymc.io/projects/docs/en/stable/)
- [Stan model example](stan_codes/ruwe_ecc_dark.py)
