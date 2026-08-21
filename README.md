# Breaking Probability — WBICS

Python implementation of the wave-breaking probability method from Steer et al.,
applied to SJTU crossing-sea experiments.

## Repository layout

```
breaking_probability_wbics/
├── data/
│   ├── SJTU_bT.pkl                          # Experimental input data
│   └── experimental_Hs_spread_cross_...txt  # Generated output table
└── src/
    ├── main_experimental.py                 # Reproduces paper results
    ├── main_synthetic.py                    # Synthetic JONSWAP sensitivity study
    ├── aux_experimental_explore_fmax.py     # Frequency cutoff sensitivity
    └── utilities/
        ├── universal/
        │   ├── classes.py                   # WaveSpectrum / WaveSpectraCollection
        │   └── engine.py                    # Breaking probability engine
        ├── idealised_spectra/
        │   └── jonswap.py                   # JONSWAP spectrum generator
        └── experimental/
            ├── main_experimental_helpers.py # Directional PDF, linear fit
            └── plots.py                     # Plotting routines
```

## Running the scripts

All scripts must be run from the `src/` directory:

```bash
cd breaking_probability_wbics/src
python3 main_experimental.py
```

---

## `main_experimental.py` — reproduces paper results

This is the primary script. It reproduces Figure 4 from Steer et al. using the
SJTU experimental dataset.

### Input

`data/SJTU_bT.pkl` — a pickle file containing, for each experiment:

| Field                  | Description                                       |
|------------------------|---------------------------------------------------|
| `bT`                   | Experimental breaking probability parameter       |
| `spread`               | Directional spreading angle σ_θ (degrees)         |
| `cross`                | Crossing angle θ_× (degrees)                      |
| `frequency_vector_hz`  | Frequency vector for each experiment (Hz)         |
| `variance_spectrum`    | 1D variance density spectrum S(f) (m²/Hz)         |

### Method

1. A 2D directional spectrum S(f, θ) is constructed for each experiment by
   combining the measured 1D variance spectrum with a two-partition wrapped-normal
   directional distribution parameterised by `spread` and `cross`.

2. Breaking probability p_B is computed from the directional Longuet-Higgins slope
   PDF, integrating the probability of the slope magnitude exceeding the McAllister
   (2024) empirical non-breaking threshold, which is a function of the directional
   spreading parameter Ω_0.

3. This is repeated twice:
   - **force=1** — using the measured spread and crossing angles from the pickle.
   - **force=2** — with all experiments forced to σ_θ = 10°, θ_× = 0° (counter-factual).

4. A linear regression p_B(modelled) = m · p_B(experimental) is fitted for each
   condition and plotted with 95% confidence intervals.

### Key parameters

| Parameter        | Default | Description                                              |
|------------------|---------|----------------------------------------------------------|
| `F_MAX_HZ`       | 7.0     | Frequency cutoff applied to experimental spectra (Hz)    |
| `N_DIRECTIONS`   | 144     | Number of directional bins for 2D spectrum               |
| `OMEGA_0_SCALING`| 1.0     | Sensitivity scaling applied to Ω_0 before threshold calc |
| `SLOPE_2`        | 2.0     | Upper slope limit for PDF integration                    |
| `SLOPE_INTERVAL` | 0.01    | Resolution of slope magnitude vector                     |

### Outputs

- Printed and saved summary table: Hs, spread, cross, Ω_0, p_B for every
  experiment and force condition → `data/experimental_Hs_spread_cross_Omega0_table.txt`
- **Figure 1**: Experimental bT vs modelled p_B (two panels: force=1 and force=2),
  with linear fit, 95% CI, and 99% prediction interval.
- **Figure 2**: Modelled p_B vs Hs scatter plot (force=1 data only).

---

## `main_synthetic.py` — synthetic sensitivity study

Sweeps over a user-defined grid of spreading and crossing angles using a synthetic
JONSWAP spectrum (no experimental data required). Produces contour maps of p_B as
a function of (θ_×, σ_θ) for a range of frequency cutoffs `F_MAX_HZ_VALUES`.

Each surface is normalised first by its (σ_θ=0, θ_×=0) entry, then by the surface
at the highest frequency cutoff, to highlight convergence.

Key parameters at the top of the script: `HS`, `TP`, `GAMMA`, `F_MAX_HZ_VALUES`,
`SPREAD_DEG_VALUES`, `CROSS_DEG_VALUES`.

---

## `aux_experimental_explore_fmax.py` — frequency cutoff sensitivity

Runs the force=1 experimental workflow over a range of frequency cutoffs
(`F_MAX_HZ_MIN`, `F_MAX_HZ_MAX`, `F_MAX_HZ_INCREMENT`) and reports how the linear
fit R² changes with cutoff. Useful for choosing an appropriate `F_MAX_HZ` for
`main_experimental.py`.

Outputs:
- Printed R² table.
- Line plot of R² vs f_max.

---

## Experimental Data

This repository includes supplementary data for Steer et al. (2026), published in
Geophysical Research Letters. The data file (`data/SJTU_bT.pkl`) contains
measurements from the experiments presented in that paper, stored as a Python
dictionary with the following fields:

| Field                 | Shape       | Description                                                                  |
|-----------------------|-------------|------------------------------------------------------------------------------|
| `bT`                  | (1 × 28)    | Breaking probability observed for each experimental condition                |
| `spread`              | (1 × 28)    | Directional spreading width in degrees for each experimental condition       |
| `cross`               | (1 × 28)    | Directional crossing angle in degrees for each experimental condition        |
| `frequency_vector_hz` | (25601 × 28)| Frequency vector in Hz for each experimental condition (one per column)      |
| `variance_spectrum`   | (25601 × 28)| Variance spectrum in m²/Hz for each experimental condition (one per column)  |

The data file originated as a MATLAB structure named `data_out` and was converted
to Python pickle format for use with this codebase.
