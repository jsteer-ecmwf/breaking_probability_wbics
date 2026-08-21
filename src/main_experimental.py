"""Calculate breaking probability of experiments run at SJTU.

This module reproduces the analysis from Steer et al. using experimental wave
spectrum data. It processes a pickle file containing:
  - Variance spectra recorded at the facility
  - Crossing angles for each experiment
  - Spreading angles of component wave systems

Workflow:
  1. Generate 2D directional spectra using 1D variance spectra with wrapped-normal
     directional distributions based on given crossing and spreading angles.
  2. Compute breaking probability using full directional information from
     experiments (force=1).
  3. Compute breaking probability using prescribed directional parameters
     (force=2: 10° spreading, 0° crossing) for sensitivity comparison.
  4. Fit linear regression between experimental bT parameter and breaking
     probability for each condition with 95% confidence intervals.

Outputs:
  - Summary table of experimental parameters and results (printed and saved).
  - Figure 4: Experimental vs modelled breaking probability with fit statistics.
  - Scatter plot: Breaking probability vs significant wave height (force=1 data).

Reference:
    Steer et al. (see extractData_crossanglecolor_maxPDF_master.m in MATLAB)

Usage:
    python3 main_experimental.py  (run from src/ directory)
"""

import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Make 'utilities' importable when running directly from src/
sys.path.insert(0, str(Path(__file__).parents[2]))

from utilities.universal import engine as wave_engine
from utilities.universal.classes import WaveSpectrum, WaveSpectraCollection
from utilities.experimental.main_experimental_helpers import (
    directional_pdf,
    fit_linear_with_ci,
)
from utilities.experimental.plots import plot_experimental_comparison

# ---------------------------------------------------------------------------
# Hardcoded paths / parameters  (match MATLAB)
# ---------------------------------------------------------------------------
PICKLE_FILE = Path(__file__).parents[1] / "data" / "SJTU_bT.pkl"
TABLE_OUTPUT_FILE = Path(__file__).parents[1] / "data" / "experimental_Hs_spread_cross_Omega0_table.txt"
N_DIRECTIONS = 144 # Number of directions to use for 2D wave spectra
F_MIN_HZ = 0.0 # Cutoff frequency to apply to experimental variance spectra
F_MAX_HZ = 7.0 # Cutoff frequency to apply to experimental variance spectra
OMEGA_0_SCALING = 1.0 # Scales Omega_0 before max-slope/breaking calculations (used for sensitivity analysis)
SLOPE_2 = 2.0 # Maximum slope upto which to integrate the slope PDF
SLOPE_INTERVAL = 0.01 # Interval of the slope vector of the slope PDF
LOG_PB_AXIS = False # If True, use log scale on the p_B axis of the scatter plot

wave_engine.SLOPE_INTERVAL = SLOPE_INTERVAL

# ---------------------------------------------------------------------------
# Load pickle
# ---------------------------------------------------------------------------
print(f"Loading: {PICKLE_FILE}")
with open(PICKLE_FILE, "rb") as fh:
    data = pickle.load(fh)

bT_all = np.asarray(data["bT"], dtype=float)
spread_all = np.asarray(data["spread"], dtype=float)
cross_all = np.asarray(data["cross"], dtype=float)
freq_all = np.asarray(data["frequency_vector_hz"], dtype=float)
variance_all = np.asarray(data["variance_spectrum"], dtype=float)

n_experiments = int(bT_all.size)

# Direction grid: N_DIRECTIONS bins on [-180, 180)
directions_deg = np.linspace(-180.0, 180.0, N_DIRECTIONS + 1)[:-1]
directions_rad = np.deg2rad(directions_deg)

force_results = []
experiment_summary_rows = []
wave_spectra_force1 = []  # Store WaveSpectrum objects from force=1
pb_by_force = {}  # pb_by_force[force] = array of breaking probabilities
x_fit = np.linspace(0.0, 0.4, 20)

for force in (1, 2):
    slope_exceedance = np.zeros(n_experiments, dtype=float)

    for i in range(n_experiments):
        freq = freq_all[:, i]
        spec_1d = variance_all[:, i]

        mask = (freq > F_MIN_HZ) & (freq < F_MAX_HZ)
        freq = freq[mask]
        spec_1d = spec_1d[mask]

        spread_deg = float(spread_all[i])
        cross_deg = float(cross_all[i])
        if force == 2:
            spread_deg = 10.0
            cross_deg = 0.0

        D = directional_pdf(
            directions_rad,
            spread_deg,
            cross_deg,
            warning_context=(
                f"experiment={i + 1}, force={force}, "
                f"spread_deg={spread_deg:.0f}, cross_deg={cross_deg:.0f}"
            ),
        )
        spectrum_2d = np.outer(spec_1d, D)

        ws = WaveSpectrum(
            spectrum_2d=spectrum_2d,
            frequencies_hz=freq,
            directions_deg=directions_deg,
            units="m^2/Hz/rad",
            source_file=str(PICKLE_FILE),
            metadata={
                "experiment_index": i,
                "force": force,
                "spread_deg": spread_deg,
                "cross_deg": cross_deg,
            },
        )

        omega_0 = ws.Omega_0(sensitivity_proportion=OMEGA_0_SCALING)
        hs = ws.swh()

        breaking_probability = ws.breaking_probability(slope2=SLOPE_2)
        slope_exceedance[i] = breaking_probability

        # Store WaveSpectrum objects from force=1 for later plotting
        if force == 1:
            wave_spectra_force1.append(ws)

        experiment_summary_rows.append(
            {
                "experiment": i + 1,
                "force": force,
                "hs": hs,
                "spread_deg": int(round(spread_deg)),
                "cross_deg": int(round(cross_deg)),
                "omega_0": 1 - omega_0,
                "max_slope": ws.max_slope_value,
                "breaking_probability": breaking_probability,
            }
        )

    fit = fit_linear_with_ci(bT_all, slope_exceedance, x_fit)
    m_fit_0 = float(np.dot(bT_all, slope_exceedance) / np.dot(bT_all, bT_all))
    pb_by_force[force] = slope_exceedance.copy()

    force_results.append(
        {
            "force": force,
            "y": slope_exceedance,
            "slope": fit["slope"],
            "r2": fit["r2"],
            "m_fit_0": m_fit_0,
            "y_pred": fit["y_pred"],
            "ci_low": fit["ci_low"],
            "ci_high": fit["ci_high"],
            "beta": fit["beta"],
            "xtx_inv": fit["xtx_inv"],
            "sigma2": fit["sigma2"],
            "dof": fit["dof"],
        }
    )

    print(
        f"force={force}: R^2 = {fit['r2']:.2f}, "
        f"m = {fit['slope']:.2f}, m0 = {m_fit_0:.2f}"
    )


# pb_ratio[i] = pb_force1[i] / pb_force2[i]; nan where force=2 pb is zero
with np.errstate(invalid="ignore", divide="ignore"):
    pb_ratio = np.where(pb_by_force[2] > 0.0, pb_by_force[1] / pb_by_force[2], np.nan)

table_header = (
    f"{'experiment':>10} {'Hs_m':>8} {'spread_deg':>11} {'cross_deg':>10} {'Omega_0':>9} {'max_slope':>10} {'pb_exp':>8} {'pb_mod':>8} {'pb_ref':>8} {'pb_ratio':>9}"
)
table_separator = "-" * len(table_header)
table_lines = [
    "Experimental summary table (pb_exp=measured bT, pb_mod=force1, pb_ref=force2, pb_ratio=pb_mod/pb_ref)",
    table_header,
    table_separator,
]

# experiment_summary_rows contains force=1 rows first (experiments 1..n), then force=2
rows_f1 = [r for r in experiment_summary_rows if r["force"] == 1]
for row in rows_f1:
    i = row["experiment"] - 1
    pb_ref = pb_by_force[2][i]
    ratio_str = f"{pb_ratio[i]:>9.3f}" if np.isfinite(pb_ratio[i]) else f"{'---':>9}"
    table_lines.append(
        f"{row['experiment']:>10d} {row['hs']:>8.2f} "
        f"{row['spread_deg']:>11d} {row['cross_deg']:>10d} {row['omega_0']:>9.2f} "
        f"{row['max_slope']:>10.4f} "
        f"{bT_all[i]:>8.4f} {row['breaking_probability']:>8.4f} {pb_ref:>8.4f} {ratio_str}"
    )

table_text = "\n".join(table_lines)

with open(TABLE_OUTPUT_FILE, "w", encoding="ascii") as fh:
    fh.write(table_text + "\n")

print("\n" + table_text)
print(f"\nWrote table to: {TABLE_OUTPUT_FILE}")

plot_experimental_comparison(
    force_results=force_results,
    bT_all=bT_all,
    cross_all=cross_all,
    spread_all=spread_all,
    x_fit=x_fit,
    log_pb_axis=LOG_PB_AXIS,
)

# Plot breaking probability vs Hs for the original data (force=1)
wave_spectra_collection = WaveSpectraCollection(spectra=wave_spectra_force1)
wave_spectra_collection.plot_breaking_probability_vs_swh()

# Plot force=1 / force=2 pb ratio vs bT
fig_ratio, ax_ratio = plt.subplots(figsize=(7, 5), constrained_layout=True)
valid = np.isfinite(pb_ratio)
sc_ratio = ax_ratio.scatter(
    bT_all[valid], pb_ratio[valid],
    c=spread_all[valid], cmap="viridis", s=36, edgecolors="k", linewidths=0.5,
)
ax_ratio.axhline(1.0, color="#111111", linewidth=1.5, linestyle="--")
cbar_ratio = fig_ratio.colorbar(sc_ratio, ax=ax_ratio, pad=0.02)
cbar_ratio.set_label("Directional spread $\\sigma_\\theta$ (deg)")
ax_ratio.set_xlabel("$p_B$ (experimental)")
ax_ratio.set_ylabel("$p_B$ (force=1) / $p_B$ (force=2)")
ax_ratio.set_title("Effect of directional information on breaking probability")
plt.show()
