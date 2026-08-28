"""Explore sensitivity of breaking probability to the spectral frequency cutoff.

Runs the same experimental workflow as main_experimental.py (force=1 only) over
a range of F_MAX_HZ values and collects the results into a single table where
each F_MAX_HZ has its own pb column, making it easy to compare how the cutoff
affects the result for each experiment.

Usage:
    python3 aux_experimental_explore_fmax.py  (run from src/ directory)
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parents[2]))

from utilities.universal import engine as wave_engine
from utilities.universal.classes import WaveSpectrum
from utilities.experimental.main_experimental_helpers import directional_pdf, fit_linear_with_ci
# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
PICKLE_FILE = Path(__file__).parents[1] / "data" / "SJTU_bT.pkl"

F_MAX_HZ_MIN = 2.0
F_MAX_HZ_MAX = 7.0
F_MAX_HZ_INCREMENT = 1.0
F_MAX_HZ_VALUES = list(np.arange(F_MAX_HZ_MIN, F_MAX_HZ_MAX + F_MAX_HZ_INCREMENT, F_MAX_HZ_INCREMENT))

N_DIRECTIONS = 144
OMEGA_0_SCALING = 1.0
SLOPE_2 = 2
SLOPE_INTERVAL = 0.01

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

directions_deg = np.linspace(-180.0, 180.0, N_DIRECTIONS + 1)[:-1]
directions_rad = np.deg2rad(directions_deg)

# ---------------------------------------------------------------------------
# Sweep over F_MAX_HZ values
# ---------------------------------------------------------------------------
# pb_results[f_max_hz][experiment_index] = breaking_probability
pb_results = {}

for f_max_hz in F_MAX_HZ_VALUES:
    pb_col = np.zeros(n_experiments, dtype=float)

    for i in range(n_experiments):
        freq = freq_all[:, i]
        spec_1d = variance_all[:, i]

        mask = freq < f_max_hz
        freq = freq[mask]
        spec_1d = spec_1d[mask]

        D = directional_pdf(
            directions_rad,
            float(spread_all[i]),
            float(cross_all[i]),
            warning_context=f"experiment={i + 1}, f_max={f_max_hz}",
        )
        spectrum_2d = np.outer(spec_1d, D)

        ws = WaveSpectrum(
            spectrum_2d=spectrum_2d,
            frequencies_hz=freq,
            directions_deg=directions_deg,
            units="m^2/Hz/rad",
            source_file=str(PICKLE_FILE),
            metadata={"experiment_index": i, "f_max_hz": f_max_hz},
        )

        ws.Omega_0(sensitivity_proportion=OMEGA_0_SCALING)
        pb_col[i] = ws.breaking_probability(slope2=SLOPE_2)

    pb_results[f_max_hz] = pb_col
    print(f"f_max={f_max_hz} Hz done")

# ---------------------------------------------------------------------------
# R^2 convergence plot
# ---------------------------------------------------------------------------
x_fit = np.linspace(0.0, 0.4, 20)
fit_by_fmax = {f: fit_linear_with_ci(bT_all, pb_results[f], x_fit) for f in F_MAX_HZ_VALUES}
r2_values = [fit_by_fmax[f]["r2"] for f in F_MAX_HZ_VALUES]
gradient_values = [fit_by_fmax[f]["slope"] for f in F_MAX_HZ_VALUES]

r2_header = f"{'f_max_hz':>12} {'R2':>8} {'gradient':>10}"
r2_separator = "-" * len(r2_header)
r2_lines = ["R^2 vs F_MAX_HZ", r2_header, r2_separator]
r2_lines += [
    f"{f:>12.2f} {r2:>8.4f} {m:>10.4f}"
    for f, r2, m in zip(F_MAX_HZ_VALUES, r2_values, gradient_values)
]
print("\n" + "\n".join(r2_lines))

fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
ax.plot(F_MAX_HZ_VALUES, r2_values, marker="o", linewidth=2.0, color="#1f77b4")
ax.set_xlabel("$f_{max}$ (Hz)")
ax.set_ylabel("$R^2$")
ax.set_title("Linear fit $R^2$ vs frequency cutoff")
plt.show()
