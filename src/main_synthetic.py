"""Compute breaking probability over a grid of crossing and spreading angles.

Uses a synthetic JONSWAP spectrum so no experimental data is required.
Directional distributions are built using the same wrapped-normal PDF as the
experimental workflow.

Output: contour map of breaking probability p_B(cross_deg, spread_deg).

Usage:
    python3 main_synthetic.py  (run from src/ directory)
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2]))

from utilities.universal import engine as wave_engine
from utilities.universal.classes import WaveSpectrum
from utilities.idealised_spectra import jonswap_spectrum
from utilities.experimental.main_experimental_helpers import directional_pdf

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
HS = 0.4          # Significant wave height (m)
TP = 2         # Peak period (s)
GAMMA = 3.3       # JONSWAP peak enhancement factor
F_MIN_HZ = 0.01   # Lower frequency limit (Hz)
F_MAX_HZ_VALUES = [7.0,]  # Upper frequency limits to compare (Hz)
F_DELTA_HZ = 0.01 # Frequency resolution (Hz)
N_DIRECTIONS = 144  # Number of directional bins

CROSS_DEG_VALUES = np.array([0, 180], dtype=float)
SPREAD_DEG_VALUES  = np.array([25, 30, 40, 50], dtype=float)

# SPREAD_DEG_VALUES = np.arange(0.0, 55.0, 5.0)   # Directional spread angles (deg)
# CROSS_DEG_VALUES = np.arange(0.0, 65.0, 5.0)    # Crossing angles (deg)

OMEGA_0_SCALING = 1.0 # Used for sensitivity analysis
SLOPE_2 = 4.0
SLOPE_INTERVAL = 0.01
PRINT_OMEGA0 = True
PRINT_MAXSLOPE = True

wave_engine.SLOPE_INTERVAL = SLOPE_INTERVAL

# ---------------------------------------------------------------------------
# Fixed grids
# ---------------------------------------------------------------------------
directions_deg = np.linspace(-180.0, 180.0, N_DIRECTIONS + 1)[:-1]
directions_rad = np.deg2rad(directions_deg)
cross_grid, spread_grid = np.meshgrid(CROSS_DEG_VALUES, SPREAD_DEG_VALUES)

# ---------------------------------------------------------------------------
# Sweep over F_MAX_HZ values and crossing/spread angles
# ---------------------------------------------------------------------------
all_pb_grids = []
all_max_slope_grids = []
omega0_grid = np.full((len(SPREAD_DEG_VALUES), len(CROSS_DEG_VALUES)), np.nan)

for f_max_hz in F_MAX_HZ_VALUES:
    frequencies_hz = np.arange(F_MIN_HZ, f_max_hz + F_DELTA_HZ, F_DELTA_HZ)
    spec_1d = jonswap_spectrum(frequencies_hz, hs=HS, tp=TP, gamma=GAMMA)
    pb_grid = np.full((len(SPREAD_DEG_VALUES), len(CROSS_DEG_VALUES)), np.nan)
    max_slope_grid = np.full((len(SPREAD_DEG_VALUES), len(CROSS_DEG_VALUES)), np.nan)

    for i, spread_deg in enumerate(SPREAD_DEG_VALUES):
        for j, cross_deg in enumerate(CROSS_DEG_VALUES):
            D = directional_pdf(
                directions_rad,
                spread_deg,
                cross_deg,
                warning_context=f"f_max={f_max_hz}, spread={spread_deg:.0f}, cross={cross_deg:.0f}",
            )
            spectrum_2d = np.outer(spec_1d, D)

            ws = WaveSpectrum(
                spectrum_2d=spectrum_2d,
                frequencies_hz=frequencies_hz,
                directions_deg=directions_deg,
                units="m^2/Hz/rad",
                metadata={"spread_deg": spread_deg, "cross_deg": cross_deg},
            )

            ws.Omega_0(sensitivity_proportion=OMEGA_0_SCALING)
            pb_grid[i, j] = ws.breaking_probability(slope2=SLOPE_2)
            max_slope_grid[i, j] = ws.max_slope_value
            # omega_0 is the same for all f_max; only store on first pass
            if omega0_grid[i, j] != omega0_grid[i, j]:  # is nan
                omega0_grid[i, j] = 1 - ws.omega_0_value

    all_pb_grids.append(pb_grid)
    all_max_slope_grids.append(max_slope_grid)
    print(f"f_max={f_max_hz} Hz done")

if PRINT_OMEGA0:
    col_w = 10
    cross_header = "".join(f"{'cross=' + str(int(c)):>{col_w}}" for c in CROSS_DEG_VALUES)
    header = f"{'spread_deg':>12}{cross_header}"
    print("\nOmega_0 table (independent of f_max):")
    print(header)
    print("-" * len(header))
    for i, spread_deg in enumerate(SPREAD_DEG_VALUES):
        row = "".join(f"{omega0_grid[i, j]:>{col_w}.4f}" for j in range(len(CROSS_DEG_VALUES)))
        print(f"{spread_deg:>12.1f}{row}")

if PRINT_MAXSLOPE:
    col_w = 10
    cross_header = "".join(f"{'cross=' + str(int(c)):>{col_w}}" for c in CROSS_DEG_VALUES)
    header = f"{'spread_deg':>12}{cross_header}"
    for f_max_hz, max_slope_grid in zip(F_MAX_HZ_VALUES, all_max_slope_grids):
        print(f"\nmax_slope table for f_max={f_max_hz} Hz:")
        print(header)
        print("-" * len(header))
        for i, spread_deg in enumerate(SPREAD_DEG_VALUES):
            row = "".join(f"{max_slope_grid[i, j]:>{col_w}.4f}" for j in range(len(CROSS_DEG_VALUES)))
            print(f"{spread_deg:>12.1f}{row}")

# Shared colour scale across all panels
vmax = max(np.nanmax(g) for g in all_pb_grids)
levels = np.linspace(0.0, vmax, 20)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
n_panels = len(F_MAX_HZ_VALUES)
fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5), constrained_layout=True)
if n_panels == 1:
    axes = [axes]

cf = None
for ax, pb_grid, f_max_hz in zip(axes, all_pb_grids, F_MAX_HZ_VALUES):
    cf = ax.contourf(cross_grid, spread_grid, pb_grid, levels=levels, cmap="viridis")
    cs = ax.contour(cross_grid, spread_grid, pb_grid, levels=levels, colors="white", linewidths=0.4, alpha=0.4)
    ax.clabel(cs, fmt="%.3f", fontsize=7)
    ax.set_xlabel("Crossing angle $\\theta_\\times$ (deg)")
    ax.set_ylabel("Directional spread $\\sigma_\\theta$ (deg)")
    ax.set_title(f"$f_{{max}} = {f_max_hz}$ Hz", fontsize=11)

cbar = fig.colorbar(cf, ax=axes, pad=0.02, shrink=0.8)
cbar.set_label("Breaking probability $p_B$")
fig.suptitle(
    f"Breaking probability: $H_s={HS}$ m, $T_p={TP}$ s, $\\gamma={GAMMA}$",
    fontsize=13,
)
plt.show()
