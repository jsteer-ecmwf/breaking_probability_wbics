"""Plotting utilities for main_experimental workflow."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm

from utilities.experimental.main_experimental_helpers import prediction_interval_observation


def plot_experimental_comparison(
    force_results: list[dict],
    bT_all: np.ndarray,
    cross_all: np.ndarray,
    spread_all: np.ndarray,
    x_fit: np.ndarray,
    log_pb_axis: bool = False,
) -> None:
    """Render the experimental comparison figure used by main_experimental."""
    # Dynamically assign markers to unique spread values
    markers = ("^", "o", "s", "D", "v")  # Triangle, circle, square, diamond, inverted triangle
    unique_spreads = sorted(np.unique(spread_all))
    spread_to_marker = {float(spread): markers[i % len(markers)] for i, spread in enumerate(unique_spreads)}
    plot_marker_types = np.array([spread_to_marker[float(s)] for s in spread_all])
    plt.style.use("seaborn-v0_8-whitegrid")

    cmap = plt.get_cmap("viridis", 6)
    norm = BoundaryNorm(np.linspace(0.0, 60.0, cmap.N + 1), cmap.N)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.2, 8.0),
        sharex=True,
        constrained_layout=True,
    )
    fig.suptitle("Experimental vs modelled breaking probability", fontsize=14, fontweight="semibold")

    sc = None

    for row, result in enumerate(force_results):
        ax = axes[row]
        y = result["y"]

        ax.set_facecolor("#f6f7f9")
        ax.grid(False)

        band = ax.fill_between(
            x_fit,
            result["ci_low"],
            result["ci_high"],
            color="#6c757d",
            alpha=0.22,
            label="95% CI",
            zorder=1,
        )
        fit_line, = ax.plot(
            x_fit,
            result["y_pred"],
            color="#111111",
            linewidth=2.0,
            label="Best fit",
            zorder=2,
        )
        x_val_eg = 0.35
        y_val_eg, y_pi_low, y_pi_high = prediction_interval_observation(result, x_val_eg, alpha=0.01)
        pred_bar = ax.errorbar(
            x_val_eg,
            y_val_eg,
            yerr=[[y_val_eg - y_pi_low], [y_pi_high - y_val_eg]],
            fmt="k.",
            capsize=3,
            label="99% Pred. Int.",
            zorder=5,
        )

        # Plot each unique spread with its assigned marker
        for spread_val in unique_spreads:
            marker = spread_to_marker[spread_val]
            mask = plot_marker_types == marker
            
            # Black outline scatter
            ax.scatter(
                bT_all[mask],
                y[mask],
                s=60,
                marker=marker,
                c="k",
                alpha=0.9,
                zorder=3,
            )
            # Colored interior scatter
            if np.any(mask):
                sc = ax.scatter(
                    bT_all[mask],
                    y[mask],
                    s=48,
                    marker=marker,
                    c=cross_all[mask],
                    cmap=cmap,
                    norm=norm,
                    edgecolors="white",
                    linewidths=0.5,
                    zorder=4,
                )

        ax.set_ylabel("$p_B$")
        if log_pb_axis:
            ax.set_yscale("log")
        else:
            ax.set_ylim(bottom=0.0)
        ax.text(
            0.04,
            0.90,
            f"$m = {result['slope']:.2f}, R^2 = {result['r2']:.2f}$",
            transform=ax.transAxes,
            fontsize=10,
            va="top",
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": "#d0d4da",
                "alpha": 0.9,
            },
        )

        if row == 0:
            leg = ax.legend(
                [fit_line, band, pred_bar],
                ["Best fit", "95% CI", "99% Pred. Int."],
                loc="lower right",
                frameon=True,
                framealpha=0.95,
                fontsize=9,
            )
            leg.get_frame().set_linewidth(0.0)
            leg.get_frame().set_edgecolor("none")

    axes[1].set_xlabel("$p_B$ (experimental)")
    axes[0].set_title("Original directional parameters", fontsize=11)
    axes[1].set_title("Forced case: $\\sigma_\\theta = 10^\\circ$, $\\theta_\\times = 0^\\circ$", fontsize=11)
    axes[0].text(0.01, 0.98, "a", transform=axes[0].transAxes, va="top", ha="left", fontweight="bold")
    axes[1].text(0.01, 0.98, "b", transform=axes[1].transAxes, va="top", ha="left", fontweight="bold")

    # Legend for directional spread using distinct markers
    scatter_handles = []
    scatter_labels = []
    for spread_val in unique_spreads:
        marker = spread_to_marker[spread_val]
        scatter_handles.append(
            axes[1].scatter([], [], s=48, marker=marker, c="k", alpha=0.85, edgecolors="white", linewidths=0.5)
        )
        scatter_labels.append(f"$\\sigma_\\theta = {spread_val:.0f}^\\circ$")
    
    axes[1].legend(
        scatter_handles,
        scatter_labels,
        loc="lower right",
        fontsize=9,
        frameon=True,
        framealpha=0.95,
    )
    leg2 = axes[1].get_legend()
    leg2.get_frame().set_linewidth(0.0)
    leg2.get_frame().set_edgecolor("none")

    cbar = fig.colorbar(sc, ax=axes, orientation="vertical", shrink=0.9, pad=0.02)
    cbar.set_label("Crossing angle $\\theta_\\times$ (deg)")
    cbar.set_ticks(np.linspace(0.0, 60.0, 4))

    plt.show(block=False)
