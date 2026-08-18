"""Plotting utilities for wave spectra fields."""

from typing import List

import matplotlib.pyplot as plt
import numpy as np

from utilities.universal.classes import WaveSpectrum


def _get_field_value(wave_spectrum: WaveSpectrum, field_name: str) -> float:
    """Compute scalar field value for one WaveSpectrum."""
    if field_name == "swh":
        return wave_spectrum.swh()
    if field_name == "breaking_probability":
        return wave_spectrum.breaking_probability()
    raise ValueError(
        "Unsupported field_name. Expected 'swh' or 'breaking_probability'."
    )


def plot_wave_spectra_field(
    wave_spectra: List[WaveSpectrum],
    field_name: str = "swh",
    missing_value: float = None,
    discrete: bool = False,
):
    """Plot field values from spectra on a world map.

    The function expects each WaveSpectrum metadata dict to include
    'latitude' and 'longitude'.

    Args:
        discrete: If True, plot as discrete marker points. If False, plot
            as a gridded field using pcolormesh.
    """
    try:
        import cartopy.crs as ccrs
    except ImportError as exc:
        raise ImportError(
            "cartopy is required for map plotting. Install cartopy first."
        ) from exc

    lats = []
    lons = []
    vals = []

    for ws in wave_spectra:
        lat = ws.metadata.get("latitude")
        lon = ws.metadata.get("longitude")
        if lat is None or lon is None:
            continue

        value = float(_get_field_value(ws, field_name))
        if missing_value is not None and np.isclose(value, missing_value):
            continue

        lats.append(float(lat))
        lons.append(float(lon))
        vals.append(value)

    if not vals:
        raise ValueError(
            "No plottable spectra found. Ensure metadata contains latitude/longitude."
        )

    lats_arr = np.asarray(lats, dtype=float)
    lons_arr = np.asarray(lons, dtype=float)
    vals_arr = np.asarray(vals, dtype=float)

    fig = plt.figure(figsize=(11, 5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines(linewidth=0.8)
    ax.set_global()

    if discrete:
        artist = ax.scatter(
            lons_arr,
            lats_arr,
            c=vals_arr,
            s=24,
            cmap="viridis",
            transform=ccrs.PlateCarree(),
        )
    else:
        unique_lats = np.unique(lats_arr)
        unique_lons = np.unique(lons_arr)
        field = np.full((unique_lats.size, unique_lons.size), np.nan, dtype=float)

        lat_index = {v: i for i, v in enumerate(unique_lats)}
        lon_index = {v: j for j, v in enumerate(unique_lons)}

        for lat, lon, val in zip(lats_arr, lons_arr, vals_arr):
            i = lat_index[lat]
            j = lon_index[lon]
            field[i, j] = val

        lon_grid, lat_grid = np.meshgrid(unique_lons, unique_lats)
        masked_field = np.ma.masked_invalid(field)

        artist = ax.pcolormesh(
            lon_grid,
            lat_grid,
            masked_field,
            shading="auto",
            cmap="viridis",
            transform=ccrs.PlateCarree(),
        )

    cbar = plt.colorbar(artist, ax=ax, orientation="vertical", shrink=0.8, pad=0.03)
    cbar.set_label(field_name)
    ax.set_title(f"Wave spectra field: {field_name}")
    plt.tight_layout()

    return fig, ax
