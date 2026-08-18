"""Wave spectrum data classes for breaking probability workflows."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional

import numpy as np


@dataclass
class WaveSpectrum:
    """Container for a 2D directional wave spectrum.

    Attributes:
        spectrum_2d: 2D spectrum matrix with shape (n_frequencies, n_directions).
        frequencies_hz: Frequency bins in Hz.
        directions_deg: Direction bins in degrees.
        units: Units of the spectrum.
        source_file: Original data source path.
        valid_time: Optional validity time.
        metadata: Optional extra metadata.
        breaking_probability_value: Cached result from breaking probability method.
    """

    spectrum_2d: np.ndarray
    frequencies_hz: np.ndarray
    directions_deg: np.ndarray
    units: str = "m^2/Hz/rad"
    source_file: Optional[str] = None
    valid_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    breaking_probability_value: Optional[float] = None
    wave_breaking_moments: Optional[Dict[str, float]] = None
    omega_0_value: Optional[float] = None
    mean_wave_direction_value: Optional[float] = None
    max_slope_value: Optional[float] = None
    breaking_threshold: Optional[float] = None
    depth: float = 9999.0
    compute_derived_on_init: bool = False
    omega_0_sensitivity_proportion: float = 1.0

    def __post_init__(self) -> None:
        if self.spectrum_2d.ndim != 2:
            raise ValueError("spectrum_2d must be a 2D array.")
        if self.spectrum_2d.shape[0] != self.frequencies_hz.shape[0]:
            raise ValueError("spectrum_2d rows must match frequencies_hz length.")
        if self.spectrum_2d.shape[1] != self.directions_deg.shape[0]:
            raise ValueError("spectrum_2d columns must match directions_deg length.")
        if self.compute_derived_on_init and self.units == "m^2/Hz/rad":
            self.mean_wave_direction()
            self.Omega_0()
            self.max_slope()

    def _require_variance_units(self, method_name: str) -> None:
        """Guard methods that assume variance-density spectrum units."""
        expected_units = "m^2/Hz/rad"
        if self.units != expected_units:
            raise ValueError(
                f"{method_name} requires units '{expected_units}', got '{self.units}'."
            )

    def breaking_probability(self, slope2: float = 2.0) -> float:
        """Compute and cache breaking probability using the standalone engine."""
        from utilities.universal.engine import breaking_probability

        self._require_variance_units("breaking_probability")
        pb = breaking_probability(self, slope2=slope2)
        self.breaking_probability_value = pb
        return pb

    def swh(self) -> float:
        """Compute significant wave height from the 2D directional spectrum.

        Uses a simple discrete directional integral followed by a trapezoidal
        integral over frequency:
            E(f) = integral S(f, theta) dtheta
            m0   = integral E(f) df
            Hs   = 4 * sqrt(m0)
        """
        self._require_variance_units("swh")

        if self.directions_deg.size < 2:
            raise ValueError("At least two direction bins are required for SWH.")

        dtheta = (2.0 * np.pi) / float(self.directions_deg.size)
        spectrum_1d = np.nansum(self.spectrum_2d, axis=1) * dtheta
        if hasattr(np, "trapezoid"):
            m0 = np.trapezoid(spectrum_1d, x=self.frequencies_hz)
        else:
            m0 = np.trapz(spectrum_1d, x=self.frequencies_hz)
        return float(4.0 * np.sqrt(max(m0, 0.0)))

    def variance_to_amplitude_spectrum(self) -> "WaveSpectrum":
        """Return an amplitude-spectrum WaveSpectrum converted from variance form."""
        from utilities.universal.engine import variance_to_amplitude_spectrum

        self._require_variance_units("variance_to_amplitude_spectrum")
        return variance_to_amplitude_spectrum(self)

    def _prepared_directional_variance(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (angle_vector_rad, frequency_vector_hz, spectrum_sorted)."""
        self._require_variance_units("_prepared_directional_variance")

        frequency_vector_hz = np.asarray(self.frequencies_hz, dtype=float).reshape(-1)
        directions_deg = np.asarray(self.directions_deg, dtype=float).reshape(-1)
        directional_variance_spectrum = np.asarray(self.spectrum_2d, dtype=float)

        angle_vector = np.deg2rad(((directions_deg + 180.0) % 360.0) - 180.0)
        sort_index = np.argsort(angle_vector)
        angle_vector = angle_vector[sort_index]
        directional_variance_spectrum = directional_variance_spectrum[:, sort_index]
        return angle_vector, frequency_vector_hz, directional_variance_spectrum

    def mean_wave_direction(self) -> float:
        """Compute and cache mean wave direction (radians)."""
        angle_vector, frequency_vector_hz, directional_variance_spectrum = self._prepared_directional_variance()

        direction_matrix, _ = np.meshgrid(angle_vector, frequency_vector_hz, indexing="xy")
        integrand_a = directional_variance_spectrum * np.cos(direction_matrix)
        integrand_b = directional_variance_spectrum * np.sin(direction_matrix)

        a_frequency = np.trapezoid(integrand_a, x=frequency_vector_hz, axis=0)
        b_frequency = np.trapezoid(integrand_b, x=frequency_vector_hz, axis=0)
        a = float(np.trapezoid(a_frequency, x=angle_vector))
        b = float(np.trapezoid(b_frequency, x=angle_vector))
        mwd = float(np.arctan2(b, a))
        self.mean_wave_direction_value = mwd
        return mwd

    def spread_calculations(self) -> float:
        """Compute and cache directional spread parameter Omega_0."""
        angle_vector, frequency_vector_hz, directional_variance_spectrum = self._prepared_directional_variance()

        spectrum_direction_freq = directional_variance_spectrum.T
        frequency_integral = np.trapezoid(
            spectrum_direction_freq,
            x=frequency_vector_hz,
            axis=1,
        )
        a0 = float(np.trapezoid(frequency_integral, x=angle_vector))
        if a0 <= 0.0 or not np.isfinite(a0):
            self.omega_0_value = 0.0
            return 0.0

        theta_0 = self.mean_wave_direction()
        omega = frequency_integral / a0
        integrand = np.cos(angle_vector - theta_0) * omega
        omega_0 = float(np.trapezoid(integrand, x=angle_vector))
        self.omega_0_value = omega_0
        return omega_0

    def experimental_steepness_boundary_mcallister(self, omega_0: Optional[float] = None) -> float:
        """Compute and cache maximum non-breaking slope from Omega_0."""
        if omega_0 is None:
            omega_0 = self.Omega_0()

        e1 = 1.456
        e2 = -1.127
        e3 = -0.6423
        e4 = -3.781
        max_slope = float(e1 * np.exp(e2 * omega_0) + e3 * np.exp(e4 * omega_0))
        self.max_slope_value = max_slope
        self.breaking_threshold = max_slope
        return max_slope

    def max_slope(self) -> float:
        """Compute and cache max non-breaking slope (McAllister 2024 boundary)."""
        return self.experimental_steepness_boundary_mcallister()

    def Omega_0(self, sensitivity_proportion: Optional[float] = None) -> float:
        """Return Omega_0 with optional sensitivity scaling and cache it on the object.

        If sensitivity_proportion is provided, it is stored on the object and
        reused in subsequent calls where sensitivity_proportion is omitted.
        """
        if sensitivity_proportion is not None:
            self.omega_0_sensitivity_proportion = float(sensitivity_proportion)

        omega_0_raw = self.spread_calculations()
        omega_0_scaled = omega_0_raw * float(self.omega_0_sensitivity_proportion)
        self.omega_0_value = omega_0_scaled
        return omega_0_scaled

    def MWD(self) -> float:
        """Compatibility alias returning mean wave direction in radians."""
        return self.mean_wave_direction()


@dataclass
class WaveSpectraCollection:
    """Collection wrapper around a list of WaveSpectrum objects."""

    spectra: List[WaveSpectrum]

    def __len__(self) -> int:
        return len(self.spectra)

    def __iter__(self) -> Iterator[WaveSpectrum]:
        return iter(self.spectra)

    def __getitem__(self, index: int) -> WaveSpectrum:
        return self.spectra[index]

    def field_values(self, field_name: str) -> List[float]:
        """Return one scalar value per spectrum for the requested field."""
        if field_name == "swh":
            return [ws.swh() for ws in self.spectra]
        if field_name == "breaking_probability":
            return [ws.breaking_probability() for ws in self.spectra]
        raise ValueError(
            "Unsupported field_name. Expected 'swh' or 'breaking_probability'."
        )

    def plot_field(
        self,
        field_name: str = "swh",
        missing_value: Optional[float] = None,
        discrete: bool = False,
    ):
        """Plot a scalar field from all spectra on a cartopy map."""
        from utilities.universal.plots import plot_wave_spectra_field

        return plot_wave_spectra_field(
            self.spectra,
            field_name=field_name,
            missing_value=missing_value,
            discrete=discrete,
        )

    def plot_breaking_probability_vs_swh(self):
        """Scatter plot of breaking probability against significant wave height."""
        import matplotlib.pyplot as plt

        swh_values = np.array([ws.swh() for ws in self.spectra], dtype=float)
        pb_values = np.array([ws.breaking_probability() for ws in self.spectra], dtype=float)
        positive_mask = pb_values > 0.0

        fig, ax = plt.subplots(figsize=(7.0, 5.0))
        ax.scatter(swh_values[positive_mask], pb_values[positive_mask], s=22, alpha=0.8, color="#1f77b4")
        ax.set_yscale("log")
        ax.set_xlabel("Significant wave height, $H_s$ (m)")
        ax.set_ylabel("Breaking probability")
        ax.set_title("Breaking probability vs significant wave height")
        fig.tight_layout()
        return fig, ax