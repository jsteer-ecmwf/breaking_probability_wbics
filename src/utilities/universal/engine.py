"""Breaking probability engine for directional wave spectra.

Spectra are represented as continuous directional variance density
S(f, theta) with shape (frequency, angle), angles in radians on
[-pi, pi), sorted ascending. GRIB spectra are normalised to that
convention at the engine boundary.
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Union
import warnings

import numpy as np

from utilities.universal.classes import WaveSpectrum


GRAVITY = 9.81
SLOPE_INTERVAL = 0.01


@dataclass
class _SpectrumArrays:
	"""Internal intermediate representation of a wave spectrum."""

	angle_vector: np.ndarray
	frequency_vector_hz: np.ndarray
	wavenumber_vector: np.ndarray
	directional_variance_spectrum: np.ndarray


def breaking_probability(
	wave_spectrum: WaveSpectrum,
	peaks_flag: int = 0,
	return_diagnostics: bool = False,
	slope2: float = 2.0,
) -> Union[float, Dict[str, float]]:
	"""Compute breaking probability for a WaveSpectrum.

	Returns the scalar m0_breaking, or a diagnostics dict when
	return_diagnostics is True.
	"""
	wave_spectrum._require_variance_units("breaking_probability")

	if peaks_flag not in (0, 1):
		raise ValueError("peaks_flag must be 0 or 1.")

	if is_zero_spectrum(wave_spectrum):
		diagnostics = zero_breaking_diagnostics(peaks_flag)
		wave_spectrum.breaking_probability_value = diagnostics["m0_breaking"]
		wave_spectrum.wave_breaking_moments = {
			"m0_breaking": diagnostics["m0_breaking"],
			"m1_breaking": diagnostics["m1_breaking"],
			"mss": diagnostics["mss"],
			"peaks_flag": float(peaks_flag),
		}
		wave_spectrum.omega_0_value = diagnostics["Omega_0"]
		wave_spectrum.max_slope_value = diagnostics["breaking_threshold"]
		wave_spectrum.breaking_threshold = diagnostics["breaking_threshold"]
		if return_diagnostics:
			return diagnostics
		return diagnostics["m0_breaking"]

	spectrum_arrays = _prepare_spectrum(wave_spectrum)
	diagnostics = _compute_breaking_moments(
		wave_spectrum,
		spectrum_arrays,
		peaks_flag=peaks_flag,
		slope2=slope2,
	)

	wave_spectrum.breaking_probability_value = diagnostics["m0_breaking"]
	wave_spectrum.wave_breaking_moments = {
		"m0_breaking": diagnostics["m0_breaking"],
		"m1_breaking": diagnostics["m1_breaking"],
		"mss": diagnostics["mss"],
		"peaks_flag": float(peaks_flag),
	}
	wave_spectrum.omega_0_value = diagnostics["Omega_0"]
	wave_spectrum.max_slope_value = diagnostics["breaking_threshold"]
	wave_spectrum.breaking_threshold = diagnostics["breaking_threshold"]

	if return_diagnostics:
		return diagnostics

	return diagnostics["m0_breaking"]


def wave_breaking_moments(
	wave_spectrum: WaveSpectrum,
	peaks_flag: int = 0,
	slope2: float = 2.0,
) -> Tuple[float, float, float, float, float]:
	"""Return breaking moments (m0, m1, mss, Omega_0, breaking_threshold) for a WaveSpectrum."""
	wave_spectrum._require_variance_units("wave_breaking_moments")
	if is_zero_spectrum(wave_spectrum):
		diagnostics = zero_breaking_diagnostics(peaks_flag)
		return (
			diagnostics["m0_breaking"],
			diagnostics["m1_breaking"],
			diagnostics["mss"],
			diagnostics["Omega_0"],
			diagnostics["breaking_threshold"],
		)
	diagnostics = _compute_breaking_moments(
		wave_spectrum,
		_prepare_spectrum(wave_spectrum),
		peaks_flag=peaks_flag,
		slope2=slope2,
	)
	return (
		diagnostics["m0_breaking"],
		diagnostics["m1_breaking"],
		diagnostics["mss"],
		diagnostics["Omega_0"],
		diagnostics["breaking_threshold"],
	)


def is_zero_spectrum(wave_spectrum: WaveSpectrum) -> bool:
	"""Return True for land/empty spectra with no positive finite energy."""
	spectrum = np.asarray(wave_spectrum.spectrum_2d, dtype=float)
	return not np.any(np.isfinite(spectrum) & (spectrum > 0.0))


def zero_breaking_diagnostics(peaks_flag: int = 0) -> Dict[str, float]:
	"""Diagnostics used for land/empty spectra."""
	return {
		"pb": 0.0,
		"m0_breaking": 0.0,
		"m1_breaking": 0.0,
		"mss": 0.0,
		"Omega_0": 0.0,
		"breaking_threshold": 0.0,
		"peaks_flag": float(peaks_flag),
	}


def _prepare_spectrum(wave_spectrum: WaveSpectrum) -> _SpectrumArrays:
	"""Normalise a WaveSpectrum to the internal array representation.

	Input directions may be 0..360 degrees. The output angle vector is
	radians on [-pi, pi), sorted ascending, and spectrum columns are
	reordered to match.
	"""
	wave_spectrum._require_variance_units("_prepare_spectrum")

	spectrum = np.asarray(wave_spectrum.spectrum_2d, dtype=float)
	frequency_vector_hz = np.asarray(wave_spectrum.frequencies_hz, dtype=float).reshape(-1)
	directions_deg = np.asarray(wave_spectrum.directions_deg, dtype=float).reshape(-1)

	if spectrum.shape != (frequency_vector_hz.size, directions_deg.size):
		raise ValueError("Spectrum shape must be (n_frequencies, n_directions).")
	if frequency_vector_hz.size < 2:
		raise ValueError("At least two frequency bins are required.")
	if directions_deg.size < 2:
		raise ValueError("At least two direction bins are required.")

	angle_vector = np.deg2rad(((directions_deg + 180.0) % 360.0) - 180.0)
	sort_index = np.argsort(angle_vector)
	angle_vector = angle_vector[sort_index]
	directional_variance_spectrum = spectrum[:, sort_index]

	wavenumber_vector = deep_water_wavenumber_vector(frequency_vector_hz)

	return _SpectrumArrays(
		angle_vector=angle_vector,
		frequency_vector_hz=frequency_vector_hz,
		wavenumber_vector=wavenumber_vector,
		directional_variance_spectrum=directional_variance_spectrum,
	)


def _compute_breaking_moments(
	wave_spectrum: WaveSpectrum,
	spectrum_arrays: _SpectrumArrays,
	peaks_flag: int = 0,
	slope2: float = 2.0,
) -> Dict[str, float]:
	"""Compute wave breaking moments from internal spectrum arrays."""
	if peaks_flag not in (0, 1):
		raise ValueError("peaks_flag must be 0 or 1.")

	direction_matrix, wavenumber_matrix = _domain_matrices(spectrum_arrays)
	directional_amplitude_spectrum = variance_to_amplitude_array(
		spectrum_arrays.angle_vector,
		spectrum_arrays.frequency_vector_hz,
		spectrum_arrays.directional_variance_spectrum,
	)

	omega_0 = wave_spectrum.Omega_0()
	maximum_slope = wave_spectrum.max_slope()

	slope_magnitude_vector, breaking_slope_pdf, mss = longuet_higgins_slope_pdf_arrays(
		direction_matrix,
		wavenumber_matrix,
		directional_amplitude_spectrum,
		slope1=maximum_slope,
		slope2=slope2,
		use_unidirectional_limit=bool(np.isclose(omega_0, 1.0, atol=1.0e-12)),
		warning_context=_warning_context_from_metadata(wave_spectrum.metadata),
	)
	mu = pdf_moments(slope_magnitude_vector, breaking_slope_pdf, c=0.0, n=2)
	m0_breaking = float(mu[0])
	m1_breaking = float(mu[1])

	if peaks_flag == 1:
		full_slope_vector, full_slope_pdf, _ = longuet_higgins_slope_pdf_arrays(
			direction_matrix,
			wavenumber_matrix,
			directional_amplitude_spectrum,
			slope1=0.0,
			slope2=2.0,
			use_unidirectional_limit=bool(np.isclose(omega_0, 1.0, atol=1.0e-12)),
			warning_context=_warning_context_from_metadata(wave_spectrum.metadata),
		)
		weighted_pdf = full_slope_pdf * full_slope_vector
		weighted_integral = float(np.trapezoid(weighted_pdf, x=full_slope_vector))
		if weighted_integral > 0.0:
			weighted_pdf = weighted_pdf / weighted_integral
			lower_index = int(np.argmin(np.abs(full_slope_vector - maximum_slope)))
			m0_breaking = float(
				np.trapezoid(
					weighted_pdf[lower_index:],
					x=full_slope_vector[lower_index:],
				)
			)
			m1_breaking = m0_breaking

	return {
		"pb": m0_breaking,
		"m0_breaking": m0_breaking,
		"m1_breaking": m1_breaking,
		"mss": float(mss),
		"Omega_0": float(omega_0),
		"breaking_threshold": float(maximum_slope),
	}


def slope_pdf_output(wave_spectrum: WaveSpectrum) -> Tuple[np.ndarray, np.ndarray]:
	"""Return the full slope PDF over [0, 2] for a WaveSpectrum."""
	spectrum_arrays = _prepare_spectrum(wave_spectrum)
	direction_matrix, wavenumber_matrix = _domain_matrices(spectrum_arrays)
	amplitude_spectrum = variance_to_amplitude_array(
		spectrum_arrays.angle_vector,
		spectrum_arrays.frequency_vector_hz,
		spectrum_arrays.directional_variance_spectrum,
	)
	slope_magnitude_vector, slope_pdf, _ = longuet_higgins_slope_pdf_arrays(
		direction_matrix,
		wavenumber_matrix,
		amplitude_spectrum,
		slope1=0.0,
		slope2=2.0,
		use_unidirectional_limit=bool(np.isclose(wave_spectrum.Omega_0(), 1.0, atol=1.0e-12)),
		warning_context=_warning_context_from_metadata(wave_spectrum.metadata),
	)
	return slope_magnitude_vector, slope_pdf


def variance_to_amplitude_spectrum(wave_spectrum: WaveSpectrum) -> WaveSpectrum:
	"""Return the amplitude spectrum as a WaveSpectrum."""
	spectrum_arrays = _prepare_spectrum(wave_spectrum)
	amplitude_spectrum = variance_to_amplitude_array(
		spectrum_arrays.angle_vector,
		spectrum_arrays.frequency_vector_hz,
		spectrum_arrays.directional_variance_spectrum,
	)
	return WaveSpectrum(
		spectrum_2d=amplitude_spectrum,
		frequencies_hz=spectrum_arrays.frequency_vector_hz.copy(),
		directions_deg=np.rad2deg(spectrum_arrays.angle_vector),
		units="m",
		source_file=wave_spectrum.source_file,
		valid_time=wave_spectrum.valid_time,
		metadata=dict(wave_spectrum.metadata),
		depth=wave_spectrum.depth,
	)


def deep_water_wavenumber_vector(
	frequency_vector_hz: np.ndarray,
	gravity: float = GRAVITY,
) -> np.ndarray:
	"""Return the deep-water wavenumber vector: k = (2*pi*f)^2 / g."""
	frequency_vector_hz = np.asarray(frequency_vector_hz, dtype=float)
	return (2.0 * np.pi * frequency_vector_hz) ** 2 / gravity


def _domain_matrices(
	spectrum_arrays: _SpectrumArrays,
) -> Tuple[np.ndarray, np.ndarray]:
	"""Return (direction_matrix, wavenumber_matrix) meshgrids."""
	return np.meshgrid(
		spectrum_arrays.angle_vector,
		spectrum_arrays.wavenumber_vector,
		indexing="xy",
	)


def variance_to_amplitude_array(
	angle_vector: np.ndarray,
	frequency_vector_hz: np.ndarray,
	directional_variance_spectrum: np.ndarray,
) -> np.ndarray:
	"""Convert a directional variance spectrum to an amplitude spectrum."""
	angle_vector = np.asarray(angle_vector, dtype=float).reshape(-1)
	frequency_vector_hz = np.asarray(frequency_vector_hz, dtype=float).reshape(-1)
	directional_variance_spectrum = np.asarray(directional_variance_spectrum, dtype=float)

	if directional_variance_spectrum.shape != (frequency_vector_hz.size, angle_vector.size):
		raise ValueError("directional_variance_spectrum must have shape (frequency, angle).")

	df = np.diff(frequency_vector_hz)
	df = np.append(df, df[-1])
	if angle_vector.size == 1:
		dang = np.array([1.0], dtype=float)
	else:
		dang = np.diff(angle_vector)
		dang = np.append(dang, dang[-1])

	dang_grid, df_grid = np.meshgrid(dang, df, indexing="xy")
	return np.sqrt(2.0 * np.maximum(directional_variance_spectrum, 0.0) * dang_grid * df_grid)


def longuet_higgins_slope_pdf(
	wave_spectrum: WaveSpectrum,
	slope1: float,
	slope2: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
	"""Return the Longuet-Higgins slope PDF for a WaveSpectrum."""
	spectrum_arrays = _prepare_spectrum(wave_spectrum)
	direction_matrix, wavenumber_matrix = _domain_matrices(spectrum_arrays)
	amplitude_spectrum = variance_to_amplitude_array(
		spectrum_arrays.angle_vector,
		spectrum_arrays.frequency_vector_hz,
		spectrum_arrays.directional_variance_spectrum,
	)
	return longuet_higgins_slope_pdf_arrays(
		direction_matrix,
		wavenumber_matrix,
		amplitude_spectrum,
		slope1=slope1,
		slope2=slope2,
		use_unidirectional_limit=bool(np.isclose(wave_spectrum.Omega_0(), 1.0, atol=1.0e-12)),
		warning_context=_warning_context_from_metadata(wave_spectrum.metadata),
	)


def longuet_higgins_slope_pdf_arrays(
	direction_matrix: np.ndarray,
	wavenumber_matrix: np.ndarray,
	directional_amplitude_spectrum: np.ndarray,
	slope1: float,
	slope2: float,
	use_unidirectional_limit: bool = False,
	warning_context: str | None = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
	"""Compute the Longuet-Higgins directional slope PDF."""
	slope_magnitude_vector, slope_direction_vector = slope_vector(slope1, slope2)
	slope_direction_matrix, slope_magnitude_matrix = np.meshgrid(
		slope_direction_vector,
		slope_magnitude_vector,
		indexing="xy",
	)
	m20, m02, m11, _xi, delta2, _gamma = longuet_higgins_steepness_integrals(
		direction_matrix,
		wavenumber_matrix,
		directional_amplitude_spectrum,
	)
	if use_unidirectional_limit:
		warnings.warn(
			"\n"
			"========================================\n"
			"UNIDIRECTIONALITY WARNING\n"
			"========================================\n"
			"Omega_0 is effectively 1.\n"
			f"{_warning_context_line(warning_context)}"
			"Using the long-crested/unidirectional Longuet-Higgins slope-PDF limit\n"
			"instead of the directional formula.\n"
			"========================================\n",
			RuntimeWarning,
			stacklevel=2,
		)
		slope_pdf = longuet_higgins_pdf_unidirectional(
			slope_magnitude_vector,
			m20,
			m02,
		)
	else:
		slope_pdf = longuet_higgins_pdf_directional(
			slope_magnitude_matrix,
			slope_direction_matrix,
			delta2,
			m20,
			m02,
			m11,
		)
	return slope_magnitude_vector, slope_pdf, float(np.sqrt(max(m20 + m02, 0.0)))


def _warning_context_from_metadata(metadata: Dict[str, object]) -> str | None:
	"""Build a compact warning context string from WaveSpectrum metadata."""
	parts = []
	if "experiment_index" in metadata:
		parts.append(f"experiment={int(metadata['experiment_index']) + 1}")
	if "force" in metadata:
		parts.append(f"force={metadata['force']}")
	if "spread_deg" in metadata:
		parts.append(f"spread_deg={float(metadata['spread_deg']):.0f}")
	if "cross_deg" in metadata:
		parts.append(f"cross_deg={float(metadata['cross_deg']):.0f}")
	if not parts:
		return None
	return ", ".join(parts)


def _warning_context_line(warning_context: str | None) -> str:
	"""Return formatted warning context line for terminal output."""
	if not warning_context:
		return ""
	return f"Context: {warning_context}\n"


def longuet_higgins_steepness_integrals(
	direction_matrix: np.ndarray,
	wavenumber_matrix: np.ndarray,
	directional_amplitude_spectrum: np.ndarray,
) -> Tuple[float, float, float, np.ndarray, float, float]:
	"""Compute the Longuet-Higgins steepness spectral moments."""
	kx = wavenumber_matrix * np.cos(direction_matrix)
	ky = wavenumber_matrix * np.sin(direction_matrix)

	m20_integrand = 0.5 * (directional_amplitude_spectrum ** 2) * (kx ** 2)
	m02_integrand = 0.5 * (directional_amplitude_spectrum ** 2) * (ky ** 2)
	m11_integrand = 0.5 * (directional_amplitude_spectrum ** 2) * ky * kx

	m20 = float(np.sum(m20_integrand))
	m02 = float(np.sum(m02_integrand))
	m11 = float(np.sum(m11_integrand))

	xi = np.array([[m20, m11], [m11, m02]], dtype=float)
	delta2 = float(np.linalg.det(xi))

	m2_max = (m20 + m02) - np.sqrt((m20 - m02) ** 2 + 4.0 * m11 ** 2)
	m2_min = (m20 + m02) + np.sqrt((m20 - m02) ** 2 + 4.0 * m11 ** 2)
	if m2_min <= 0.0:
		gamma = 0.0
	else:
		gamma = float(np.sqrt(max(m2_max / m2_min, 0.0)))

	return m20, m02, m11, xi, delta2, gamma


def longuet_higgins_pdf_directional(
	slope_magnitude_matrix: np.ndarray,
	slope_direction_matrix: np.ndarray,
	delta2: float,
	m20: float,
	m02: float,
	m11: float,
) -> np.ndarray:
	"""Evaluate the directional Longuet-Higgins slope PDF."""
	delta2 = float(delta2)
	if delta2 <= 0.0 or not np.isfinite(delta2):
		return np.zeros(slope_magnitude_matrix.shape[0], dtype=float)

	trig_term = (
		m02 * np.cos(slope_direction_matrix) ** 2
		- 2.0 * m11 * np.cos(slope_direction_matrix) * np.sin(slope_direction_matrix)
		+ m20 * np.sin(slope_direction_matrix) ** 2
	)
	p_dir = (
		slope_magnitude_matrix
		/ (2.0 * np.pi * np.sqrt(delta2))
		* np.exp(-(slope_magnitude_matrix ** 2) * trig_term / (2.0 * delta2))
	)
	slope_direction_vector = slope_direction_matrix[0, :]
	return np.trapezoid(p_dir, x=slope_direction_vector, axis=1)


def longuet_higgins_pdf_unidirectional(
	slope_magnitude_vector: np.ndarray,
	m20: float,
	m02: float,
) -> np.ndarray:
	"""Return the long-crested limit of the Longuet-Higgins slope-magnitude PDF.

	This is the 1-D Gaussian-slope limit used when Omega_0 is effectively 1,
	so the directional PDF becomes singular in its 2-D form.
	"""
	slope_magnitude_vector = np.asarray(slope_magnitude_vector, dtype=float).reshape(-1)
	slope_variance = float(m20 + m02)
	if slope_variance <= 0.0 or not np.isfinite(slope_variance):
		return np.zeros_like(slope_magnitude_vector)

	eta = slope_magnitude_vector / np.sqrt(slope_variance)
	return np.sqrt(2.0 / np.pi) * np.exp(-0.5 * eta ** 2) / np.sqrt(slope_variance)


def slope_vector(slope1: float, slope2: float) -> Tuple[np.ndarray, np.ndarray]:
	"""Return slope magnitude and direction vectors for the PDF integration domain."""
	if slope2 <= slope1:
		raise ValueError("slope2 must be greater than slope1.")
	if SLOPE_INTERVAL <= 0.0:
		raise ValueError("SLOPE_INTERVAL must be positive.")

	n_steps = int(np.floor((slope2 - slope1) / SLOPE_INTERVAL))
	slope_magnitude_vector = slope1 + SLOPE_INTERVAL * np.arange(n_steps + 1)
	slope_direction_vector = np.linspace(-np.pi, np.pi, 1001)[:-1]
	return slope_magnitude_vector, slope_direction_vector


def pdf_moments(x: np.ndarray, fx: np.ndarray, c: float = 0.0, n: int = 2) -> np.ndarray:
	"""Compute moments of a PDF up to order n."""
	x = np.asarray(x, dtype=float).reshape(-1)
	fx = np.asarray(fx, dtype=float).reshape(-1)

	if x.size != fx.size:
		raise ValueError("x and fx must have the same size.")

	mu = np.ones(n + 1, dtype=float)
	if n >= 1:
		mu[1] = 0.0

	total = float(np.trapezoid(fx, x=x))
	if total <= 0.0 or not np.isfinite(total):
		return np.zeros(n + 1, dtype=float)

	for order in range(n + 1):
		y = ((x - (c + mu[1])) ** order) * (fx / mu[0])
		denominator = np.sqrt(mu[2]) ** order if n >= 2 else 1.0
		mu[order] = float(np.trapezoid(y, x=x) / denominator)

	return mu
