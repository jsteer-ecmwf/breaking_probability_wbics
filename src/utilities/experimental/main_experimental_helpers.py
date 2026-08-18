"""Helper functions for main_experimental workflow."""

import warnings

import numpy as np

try:
    from scipy.stats import t as student_t
except Exception:  # pragma: no cover - fallback when SciPy is unavailable
    student_t = None


def wrapped_normal_pdf(
    theta: np.ndarray,
    mu: float,
    sigma: float,
    warning_context: str | None = None,
) -> np.ndarray:
    """Wrapped-normal PDF matching MATLAB wrapToPi-based implementation."""
    theta = np.asarray(theta, dtype=float).reshape(-1)
    theta_diff = (theta - mu + np.pi) % (2.0 * np.pi) - np.pi

    if sigma <= 0.0:
        if theta.size == 0:
            return np.array([], dtype=float)
        if theta.size == 1:
            warnings.warn(
                "\n"
                "========================================\n"
                "UNIDIRECTIONALITY WARNING\n"
                "========================================\n"
                "spread=0 detected.\n"
                f"{_warning_context_suffix(warning_context)}"
                "Assuming a unidirectional sea.\n"
                "Placing all directional energy in the only available theta bin.\n"
                "========================================\n",
                RuntimeWarning,
                stacklevel=2,
            )
            return np.array([1.0], dtype=float)

        closest_index = int(np.argmin(np.abs(theta_diff)))
        delta_pdf = np.zeros_like(theta, dtype=float)

        trapz_weights = np.empty_like(theta, dtype=float)
        trapz_weights[0] = 0.5 * (theta[1] - theta[0])
        trapz_weights[-1] = 0.5 * (theta[-1] - theta[-2])
        if theta.size > 2:
            trapz_weights[1:-1] = 0.5 * (theta[2:] - theta[:-2])

        warnings.warn(
            "\n"
            "========================================\n"
            "UNIDIRECTIONALITY WARNING\n"
            "========================================\n"
            "spread=0 detected.\n"
            f"{_warning_context_suffix(warning_context)}"
            "Assuming a unidirectional sea.\n"
            "Replacing the wrapped normal with a delta at the nearest theta bin.\n"
            "========================================\n",
            RuntimeWarning,
            stacklevel=2,
        )
        delta_pdf[closest_index] = 1.0 / trapz_weights[closest_index]
        return delta_pdf

    return (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(
        -((theta_diff ** 2) / (2.0 * sigma ** 2))
    )


def _warning_context_suffix(warning_context: str | None) -> str:
    """Return formatted warning context text for terminal output."""
    if not warning_context:
        return ""
    return f"Context: {warning_context}\n"


def directional_pdf(
    theta: np.ndarray,
    spread_deg: float,
    cross_deg: float,
    warning_context: str | None = None,
) -> np.ndarray:
    """Two-partition directional PDF matching directionalisation_fun.m."""
    sigma = np.deg2rad(spread_deg)
    half_cross = np.deg2rad(cross_deg / 2.0)

    d1 = wrapped_normal_pdf(theta, mu=-half_cross, sigma=sigma, warning_context=warning_context)
    d2 = wrapped_normal_pdf(theta, mu=+half_cross, sigma=sigma, warning_context=warning_context)
    combined = d1 + d2

    return combined / np.trapezoid(combined, x=theta)


def fit_linear_with_ci(x: np.ndarray, y: np.ndarray, x_fit: np.ndarray) -> dict:
    """Fit y = b0 + b1*x and return prediction + 95% CI for mean response."""
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    x_fit = np.asarray(x_fit, dtype=float).reshape(-1)

    n = x.size
    if n < 3:
        raise ValueError("At least 3 points are required for linear fit with CI.")

    x_matrix = np.column_stack([np.ones(n), x])
    xtx = x_matrix.T @ x_matrix
    xtx_inv = np.linalg.inv(xtx)
    beta = xtx_inv @ x_matrix.T @ y

    y_hat = x_matrix @ beta
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else np.nan

    dof = max(n - 2, 1)
    sigma2 = ss_res / dof
    t_crit = float(student_t.ppf(0.975, dof)) if student_t is not None else 1.96

    x_fit_matrix = np.column_stack([np.ones(x_fit.size), x_fit])
    y_pred = x_fit_matrix @ beta
    leverage = np.einsum("ij,jk,ik->i", x_fit_matrix, xtx_inv, x_fit_matrix)
    se_mean = np.sqrt(np.maximum(sigma2 * leverage, 0.0))
    ci_low = y_pred - t_crit * se_mean
    ci_high = y_pred + t_crit * se_mean

    return {
        "intercept": float(beta[0]),
        "slope": float(beta[1]),
        "r2": r2,
        "y_pred": y_pred,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "beta": beta,
        "xtx_inv": xtx_inv,
        "sigma2": float(sigma2),
        "dof": int(dof),
    }


def prediction_interval_observation(
    fit: dict,
    x_value: float,
    alpha: float = 0.01,
) -> tuple[float, float, float]:
    """Return y-hat and two-sided prediction interval for one x value."""
    x_row = np.array([1.0, float(x_value)], dtype=float)
    y_hat = float(x_row @ fit["beta"])
    leverage = float(x_row @ fit["xtx_inv"] @ x_row)
    se_pred = np.sqrt(max(fit["sigma2"] * (1.0 + leverage), 0.0))
    t_crit = float(student_t.ppf(1.0 - alpha / 2.0, fit["dof"])) if student_t is not None else 2.576
    return y_hat, y_hat - t_crit * se_pred, y_hat + t_crit * se_pred