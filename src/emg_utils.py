"""Exponentially Modified Gaussian (EMG) line-density fitting for NOx flux.

Reference implementation:
    notebooks/nox_estimation_emg.ipynb (Jin et al. 2021)

Public API
----------
emg(x, a, x0, mu_x, sigma_x, B)
    EMG line density profile, mol m^-1.
fit_emg_bootstrap(...)
    Bootstrap fit with multiple perturbed initial guesses; returns
    median parameters and stability metrics.
rotate_to_wind_frame(...)
    Project flat lat/lon arrays into wind-aligned (x_rot, y_rot) km.
bin_to_grid(...)
    Bin rotated NO2 pixels onto a regular downwind x cross-wind grid.
build_line_density(...)
    Cross-wind nansum * dy -> 1D line density profile (mol m^-1).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import ndtr


def emg(
    x: np.ndarray,
    a: float, x0: float, mu_x: float, sigma_x: float, B: float,
) -> np.ndarray:
    """EMG line density profile (Jin et al. 2021, Eq. 1).

    Parameters
    ----------
    x : array
        Downwind distance, km.
    a : float
        Integrated line density (mol m^-1).
    x0 : float
        Apparent source offset / e-folding decay length (km).  Must be > 0.
    mu_x : float
        Plume centre offset (km).
    sigma_x : float
        Gaussian dispersion width (km).  Must be > 0.
    B : float
        Background line density (mol m^-1).

    Returns
    -------
    rho : array (mol m^-1)
    """
    x = np.asarray(x, dtype=float)
    x0 = max(x0, 1e-6)
    sigma_x = max(sigma_x, 1e-6)
    exponent = mu_x / x0 + sigma_x ** 2 / (2 * x0 ** 2) - x / x0
    exponent = np.clip(exponent, -500.0, 500.0)
    phi_arg = (x - mu_x) / sigma_x - sigma_x / x0
    return (a / x0) * np.exp(exponent) * ndtr(phi_arg) + B


def rotate_to_wind_frame(
    lat: np.ndarray,
    lon: np.ndarray,
    fire_lat: float,
    fire_lon: float,
    theta_rad: float,
    m_per_deg_lat: float = 111_320.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Project lat/lon arrays into wind-aligned km coordinates.

    Returns
    -------
    x_rot : km along the downwind axis (positive = downwind of fire)
    y_rot : km perpendicular to the wind
    """
    m_per_deg_lon = m_per_deg_lat * np.cos(np.radians(fire_lat))
    x_km = (lon - fire_lon) * m_per_deg_lon / 1000.0
    y_km = (lat - fire_lat) * m_per_deg_lat / 1000.0
    cos_t = np.cos(theta_rad)
    sin_t = np.sin(theta_rad)
    x_rot = x_km * cos_t + y_km * sin_t
    y_rot = -x_km * sin_t + y_km * cos_t
    return x_rot, y_rot


def bin_to_grid(
    x_rot: np.ndarray,
    y_rot: np.ndarray,
    no2: np.ndarray,
    x_min_km: float, x_max_km: float,
    y_max_km: float,
    grid_dy_km: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bin rotated NO2 pixels onto a regular downwind x cross-wind grid.

    Returns
    -------
    x_centres : (nx,) downwind cell centres (km)
    y_centres : (ny,) cross-wind cell centres (km)
    grid_mean : (nx, ny) mean NO2 per cell (mol m^-2; NaN where no pixels)
    grid_cnt  : (nx, ny) pixel count per cell
    """
    half = grid_dy_km / 2.0
    x_edges = np.arange(x_min_km, x_max_km + 1e-9, grid_dy_km)
    y_edges = np.arange(-y_max_km - half, y_max_km + half + 1e-9, grid_dy_km)
    x_centres = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centres = 0.5 * (y_edges[:-1] + y_edges[1:])
    nx, ny = len(x_centres), len(y_centres)

    ix = np.digitize(x_rot, x_edges) - 1
    iy = np.digitize(y_rot, y_edges) - 1
    valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)

    grid_sum = np.zeros((nx, ny))
    grid_cnt = np.zeros((nx, ny), dtype=int)
    for k in np.where(valid)[0]:
        grid_sum[ix[k], iy[k]] += no2[k]
        grid_cnt[ix[k], iy[k]] += 1
    grid_mean = np.where(grid_cnt > 0, grid_sum / grid_cnt, np.nan).astype(np.float32)
    return x_centres, y_centres, grid_mean, grid_cnt


def build_line_density(
    grid_mean: np.ndarray,
    grid_cnt: np.ndarray,
    grid_dy_m: float,
) -> np.ndarray:
    """Cross-wind nansum * dy -> 1D line density (mol m^-1).

    Rows with no valid cells become NaN.
    """
    dno2 = grid_mean.astype(np.float64)
    dno2[grid_cnt == 0] = np.nan
    dno2 = np.where(np.isfinite(dno2) & (dno2 != 0), dno2, np.nan)

    profile = np.nansum(np.where(np.isfinite(dno2), dno2, 0.0), axis=1) * grid_dy_m
    has_valid = np.any(np.isfinite(dno2), axis=1)
    return np.where(has_valid, profile, np.nan)


def fit_emg_bootstrap(
    x_centres: np.ndarray,
    profile: np.ndarray,
    wind_speed_m_s: float,
    nox_no2_ratio_gamma: float,
    no2_molar_mass_g_per_mol: float,
    bootstrap_iterations: int,
    bootstrap_seed: int,
    perturbation_low: float,
    perturbation_high: float,
    curve_fit_maxfev: int,
    cv_stability_threshold: float,
    peak_offset_km_max: float,
) -> dict:
    """Run bootstrap EMG fit and return summary metrics.

    Returns a dict with keys:
        n_converged, n_attempted,
        a_mol_per_m, x0_km, mu_km, sigma_km, B_mol_per_m,
        tau_h, E_mol_per_s, E_nox_g_per_s,
        r2, cv_a, cv_E, fit_stable,
        peak_x_km, peak_within_30km,
        x_fit, y_fit, y_pred, success_params (Nx5)
    """
    finite = np.isfinite(profile)
    x_fit = x_centres[finite]
    y_fit = profile[finite]

    if len(x_fit) < 10:
        raise ValueError(
            f"Only {len(x_fit)} finite line-density points (< 10 required). "
            "Check upstream gridding / background subtraction."
        )

    y_max = float(np.nanmax(y_fit))
    y_min = float(np.nanmin(y_fit))
    x_peak_obs = float(x_fit[np.argmax(y_fit)])

    p0 = [y_max * 50.0, 50.0, x_peak_obs * 0.5, 30.0, float(y_min)]
    bounds_lo = [0.0, 1.0, -200.0, 1.0, -1e6]
    bounds_hi = [np.inf, 500.0, 200.0, 200.0, float(max(y_max, 1.0))]

    rng = np.random.default_rng(bootstrap_seed)
    success_params: list[np.ndarray] = []

    for _ in range(bootstrap_iterations):
        pert = rng.uniform(perturbation_low, perturbation_high, size=5)
        p0_p = np.clip(np.array(p0) * pert, bounds_lo, bounds_hi)
        try:
            popt, _ = curve_fit(
                emg, x_fit, y_fit, p0=p0_p,
                bounds=(bounds_lo, bounds_hi),
                maxfev=curve_fit_maxfev,
            )
            if np.all(np.isfinite(popt)) and popt[0] > 0 and popt[1] > 0:
                success_params.append(popt)
        except (RuntimeError, ValueError):
            continue

    if not success_params:
        raise RuntimeError(
            f"EMG bootstrap: 0/{bootstrap_iterations} fits converged. "
            "Likely causes: bad rotation, insufficient signal-to-background, "
            "wrong fire centroid, or domain too small."
        )

    params_arr = np.array(success_params)
    a_med, x0_med, mu_med, sig_med, B_med = np.median(params_arr, axis=0)

    # Per-bootstrap E(NOx) for CV(E)
    U_kph = wind_speed_m_s * 3.6
    E_vals_mol_s = (params_arr[:, 0] / params_arr[:, 1]) * U_kph * 1000.0 / 3600.0
    cv_E = float(np.std(E_vals_mol_s) / np.mean(E_vals_mol_s))
    cv_a = float(np.std(params_arr[:, 0]) / np.mean(params_arr[:, 0]))

    tau_h = float(x0_med / U_kph)
    E_mol_s = float((a_med / x0_med) * U_kph * 1000.0 / 3600.0)
    E_nox_g_s = float(E_mol_s * nox_no2_ratio_gamma * no2_molar_mass_g_per_mol)

    y_pred = emg(x_fit, a_med, x0_med, mu_med, sig_med, B_med)
    ss_res = float(np.sum((y_fit - y_pred) ** 2))
    ss_tot = float(np.sum((y_fit - np.mean(y_fit)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    x_dense = np.linspace(-50.0, 150.0, 10_000)
    L_dense = emg(x_dense, a_med, x0_med, mu_med, sig_med, 0.0)
    peak_x_km = float(x_dense[np.argmax(L_dense)])

    return dict(
        n_converged=len(success_params),
        n_attempted=bootstrap_iterations,
        a_mol_per_m=float(a_med),
        x0_km=float(x0_med),
        mu_km=float(mu_med),
        sigma_km=float(sig_med),
        B_mol_per_m=float(B_med),
        tau_h=tau_h,
        E_mol_per_s=E_mol_s,
        E_nox_g_per_s=E_nox_g_s,
        r2=float(r2),
        cv_a=cv_a,
        cv_E=cv_E,
        fit_stable=bool(cv_E <= cv_stability_threshold),
        peak_x_km=peak_x_km,
        peak_within_30km=bool(abs(peak_x_km) <= peak_offset_km_max),
        x_fit=x_fit, y_fit=y_fit, y_pred=y_pred,
        success_params=params_arr,
    )
