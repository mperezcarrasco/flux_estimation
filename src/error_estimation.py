#!/usr/bin/env python3
"""Monte-Carlo retrieval-error estimation for the CO (CSF) and NOx (EMG) fluxes.

This module drives the existing retrieval machinery (``csf_utils``,
``emg_utils``, ``wind_helpers``) repeatedly under randomly sampled inputs to
produce a *single combined* uncertainty distribution per emitted species, plus
the per-member input log needed for a tornado / variance-decomposition
diagnostic.

Design (agreed in the brainstorm, see CLAUDE.md "writing the paper" stage)
-------------------------------------------------------------------------
Sampled sources (one combined ensemble, reported as median + [2.5, 97.5] %):

* **Injection pressure** — drawn from a KDE fit to the AER_LH↔AER_AI matched
  pixels; the sampled pressure deterministically sets the ERA5 wind (we do NOT
  also sample the wind, to avoid double-counting the pressure→wind coupling).
* **Fire centroid** — drawn from an FRP-weighted 2-D KDE of the local MODIS
  detections; moves the rotation origin and the ERA5 grid point.
* **Analyst choices** — AI plume threshold, QA threshold, transect geometry /
  domain extent, and the background window. (The CO grid is fixed: 4 km Adams
  oversampled grid, see `oversample_co_ai_grid`.)

Deliberate decouplings (documented so they are not silently assumed):

* The injection-pressure KDE and the centroid KDE are each built **once** at the
  baseline AI threshold / centroid. Their *first-order* effect (wind speed,
  plume mask, rotation) is fully sampled every member; only the second-order
  feedback of a sampled AI threshold onto the *pressure distribution itself* is
  neglected.

Fail-loud policy (global ~/.claude/CLAUDE.md)
---------------------------------------------
Every parameter is explicit (no silent defaults for sampled quantities); shape
and value invariants are asserted at boundaries; only the *specific*
retrieval-failure exceptions (``ValueError``, ``RuntimeError`` from the fit /
CSF routines) are caught — and even then the member is recorded as a failure
and the run aborts loudly if the failure fraction exceeds ``max_fail_fraction``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import gaussian_kde

from .wind_helpers import match_alh_plume_pressures
from .csf_utils import compute_csf_dynamic
from .oversampling import GridSpec, oversample_to_grid
from .emg_utils import bin_to_grid, build_line_density, fit_emg_bootstrap, rotate_to_wind_frame

# ── Physical / geometric constants ─────────────────────────────────────────
M_PER_DEG_LAT: float = 111_320.0
EARTH_RADIUS_KM: float = 6_371.0


# ===========================================================================
# Observation-informed priors
# ===========================================================================
def build_pressure_kde(
    ai_raw: np.ndarray,
    ai_lat: np.ndarray,
    ai_lon: np.ndarray,
    alh_pres_raw: np.ndarray,
    alh_lat: np.ndarray,
    alh_lon: np.ndarray,
    fire_lat: float,
    fire_lon: float,
    ai_threshold: float,
    radius_km: float,
    min_pixels: int,
) -> dict:
    """Fit a 1-D KDE to the AER_LH injection pressures matched to the AER_AI plume.

    Reuses ``wind_helpers.match_alh_plume_pressures`` (the single source of
    truth for the 3-step match) and retains the full distribution rather than a
    scalar aggregate.

    Returns
    -------
    dict with keys ``kde`` (scipy gaussian_kde), ``samples_hpa`` (matched
    pressures), ``n``, ``median_hpa``, ``iqr_hpa``.
    """
    matched = match_alh_plume_pressures(
        ai_raw, ai_lat, ai_lon,
        alh_pres_raw, alh_lat, alh_lon,
        fire_lat, fire_lon,
        ai_threshold=ai_threshold, radius_km=radius_km,
    )
    pres = matched["pres_hpa"]
    if pres.size < min_pixels:
        raise ValueError(
            f"Only {pres.size} matched AER_LH pixels (< min_pixels={min_pixels}); "
            "cannot build a reliable injection-pressure KDE. Widen radius_km or "
            "lower ai_threshold."
        )
    q25, q75 = np.percentile(pres, [25, 75])
    return dict(
        kde=gaussian_kde(pres, bw_method="scott"),
        samples_hpa=pres,
        n=int(pres.size),
        median_hpa=float(np.median(pres)),
        mean_hpa=float(np.mean(pres)),
        iqr_hpa=float(q75 - q25),
    )


def build_centroid_kde(
    modis_csv: Path,
    fire_date: str,
    fire_lat: float,
    fire_lon: float,
    local_radius_km: float,
    min_confidence: int,
    satellite: str,
    daynight: str,
) -> dict:
    """Fit an FRP-weighted 2-D KDE to the local MODIS detections.

    Mirrors the spatial prior used in
    ``notebooks/extra/pysplit_probabilistic_trajectories.ipynb``: daytime
    MODIS detections of the requested satellite with ``confidence >=
    min_confidence`` within ``local_radius_km`` of the dominant centroid,
    weighted by FRP.

    Returns
    -------
    dict with keys ``kde`` (2-D gaussian_kde over rows [lat, lon]), ``n``,
    ``lats``, ``lons``, ``frp``.
    """
    df = pd.read_csv(modis_csv, parse_dates=["acq_date"])
    target = pd.Timestamp(fire_date).date()
    sel = df[
        (df["acq_date"].dt.date == target)
        & (df["satellite"] == satellite)
        & (df["confidence"] >= min_confidence)
        & (df["daynight"] == daynight)
    ].copy()
    if len(sel) < 3:
        raise ValueError(
            f"Only {len(sel)} MODIS detections on {target} "
            f"(sat={satellite}, conf>={min_confidence}, {daynight}); need >= 3 for a 2-D KDE."
        )
    lats = sel["latitude"].to_numpy()
    lons = sel["longitude"].to_numpy()
    frp = sel["frp"].to_numpy()

    dist = _haversine_km(fire_lat, fire_lon, lats, lons)
    local = dist <= local_radius_km
    if local.sum() < 3:
        raise ValueError(
            f"Only {int(local.sum())} MODIS detections within {local_radius_km:.0f} km "
            "of the centroid; need >= 3. Increase local_radius_km."
        )
    lats, lons, frp = lats[local], lons[local], frp[local]
    weights = frp / frp.sum()
    return dict(
        kde=gaussian_kde(np.vstack([lats, lons]), bw_method="scott", weights=weights),
        n=int(local.sum()),
        lats=lats, lons=lons, frp=frp,
    )


# ===========================================================================
# ERA5 wind: pressure -> (u, v)
# ===========================================================================
@dataclass
class Era5Slice:
    """Pre-loaded ERA5 pressure-level wind at the overpass time.

    Pressures are stored ascending; ``u``/``v`` have shape (n_levels, nlat, nlon).
    """
    lats: np.ndarray
    lons: np.ndarray
    p_sorted: np.ndarray      # (nlev,) ascending hPa
    u: np.ndarray             # (nlev, nlat, nlon)
    v: np.ndarray
    selected_time: pd.Timestamp


def load_era5_pl_slice(path: Path, overpass_utc: pd.Timestamp) -> Era5Slice:
    """Load the ERA5 pressure-level wind slice nearest ``overpass_utc`` into NumPy."""
    with xr.open_dataset(path) as ds:
        tdim = "valid_time" if "valid_time" in ds.dims else "time"
        dst = ds.sel({tdim: overpass_utc}, method="nearest")
        lat_key = "latitude" if "latitude" in dst.coords else "lat"
        lon_key = "longitude" if "longitude" in dst.coords else "lon"
        lats = dst[lat_key].values
        lons = dst[lon_key].values
        plev = dst["pressure_level"].values.ravel()  # hPa
        u = dst["u"].values
        v = dst["v"].values
        sel_t = pd.Timestamp(dst[tdim].values)

    # u/v dims are (level, lat, lon); enforce that ordering.
    assert u.shape[0] == plev.size, (
        f"ERA5 u first axis ({u.shape[0]}) != n pressure levels ({plev.size}); "
        "unexpected dimension ordering."
    )
    order = np.argsort(plev)  # ascending pressure
    return Era5Slice(
        lats=lats, lons=lons, p_sorted=plev[order],
        u=u[order], v=v[order], selected_time=sel_t,
    )


def wind_from_pressure(
    era5: Era5Slice, fire_lat: float, fire_lon: float, p_hpa: float
) -> tuple[float, float, float, float]:
    """Log-pressure-interpolate ERA5 wind to ``p_hpa`` at the nearest grid point.

    Returns ``(u, v, speed, theta_rad)`` with ``theta = atan2(v, u)`` (CCW from East).
    """
    i_lat = int(np.argmin(np.abs(era5.lats - fire_lat)))
    i_lon = int(np.argmin(np.abs(era5.lons - fire_lon)))
    u_prof = era5.u[:, i_lat, i_lon]
    v_prof = era5.v[:, i_lat, i_lon]
    log_p = np.log(era5.p_sorted)
    u = float(np.interp(np.log(p_hpa), log_p, u_prof))
    v = float(np.interp(np.log(p_hpa), log_p, v_prof))
    return u, v, float(np.hypot(u, v)), float(np.arctan2(v, u))


# ===========================================================================
# CO — Cross-Sectional Flux, gridded
# ===========================================================================
def bin_co_ai_grid(
    co_flat: np.ndarray, co_lat: np.ndarray, co_lon: np.ndarray,
    ai_flat: np.ndarray, ai_lat: np.ndarray, ai_lon: np.ndarray,
    lat_min: float, lat_max: float, lon_min: float, lon_max: float,
    grid_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bin flat CO and AER_AI pixels onto a common regular grid (nan-mean).

    Mirrors ``02_co_emission_csf.ipynb``'s ``bin_flat_to_grid`` but returns the
    raw CO and AI grids (the plume mask is applied per-member because the AI
    threshold is a sampled parameter).

    Returns ``(lat_c, lon_c, co_g, ai_g)`` with ``lat_c``/``lon_c`` ascending.
    """
    lat_edges = np.arange(lat_min, lat_max + grid_deg, grid_deg)
    lon_edges = np.arange(lon_min, lon_max + grid_deg, grid_deg)
    lat_c = (0.5 * (lat_edges[:-1] + lat_edges[1:])).astype(np.float32)
    lon_c = (0.5 * (lon_edges[:-1] + lon_edges[1:])).astype(np.float32)
    nlat, nlon = len(lat_c), len(lon_c)

    def _nanmean_bin(vals, lats, lons):
        grid = np.full((nlat, nlon), np.nan, dtype=np.float32)
        il = np.searchsorted(lat_edges[1:-1], lats)
        jl = np.searchsorted(lon_edges[1:-1], lons)
        ok = (il >= 0) & (il < nlat) & (jl >= 0) & (jl < nlon) & np.isfinite(vals)
        df = pd.DataFrame({"v": vals[ok].astype(float), "i": il[ok], "j": jl[ok]})
        for (i, j), v in df.groupby(["i", "j"])["v"].mean().items():
            grid[i, j] = float(v)
        return grid

    co_g = _nanmean_bin(co_flat, co_lat, co_lon)
    ai_g = _nanmean_bin(ai_flat.astype(np.float32), ai_lat, ai_lon)
    return lat_c, lon_c, co_g, ai_g


def oversample_co_ai_grid(
    co: dict, ai: dict, grid: GridSpec, radius_km: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Oversample CO and AER_AI pixels onto ``grid`` (Adams et al. 2019, see src.oversampling).

    ``co`` / ``ai`` are loader dicts with ``with_corners=True`` (keys ``co``/``ai``,
    ``lat_b``, ``lon_b``); ``ai["ai"]`` should already be clipped to the plume
    boundary. Returns ``(lat_c, lon_c, co_g, ai_g, diag)``: ascending centres (deg),
    CO (mol m⁻²) and AI grids (nlat, nlon), and per-species oversampling diagnostics.
    """
    for name, d, key in (("CO", co, "co"), ("AER_AI", ai, "ai")):
        assert "lat_b" in d and "lon_b" in d, f"{name} dict lacks pixel corners (load with_corners=True)."
        assert d["lat_b"].shape == (d[key].size, 4), f"{name}: corners do not match values."
    co_g, diag_co = oversample_to_grid(co["co"], co["lat_b"], co["lon_b"], grid, radius_km)
    ai_g, diag_ai = oversample_to_grid(ai["ai"].astype(np.float64), ai["lat_b"], ai["lon_b"],
                                       grid, radius_km)
    return grid.lat, grid.lon, co_g, ai_g, {"co": diag_co, "ai": diag_ai}


def co_flux_member(
    lat: np.ndarray, lon: np.ndarray, co_g: np.ndarray, ai_g: np.ndarray,
    u_wind: float, v_wind: float, theta_rad: float,
    ai_threshold: float,
    bg_x_min_km: float, bg_x_max_km: float,
    n_transects: int, transect_half_width_km: float,
    min_bg_pixels: int, min_valid_fraction: float, sample_spacing_m: float,
) -> dict:
    """One CSF CO-flux retrieval under a sampled parameter set.

    The wind is spatially uniform at the sampled injection pressure
    (``u_wind``/``v_wind``), matching the AER_LH variant that the manuscript
    reports as the headline CO number.

    ``lat``/``lon``/``co_g``/``ai_g`` come from `oversample_co_ai_grid` (4 km
    Adams oversampled grid); ``sample_spacing_m`` is the transect sample spacing
    (the grid cell size) and ``min_valid_fraction`` the transect coverage gate
    (accept if more than this fraction of samples is finite).

    ``ai_g`` is expected to already be clipped to the saved plume boundary
    (data/{MMDD}/plume_boundary.geojson) by the caller before gridding — this
    function no longer applies its own ad hoc latitude clip; that spatial
    gating now lives entirely in the boundary.

    Returns a dict with ``flux_kgs``, ``bg_mmol_m2``, ``n_plume``,
    ``n_valid_transects``. Raises ``ValueError`` (propagated from the CSF
    routine, or for an empty plume / background window) so the caller can record
    the member as a controlled failure.
    """
    co_finite = np.isfinite(co_g)
    plume = (ai_g > ai_threshold) & co_finite
    if not plume.any():
        raise ValueError(f"No plume pixels at AI > {ai_threshold}.")

    # Downwind projection (km) for the background window, using the sampled wind direction.
    lat_c = float(lat.mean())
    lon_c = float(lon.mean())
    lo_g, la_g = np.meshgrid(lon, lat)
    x_m = (lo_g - lon_c) * np.cos(np.radians(lat_c)) * M_PER_DEG_LAT
    y_m = (la_g - lat_c) * M_PER_DEG_LAT
    x_rot_km = (x_m * np.cos(theta_rad) + y_m * np.sin(theta_rad)) / 1000.0

    bg_mask = (
        (x_rot_km >= bg_x_min_km) & (x_rot_km <= bg_x_max_km)
        & co_finite & (co_g > 0)
    )
    if int(bg_mask.sum()) < min_bg_pixels:
        raise ValueError(
            f"Background window [{bg_x_min_km:.0f}, {bg_x_max_km:.0f}] km has "
            f"{int(bg_mask.sum())} px (< {min_bg_pixels}); cannot define a stable background."
        )
    bg = float(np.nanmedian(co_g[bg_mask]))
    co_delta = co_g - bg

    u_field = np.full_like(co_g, u_wind, dtype=np.float64)
    v_field = np.full_like(co_g, v_wind, dtype=np.float64)

    _, t_flux, flux_kgs, _ = compute_csf_dynamic(
        co_delta, lat.astype(np.float64), lon.astype(np.float64),
        u_field, v_field, plume,
        n_transects=n_transects, transect_half_width_km=transect_half_width_km,
        aggregation="mean", sample_spacing_m=sample_spacing_m,
        min_valid_fraction=min_valid_fraction,
    )
    return dict(
        flux_kgs=float(flux_kgs),
        bg_mmol_m2=float(bg * 1e3),
        n_plume=int(plume.sum()),
        n_valid_transects=int(np.isfinite(t_flux).sum()),
    )


# ===========================================================================
# NOx — EMG line-density fit, with a pixel bootstrap
# ===========================================================================
def nox_flux_member(
    no2_lat: np.ndarray, no2_lon: np.ndarray, no2_vals: np.ndarray,
    fire_lat: float, fire_lon: float,
    u_wind: float, v_wind: float, wind_speed: float,
    extraction_radius_km: float,
    downwind_min_km: float, downwind_max_km: float, crosswind_km: float,
    grid_dy_km: float,
    nox_no2_ratio_gamma: float, no2_molar_mass_g_per_mol: float,
    rng: np.random.Generator, bootstrap_iters: int,
    min_x0_km: float,
) -> dict:
    """One EMG NOx retrieval under a sampled parameter set.

    Uses the full in-scene NO2 pixel set (no bootstrap resampling). Pixel
    bootstrap was removed because a single draw systematically undersamples
    sparse plume-peak cells, introducing a downward bias in E that dominates
    the physical parameter uncertainty. The EMG fit stability is still assessed
    internally via perturbed initial-condition runs (``fit_emg_bootstrap``).

    No explicit background is subtracted: the EMG additive offset B absorbs a
    spatially-uniform background, matching the canonical retrieval.

    Parameters
    ----------
    min_x0_km : float
        Physical lower bound on x0 (km), equal to ``grid_dy_km``. Passed to
        ``fit_emg_bootstrap`` as the ``curve_fit`` lower bound so the optimizer
        cannot fit a sub-grid decay length. After the fit, if x0 is at or below
        this floor the member is failed loudly as a degenerate fit.

    Returns a dict with ``E_nox_g_per_s``, ``tau_h``, ``r2``, ``x0_km``,
    ``n_scene``. Raises ``ValueError`` / ``RuntimeError`` for the caller to
    record as a controlled failure.
    """
    theta = float(np.arctan2(v_wind, u_wind))

    dist = _haversine_km(fire_lat, fire_lon, no2_lat, no2_lon)
    scene = (dist <= extraction_radius_km) & np.isfinite(no2_vals)
    n_scene = int(scene.sum())
    if n_scene < 50:
        raise ValueError(f"Only {n_scene} in-scene NO2 pixels (< 50); scene too sparse.")

    lat_s = no2_lat[scene]
    lon_s = no2_lon[scene]
    no2_s = no2_vals[scene]

    x_rot, y_rot = rotate_to_wind_frame(lat_s, lon_s, fire_lat, fire_lon, theta)

    x_centres, _, grid_mean, grid_cnt = bin_to_grid(
        x_rot, y_rot, no2_s,
        x_min_km=downwind_min_km, x_max_km=downwind_max_km,
        y_max_km=crosswind_km, grid_dy_km=grid_dy_km,
    )

    # No explicit background subtraction: the EMG additive offset B absorbs a
    # spatially-uniform background, so an upwind subtraction is redundant for the
    # flux. This matches the canonical retrieval (notebook 03 / run_emg_pipeline),
    # which fits the line density of grid_mean directly.
    profile = build_line_density(grid_mean, grid_cnt, grid_dy_km * 1000.0)

    fit = fit_emg_bootstrap(
        x_centres, profile,
        wind_speed_m_s=wind_speed,
        nox_no2_ratio_gamma=nox_no2_ratio_gamma,
        no2_molar_mass_g_per_mol=no2_molar_mass_g_per_mol,
        bootstrap_iterations=bootstrap_iters,
        bootstrap_seed=int(rng.integers(0, 2**31 - 1)),
        perturbation_low=0.8, perturbation_high=1.2,
        curve_fit_maxfev=5000,
        cv_stability_threshold=0.5,
        peak_offset_km_max=30.0,
        x0_min_km=min_x0_km,
    )

    # Fail loudly if x0 is at the lower bound: the optimizer could not find a
    # physically resolvable decay length, meaning the plume geometry in this
    # rotated frame is inconsistent with the EMG assumption. Returning E∝1/x0
    # from a boundary solution would be a mathematical artifact, not a physical
    # uncertainty. The 1% tolerance catches floating-point boundary proximity.
    if fit["x0_km"] <= min_x0_km * 1.01:
        raise ValueError(
            f"EMG degenerate fit: x0={fit['x0_km']:.3f} km is at the lower "
            f"bound ({min_x0_km:.3f} km). Plume geometry likely inconsistent "
            f"with EMG assumption (wind_speed={wind_speed:.2f} m/s, "
            f"fire_lon={fire_lon:.3f}, R2={fit['r2']:.3f})."
        )

    return dict(
        E_nox_g_per_s=float(fit["E_nox_g_per_s"]),
        tau_h=float(fit["tau_h"]),
        r2=float(fit["r2"]),
        x0_km=float(fit["x0_km"]),
        a_mol_per_m=float(fit["a_mol_per_m"]),
        # Fit-quality diagnostics — logged (not gated) so degenerate fits can be
        # filtered post-hoc during reanalysis (user chose to report the raw ensemble).
        cv_E=float(fit["cv_E"]),
        peak_x_km=float(fit["peak_x_km"]),
        fit_stable=bool(fit["fit_stable"]),
        peak_within_30km=bool(fit["peak_within_30km"]),
        n_converged=int(fit["n_converged"]),
        n_scene=n_scene,
    )


# ===========================================================================
# Sensitivity decomposition (tornado)
# ===========================================================================
def tornado_sensitivity(
    inputs: pd.DataFrame, output: np.ndarray, output_name: str
) -> pd.DataFrame:
    """Rank sampled inputs by their contribution to the output variance.

    Two complementary, ensemble-based measures (no extra model runs):

    * ``spearman`` — rank correlation of each input with the output (robust to
      monotone non-linearity).
    * ``std_beta`` — standardised OLS regression coefficient (z-scored inputs
      and output); ``beta**2`` approximates the first-order variance share when
      inputs are near-independent (they are, by construction of the sampler).

    Constant columns (e.g. a parameter that did not actually vary) are dropped
    with no error. Returns a DataFrame sorted by ``abs(std_beta)`` descending.
    """
    assert len(inputs) == len(output), "inputs and output length mismatch."
    finite = np.isfinite(output)
    X = inputs.loc[finite].copy()
    y = np.asarray(output)[finite].astype(float)

    # Drop non-varying inputs.
    varying = [c for c in X.columns if np.nanstd(X[c].to_numpy(dtype=float)) > 0]
    X = X[varying]

    # Spearman via rank-transform Pearson.
    yr = pd.Series(y).rank().to_numpy()
    spearman = {
        c: float(np.corrcoef(pd.Series(X[c].to_numpy(dtype=float)).rank().to_numpy(), yr)[0, 1])
        for c in varying
    }

    # Standardised OLS.
    Xz = (X - X.mean()) / X.std()
    yz = (y - y.mean()) / y.std()
    Xmat = np.column_stack([np.ones(len(yz)), Xz.to_numpy()])
    beta, *_ = np.linalg.lstsq(Xmat, yz, rcond=None)
    std_beta = dict(zip(varying, beta[1:]))

    out = pd.DataFrame({
        "parameter": varying,
        "spearman": [spearman[c] for c in varying],
        "std_beta": [std_beta[c] for c in varying],
    })
    out["abs_std_beta"] = out["std_beta"].abs()
    out["var_share_pct"] = 100.0 * out["std_beta"] ** 2 / (out["std_beta"] ** 2).sum()
    out = out.sort_values("abs_std_beta", ascending=False).reset_index(drop=True)
    out.attrs["output_name"] = output_name
    return out


# ===========================================================================
# Helpers
# ===========================================================================
def _haversine_km(
    lat1: float, lon1: float, lat2_arr: np.ndarray, lon2_arr: np.ndarray
) -> np.ndarray:
    lat1r, lon1r = np.deg2rad(lat1), np.deg2rad(lon1)
    lat2r = np.deg2rad(lat2_arr)
    lon2r = np.deg2rad(lon2_arr)
    a = (np.sin((lat2r - lat1r) / 2) ** 2
         + np.cos(lat1r) * np.cos(lat2r) * np.sin((lon2r - lon1r) / 2) ** 2)
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def summarize(values: np.ndarray, name: str, unit: str) -> dict:
    """Median + 95% CI summary of a finite-filtered ensemble."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        raise ValueError(f"No finite values to summarize for {name}.")
    p2_5, p50, p97_5 = np.percentile(v, [2.5, 50, 97.5])
    return dict(
        name=name, unit=unit, n=int(v.size),
        median=float(p50), p2_5=float(p2_5), p97_5=float(p97_5),
        mean=float(v.mean()), std=float(v.std()),
        rel_err_pct=float(100.0 * 0.5 * (p97_5 - p2_5) / p50) if p50 != 0 else float("nan"),
    )


# ===========================================================================
# Plume-alignment gate (moved from nb07/nb08, Oct 2026; numerically identical)
# ===========================================================================
KM_PER_DEG: float = 111.32   # km per degree (latitude; longitude × cos(lat)) used by the gate

ALIGNMENT_GATE_KEYS: tuple[str, ...] = (
    "pca_radius_km", "far_field_min_km", "max_wind_plume_angle_deg",
    "min_pca_pixels", "min_far_field_pixels",
)


def load_alignment_gate_config(path: Path) -> dict:
    """Read config/mc_alignment_gate.yaml; fail loudly on missing or unknown keys."""
    import yaml
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    missing = set(ALIGNMENT_GATE_KEYS) - set(cfg)
    unknown = set(cfg) - set(ALIGNMENT_GATE_KEYS)
    if missing or unknown:
        raise ValueError(f"{path}: missing keys {sorted(missing)}, unknown keys {sorted(unknown)}")
    return {k: float(cfg[k]) if k.endswith("_km") or k.endswith("_deg") else int(cfg[k])
            for k in ALIGNMENT_GATE_KEYS}


def haversine_km_from_point(lat1: float, lon1: float,
                            lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Great-circle distance (km) from one point to arrays of points (R = 6371 km)."""
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2
         + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2)
    return EARTH_RADIUS_KM * 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def plume_axis_pca(
    center_lat: float, center_lon: float,
    plume_lat: np.ndarray, plume_lon: np.ndarray, plume_ai: np.ndarray,
    *, radius_km: float, far_field_min_km: float,
    min_pixels: int, min_far_pixels: int,
) -> float:
    """AI-weighted PCA plume axis around a (sampled) fire centroid.

    plume_lat/lon/ai : (n,) AER_AI plume pixels (already AI > threshold and inside
                       the plume boundary), deg / unitless AI.
    Uses pixels within ``radius_km`` of the centroid; the first principal component
    of their AI-weighted positions (local x east, y north, km) is the axis, and its
    sign points toward the AI-weighted mean of pixels farther than
    ``far_field_min_km``.

    Returns θ_plume in radians (atan2(y, x) convention, CCW from east, downwind).
    Raises ValueError (message starts "PCA:") if fewer than ``min_pixels`` pixels
    are within the radius or fewer than ``min_far_pixels`` are in the far field.
    """
    assert plume_lat.shape == plume_lon.shape == plume_ai.shape and plume_lat.ndim == 1
    dist = haversine_km_from_point(center_lat, center_lon, plume_lat, plume_lon)
    within = dist <= radius_km
    n_pca = int(within.sum())
    if n_pca < min_pixels:
        raise ValueError(
            f"PCA: only {n_pca} plume pixels within {radius_km:.0f} km of "
            f"({center_lat:.3f}, {center_lon:.3f}) — sampled centroid outside plume."
        )
    cos_lat = np.cos(np.radians(center_lat))
    x_km = (plume_lon[within] - center_lon) * cos_lat * KM_PER_DEG   # east +
    y_km = (plume_lat[within] - center_lat) * KM_PER_DEG             # north +
    w = plume_ai[within]
    W = w / w.sum()
    x_c, y_c = float(W @ x_km), float(W @ y_km)
    dx, dy = x_km - x_c, y_km - y_c
    cov = np.array([[W @ (dx * dx), W @ (dx * dy)],
                    [W @ (dx * dy), W @ (dy * dy)]])
    _, eigvecs = np.linalg.eigh(cov)          # ascending eigenvalues
    axis = eigvecs[:, -1].copy()              # largest eigenvector = first PC
    r = np.sqrt(x_km ** 2 + y_km ** 2)
    far = r > far_field_min_km
    if far.sum() < min_far_pixels:
        raise ValueError(
            f"PCA: only {int(far.sum())} far-field pixels (>{far_field_min_km:.0f} km) "
            "for sign resolution."
        )
    x_ff = float(np.average(x_km[far], weights=w[far]))
    y_ff = float(np.average(y_km[far], weights=w[far]))
    if axis[0] * x_ff + axis[1] * y_ff < 0:
        axis = -axis
    return float(np.arctan2(axis[1], axis[0]))


def angular_diff_deg(theta1_rad: float, theta2_rad: float) -> float:
    """Smallest angle between two directions, in degrees (0–180)."""
    diff = float(np.degrees(theta1_rad - theta2_rad)) % 360.0
    return min(diff, 360.0 - diff)
