"""End-to-end pipeline orchestrators for the EMG (NOx) and CSF (CO) commands.

Each ``run_*_pipeline`` function takes a fully resolved configuration dict
and a fire centroid, runs the corresponding notebook pipeline, writes
diagnostic plots and a JSON results file, and returns the results dict.

Reference notebooks:
    notebooks/nox_estimation_emg.ipynb
    notebooks/co_estimation_csf.ipynb
"""

from __future__ import annotations

import glob
import json
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import shapely
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

from src import plotting
from src import plume_helpers as ph
from src.csf_utils import (
    build_dynamic_transects,
    compute_csf_dynamic,
    compute_local_wind_direction,
    CO_MOLAR_MASS_KG_PER_MOL,
)
from src.data_helpers import (
    load_aer_ai, load_aer_lh, load_co_l2, load_no2_l2,
)
from src.oversampling import oversample_to_grid, regular_grid_km
from src.emg_utils import (
    bin_to_grid, build_line_density, fit_emg_bootstrap,
    rotate_to_wind_frame,
)
from src.wind_helpers import select_injection_pressure


# --------------------------------------------------------------------- shared

def _find_all(pattern_glob: str, label: str) -> list[Path]:
    """Resolve all L2 overpass files matching a glob, fail loudly if none."""
    matches = sorted(glob.glob(pattern_glob))
    if len(matches) == 0:
        raise FileNotFoundError(
            f"{label}: no file matched pattern {pattern_glob}"
        )
    return [Path(m) for m in matches]


def _load_and_concat(paths: list[Path], loader: Callable, **kwargs) -> dict:
    """Concatenate a loader's output across multiple L2 overpass files."""
    chunks: dict[str, list] = defaultdict(list)
    scalars: dict[str, Any] = {}
    for path in paths:
        data = loader(path, **kwargs)
        if data is None:
            continue  # skip orbits with no valid pixels in the ROI
        for k, v in data.items():
            if isinstance(v, np.ndarray):
                chunks[k].append(v)
            else:
                scalars[k] = v
    out = {k: np.concatenate(v) for k, v in chunks.items()}
    out.update(scalars)
    return out


def _load_plume_boundary_or_raise(data_root: Path, target_day: str) -> ph.PlumeBoundary:
    """Load the saved plume boundary for a date, failing loudly if missing.

    Boundaries are hand-reviewed (see notebooks/plume_boundary_inspection.ipynb)
    and only exist for dates where AER_AI actually produces a consistent
    plume detection at the fixed threshold — there is no safe automatic
    fallback for a missing boundary.
    """
    boundary_path = data_root / target_day / "plume_boundary.geojson"
    if not boundary_path.is_file():
        raise FileNotFoundError(
            f"No plume boundary found at {boundary_path}. Run "
            "notebooks/plume_boundary_inspection.ipynb for this date first — "
            "there is no automatic fallback (AER_AI does not reliably produce "
            "a consistent plume at a fixed threshold on every date)."
        )
    return ph.load_plume_boundary(boundary_path)


def _clip_ai_to_boundary(ai_data: dict, boundary: ph.PlumeBoundary) -> np.ndarray:
    """Clip AI below threshold outside the plume boundary (in place on a copy).

    Returns the clipped AI array; every downstream `ai > threshold` test
    (injection pressure, grid plume mask) becomes boundary-consistent.
    """
    in_boundary = shapely.contains_xy(boundary.boundary, ai_data["lon"], ai_data["lat"])
    return np.where(in_boundary, ai_data["ai"],
                    np.minimum(ai_data["ai"], boundary.threshold - 1e-3))


def _interp_log_p(
    p_target_hpa: float,
    p_sorted: np.ndarray,
    u_sorted: np.ndarray,
    v_sorted: np.ndarray,
) -> tuple[float, float]:
    """Log-pressure interpolation of ERA5 PL winds to a target pressure."""
    log_p = np.log(p_sorted)
    u_i = float(np.interp(np.log(p_target_hpa), log_p, u_sorted))
    v_i = float(np.interp(np.log(p_target_hpa), log_p, v_sorted))
    return u_i, v_i


def _era5_pl_wind_at_fire(
    era5_pl_path: Path,
    overpass_utc: pd.Timestamp,
    fire_lat: float,
    fire_lon: float,
    p_target_hpa: float,
) -> dict:
    """Interpolate ERA5 pressure-level winds to (overpass time, fire grid pt, p)."""
    if not era5_pl_path.exists():
        raise FileNotFoundError(
            f"ERA5 pressure-level file not found: {era5_pl_path}"
        )
    ds = xr.open_dataset(era5_pl_path)
    try:
        time_dim = "valid_time" if "valid_time" in ds.dims else "time"
        ds_t = ds.sel({time_dim: overpass_utc}, method="nearest")
        sel_t = pd.Timestamp(ds_t[time_dim].values)
        if abs(sel_t - overpass_utc) > pd.Timedelta("1h"):
            raise ValueError(
                f"ERA5 nearest time {sel_t} is more than 1 h from "
                f"overpass {overpass_utc}.  Check file coverage."
            )
        lat_key = "latitude" if "latitude" in ds_t.coords else "lat"
        lon_key = "longitude" if "longitude" in ds_t.coords else "lon"
        era5_lats = ds_t[lat_key].values
        era5_lons = ds_t[lon_key].values
        i_lat = int(np.argmin(np.abs(era5_lats - fire_lat)))
        i_lon = int(np.argmin(np.abs(era5_lons - fire_lon)))
        u_profile = ds_t["u"].isel(**{lat_key: i_lat, lon_key: i_lon}).values.ravel()
        v_profile = ds_t["v"].isel(**{lat_key: i_lat, lon_key: i_lon}).values.ravel()
        p_levels = ds_t["pressure_level"].values.ravel()
    finally:
        ds.close()

    sort_idx = np.argsort(p_levels)
    p_sorted = p_levels[sort_idx]
    u_sorted = u_profile[sort_idx]
    v_sorted = v_profile[sort_idx]
    u, v = _interp_log_p(p_target_hpa, p_sorted, u_sorted, v_sorted)
    return dict(
        u_m_s=u, v_m_s=v,
        speed_m_s=float(np.hypot(u, v)),
        theta_rad=float(np.arctan2(v, u)),
        era5_grid_lat=float(era5_lats[i_lat]),
        era5_grid_lon=float(era5_lons[i_lon]),
        era5_selected_time=str(sel_t),
        # Keep sorted arrays for downstream sensitivity analyses if needed:
        p_sorted=p_sorted.tolist(),
        u_sorted=u_sorted.tolist(),
        v_sorted=v_sorted.tolist(),
    )


def _era5_sfc_plume_mean_wind(
    era5_sfc_path: Path,
    overpass_utc: pd.Timestamp,
    lat_c: np.ndarray,
    lon_c: np.ndarray,
    plume: np.ndarray,
) -> dict:
    """ERA5 10 m wind regridded to the retrieval grid, vector-averaged over plume cells.

    Reproduces notebook 02's background-projection wind exactly: linear regrid
    of u10/v10 onto the CO grid, then the vector mean over plume cells with a
    valid interpolated wind. This wind orients the upwind background window
    ONLY — the transect integration wind remains the AER_LH pressure-level
    wind (notebook 02 'Change D bg + AER_LH wind' variant).
    """
    if not era5_sfc_path.exists():
        raise FileNotFoundError(
            f"ERA5 10 m surface-wind file not found: {era5_sfc_path} "
            "(required by background.wind_source = 'era5_10m_plume_mean')."
        )
    with xr.open_dataset(era5_sfc_path) as ds:
        tdim = "valid_time" if "valid_time" in ds.dims else "time"
        dst = ds.sel({tdim: overpass_utc}, method="nearest")
        sel_t = pd.Timestamp(dst[tdim].values)
        src_lat = dst["latitude"].values
        src_lon = dst["longitude"].values
        u10_raw = dst["u10"].values
        v10_raw = dst["v10"].values
    if abs(sel_t - overpass_utc) > pd.Timedelta("1h"):
        raise ValueError(
            f"ERA5 surface nearest time {sel_t} is more than 1 h from "
            f"overpass {overpass_utc}. Check file coverage."
        )
    # RegularGridInterpolator needs ascending axes; ERA5 latitude is descending.
    if src_lat[0] > src_lat[-1]:
        src_lat = src_lat[::-1]
        u10_raw = u10_raw[::-1]
        v10_raw = v10_raw[::-1]
    assert np.all(np.diff(src_lat) > 0) and np.all(np.diff(src_lon) > 0), (
        "ERA5 surface lat/lon axes not strictly ascending after flip."
    )
    lo_g, la_g = np.meshgrid(lon_c, lat_c)
    pts = np.column_stack([la_g.ravel(), lo_g.ravel()])
    u_fn = RegularGridInterpolator((src_lat, src_lon), u10_raw,
                                   method="linear", bounds_error=False, fill_value=np.nan)
    v_fn = RegularGridInterpolator((src_lat, src_lon), v10_raw,
                                   method="linear", bounds_error=False, fill_value=np.nan)
    u10 = u_fn(pts).reshape(len(lat_c), len(lon_c)).astype(np.float32)
    v10 = v_fn(pts).reshape(len(lat_c), len(lon_c)).astype(np.float32)
    _pw = plume & np.isfinite(u10)
    assert _pw.any(), "No plume pixels with valid ERA5 10 m wind."
    u_sfc = float(u10[_pw].mean())
    v_sfc = float(v10[_pw].mean())
    return dict(
        u_m_s=u_sfc, v_m_s=v_sfc,
        speed_m_s=float(np.hypot(u_sfc, v_sfc)),
        theta_rad=float(np.arctan2(v_sfc, u_sfc)),
        era5_selected_time=str(sel_t),
    )


def _find_overpass_time(
    no2_data: dict, fire_lat: float, fire_lon: float,
) -> pd.Timestamp:
    """UTC time of the L2 pixel closest to the fire centroid.

    `no2_data["time"]` is per-pixel (see `data_helpers.load_no2_l2`), already
    positionally aligned with `lat`/`lon` — no scanline-index arithmetic
    needed (that arithmetic is invalid once pixels are QA/domain-filtered
    and/or multiple overpasses are concatenated).
    """
    lats = no2_data["lat"]
    lons = no2_data["lon"]
    times = no2_data["time"]
    dlat = lats - fire_lat
    dlon = (lons - fire_lon) * np.cos(np.radians(fire_lat))
    closest_pixel_idx = int(np.argmin(dlat ** 2 + dlon ** 2))
    raw_time = times[closest_pixel_idx]
    if isinstance(raw_time, (bytes, np.bytes_)):
        raw_time = raw_time.decode("utf-8")
    return pd.Timestamp(raw_time).tz_localize(None)


def _ensure_dirs(out_root: Path) -> tuple[Path, Path]:
    plots_dir = out_root / "plots"
    results_dir = out_root / "results"
    plots_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir, results_dir


def _serialize(obj: Any) -> Any:
    """Convert numpy scalars / arrays / Path / Timestamp for json.dump."""
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (Path, pd.Timestamp)):
        return str(obj)
    return obj


# --------------------------------------------------------------------- NOx

def run_emg_pipeline(
    fire_date: str,           # 'YYYY-MM-DD'
    fire_lat: float,
    fire_lon: float,
    config: dict,
    data_root: Path,
    output_root: Path,
) -> dict:
    """Reproduce notebooks/nox_estimation_emg.ipynb end-to-end.

    Returns the results dict and writes JSON + 3 PNG plots under
    ``output_root / 'plots' or 'results'``.
    """
    target_day = "".join(fire_date.split("-")[1:])      # e.g. '0118'
    yyyymmdd = "".join(fire_date.split("-"))            # e.g. '20260118'
    plots_dir, results_dir = _ensure_dirs(output_root)

    tropomi_cfg = config["tropomi"]
    domain = config["domain"]
    inj_cfg = config["injection"]
    scene_cfg = config["scene"]
    emg_cfg = config["emg"]
    phys = config["physics"]

    # ── 1. Resolve L2 paths (all overpasses) ---------------------------------
    no2_paths = _find_all(
        str(data_root / target_day / "S5P_OFFL_L2__NO2____*.nc"), "NO2 L2"
    )
    aer_ai_paths = _find_all(
        str(data_root / target_day / "S5P_OFFL_L2__AER_AI_*.nc"), "AER_AI L2"
    )
    aer_lh_paths = _find_all(
        str(data_root / target_day / "S5P_OFFL_L2__AER_LH_*.nc"), "AER_LH L2"
    )

    # ── 2. Load + concatenate L2 swaths --------------------------------------
    qa_uint8 = int(tropomi_cfg["qa_threshold"] * 100)
    roi_kwargs = dict(
        lat_min=domain["lat_min"], lat_max=domain["lat_max"],
        lon_min=domain["lon_min"], lon_max=domain["lon_max"],
        qa_threshold=qa_uint8,
    )
    no2_data = _load_and_concat(no2_paths, load_no2_l2, **roi_kwargs)
    ai_data = _load_and_concat(aer_ai_paths, load_aer_ai, **roi_kwargs)
    alh_data = _load_and_concat(aer_lh_paths, load_aer_lh, **roi_kwargs)

    # ── 2b. Plume boundary: clip AI outside the saved blob-clustered
    # inclusion geometry down below threshold, replacing the fixed
    # tropomi.ai_threshold config value everywhere downstream.
    plume_boundary = _load_plume_boundary_or_raise(data_root, target_day)
    ai_threshold = plume_boundary.threshold
    ai_data["ai"] = _clip_ai_to_boundary(ai_data, plume_boundary)

    # ── 3. Aerosol injection pressure ---------------------------------------
    psel = select_injection_pressure(
        ai_raw=ai_data["ai"], ai_lat=ai_data["lat"], ai_lon=ai_data["lon"],
        alh_pres_raw=alh_data["alh_pres"],
        alh_lat=alh_data["lat"], alh_lon=alh_data["lon"],
        fire_lat=fire_lat, fire_lon=fire_lon,
        ai_threshold=ai_threshold,
        radius_km=float(inj_cfg["radius_km"]),
        min_pixels=int(inj_cfg["min_pixels"]),
        aggregation=str(inj_cfg["aggregation"]),
        fallback_hpa=float(inj_cfg["fallback_hpa"]),
    )
    inj_pres_hpa = float(psel["injection_pressure_hpa"])

    # ── 4. Overpass time + ERA5 wind at injection pressure ------------------
    overpass = _find_overpass_time(no2_data, fire_lat, fire_lon)
    era5_pl_path = data_root / "ERA5_winds_pressure_levels.nc"
    wind = _era5_pl_wind_at_fire(
        era5_pl_path, overpass, fire_lat, fire_lon, inj_pres_hpa,
    )

    # ── 5. Scene extraction (within EXTRACTION_RADIUS_KM of fire) -----------
    earth_r = float(scene_cfg["earth_radius_km"])
    phi0, phi_all = np.radians(fire_lat), np.radians(no2_data["lat"])
    dphi = phi_all - phi0
    dlambda = np.radians(no2_data["lon"] - fire_lon)
    a = (np.sin(dphi / 2) ** 2
         + np.cos(phi0) * np.cos(phi_all) * np.sin(dlambda / 2) ** 2)
    dist_km = 2 * earth_r * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

    scene_mask = (
        (dist_km <= float(scene_cfg["extraction_radius_km"]))
        & np.isfinite(no2_data["no2"])
    )
    if not scene_mask.any():
        raise ValueError(
            f"No valid NO2 pixels within "
            f"{scene_cfg['extraction_radius_km']:.0f} km of fire centroid."
        )
    lat_s = no2_data["lat"][scene_mask]
    lon_s = no2_data["lon"][scene_mask]
    no2_s = no2_data["no2"][scene_mask]

    # ── 6. Wind-frame rotation ---------------------------------------------
    x_rot, y_rot = rotate_to_wind_frame(
        lat_s, lon_s, fire_lat, fire_lon, wind["theta_rad"],
    )

    # ── 7. Bin to regular grid ---------------------------------------------
    grid_dy_km = float(scene_cfg["grid_dy_km"])
    x_centres, y_centres, grid_mean, grid_cnt = bin_to_grid(
        x_rot, y_rot, no2_s,
        x_min_km=float(scene_cfg["downwind_min_km"]),
        x_max_km=float(scene_cfg["downwind_max_km"]),
        y_max_km=float(scene_cfg["crosswind_max_km"]),
        grid_dy_km=grid_dy_km,
    )

    # ── 8. Build line density profile --------------------------------------
    profile = build_line_density(grid_mean, grid_cnt, grid_dy_km * 1000.0)

    # ── 9. EMG bootstrap fit -----------------------------------------------
    emg_res = fit_emg_bootstrap(
        x_centres, profile,
        wind_speed_m_s=wind["speed_m_s"],
        nox_no2_ratio_gamma=float(phys["nox_no2_ratio_gamma"]),
        no2_molar_mass_g_per_mol=float(phys["no2_molar_mass_g_per_mol"]),
        bootstrap_iterations=int(emg_cfg["bootstrap_iterations"]),
        bootstrap_seed=int(emg_cfg["bootstrap_seed"]),
        perturbation_low=float(emg_cfg["perturbation_low"]),
        perturbation_high=float(emg_cfg["perturbation_high"]),
        curve_fit_maxfev=int(emg_cfg["curve_fit_maxfev"]),
        cv_stability_threshold=float(emg_cfg["cv_stability_threshold"]),
        peak_offset_km_max=float(emg_cfg["peak_offset_km_max"]),
        x0_min_km=grid_dy_km,
    )

    # ── 10. Plots ----------------------------------------------------------
    plotting.plot_nox_scene(
        no2_data["lat"], no2_data["lon"], no2_data["no2"],
        fire_lat, fire_lon, float(scene_cfg["extraction_radius_km"]),
        plots_dir / "01_nox_scene.png",
    )
    plotting.plot_nox_line_density(
        x_centres, profile,
        plots_dir / "02_nox_line_density.png",
    )
    plotting.plot_emg_fit(
        emg_res["x_fit"], emg_res["y_fit"], emg_res["y_pred"],
        emg_res["r2"], emg_res["tau_h"], emg_res["E_nox_g_per_s"],
        plots_dir / "03_emg_fit.png",
    )

    # ── 11. Persist results JSON -------------------------------------------
    results = {
        "command": "nox",
        "fire_date": fire_date,
        "yyyymmdd": yyyymmdd,
        "fire_lat": fire_lat,
        "fire_lon": fire_lon,
        "input_files": {
            "no2_l2":   [p.name for p in no2_paths],
            "aer_ai_l2": [p.name for p in aer_ai_paths],
            "aer_lh_l2": [p.name for p in aer_lh_paths],
            "era5_pl":  era5_pl_path.name,
        },
        "plume_boundary": {
            "path":      str(data_root / target_day / "plume_boundary.geojson"),
            "threshold": ai_threshold,
            "geom_type": plume_boundary.boundary.geom_type,
        },
        "injection_pressure": {
            "selected_hpa":    inj_pres_hpa,
            "source":          psel["pressure_source"],
            "n_alh_pixels":    int(psel["n_alh_pixels"]),
            "median_hpa":      float(psel["median_hpa"]),
            "max_hpa":         float(psel["max_hpa"]),
            "iqr_hpa":         float(psel["pressure_iqr_hpa"]),
        },
        "overpass_utc": str(overpass),
        "wind_at_injection": {
            "u_m_s":     wind["u_m_s"],
            "v_m_s":     wind["v_m_s"],
            "speed_m_s": wind["speed_m_s"],
            "theta_deg_ccw_east": float(np.degrees(wind["theta_rad"])),
            "era5_grid_lat":      wind["era5_grid_lat"],
            "era5_grid_lon":      wind["era5_grid_lon"],
            "era5_selected_time": wind["era5_selected_time"],
        },
        "scene": {
            "n_pixels_in_radius": int(scene_mask.sum()),
            "extraction_radius_km":
                float(scene_cfg["extraction_radius_km"]),
            "grid_shape":   list(grid_mean.shape),
            "grid_filled":  int((grid_cnt > 0).sum()),
        },
        "emg_fit": {
            "n_converged":   int(emg_res["n_converged"]),
            "n_attempted":   int(emg_res["n_attempted"]),
            "a_mol_per_m":   emg_res["a_mol_per_m"],
            "x0_km":         emg_res["x0_km"],
            "mu_km":         emg_res["mu_km"],
            "sigma_km":      emg_res["sigma_km"],
            "B_mol_per_m":   emg_res["B_mol_per_m"],
            "tau_h":         emg_res["tau_h"],
            "E_mol_per_s":   emg_res["E_mol_per_s"],
            "E_nox_g_per_s": emg_res["E_nox_g_per_s"],
            "E_nox_t_per_day": emg_res["E_nox_g_per_s"] * 86400.0 / 1e6,
            "r2":               emg_res["r2"],
            "cv_a":             emg_res["cv_a"],
            "cv_E":             emg_res["cv_E"],
            "fit_stable":       emg_res["fit_stable"],
            "peak_x_km":        emg_res["peak_x_km"],
            "peak_within_30km": emg_res["peak_within_30km"],
        },
        "config": config,
    }
    with open(results_dir / "run_metadata.json", "w") as fh:
        json.dump(_serialize(results), fh, indent=2)

    return results


# --------------------------------------------------------------------- CO

def run_csf_pipeline(
    fire_date: str,
    fire_lat: float,
    fire_lon: float,
    config: dict,
    data_root: Path,
    output_root: Path,
) -> dict:
    """Reproduce notebooks/co_estimation_csf.ipynb (AER_LH variant only).

    Differences vs. the notebook:
      * Surface ERA5 winds are NOT used. Both the upwind background
        projection and the dynamic-transect wind field use the
        AER_LH-pressure-level wind.
      * Only the 'Variant: Change D bg + AER_LH wind' configuration runs.
    """
    target_day = "".join(fire_date.split("-")[1:])
    yyyymmdd = "".join(fire_date.split("-"))
    plots_dir, results_dir = _ensure_dirs(output_root)

    tropomi_cfg = config["tropomi"]
    domain = config["domain"]
    inj_cfg = config["injection"]
    grid_cfg = config["grid"]
    bg_cfg = config["background"]
    csf_cfg = config["csf"]
    phys = config["physics"]

    # ── 1. Resolve L2 paths (all overpasses) ---------------------------------
    co_paths = _find_all(
        str(data_root / target_day / "S5P_OFFL_L2__CO_____*.nc"), "CO L2"
    )
    aer_ai_paths = _find_all(
        str(data_root / target_day / "S5P_OFFL_L2__AER_AI_*.nc"), "AER_AI L2"
    )
    aer_lh_paths = _find_all(
        str(data_root / target_day / "S5P_OFFL_L2__AER_LH_*.nc"), "AER_LH L2"
    )

    # ── 2. Load + concatenate L2 swaths --------------------------------------
    qa_uint8 = int(tropomi_cfg["qa_threshold"] * 100)
    roi_kwargs = dict(
        lat_min=domain["lat_min"], lat_max=domain["lat_max"],
        lon_min=domain["lon_min"], lon_max=domain["lon_max"],
        qa_threshold=qa_uint8,
    )
    # Pixel corners are needed for the footprint oversampling (step 5).
    co_data = _load_and_concat(co_paths, load_co_l2, **roi_kwargs, with_corners=True)
    ai_data = _load_and_concat(aer_ai_paths, load_aer_ai, **roi_kwargs, with_corners=True)
    alh_data = _load_and_concat(aer_lh_paths, load_aer_lh, **roi_kwargs)

    # ── 2b. Plume boundary: clip AI outside the saved blob-clustered
    # inclusion geometry down below threshold, replacing the fixed
    # tropomi.ai_threshold config value + plume_mask.clip_north_idx hack.
    plume_boundary = _load_plume_boundary_or_raise(data_root, target_day)
    ai_threshold = plume_boundary.threshold
    ai_data["ai"] = _clip_ai_to_boundary(ai_data, plume_boundary)

    # ── 3. Aerosol injection pressure ---------------------------------------
    psel = select_injection_pressure(
        ai_raw=ai_data["ai"], ai_lat=ai_data["lat"], ai_lon=ai_data["lon"],
        alh_pres_raw=alh_data["alh_pres"],
        alh_lat=alh_data["lat"], alh_lon=alh_data["lon"],
        fire_lat=fire_lat, fire_lon=fire_lon,
        ai_threshold=ai_threshold,
        radius_km=float(inj_cfg["radius_km"]),
        min_pixels=int(inj_cfg["min_pixels"]),
        aggregation=str(inj_cfg["aggregation"]),
        fallback_hpa=float(inj_cfg["fallback_hpa"]),
    )
    inj_pres_hpa = float(psel["injection_pressure_hpa"])

    # ── 4. Solar-time overpass + ERA5 PL wind at injection pressure ---------
    # TROPOMI ascending-node equator crossing is 13:30 local solar time, so
    # UTC = 13.5 - lon/15 (cross-checked against the per-pixel time_utc lookup
    # used by the NOx pipeline: 18:20 vs measured 18:21 UTC for the 0118 fire).
    overpass_utc_h = 13.5 - fire_lon / 15.0
    overpass = (pd.Timestamp(fire_date) + pd.Timedelta(hours=overpass_utc_h))
    era5_pl_path = data_root / "ERA5_winds_pressure_levels.nc"
    wind = _era5_pl_wind_at_fire(
        era5_pl_path, overpass, fire_lat, fire_lon, inj_pres_hpa,
    )

    # ── 5. Oversample CO + AER_AI onto the 4 km grid (Adams et al. 2019) ----
    # Griffin et al. (2024) gridding: cell_km × cell_km cells at the fire
    # latitude; each grid point inside a pixel footprint takes its value, then
    # cells average all such values within oversample_radius_km (weight 1/area).
    cell_km = float(grid_cfg["cell_km"])
    oversample_radius_km = float(grid_cfg["oversample_radius_km"])
    grid = regular_grid_km(domain["lat_min"], domain["lat_max"],
                           domain["lon_min"], domain["lon_max"],
                           cell_km=cell_km, ref_lat=fire_lat)
    lat_c, lon_c = grid.lat, grid.lon
    nlat, nlon = grid.nlat, grid.nlon

    co_grid, os_diag_co = oversample_to_grid(
        co_data["co"], co_data["lat_b"], co_data["lon_b"], grid, oversample_radius_km)
    # ai_data["ai"] is already clipped below ai_threshold outside the plume
    # boundary (see step 2b), so this AI>threshold test is automatically
    # boundary-consistent — no separate spatial clip needed here.
    ai_grid, os_diag_ai = oversample_to_grid(
        ai_data["ai"].astype(np.float64), ai_data["lat_b"], ai_data["lon_b"],
        grid, oversample_radius_km)

    plume = (ai_grid > ai_threshold) & np.isfinite(co_grid)
    plume_raw_count = int(plume.sum())

    if not plume.any():
        raise ValueError(
            f"No plume pixels (AI > {ai_threshold} inside the saved boundary). "
            "Check the plume boundary or domain."
        )
    if not np.isfinite(co_grid).any():
        raise ValueError("All CO grid cells are NaN.  Check QA threshold.")

    # Uniform AER_LH-wind field for the dynamic-transect machinery
    u_field = np.full_like(co_grid, wind["u_m_s"], dtype=np.float32)
    v_field = np.full_like(co_grid, wind["v_m_s"], dtype=np.float32)

    # ── 6. Change D background (far-upwind median) -------------------------
    # The upwind background window is oriented along a downwind axis. Notebook
    # 02 defines that axis from the ERA5 10 m plume-mean wind (surface flow),
    # NOT the AER_LH pressure-level wind that drives the transects — the two
    # differ by ~40 deg on 18 Jan, and the choice changes which upwind pixels
    # land in the [bg_x_min, bg_x_max] band and hence the background value.
    # `background.wind_source` selects which: 'era5_10m_plume_mean' reproduces
    # the manuscript CO number; 'injection_pressure' reuses the transect wind.
    bg_wind_source = str(bg_cfg["wind_source"])
    if bg_wind_source == "era5_10m_plume_mean":
        era5_sfc_path = data_root / "ERA5_winds.nc"
        bg_wind = _era5_sfc_plume_mean_wind(
            era5_sfc_path, overpass, lat_c, lon_c, plume,
        )
    elif bg_wind_source == "injection_pressure":
        bg_wind = dict(theta_rad=wind["theta_rad"], speed_m_s=wind["speed_m_s"],
                       era5_selected_time=wind["era5_selected_time"])
    else:
        raise ValueError(
            f"background.wind_source must be 'era5_10m_plume_mean' or "
            f"'injection_pressure'; got {bg_wind_source!r}."
        )

    lat_centre = float(lat_c.mean())
    m_per_deg_lat = float(phys["m_per_deg_lat"])
    lo_g, la_g = np.meshgrid(lon_c, lat_c)
    lon_centre = float(lon_c.mean())
    x_m = (lo_g - lon_centre) * np.cos(np.radians(lat_centre)) * m_per_deg_lat
    y_m = (la_g - lat_centre) * m_per_deg_lat
    cos_t = np.cos(bg_wind["theta_rad"])
    sin_t = np.sin(bg_wind["theta_rad"])
    x_rot_km = (x_m * cos_t + y_m * sin_t) / 1000.0

    bg_x_min = float(bg_cfg["bg_x_min_km"])
    bg_x_max = float(bg_cfg["bg_x_max_km"])
    bg_min_pix = int(bg_cfg["min_pixels"])
    bg_window = (
        (x_rot_km >= bg_x_min) & (x_rot_km <= bg_x_max)
        & np.isfinite(co_grid) & (co_grid > 0)
    )
    n_bg = int(bg_window.sum())
    if n_bg >= bg_min_pix:
        bg_co = float(np.nanmedian(co_grid[bg_window]))
        bg_source = (f"median of {n_bg} px in [{bg_x_min:.0f}, "
                     f"{bg_x_max:.0f}] km upwind")
    else:
        bg_co = float(np.nanpercentile(co_grid[np.isfinite(co_grid) & (co_grid > 0)], 1))
        bg_source = (f"1st-percentile fallback ({n_bg} px in window, "
                     f"min required {bg_min_pix})")
        warnings.warn(
            f"Change D window has only {n_bg} px (< {bg_min_pix}). "
            "Using 1st-percentile of all positive CO as background.",
            UserWarning,
        )
    co_delta = co_grid - bg_co

    # ── 7. CSF integration --------------------------------------------------
    n_transects = int(csf_cfg["n_transects"])
    half_width = float(csf_cfg["transect_half_width_km"])
    aggregation = str(csf_cfg["aggregation"])

    slab_pos, wind_angles, u_slab, v_slab = compute_local_wind_direction(
        u_field, v_field, plume, lat_c, lon_c, n_transects=n_transects,
    )
    sample_spacing_m = float(csf_cfg["sample_spacing_km"]) * 1e3
    min_valid_fraction = float(csf_cfg["min_valid_fraction"])
    transects = build_dynamic_transects(
        lat_c, lon_c, plume, wind_angles, u_slab, v_slab,
        n_transects=n_transects, transect_half_width_km=half_width,
        sample_spacing_m=sample_spacing_m,
    )
    slab_pos2, t_flux_mol_s, flux_mean_kg_s, flux_mean_Gg_yr = compute_csf_dynamic(
        co_delta, lat_c, lon_c, u_field, v_field, plume,
        n_transects=n_transects, transect_half_width_km=half_width,
        aggregation=aggregation, sample_spacing_m=sample_spacing_m,
        min_valid_fraction=min_valid_fraction,
    )
    t_flux_kg_s = t_flux_mol_s * CO_MOLAR_MASS_KG_PER_MOL

    n_valid = int(np.sum(np.isfinite(t_flux_kg_s)))
    if n_valid == 0:
        raise RuntimeError("CSF: all transects returned NaN flux.")

    # ── 8. Plots ------------------------------------------------------------
    plotting.plot_co_scene(
        co_data["lat"], co_data["lon"], co_data["co"],
        fire_lat, fire_lon, float(inj_cfg["radius_km"]),
        plots_dir / "01_co_scene.png",
    )
    plotting.plot_csf_diagnostic(
        lat_grid=lat_c, lon_grid=lon_c,
        co_grid_mol_m2=co_grid, plume_mask=plume,
        transects=transects, transect_fluxes_kg_s=t_flux_kg_s,
        slab_pos_m=slab_pos2, flux_mean_kg_s=flux_mean_kg_s,
        fire_lat=fire_lat, fire_lon=fire_lon,
        inj_pres_hpa=inj_pres_hpa, wind_speed_m_s=wind["speed_m_s"],
        bg_co_mol_m2=bg_co, fire_date=fire_date,
        n_transects=n_transects,
        output_path=plots_dir / "02_csf_diagnostic.png",
    )

    # ── 9. Persist results JSON --------------------------------------------
    results = {
        "command": "co",
        "fire_date": fire_date,
        "yyyymmdd": yyyymmdd,
        "fire_lat": fire_lat,
        "fire_lon": fire_lon,
        "input_files": {
            "co_l2":     [p.name for p in co_paths],
            "aer_ai_l2": [p.name for p in aer_ai_paths],
            "aer_lh_l2": [p.name for p in aer_lh_paths],
            "era5_pl":   era5_pl_path.name,
        },
        "plume_boundary": {
            "path":      str(data_root / target_day / "plume_boundary.geojson"),
            "threshold": ai_threshold,
            "geom_type": plume_boundary.boundary.geom_type,
        },
        "injection_pressure": {
            "selected_hpa":    inj_pres_hpa,
            "source":          psel["pressure_source"],
            "n_alh_pixels":    int(psel["n_alh_pixels"]),
            "median_hpa":      float(psel["median_hpa"]),
            "max_hpa":         float(psel["max_hpa"]),
            "iqr_hpa":         float(psel["pressure_iqr_hpa"]),
        },
        "overpass_utc_solar_time": str(overpass),
        "wind_at_injection": {
            "u_m_s":     wind["u_m_s"],
            "v_m_s":     wind["v_m_s"],
            "speed_m_s": wind["speed_m_s"],
            "theta_deg_ccw_east": float(np.degrees(wind["theta_rad"])),
            "era5_grid_lat":      wind["era5_grid_lat"],
            "era5_grid_lon":      wind["era5_grid_lon"],
            "era5_selected_time": wind["era5_selected_time"],
        },
        "grid": {
            "shape":          [int(nlat), int(nlon)],
            "cell_km":        cell_km,
            "dlat_deg":       grid.dlat,
            "dlon_deg":       grid.dlon,
            "ref_lat":        grid.ref_lat,
            "oversample_radius_km": oversample_radius_km,
            "oversampling_diag_co": os_diag_co,
            "oversampling_diag_ai": os_diag_ai,
            "co_filled_cells": int(np.isfinite(co_grid).sum()),
            "plume_pixels_raw":  plume_raw_count,
            "plume_pixels_used": int(plume.sum()),
        },
        "background": {
            "bg_co_mol_m2":      bg_co,
            "bg_co_mmol_m2":     bg_co * 1e3,
            "bg_x_min_km":       bg_x_min,
            "bg_x_max_km":       bg_x_max,
            "n_window_pixels":   n_bg,
            "source":            bg_source,
            "wind_source":       bg_wind_source,
            "wind_theta_deg_ccw_east": float(np.degrees(bg_wind["theta_rad"])),
            "wind_speed_m_s":    float(bg_wind["speed_m_s"]),
            "wind_era5_selected_time": bg_wind["era5_selected_time"],
        },
        "csf": {
            "n_transects":          n_transects,
            "transect_half_width_km": half_width,
            "aggregation":          aggregation,
            "sample_spacing_m":     transects[0].ds_meters,
            "n_samples_per_transect": int(len(transects[0].positions_lat)),
            "min_valid_fraction":   min_valid_fraction,
            "n_valid_transects":    n_valid,
            "transect_flux_kg_s":   t_flux_kg_s.tolist(),
            "transect_pos_km":      (slab_pos2 / 1e3).tolist(),
            "flux_mean_kg_s":       flux_mean_kg_s,
            "flux_mean_t_per_day":  flux_mean_kg_s * 86400.0 / 1000.0,
            "flux_mean_Gg_per_yr":  flux_mean_Gg_yr,
        },
        "config": config,
    }
    with open(results_dir / "run_metadata.json", "w") as fh:
        json.dump(_serialize(results), fh, indent=2)

    return results
