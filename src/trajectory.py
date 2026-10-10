"""ERA5-driven kinematic trajectory ensemble for wildfire plume pathways.

Consolidates the Monte-Carlo trajectory machinery previously embedded in
``notebooks/extra/pysplit_probabilistic_trajectories.ipynb`` so it can be
reused as a thin library call. The ensemble answers ONE question:
the transport *pathway* (and an emission-independent corridor mask). It does
not measure dispersion or detectability — those come from the observed columns
(notebook 06) and the HYSPLIT forward run.

Design notes
------------
* Isobaric kinematic trajectories (pressure held constant after release).
  The release height is observationally pinned from AER_LH; vertical motion is
  neglected (documented limitation).
* Ensemble spread reflects *release* uncertainty (injection pressure, launch
  hour, source location), NOT meteorological/transport uncertainty — a single
  ERA5 realization drives every member. The corridor is therefore a lower
  bound on true pathway spread.
* Reuses ``wind_helpers`` primitives and ``wind_helpers.match_alh_plume_pressures``
  so the injection-pressure distribution shares one code path with the scalar
  ``select_injection_pressure`` used in the emission retrievals.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import gaussian_kde

from .wind_helpers import _haversine_km, match_alh_plume_pressures

R_EARTH_M: float = 6_371_000.0


# ──────────────────────────────────────────────────────────────────────────────
# ERA5 wind field
# ──────────────────────────────────────────────────────────────────────────────
class ERA5WindField:
    """Trilinear (time, pressure, lat, lon) interpolator over ERA5 u/v winds.

    Built once per ERA5 file; ``wind_at`` returns (u, v) in m/s. Axes are
    flipped to strictly ascending as required by ``RegularGridInterpolator``.
    """

    def __init__(self, era5_file: Path) -> None:
        era5_file = Path(era5_file)
        if not era5_file.exists():
            raise FileNotFoundError(f"ERA5 file not found: {era5_file}")
        ds = xr.open_dataset(era5_file)

        t_np = ds.valid_time.values.astype("datetime64[s]").astype(np.float64)
        if not np.all(np.diff(t_np) > 0):
            raise ValueError("ERA5 time axis is not strictly ascending.")

        pres_asc = ds.pressure_level.values.astype(np.float64)[::-1].copy()
        lat_asc = ds.latitude.values.astype(np.float64)[::-1].copy()
        lon_arr = ds.longitude.values.astype(np.float64)
        for name, ax in (("pressure", pres_asc), ("latitude", lat_asc), ("longitude", lon_arr)):
            if not np.all(np.diff(ax) > 0):
                raise ValueError(f"ERA5 {name} axis not ascending after flip.")

        u_arr = ds.u.values.astype(np.float32)[:, ::-1, ::-1, :]
        v_arr = ds.v.values.astype(np.float32)[:, ::-1, ::-1, :]
        ds.close()

        points = (t_np, pres_asc, lat_asc, lon_arr)
        self._u = RegularGridInterpolator(points, u_arr, method="linear",
                                          bounds_error=False, fill_value=np.nan)
        self._v = RegularGridInterpolator(points, v_arr, method="linear",
                                          bounds_error=False, fill_value=np.nan)
        self.pres_min = float(pres_asc.min())
        self.pres_max = float(pres_asc.max())
        self.lat_min = float(lat_asc.min())
        self.lat_max = float(lat_asc.max())
        self.lon_min = float(lon_arr.min())
        self.lon_max = float(lon_arr.max())
        self.n_times = int(len(t_np))

    def wind_at(self, lat: float, lon: float, pres_hpa: float, t_epoch_s: float):
        """Return (u, v) in m/s at (lat, lon, pres_hPa, epoch seconds)."""
        pt = np.array([[t_epoch_s, pres_hpa, lat, lon]])
        return float(self._u(pt)), float(self._v(pt))


# ──────────────────────────────────────────────────────────────────────────────
# RK4 isobaric advection
# ──────────────────────────────────────────────────────────────────────────────
def _dpos(wf: ERA5WindField, lat, lon, pres, t_s):
    u, v = wf.wind_at(lat, lon, pres, t_s)
    if np.isnan(u) or np.isnan(v):
        return np.nan, np.nan
    dlat_dt = np.degrees(v / R_EARTH_M)
    dlon_dt = np.degrees(u / (R_EARTH_M * np.cos(np.radians(lat))))
    return dlat_dt, dlon_dt


def _rk4_step(wf: ERA5WindField, lat, lon, pres, t_s, dt_s):
    k1l, k1o = _dpos(wf, lat, lon, pres, t_s)
    if np.isnan(k1l):
        return np.nan, np.nan
    k2l, k2o = _dpos(wf, lat + 0.5 * dt_s * k1l, lon + 0.5 * dt_s * k1o, pres, t_s + 0.5 * dt_s)
    if np.isnan(k2l):
        return np.nan, np.nan
    k3l, k3o = _dpos(wf, lat + 0.5 * dt_s * k2l, lon + 0.5 * dt_s * k2o, pres, t_s + 0.5 * dt_s)
    if np.isnan(k3l):
        return np.nan, np.nan
    k4l, k4o = _dpos(wf, lat + dt_s * k3l, lon + dt_s * k3o, pres, t_s + dt_s)
    if np.isnan(k4l):
        return np.nan, np.nan
    new_lat = lat + (dt_s / 6.0) * (k1l + 2 * k2l + 2 * k3l + k4l)
    new_lon = lon + (dt_s / 6.0) * (k1o + 2 * k2o + 2 * k3o + k4o)
    return new_lat, new_lon


def advect(
    wf: ERA5WindField,
    lat0: float,
    lon0: float,
    pres_hpa: float,
    launch_dt: datetime,
    n_hours: int,
    dt_s: int,
) -> dict:
    """Integrate one Lagrangian parcel forward with RK4 (isobaric, ERA5 winds).

    Returns dict: time_h, lat, lon (arrays, NaN after the parcel leaves the
    ERA5 domain), launch_dt, pres_hPa, lat0, lon0.
    """
    t0_s = float(np.datetime64(launch_dt.strftime("%Y-%m-%dT%H:%M:%S"), "s").astype(np.float64))
    n_steps = int(n_hours * 3600 / dt_s)
    lats = np.full(n_steps + 1, np.nan)
    lons = np.full(n_steps + 1, np.nan)
    time_h = np.arange(n_steps + 1, dtype=np.float64) * dt_s / 3600.0
    lats[0], lons[0] = lat0, lon0
    lat, lon, t_s = lat0, lon0, t0_s
    for i in range(n_steps):
        new_lat, new_lon = _rk4_step(wf, lat, lon, pres_hpa, t_s, dt_s)
        if np.isnan(new_lat):
            break
        lat, lon, t_s = new_lat, new_lon, t_s + dt_s
        lats[i + 1], lons[i + 1] = lat, lon
    return dict(time_h=time_h, lat=lats, lon=lons, launch_dt=launch_dt,
                pres_hPa=pres_hpa, lat0=lat0, lon0=lon0)


# ──────────────────────────────────────────────────────────────────────────────
# Observation-informed priors
# ──────────────────────────────────────────────────────────────────────────────
def build_pressure_kde(
    ai_data: dict,
    alh_data: dict,
    fire_lat: float,
    fire_lon: float,
    ai_threshold: float = 2.0,
    radius_km: float = 300.0,
    min_pixels: int = 5,
) -> tuple[gaussian_kde, np.ndarray, dict]:
    """KDE of AER_LH injection pressure (hPa) over AER_AI-matched plume pixels.

    Shares the matching code path with ``select_injection_pressure`` via
    ``match_alh_plume_pressures`` (no duplication). Returns (kde, valid_hpa,
    stats).
    """
    matched = match_alh_plume_pressures(
        ai_data["ai"], ai_data["lat"], ai_data["lon"],
        alh_data["alh_pres"], alh_data["lat"], alh_data["lon"],
        fire_lat, fire_lon, ai_threshold=ai_threshold, radius_km=radius_km,
    )
    valid_hpa = matched["pres_hpa"]
    if len(valid_hpa) < min_pixels:
        raise ValueError(
            f"Only {len(valid_hpa)} matched AER_LH pixels (min={min_pixels}); "
            "cannot build a pressure KDE. Increase radius_km or lower ai_threshold."
        )
    kde = gaussian_kde(valid_hpa, bw_method="scott")
    q25, q75 = np.percentile(valid_hpa, [25, 75])
    stats = dict(
        n=int(len(valid_hpa)),
        mean=float(np.mean(valid_hpa)), median=float(np.median(valid_hpa)),
        std=float(np.std(valid_hpa)), iqr=float(q75 - q25),
        p05=float(np.percentile(valid_hpa, 5)), p95=float(np.percentile(valid_hpa, 95)),
        bw_factor=float(kde.factor),
    )
    return kde, valid_hpa, stats


def build_spatial_frp_kde(
    fires_df: pd.DataFrame,
    fire_lat: float,
    fire_lon: float,
    local_radius_km: float = 200.0,
    min_pixels: int = 3,
) -> tuple[gaussian_kde, dict]:
    """FRP-weighted 2D KDE over (lat, lon) of the local fire complex.

    ``fires_df`` is expected from ``fire_clustering.load_modis_csv`` (already
    date/satellite/confidence/daytime filtered). Pixels beyond
    ``local_radius_km`` of the fire centre are excluded so Scott's-rule
    bandwidth reflects the complex, not continental spread.
    """
    for col in ("latitude", "longitude", "frp"):
        if col not in fires_df.columns:
            raise ValueError(f"fires_df missing required column {col!r}")
    lat = fires_df["latitude"].to_numpy()
    lon = fires_df["longitude"].to_numpy()
    frp = fires_df["frp"].to_numpy()
    local = _haversine_km(fire_lat, fire_lon, lat, lon) <= local_radius_km
    if int(local.sum()) < min_pixels:
        raise ValueError(
            f"Only {int(local.sum())} fire pixels within {local_radius_km} km; "
            "cannot build a 2D spatial KDE."
        )
    w = frp[local] / frp[local].sum()
    kde = gaussian_kde(np.vstack([lat[local], lon[local]]), bw_method="scott", weights=w)
    info = dict(
        n_pixels=int(local.sum()),
        frp_centroid_lat=float(np.sum(w * lat[local])),
        frp_centroid_lon=float(np.sum(w * lon[local])),
        frp_sum_mw=float(frp[local].sum()),
        bw_factor=float(kde.factor),
    )
    return kde, info


def sample_release_params(
    kde_pres: gaussian_kde,
    kde_spatial: gaussian_kde,
    n_samples: int,
    seed: int,
    launch_hour_min: float,
    launch_hour_max: float,
    pres_clip: tuple[float, float],
    lat_clip: tuple[float, float],
    lon_clip: tuple[float, float],
) -> dict:
    """Draw N release parameter sets from the observation-informed priors.

    Pressure ~ KDE(AER_LH); launch hour ~ Uniform[min,max] (no observational
    constraint); source ~ FRP-weighted 2D KDE. Samples are clipped to the ERA5
    pressure range / domain; clip counts are returned for logging.
    """
    rng = np.random.default_rng(seed)
    raw_p = kde_pres.resample(n_samples, seed=seed).flatten()
    pres = np.clip(raw_p, *pres_clip)
    hours = rng.uniform(launch_hour_min, launch_hour_max, n_samples)
    raw_s = kde_spatial.resample(n_samples, seed=seed + 1)  # (2, N)
    lat = np.clip(raw_s[0], *lat_clip)
    lon = np.clip(raw_s[1], *lon_clip)
    return dict(
        pres=pres, hours=hours, lat=lat, lon=lon,
        n_clipped_pres=int(np.sum((raw_p < pres_clip[0]) | (raw_p > pres_clip[1]))),
        n_clipped_spatial=int(np.sum(
            (raw_s[0] < lat_clip[0]) | (raw_s[0] > lat_clip[1])
            | (raw_s[1] < lon_clip[0]) | (raw_s[1] > lon_clip[1])
        )),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Ensemble run + corridor
# ──────────────────────────────────────────────────────────────────────────────
def run_ensemble(
    wf: ERA5WindField,
    params: dict,
    ref_date: datetime,
    run_hours: int,
    dt_s: int,
) -> list[dict]:
    """Advect all members; returns a list of trajectory dicts (one per sample)."""
    n = len(params["pres"])
    ensemble = []
    for i in range(n):
        launch_dt = ref_date + timedelta(hours=float(params["hours"][i]))
        traj = advect(wf, float(params["lat"][i]), float(params["lon"][i]),
                      float(params["pres"][i]), launch_dt, run_hours, dt_s)
        traj["sample_idx"] = i
        ensemble.append(traj)
    return ensemble


def corridor_probability(
    ensemble: list[dict],
    lat_edges: np.ndarray,
    lon_edges: np.ndarray,
    time_window_h: Optional[tuple[float, float]] = None,
) -> np.ndarray:
    """Fraction of ensemble members that visit each grid cell.

    A member contributes at most once per cell (presence, not dwell time), so
    the result is a per-cell visit probability in [0, 1]. If ``time_window_h``
    is given, only positions with elapsed time in [t0, t1) are counted (for an
    age-resolved corridor); otherwise all elapsed times are used.

    Positions outside the histogram grid are excluded, NOT clamped into the
    border cells: the ERA5 domain trajectories roam is larger than a typical
    corridor grid, and clamping inflates visit probability along the grid
    edges (out-of-grid excursions previously piled up in the border row/column).
    """
    nlat, nlon = len(lat_edges) - 1, len(lon_edges) - 1
    count = np.zeros((nlat, nlon), dtype=np.float64)
    n_members = len(ensemble)
    for traj in ensemble:
        lat, lon, th = traj["lat"], traj["lon"], traj["time_h"]
        ok = np.isfinite(lat) & np.isfinite(lon)
        if time_window_h is not None:
            ok &= (th >= time_window_h[0]) & (th < time_window_h[1])
        if not ok.any():
            continue
        il = np.searchsorted(lat_edges, lat[ok], side="right") - 1
        jl = np.searchsorted(lon_edges, lon[ok], side="right") - 1
        in_grid = (il >= 0) & (il < nlat) & (jl >= 0) & (jl < nlon)
        if not in_grid.any():
            continue
        visited = np.zeros((nlat, nlon), dtype=bool)
        visited[il[in_grid], jl[in_grid]] = True
        count += visited
    return count / float(n_members)
