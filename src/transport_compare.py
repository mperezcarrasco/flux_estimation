"""HYSPLIT forward run vs observed TROPOMI CO: gridding, regridding and footprint comparison.

Moved from notebooks/06_hysplit_forward_validation.ipynb (Oct 2026) so nb06, the
transport-figure notebook (co_transport_temporal_coherence.ipynb) and the tests
share one implementation and one config (config/hysplit_comparison.yaml).
Numerically identical to the previous nb06 cells.

Method (manuscript Sect. "HYSPLIT Forward Dispersion model" + Appendix):
* Observed CO: TROPOMI L2 CO (QA ≥ 0.5) binned to a regular 0.25° analysis grid
  (cell = mean of pixel centres in the cell); enhancement = grid minus the
  day's ``bg_percentile`` percentile.
* Model: cdump layer column (kg m⁻²) / M_CO → mol m⁻², put on the analysis grid
  by index lookup (each analysis cell takes the model cell containing its
  centre; 0 outside the model grid). No interpolation, so no tie ambiguity.
* Footprint = model column > T and source distance > near_source_km. Model and
  observed medians are compared over the SAME cells (footprint ∩ valid obs);
  reach = p95/p99 source distance of each field's own footprint.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .data_helpers import load_co_l2
from .wind_helpers import _haversine_km


@dataclass(frozen=True)
class ComparisonConfig:
    """Parsed config/hysplit_comparison.yaml (all values explicit, no defaults)."""

    fire_lat: float
    fire_lon: float
    level_top_m: float
    co_molar_mass_kg_per_mol: float
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    res_deg: float
    qa_uint8: int
    bg_percentile: float
    threshold_mol_m2: float
    near_source_km: float
    overpasses: tuple[dict, ...]   # {"folder", "offset_d", "time_utc": datetime}


def load_comparison_config(path: Path) -> ComparisonConfig:
    """Read and validate config/hysplit_comparison.yaml (KeyError on any missing key)."""
    import yaml
    with open(path) as fh:
        c = yaml.safe_load(fh)
    g, o, f = c["analysis_grid"], c["observations"], c["footprint"]
    ovs = tuple({"folder": str(v["folder"]), "offset_d": int(v["offset_d"]),
                 "time_utc": datetime.fromisoformat(str(v["time_utc"]))}
                for v in c["overpasses"])
    assert ovs, f"{path}: no overpasses."
    assert 0.0 < float(o["qa_threshold"]) <= 1.0, "qa_threshold must be in (0, 1]."
    return ComparisonConfig(
        fire_lat=float(c["source"]["fire_lat"]), fire_lon=float(c["source"]["fire_lon"]),
        level_top_m=float(c["column"]["level_top_m"]),
        co_molar_mass_kg_per_mol=float(c["column"]["co_molar_mass_kg_per_mol"]),
        lat_min=float(g["lat_min"]), lat_max=float(g["lat_max"]),
        lon_min=float(g["lon_min"]), lon_max=float(g["lon_max"]), res_deg=float(g["res_deg"]),
        qa_uint8=int(round(float(o["qa_threshold"]) * 100)),
        bg_percentile=float(o["bg_percentile"]),
        threshold_mol_m2=float(f["column_threshold_mol_m2"]),
        near_source_km=float(f["near_source_km"]),
        overpasses=ovs,
    )


@dataclass(frozen=True)
class AnalysisGrid:
    """Regular lat/lon analysis grid. Edges: np.arange(min, max + res, res)."""

    lat_edges: np.ndarray
    lon_edges: np.ndarray
    lat_ctrs: np.ndarray
    lon_ctrs: np.ndarray


def analysis_grid(cfg: ComparisonConfig) -> AnalysisGrid:
    """Analysis grid from the config (same construction as the former nb06 cell)."""
    lat_edges = np.arange(cfg.lat_min, cfg.lat_max + cfg.res_deg, cfg.res_deg)
    lon_edges = np.arange(cfg.lon_min, cfg.lon_max + cfg.res_deg, cfg.res_deg)
    return AnalysisGrid(lat_edges, lon_edges,
                        0.5 * (lat_edges[:-1] + lat_edges[1:]),
                        0.5 * (lon_edges[:-1] + lon_edges[1:]))


def source_distance_km(cfg: ComparisonConfig, grid: AnalysisGrid) -> np.ndarray:
    """(nlat, nlon) great-circle distance (km) of each analysis-cell centre from the source."""
    lon2d, lat2d = np.meshgrid(grid.lon_ctrs, grid.lat_ctrs)
    return _haversine_km(cfg.fire_lat, cfg.fire_lon,
                         lat2d.ravel(), lon2d.ravel()).reshape(lat2d.shape)


def bin_observed_co(files: list[Path], cfg: ComparisonConfig, grid: AnalysisGrid) -> np.ndarray:
    """Observed TROPOMI CO total column (mol m⁻²) on the analysis grid.

    All files (overpasses of one day) are concatenated; each cell is the mean of
    the pixel centres it contains (NaN if none). Raises FileNotFoundError if no files.
    """
    files = sorted(Path(f) for f in files)
    if not files:
        raise FileNotFoundError("No CO L2 files given for this day.")
    co, la, lo = [], [], []
    for f in files:
        c = load_co_l2(f, lat_min=cfg.lat_min, lat_max=cfg.lat_max,
                       lon_min=cfg.lon_min, lon_max=cfg.lon_max, qa_threshold=cfg.qa_uint8)
        co.append(c["co"]); la.append(c["lat"]); lo.append(c["lon"])
    co_f = np.concatenate(co); la = np.concatenate(la); lo = np.concatenate(lo)
    g = np.full((len(grid.lat_ctrs), len(grid.lon_ctrs)), np.nan)
    il = np.searchsorted(grid.lat_edges[1:-1], la)
    jl = np.searchsorted(grid.lon_edges[1:-1], lo)
    ok = np.isfinite(co_f)
    df = pd.DataFrame({"v": co_f[ok], "i": il[ok], "j": jl[ok]})
    for (i, j), v in df.groupby(["i", "j"])["v"].mean().items():
        g[i, j] = v
    return g


def observed_enhancement(obs_grid: np.ndarray, bg_percentile: float) -> tuple[np.ndarray, float]:
    """Enhancement = grid − nanpercentile(grid, bg_percentile). Returns (enhancement, background)."""
    bg = float(np.nanpercentile(obs_grid, bg_percentile))
    return obs_grid - bg, bg


def model_column_mol_m2(ds, cfg: ComparisonConfig):
    """cdump dataset → CO column (mol m⁻²), xarray (time, y, x)."""
    from .hysplit_helpers import cdump_to_column
    return cdump_to_column(ds, "CO", level_thickness_m=cfg.level_top_m) / cfg.co_molar_mass_kg_per_mol


class IndexRegrid:
    """Model (cdump) grid → analysis grid by containing-cell index lookup.

    Each analysis cell takes the value of the model cell that contains its centre;
    analysis cells outside the model grid get 0. Requires a rectilinear model grid.
    """

    def __init__(self, mod_lat2d: np.ndarray, mod_lon2d: np.ndarray, grid: AnalysisGrid):
        mod_lat2d = np.asarray(mod_lat2d, float); mod_lon2d = np.asarray(mod_lon2d, float)
        lat1d, lon1d = mod_lat2d[:, 0], mod_lon2d[0, :]
        assert (np.allclose(mod_lat2d - lat1d[:, None], 0)
                and np.allclose(mod_lon2d - lon1d[None, :], 0)), \
            "cdump grid is not rectilinear; the index regrid assumes a regular lat/lon grid"
        dlat = float(np.median(np.diff(lat1d))); dlon = float(np.median(np.diff(lon1d)))
        lat_e = np.concatenate([[lat1d[0] - dlat / 2], (lat1d[:-1] + lat1d[1:]) / 2, [lat1d[-1] + dlat / 2]])
        lon_e = np.concatenate([[lon1d[0] - dlon / 2], (lon1d[:-1] + lon1d[1:]) / 2, [lon1d[-1] + dlon / 2]])
        self._n_lat, self._n_lon = len(lat1d), len(lon1d)
        self._ii = np.searchsorted(lat_e, grid.lat_ctrs) - 1
        self._jj = np.searchsorted(lon_e, grid.lon_ctrs) - 1
        self._ri = (self._ii >= 0) & (self._ii < self._n_lat)
        self._rj = (self._jj >= 0) & (self._jj < self._n_lon)

    def __call__(self, field2d: np.ndarray) -> np.ndarray:
        """(ny, nx) model field → (nlat, nlon) analysis-grid field."""
        v = np.asarray(field2d, float)
        assert v.shape == (self._n_lat, self._n_lon), f"field shape {v.shape} != model grid"
        m = v[np.ix_(np.clip(self._ii, 0, self._n_lat - 1), np.clip(self._jj, 0, self._n_lon - 1))].copy()
        m[~self._ri, :] = 0.0
        m[:, ~self._rj] = 0.0
        return m


def _reach_km(dist: np.ndarray, mask: np.ndarray, pct: float) -> float:
    return round(float(np.percentile(dist[mask], pct)), 0) if mask.any() else np.nan


def footprint_comparison(mod: np.ndarray, enh: np.ndarray, dist: np.ndarray,
                         *, threshold_mol_m2: float, near_source_km: float) -> dict:
    """Model vs observed CO within the modeled footprint (one overpass).

    mod, enh, dist : (nlat, nlon) model column (mol m⁻²), observed enhancement
    (mol m⁻², NaN = no obs) and source distance (km) on the analysis grid.
    Returns the table row (rounded exactly as in model_obs_comparison.csv).
    Raises AssertionError if the footprint is empty or has no valid obs.
    """
    far = dist > near_source_km
    region = (mod > threshold_mol_m2) & far              # modeled plume footprint
    obs_in = region & np.isfinite(enh)                   # footprint cells with valid obs
    obs_reg = np.isfinite(enh) & (enh > threshold_mol_m2) & far   # observed footprint, same T
    assert region.any(), f"empty modeled footprint at T={threshold_mol_m2:.0e} mol/m2"
    assert obs_in.any(), "no valid obs inside the modeled footprint"
    return {
        "n_region_cells":             int(region.sum()),
        "n_obs_valid_in_region":      int(obs_in.sum()),
        "mod_median_molm2":           round(float(np.median(mod[obs_in])), 5),
        "obs_median_molm2":           round(float(np.median(enh[obs_in])), 5),
        "obs_over_mod":               round(float(np.median(enh[obs_in]) / np.median(mod[obs_in])), 2),
        "mod_footprint_median_molm2": round(float(np.median(mod[region])), 5),
        "mod_reach_p95_km":           _reach_km(dist, region, 95),
        "mod_reach_p99_km":           _reach_km(dist, region, 99),
        "obs_reach_p95_km":           _reach_km(dist, obs_reg, 95),
        "obs_reach_p99_km":           _reach_km(dist, obs_reg, 99),
    }


def footprint_median_series(col_mol, regrid: IndexRegrid, dist: np.ndarray,
                            *, threshold_mol_m2: float, near_source_km: float) -> np.ndarray:
    """Median modeled column (mol m⁻²) inside the footprint at every cdump time (NaN if empty)."""
    far = dist > near_source_km
    out = np.full(col_mol.sizes["time"], np.nan)
    for k in range(col_mol.sizes["time"]):
        m = regrid(col_mol.isel(time=k).values)
        region = (m > threshold_mol_m2) & far
        if region.any():
            out[k] = float(np.median(m[region]))
    return out


def comparison_table(col_mol, mod_times: pd.DatetimeIndex, regrid: IndexRegrid,
                     data_dir: Path, cfg: ComparisonConfig, grid: AnalysisGrid) -> pd.DataFrame:
    """Full model_obs_comparison table: one row per configured overpass.

    For each overpass the model time step nearest to the overpass time is used.
    """
    dist = source_distance_km(cfg, grid)
    rows = []
    for ov in cfg.overpasses:
        k = int(np.argmin(np.abs(mod_times - pd.Timestamp(ov["time_utc"]))))
        obs = bin_observed_co(sorted((Path(data_dir) / ov["folder"]).glob("S5P_OFFL_L2__CO_____*.nc")),
                              cfg, grid)
        enh, _ = observed_enhancement(obs, cfg.bg_percentile)
        row = {"offset_d": ov["offset_d"], "model_time": str(mod_times[k])}
        row.update(footprint_comparison(regrid(col_mol.isel(time=k).values), enh, dist,
                                        threshold_mol_m2=cfg.threshold_mol_m2,
                                        near_source_km=cfg.near_source_km))
        rows.append(row)
    return pd.DataFrame(rows)
