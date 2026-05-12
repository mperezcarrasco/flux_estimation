"""MODIS active-fire clustering and dominant-cluster selection.

Produces a daily fire-centroid CSV from a raw FIRMS MCD14DL download
(``fire_nrt_M-C61_*.csv``). The output CSV is the input format expected
by ``main.py nox`` and ``main.py co``.

Reference implementation:
    notebooks/temporal_fire_hotspot_detection.ipynb

Notes
-----
* Single-linkage / complete-linkage hierarchical clustering on Haversine
  distances, with cut-distance threshold from configuration.
* FRP-weighted centroid per cluster.
* Dominant cluster = max mean-FRP cluster within ``search_radius_km`` of an
  anchor point and above ``min_frp_mw``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

# Columns expected from a FIRMS MCD14DL CSV download
EXPECTED_COLS = [
    "latitude", "longitude", "brightness", "scan", "track",
    "acq_date", "acq_time", "satellite", "instrument",
    "confidence", "version", "bright_t31", "frp", "daynight",
]

# Output schema (consumed by main.py nox / co via --centroid-csv)
CENTROID_OUTPUT_COLS = [
    "date", "fire_lat", "fire_lon", "frp_mw",
    "n_pixels", "n_clusters_total", "dist_to_anchor_km",
]


def load_modis_csv(
    csv_path: Path,
    date_start: pd.Timestamp,
    date_end: pd.Timestamp,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    min_confidence: int,
    daytime_only: bool,
    satellite: Optional[str],
) -> pd.DataFrame:
    """Load a FIRMS MCD14DL CSV and apply the standard filters.

    Returns a DataFrame indexed sequentially with columns: latitude,
    longitude, frp, acq_date, acq_time, datetime_utc, satellite, daynight,
    confidence.

    Raises
    ------
    FileNotFoundError
        If ``csv_path`` does not exist.
    ValueError
        If the CSV is missing any required MCD14DL column or no rows
        survive the filters.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"MODIS CSV not found: {csv_path}")

    df_raw = pd.read_csv(
        csv_path,
        parse_dates=["acq_date"],
        dtype={"acq_time": str, "confidence": int, "version": str},
    )
    missing = [c for c in EXPECTED_COLS if c not in df_raw.columns]
    if missing:
        raise ValueError(
            f"MCD14DL file {csv_path.name} missing required columns: {missing}"
        )

    df = df_raw.copy()
    df = df[(df["acq_date"] >= date_start) & (df["acq_date"] <= date_end)]
    df = df[
        (df["latitude"] >= lat_min) & (df["latitude"] <= lat_max)
        & (df["longitude"] >= lon_min) & (df["longitude"] <= lon_max)
    ]
    df = df[df["confidence"] >= min_confidence]
    if daytime_only:
        df = df[df["daynight"] == "D"]
    if satellite is not None:
        df = df[df["satellite"] == satellite]

    if len(df) == 0:
        raise ValueError(
            f"No MODIS detections survived filtering "
            f"(domain {lat_min}..{lat_max} N, {lon_min}..{lon_max} E, "
            f"conf>={min_confidence}, daytime={daytime_only}, "
            f"sat={satellite}, dates {date_start.date()}..{date_end.date()})."
        )

    df = df.copy()
    df["acq_time"] = df["acq_time"].astype(str).str.zfill(4)
    df["datetime_utc"] = pd.to_datetime(
        df["acq_date"].dt.strftime("%Y-%m-%d") + " "
        + df["acq_time"].str[:2] + ":" + df["acq_time"].str[2:],
        format="%Y-%m-%d %H:%M",
    )
    return df.reset_index(drop=True)


def cluster_fires(
    daily_df: pd.DataFrame,
    cluster_radius_km: float,
    link_method: str = "complete",
    earth_radius_km: float = 6371.0,
) -> pd.DataFrame:
    """Hierarchical-linkage clustering of MODIS detections for one day.

    Detections within ``cluster_radius_km`` (great-circle) are merged
    into the same cluster.  The FRP-weighted centroid of each cluster
    is reported.

    Parameters
    ----------
    daily_df : pd.DataFrame
        Filtered detections for a single day; must contain
        ``latitude``, ``longitude``, ``frp``.
    cluster_radius_km : float
        Cut-distance threshold (km).  20 km is the Jin et al. (2021)
        single-linkage convention.
    link_method : {'single', 'complete', 'average'}
        scipy.cluster.hierarchy.linkage method.
    earth_radius_km : float

    Returns
    -------
    pd.DataFrame with columns: cluster_id, n_detections, mean_frp_mw,
        max_frp_mw, frp_centroid_lat, frp_centroid_lon
        (sorted by mean_frp_mw descending).
    """
    cols = ["cluster_id", "n_detections", "mean_frp_mw", "max_frp_mw",
            "frp_centroid_lat", "frp_centroid_lon"]
    if len(daily_df) == 0:
        return pd.DataFrame(columns=cols)

    if len(daily_df) == 1:
        row = daily_df.iloc[0]
        return pd.DataFrame([{
            "cluster_id": 1,
            "n_detections": 1,
            "mean_frp_mw": float(row["frp"]),
            "max_frp_mw": float(row["frp"]),
            "frp_centroid_lat": float(row["latitude"]),
            "frp_centroid_lon": float(row["longitude"]),
        }])

    lats = np.deg2rad(daily_df["latitude"].values)
    lons = np.deg2rad(daily_df["longitude"].values)
    dlat = lats[:, None] - lats[None, :]
    dlon = lons[:, None] - lons[None, :]
    a = (np.sin(dlat / 2) ** 2
         + np.cos(lats[:, None]) * np.cos(lats[None, :])
         * np.sin(dlon / 2) ** 2)
    dist_matrix_km = 2.0 * earth_radius_km * np.arcsin(
        np.sqrt(np.clip(a, 0.0, 1.0))
    )
    dists_km = squareform(dist_matrix_km)
    Z = linkage(dists_km, method=link_method)
    labels = fcluster(Z, t=cluster_radius_km, criterion="distance")

    out = daily_df.copy()
    out["cluster_id"] = labels
    rows = []
    for cid, grp in out.groupby("cluster_id"):
        w = grp["frp"].values
        rows.append({
            "cluster_id": int(cid),
            "n_detections": int(len(grp)),
            "mean_frp_mw": float(w.mean()),
            "max_frp_mw": float(w.max()),
            "frp_centroid_lat": float(np.average(grp["latitude"], weights=w)),
            "frp_centroid_lon": float(np.average(grp["longitude"], weights=w)),
        })
    return (
        pd.DataFrame(rows)
        .sort_values("mean_frp_mw", ascending=False)
        .reset_index(drop=True)
    )


def select_dominant_cluster(
    clusters_df: pd.DataFrame,
    search_lat: float,
    search_lon: float,
    search_radius_km: float,
    min_frp_mw: float,
    earth_radius_km: float = 6371.0,
) -> Optional[pd.Series]:
    """Pick the highest-FRP cluster within a circular search region.

    Returns ``None`` if no cluster qualifies (no fluxes for that day).
    """
    if len(clusters_df) == 0:
        return None

    lat_r = np.deg2rad(clusters_df["frp_centroid_lat"].values)
    lon_r = np.deg2rad(clusters_df["frp_centroid_lon"].values)
    lat0 = np.deg2rad(search_lat)
    lon0 = np.deg2rad(search_lon)
    dlat = lat_r - lat0
    dlon = lon_r - lon0
    a = (np.sin(dlat / 2) ** 2
         + np.cos(lat0) * np.cos(lat_r) * np.sin(dlon / 2) ** 2)
    dist_km = 2.0 * earth_radius_km * np.arcsin(np.sqrt(a))

    cdf = clusters_df.copy()
    cdf["dist_to_anchor_km"] = dist_km
    qualifying = cdf[
        (cdf["dist_to_anchor_km"] <= search_radius_km)
        & (cdf["mean_frp_mw"] >= min_frp_mw)
    ]
    if len(qualifying) == 0:
        return None
    return qualifying.loc[qualifying["mean_frp_mw"].idxmax()]


def run_clustering(
    modis_csv: Path,
    date_start: pd.Timestamp,
    date_end: pd.Timestamp,
    config: dict,
) -> pd.DataFrame:
    """End-to-end: filter MODIS CSV, cluster each day, pick dominant.

    Returns one row per date in [``date_start``, ``date_end``] for which
    a dominant cluster exists.  Days with no qualifying cluster are
    omitted from the output (CLI surfaces these as warnings).

    Parameters
    ----------
    modis_csv : Path
    date_start, date_end : pd.Timestamp
    config : dict
        Parsed YAML config (see ``config/default_cluster.yaml``).

    Returns
    -------
    pd.DataFrame with columns CENTROID_OUTPUT_COLS.
    """
    domain = config["domain"]
    modis_cfg = config["modis"]
    cluster_cfg = config["clustering"]
    dom_cfg = config["dominant"]

    df = load_modis_csv(
        modis_csv, date_start, date_end,
        lat_min=domain["lat_min"], lat_max=domain["lat_max"],
        lon_min=domain["lon_min"], lon_max=domain["lon_max"],
        min_confidence=int(modis_cfg["min_confidence"]),
        daytime_only=bool(modis_cfg["daytime_only"]),
        satellite=modis_cfg.get("satellite"),
    )

    rows: list[dict] = []
    skipped: list[str] = []
    for day in pd.date_range(date_start, date_end, freq="D"):
        day_df = df[df["acq_date"].dt.normalize() == day]
        clusters = cluster_fires(
            day_df,
            cluster_radius_km=float(cluster_cfg["cluster_radius_km"]),
            link_method=str(cluster_cfg["link_method"]),
            earth_radius_km=float(cluster_cfg["earth_radius_km"]),
        )
        dom = select_dominant_cluster(
            clusters,
            search_lat=float(dom_cfg["search_lat"]),
            search_lon=float(dom_cfg["search_lon"]),
            search_radius_km=float(dom_cfg["search_radius_km"]),
            min_frp_mw=float(dom_cfg["min_frp_mw"]),
            earth_radius_km=float(cluster_cfg["earth_radius_km"]),
        )
        if dom is None:
            skipped.append(day.strftime("%Y-%m-%d"))
            continue

        rows.append({
            "date":               day.strftime("%Y-%m-%d"),
            "fire_lat":           float(dom["frp_centroid_lat"]),
            "fire_lon":           float(dom["frp_centroid_lon"]),
            "frp_mw":             float(dom["mean_frp_mw"]),
            "n_pixels":           int(dom["n_detections"]),
            "n_clusters_total":   int(len(clusters)),
            "dist_to_anchor_km":  float(dom["dist_to_anchor_km"]),
        })

    if len(rows) == 0:
        raise RuntimeError(
            "No dominant cluster found on any day in the window. "
            "Check anchor location, search radius, and min_frp_mw."
        )

    out = pd.DataFrame(rows, columns=CENTROID_OUTPUT_COLS)
    if skipped:
        import warnings
        warnings.warn(
            f"No dominant cluster on these dates (skipped): "
            f"{', '.join(skipped)}",
            UserWarning,
        )
    return out


def load_centroid_csv(centroid_csv: Path) -> pd.DataFrame:
    """Load a centroid CSV produced by ``run_clustering`` and validate it.

    Returns the DataFrame with ``date`` parsed as datetime.

    Raises
    ------
    FileNotFoundError, ValueError
    """
    centroid_csv = Path(centroid_csv)
    if not centroid_csv.exists():
        raise FileNotFoundError(f"Centroid CSV not found: {centroid_csv}")
    df = pd.read_csv(centroid_csv, parse_dates=["date"])
    required = {"date", "fire_lat", "fire_lon"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Centroid CSV {centroid_csv.name} missing required columns: "
            f"{sorted(missing)}.  Run `main.py cluster` to regenerate."
        )
    return df


def lookup_centroid(
    centroid_csv: Path,
    target_date: pd.Timestamp,
) -> tuple[float, float]:
    """Return (lat, lon) for a given date from a centroid CSV.

    Raises
    ------
    KeyError
        If the date is absent from the CSV.
    """
    df = load_centroid_csv(centroid_csv)
    target = pd.Timestamp(target_date).normalize()
    hit = df[df["date"].dt.normalize() == target]
    if len(hit) == 0:
        avail = sorted(df["date"].dt.strftime("%Y-%m-%d").unique())
        raise KeyError(
            f"No centroid for date {target.date()} in {centroid_csv}. "
            f"Available dates: {avail}"
        )
    if len(hit) > 1:
        raise ValueError(
            f"Multiple centroids for date {target.date()} in {centroid_csv}."
        )
    row = hit.iloc[0]
    return float(row["fire_lat"]), float(row["fire_lon"])
