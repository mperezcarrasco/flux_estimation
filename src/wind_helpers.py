from scipy.spatial import cKDTree
import numpy as np
import warnings

EARTH_RADIUS_KM: float = 6_371.0            # WGS-84 mean

def _latlon_to_xyz(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    """Convert lat/lon arrays (degrees) to unit-sphere XYZ."""
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    return np.column_stack([
        np.cos(lat) * np.cos(lon),
        np.cos(lat) * np.sin(lon),
        np.sin(lat),
    ])

def _haversine_km(
    lat1: float, lon1: float,
    lat2_arr: np.ndarray, lon2_arr: np.ndarray,
) -> np.ndarray:
    lat1, lon1 = np.deg2rad(lat1), np.deg2rad(lon1)
    lat2 = np.deg2rad(lat2_arr)
    lon2 = np.deg2rad(lon2_arr)
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def select_injection_pressure(
    ai_raw: np.ndarray,
    ai_lat: np.ndarray,
    ai_lon: np.ndarray,
    alh_pres_raw: np.ndarray,
    alh_lat: np.ndarray,
    alh_lon: np.ndarray,
    fire_lat: float,
    fire_lon: float,
    ai_threshold: float = 2.0,
    radius_km: float = 150.0,
    min_pixels: int = 20,
    aggregation: str = "median",
    fallback_hpa: float = 900.0,
    EARTH_RADIUS_KM = 6371.0
) -> dict:
    """
    Select the aerosol injection pressure for ERA5 wind extraction.

    Identifies AER_LH pixels that lie (1) within `radius_km` of the fire
    centre and (2) whose nearest AER_AI neighbour (on the unit sphere) is
    inside the plume mask (AER_AI > ai_threshold), then aggregates their
    aerosol_mid_pressure values.

    Parameters
    ----------
    ai_raw : np.ndarray
        Flat array of AER_AI values (domain-cropped, fill-replaced with NaN).
    ai_lat, ai_lon : np.ndarray
        Flat arrays of AER_AI pixel coordinates, matching ai_raw.
    alh_pres_raw : np.ndarray
        Flat array of aerosol_mid_pressure values in Pa (domain-cropped,
        fill-replaced with NaN, QA-filtered).
    alh_lat, alh_lon : np.ndarray
        Flat arrays of AER_LH pixel coordinates, matching alh_pres_raw.
    fire_lat, fire_lon : float
        Fire centre coordinates (degrees).
    ai_threshold : float
        AER_AI value above which a pixel is considered part of the plume.
        Default: 2.0.
    radius_km : float
        Maximum haversine distance (km) from the fire centre for AER_LH
        pixels to be included. Default: 150.0.
    min_pixels : int
        Minimum number of matched AER_LH pixels required before falling
        back to `fallback_hpa`. Default: 20.
    aggregation : str
        How to aggregate matched pressures: 'median' (robust, default)
        or 'max' (Jin et al. 2021 convention).
    fallback_hpa : float
        Injection pressure (hPa) returned when no valid pixels are found
        or pixel count is below `min_pixels`. Default: 900.0.

    Returns
    -------
    dict with keys:
        injection_pressure_hpa : float
            Selected injection pressure in hPa.
        pressure_source : str
            Human-readable description of how the value was obtained.
        n_alh_pixels : int
            Number of AER_LH pixels used in the aggregation.
        median_hpa : float
        max_hpa : float
        pressure_iqr_hpa : float
            IQR of matched pressures. Values > 100 hPa indicate a
            vertically complex plume where a single pressure level is
            a poor approximation.

    Notes
    -----
    KD-tree matching uses exact unit-sphere XYZ coordinates (no flat-Earth
    approximation). Each AER_LH pixel is assigned its nearest AER_AI
    neighbour; if that neighbour is a plume pixel the AER_LH pixel is
    considered inside the plume. This is geometrically exact at any
    latitude and avoids an arbitrary distance cutoff.

    Jin et al. (2021) do not specify median vs max; 'median' is the
    default because Chilean plumes are spatially extended and injection
    height varies along the plume axis.

    The returned pressure is a plume-wide aggregate, not a fire-centre
    value. For plumes > ~500 km the single-level wind assumption introduces
    known error in the EMG flux estimate.
    """
    def _fallback(reason: str, n: int = 0) -> dict:
        warnings.warn(f"{reason} Falling back to {fallback_hpa:.0f} hPa.", UserWarning)
        return dict(
            injection_pressure_hpa=fallback_hpa,
            pressure_source=f"fallback_{fallback_hpa:.0f}hPa ({reason})",
            n_alh_pixels=n,
            median_hpa=fallback_hpa,
            max_hpa=fallback_hpa,
            pressure_iqr_hpa=0.0,
        )

    if aggregation not in ("median", "max"):
        raise ValueError(f"aggregation must be 'median' or 'max', got {aggregation!r}")

    # ── Step 1: AER_AI plume mask ─────────────────────────────────────────
    plume_mask = (ai_raw > ai_threshold) & np.isfinite(ai_raw)
    n_plume = int(plume_mask.sum())
    print(f"[select_injection_pressure] AER_AI plume pixels (AI > {ai_threshold}): {n_plume}")

    if n_plume == 0:
        return _fallback("no AER_AI plume pixels found")

    # ── Step 2: Radius filter on AER_LH pixels ────────────────────────────
    alh_finite = np.isfinite(alh_pres_raw) & (alh_pres_raw > 0)
    dist_km    = _haversine_km(fire_lat, fire_lon, alh_lat, alh_lon)
    in_radius  = alh_finite & (dist_km <= radius_km)

    n_radius = int(in_radius.sum())
    print(f"[select_injection_pressure] AER_LH pixels within {radius_km:.0f} km of fire: {n_radius}")

    if n_radius == 0:
        return _fallback(f"no AER_LH pixels within {radius_km:.0f} km of fire")

    alh_lat_v  = alh_lat[in_radius]
    alh_lon_v  = alh_lon[in_radius]
    alh_pres_v = alh_pres_raw[in_radius]

    # ── Step 3: KD-tree plume match (unit-sphere XYZ) ─────────────────────
    # Build tree on ALL domain AER_AI pixels (plume and non-plume).
    # Each AER_LH pixel inherits the plume membership of its nearest
    # AER_AI neighbour — no arbitrary distance cutoff required.
    ai_xyz  = _latlon_to_xyz(ai_lat, ai_lon)
    alh_xyz = _latlon_to_xyz(alh_lat_v, alh_lon_v)

    tree = cKDTree(ai_xyz)
    _, nn_idx = tree.query(alh_xyz, k=1, workers=-1)

    in_plume  = plume_mask[nn_idx]           # boolean (N_alh_v,)
    pres_hpa  = alh_pres_v[in_plume] / 100.0 # Pa → hPa
    n_matched = int(in_plume.sum())
    print(
        f"[select_injection_pressure] AER_LH pixels matched to AI plume: {n_matched}"
    )

    if n_matched < min_pixels:
        return _fallback(
            f"only {n_matched} matched pixels (min={min_pixels})", n=n_matched
        )

    # ── Step 4: Aggregate ─────────────────────────────────────────────────
    median_hpa = float(np.median(pres_hpa))
    max_hpa    = float(np.max(pres_hpa))
    q25, q75   = np.percentile(pres_hpa, [25, 75])
    iqr_hpa    = float(q75 - q25)

    print(
        f"[select_injection_pressure] Pressure stats: "
        f"median={median_hpa:.0f} hPa, max={max_hpa:.0f} hPa, IQR={iqr_hpa:.0f} hPa"
    )

    if iqr_hpa > 100:
        warnings.warn(
            f"Pressure IQR = {iqr_hpa:.0f} hPa > 100 hPa. The plume is vertically "
            "complex; a single injection pressure level is a poor approximation.",
            UserWarning,
        )

    selected = median_hpa if aggregation == "median" else max_hpa
    source   = f"aer_ai_plume_{aggregation} (n={n_matched}, r={radius_km:.0f}km)"

    return dict(
        injection_pressure_hpa=selected,
        pressure_source=source,
        n_alh_pixels=n_matched,
        median_hpa=median_hpa,
        max_hpa=max_hpa,
        pressure_iqr_hpa=iqr_hpa,
    )