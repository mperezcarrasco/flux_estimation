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


def match_alh_plume_pressures(
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
) -> dict:
    """Match AER_LH injection-pressure pixels to the AER_AI plume.

    Single source of truth for the 3-step matching used both by
    ``select_injection_pressure`` (which aggregates to a scalar) and by the
    probabilistic-trajectory notebook (which needs the full distribution to
    fit a KDE). Steps:

    1. AER_AI plume mask (``ai_raw > ai_threshold`` and finite).
    2. Radius + finiteness filter on AER_LH pixels (``<= radius_km`` of fire).
    3. KD-tree nearest-neighbour match on unit-sphere XYZ: each in-radius
       AER_LH pixel inherits the plume membership of its nearest AER_AI pixel.

    Returns
    -------
    dict with keys:
        pres_hpa : np.ndarray
            Matched injection pressures in hPa (finite only).
        n_plume : int   number of AER_AI plume pixels
        n_radius : int  number of in-radius finite AER_LH pixels
        n_matched : int number of AER_LH pixels matched to the plume
    """
    plume_mask = (ai_raw > ai_threshold) & np.isfinite(ai_raw)
    n_plume = int(plume_mask.sum())
    if n_plume == 0:
        return dict(pres_hpa=np.array([]), n_plume=0, n_radius=0, n_matched=0)

    alh_finite = np.isfinite(alh_pres_raw) & (alh_pres_raw > 0)
    dist_km = _haversine_km(fire_lat, fire_lon, alh_lat, alh_lon)
    in_radius = alh_finite & (dist_km <= radius_km)
    n_radius = int(in_radius.sum())
    if n_radius == 0:
        return dict(pres_hpa=np.array([]), n_plume=n_plume, n_radius=0, n_matched=0)

    tree = cKDTree(_latlon_to_xyz(ai_lat, ai_lon))
    _, nn_idx = tree.query(
        _latlon_to_xyz(alh_lat[in_radius], alh_lon[in_radius]), k=1, workers=-1
    )
    in_plume = plume_mask[nn_idx]
    pres_hpa = alh_pres_raw[in_radius][in_plume] / 100.0  # Pa -> hPa
    pres_hpa = pres_hpa[np.isfinite(pres_hpa)]
    return dict(
        pres_hpa=pres_hpa,
        n_plume=n_plume,
        n_radius=n_radius,
        n_matched=int(in_plume.sum()),
    )


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
    aggregation: str = "mean",
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
        How to aggregate matched pressures: 'mean' (default),
        'median' (robust), or 'max' (Jin et al. 2021 convention).
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
        mean_hpa : float
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

    Jin et al. (2021) do not specify mean vs max; 'mean' is the
    default because Chilean plumes are spatially extended and injection
    height is approximately uniform along the plume axis.

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
            mean_hpa=fallback_hpa,
            median_hpa=fallback_hpa,
            max_hpa=fallback_hpa,
            pressure_iqr_hpa=0.0,
        )

    if aggregation not in ("mean", "median", "max"):
        raise ValueError(f"aggregation must be 'mean', 'median', or 'max', got {aggregation!r}")

    # ── Steps 1-3: match AER_LH pixels to the AER_AI plume (shared helper) ─
    matched = match_alh_plume_pressures(
        ai_raw, ai_lat, ai_lon,
        alh_pres_raw, alh_lat, alh_lon,
        fire_lat, fire_lon,
        ai_threshold=ai_threshold, radius_km=radius_km,
    )
    print(f"[select_injection_pressure] AER_AI plume pixels (AI > {ai_threshold}): {matched['n_plume']}")
    if matched["n_plume"] == 0:
        return _fallback("no AER_AI plume pixels found")
    print(f"[select_injection_pressure] AER_LH pixels within {radius_km:.0f} km of fire: {matched['n_radius']}")
    if matched["n_radius"] == 0:
        return _fallback(f"no AER_LH pixels within {radius_km:.0f} km of fire")

    pres_hpa  = matched["pres_hpa"]
    n_matched = matched["n_matched"]
    print(
        f"[select_injection_pressure] AER_LH pixels matched to AI plume: {n_matched}"
    )

    if n_matched < min_pixels:
        return _fallback(
            f"only {n_matched} matched pixels (min={min_pixels})", n=n_matched
        )

    # ── Step 4: Aggregate ─────────────────────────────────────────────────
    mean_hpa   = float(np.mean(pres_hpa))
    median_hpa = float(np.median(pres_hpa))
    max_hpa    = float(np.max(pres_hpa))
    q25, q75   = np.percentile(pres_hpa, [25, 75])
    iqr_hpa    = float(q75 - q25)

    print(
        f"[select_injection_pressure] Pressure stats: "
        f"mean={mean_hpa:.0f} hPa, median={median_hpa:.0f} hPa, "
        f"max={max_hpa:.0f} hPa, IQR={iqr_hpa:.0f} hPa"
    )

    if iqr_hpa > 100:
        warnings.warn(
            f"Pressure IQR = {iqr_hpa:.0f} hPa > 100 hPa. The plume is vertically "
            "complex; a single injection pressure level is a poor approximation.",
            UserWarning,
        )

    if aggregation == "mean":
        selected = mean_hpa
    elif aggregation == "median":
        selected = median_hpa
    else:
        selected = max_hpa
    source = f"aer_ai_plume_{aggregation} (n={n_matched}, r={radius_km:.0f}km)"

    return dict(
        injection_pressure_hpa=selected,
        pressure_source=source,
        n_alh_pixels=n_matched,
        mean_hpa=mean_hpa,
        median_hpa=median_hpa,
        max_hpa=max_hpa,
        pressure_iqr_hpa=iqr_hpa,
    )