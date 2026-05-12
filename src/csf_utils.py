#!/usr/bin/env python3
"""Cross-Sectional Flux (CSF) utility functions for CO emission quantification.

This module contains all functions needed to estimate fire CO emission rates
from TROPOMI column fields using the CSF method.  It is shared between:

Dependencies
------------
numpy, scipy (griddata, RegularGridInterpolator)

Functions
---------
Fixed-frame (global mean wind direction):
  estimate_background               — upwind-box median CO background (scalar)
  estimate_latitudinal_background   — latitude-varying polynomial background (1D lat axis)
  estimate_latitudinal_background_2d — same but for 2D irregular lat_grid (oversampled grid)
  rotate_to_wind_frame              — rotate TROPOMI grid to wind-aligned frame
  compute_csf                       — integrate ΔCO × U_perp × Δy along transects
  csf_ensemble                      — parameter sweep over CSF free parameters

Griffin et al. (2024) — Gaussian plume method:
  build_local_grid         — equirectangular projection + distance-weighted oversampling
  rotate_grid              — apply 2D rotation to local Cartesian grid (lat/lon unchanged)
  fit_gaussians_per_bin    — per-bin Gaussian + offset fit to cross-wind CO profiles
  correct_wind_direction   — data-driven wind angle correction from Gaussian peak positions
  compute_csf_griffin      — flux integration using dynamic ±3σ plume boundaries

Dynamic-frame (per-transect wind direction):
  compute_local_wind_direction — slab-by-slab vector-mean wind angles
  build_dynamic_transects      — transect geometries perpendicular to local wind
  compute_csf_dynamic          — CSF with spatially variable transect orientation

Data types
----------
CO fields are in mol m⁻² (native TROPOMI units).  Unit conversion to
kg s⁻¹ and Gg yr⁻¹ is performed inside compute_csf and compute_csf_dynamic
using the module constants below.
"""

from dataclasses import dataclass
import numpy as np
from scipy.interpolate import RegularGridInterpolator

CO_MOLAR_MASS_KG_PER_MOL: float = 28.0104 / 1000.0  # kg mol⁻¹
_WIND_SPEED_FLOOR: float = 0.1
_M_PER_DEG_LAT: float = 111320.0
SECONDS_PER_YEAR: float = 365.25 * 24.0 * 3600.0
KG_PER_GG: float = 1.0e6

@dataclass
class Transect:
    """Geometry and wind metadata for one cross-sectional transect.

    Attributes
    ----------
    positions_lat : (n_points,) latitude of transect sample points (°N)
    positions_lon : (n_points,) longitude of transect sample points (°E)
    wind_angle    : local wind direction θ_k = atan2(v, u) (radians, CCW from East)
    u_perp        : slab-mean perpendicular wind component
                    U⊥ = U cos θ_k + V sin θ_k (m s⁻¹; positive = downwind)
    ds_meters     : arc-length spacing between adjacent transect sample points (m)
    slab_pos_m    : position of this slab along the reference downwind axis (m)
    """

    positions_lat: np.ndarray   # (n_points,) °N
    positions_lon: np.ndarray   # (n_points,) °E
    wind_angle: float            # radians, CCW from East
    u_perp: float                # m s⁻¹
    ds_meters: float             # m
    slab_pos_m: float            # m

def _latlon_to_cartesian(
    lat: np.ndarray,
    lon: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Convert lat/lon grid to approximate Cartesian metres relative to domain centre.

    Parameters
    ----------
    lat : (nlat,) ascending latitude centres (°N)
    lon : (nlon,) ascending longitude centres (°E)

    Returns
    -------
    x_m         : (nlat, nlon) eastward displacement from domain centre (m)
    y_m         : (nlat, nlon) northward displacement from domain centre (m)
    lat_centre  : domain-mean latitude (°N)
    lon_centre  : domain-mean longitude (°E)
    """
    lat_centre = float(lat.mean())
    lon_centre = float(lon.mean())
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    x_m = (lon_grid - lon_centre) * np.cos(np.radians(lat_centre)) * _M_PER_DEG_LAT
    y_m = (lat_grid - lat_centre) * _M_PER_DEG_LAT
    return x_m, y_m, lat_centre, lon_centre


# ---------------------------------------------------------------------------
# Fixed-frame functions
# ---------------------------------------------------------------------------


def estimate_background(
    co_field: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    u_mean: float,
    v_mean: float,
    box_fraction: float = 0.2,
) -> tuple[float, np.ndarray]:
    """Estimate background CO column from an upwind box.

    The domain is projected onto the upwind direction vector (opposite of
    transport).  The background box covers the `box_fraction` fraction of the
    domain extent with the highest upwind projection — i.e., the pixels that
    are most directly in the path of incoming air from outside the plume.
    The background value is the median of valid (finite, positive) CO pixels
    within that box.

    Parameters
    ----------
    co_field     : (nlat, nlon) CO column in mol m⁻² (NaN where unobserved)
    lat          : (nlat,) ascending latitude centres (°N)
    lon          : (nlon,) ascending longitude centres (°E)
    u_mean       : domain-mean or plume-mean eastward wind component (m s⁻¹)
    v_mean       : domain-mean or plume-mean northward wind component (m s⁻¹)
    box_fraction : fraction of domain extent used as background box (0 < x ≤ 1)

    Returns
    -------
    background_value : float — median CO in upwind box (mol m⁻²)
    bg_mask          : (nlat, nlon) bool — True for pixels in the upwind box

    Raises
    ------
    ValueError  if wind speed is too weak to define a direction, or if
                the background box contains no valid pixels.
    """
    if not (0.0 < box_fraction <= 1.0):
        raise ValueError(
            f"box_fraction must be in (0, 1]; got {box_fraction}."
        )

    wind_mag = float(np.hypot(u_mean, v_mean))
    if wind_mag < _WIND_SPEED_FLOOR:
        raise ValueError(
            f"Mean wind speed {wind_mag:.3f} m/s is too low to define a transport "
            "direction.  Cannot place an upwind background box without a clear "
            "wind vector.  Check the ERA5 data or choose a different analysis day."
        )

    x_m, y_m, lat_centre, lon_centre = _latlon_to_cartesian(lat, lon)

    # Unit vector pointing UPWIND (opposite of transport direction)
    uw = -u_mean / wind_mag
    vw = -v_mean / wind_mag

    # Signed projection of each pixel onto the upwind direction (m)
    # Large positive value = far upwind = good background candidate
    projection = x_m * uw + y_m * vw

    proj_range = float(projection.max() - projection.min())
    threshold = float(projection.max()) - box_fraction * proj_range
    bg_mask = projection >= threshold

    co_bg_pixels = co_field[bg_mask & np.isfinite(co_field) & (co_field > 0)]
    if len(co_bg_pixels) == 0:
        raise ValueError(
            f"No valid (finite, positive) CO pixels in the upwind background box "
            f"(box_fraction={box_fraction}, wind=({u_mean:.2f}, {v_mean:.2f}) m/s).  "
            "Check data coverage for this day."
        )

    background_value = float(np.median(co_bg_pixels))
    return background_value, bg_mask


def compute_csf_dynamic(
    co_delta: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    u_field: np.ndarray,
    v_field: np.ndarray,
    plume_mask: np.ndarray,
    n_transects: int = 10,
    transect_half_width_km: float = 50.0,
    wind_angle_perturbation_deg: float = 0.0,
    aggregation: str = "mean",
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """CSF emission rate with spatially variable (per-slab) wind direction.

    Computes the cross-sectional flux by placing `n_transects` transects along
    the plume, each perpendicular to the local slab-mean ERA5 wind direction.
    This is an extension of compute_csf that relaxes the assumption of a single
    global transport direction.

    Algorithm
    ---------
    1. compute_local_wind_direction: divide plume into slabs, get θ_k per slab.
    2. Perturb each θ_k by wind_angle_perturbation_deg (for ensemble members).
    3. build_dynamic_transects: construct transect geometries.
    4. For each transect, sample ΔCO along the transect using bilinear
       interpolation (RegularGridInterpolator) and integrate:
         F_k = Σ_j ΔCO(s_j) × U⊥_k × Δs
       where U⊥_k = u_slab_k cos θ_k + v_slab_k sin θ_k is the slab-mean
       perpendicular wind component.
    5. Convert mean flux from mol s⁻¹ → kg s⁻¹ → Gg yr⁻¹.

    Parameters
    ----------
    co_delta                  : (nlat, nlon) ΔCO field in mol m⁻² (NaN unobserved)
    lat                       : (nlat,) ascending latitude centres (°N)
    lon                       : (nlon,) ascending longitude centres (°E)
    u_field                   : (nlat, nlon) ERA5 u10 on TROPOMI grid (m s⁻¹)
    v_field                   : (nlat, nlon) ERA5 v10 on TROPOMI grid (m s⁻¹)
    plume_mask                : (nlat, nlon) bool — AI hotspot pixels
    n_transects               : number of slabs / transects (≥ 1)
    transect_half_width_km    : half-length of each transect in km (> 0)
    wind_angle_perturbation_deg : perturbation applied to all slab wind angles
                                 before building transects (degrees; + = CCW).
                                 Use for ensemble uncertainty quantification.

    Returns
    -------
    slab_centers_m   : (n_transects,) slab centre positions along downwind axis (m)
    transect_fluxes  : (n_transects,) per-transect flux (mol s⁻¹; NaN if no data)
    flux_kgs         : float — aggregated (mean by default) valid-transect flux in kg s⁻¹
    flux_Ggyr        : float — aggregated valid-transect flux in Gg yr⁻¹
    aggregation      : 'mean' (default) or 'median' — how to collapse valid transect
                       fluxes into the scalar emission estimate

    Raises
    ------
    ValueError  if inputs are inconsistent, if no plume pixels exist,
                if all transects return NaN flux, or if aggregation is invalid.
    """
    assert np.all(np.diff(lat) > 0), (
        "lat must be strictly ascending; got non-monotonic array"
    )
    assert np.all(np.diff(lon) > 0), (
        "lon must be strictly ascending; got non-monotonic array"
    )

    # Step 1: Per-slab wind directions
    slab_centers_m, wind_angles, u_slab, v_slab = compute_local_wind_direction(
        u_field, v_field, plume_mask, lat, lon, n_transects
    )

    # Step 2: Apply wind angle perturbation (used for ensemble sensitivity)
    wind_angles_perturbed = wind_angles + np.deg2rad(wind_angle_perturbation_deg)

    # Step 3: Build transects with (optionally perturbed) angles
    transects = build_dynamic_transects(
        lat, lon, plume_mask, wind_angles_perturbed, u_slab, v_slab,
        n_transects, transect_half_width_km,
    )

    # Step 4: Build nearest-neighbour interpolator for ΔCO on the TROPOMI grid.
    # RegularGridInterpolator requires strictly ascending coordinates.
    # lat and lon from .pt files are confirmed ascending (assertion above).
    #
    # method="nearest" is used instead of "linear" for two reasons:
    #   1. Bilinear interpolation propagates NaN: if any of the 4 surrounding
    #      pixels is NaN (cloud gap), the interpolated result is NaN even when
    #      the nearest pixel has a valid observation.  Nearest-neighbour avoids
    #      this and returns NaN only when the nearest pixel itself is missing.
    #   2. The transect sample spacing (≈ ds_meters ≈ one TROPOMI pixel) is
    #      already at the native grid resolution, so bilinear vs nearest makes
    #      a negligible difference to the integrated flux.
    co_rgi = RegularGridInterpolator(
        (lat, lon), co_delta,
        method="nearest", bounds_error=False, fill_value=np.nan,
    )

    # Step 5: Integrate along each transect
    transect_fluxes = np.full(n_transects, np.nan)

    for k, t in enumerate(transects):
        query_pts  = np.column_stack([t.positions_lat, t.positions_lon])
        co_sampled = co_rgi(query_pts)   # (n_points,) mol m⁻²

        # Integrate over all sample points with a finite ΔCO observation.
        # The plume-mask filter used in compute_csf (fixed frame) is NOT
        # applied here.  In the fixed frame the integration runs across the
        # entire 60° domain width, so the AI mask is needed to suppress
        # background noise.  Here each transect is only ~100 km wide and
        # centred on the plume centroid; outside the plume co_delta ≈ 0,
        # contributing approximately zero to the sum.  Requiring the sample
        # point to land on a sparse AI > 2.0 pixel (0.26% of domain) with a
        # coarse 28 km sample spacing would cause most transects to return NaN.
        # Do NOT filter by co_sampled > 0 either — see compute_csf docstring.
        valid = np.isfinite(co_sampled)
        if not valid.any():
            continue

        # F_k = Σ_j ΔCO(s_j) × U⊥_k × Δs   [mol s⁻¹]
        # U⊥_k (t.u_perp) is scalar: slab-mean perpendicular wind component.
        transect_fluxes[k] = float(
            np.sum(co_sampled[valid] * t.u_perp * t.ds_meters)
        )

    valid_fluxes = transect_fluxes[np.isfinite(transect_fluxes)]
    if len(valid_fluxes) == 0:
        raise ValueError(
            "All dynamic transects returned NaN flux.  Check that the ΔCO "
            "field, wind data, and transect positions overlap with the domain."
        )

    flux_mols = _apply_aggregation(valid_fluxes, aggregation)
    flux_kgs  = flux_mols * CO_MOLAR_MASS_KG_PER_MOL
    flux_Ggyr = flux_kgs * SECONDS_PER_YEAR / KG_PER_GG

    return slab_centers_m, transect_fluxes, flux_kgs, flux_Ggyr


def compute_local_wind_direction(
    u_field: np.ndarray,
    v_field: np.ndarray,
    plume_mask: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    n_transects: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-slab vector-mean wind direction along the global downwind axis.

    The plume is divided into `n_transects` equal slabs by projecting each
    plume pixel onto the global mean wind direction axis.  Within each slab,
    the vector-mean of u and v gives a local wind angle θ_k.

    Parameters
    ----------
    u_field     : (nlat, nlon) ERA5 u10 on TROPOMI grid (m s⁻¹)
    v_field     : (nlat, nlon) ERA5 v10 on TROPOMI grid (m s⁻¹)
    plume_mask  : (nlat, nlon) bool — True for AI hotspot pixels
    lat         : (nlat,) ascending latitude centres (°N)
    lon         : (nlon,) ascending longitude centres (°E)
    n_transects : number of slabs (must be ≥ 1)

    Returns
    -------
    slab_centers_m : (n_transects,) slab centre positions along global downwind axis (m)
    wind_angles    : (n_transects,) local wind angle θ_k = atan2(v_slab, u_slab) (radians)
    u_slab         : (n_transects,) slab-mean eastward wind component (m s⁻¹)
    v_slab         : (n_transects,) slab-mean northward wind component (m s⁻¹)

    Notes
    -----
    Slabs with no plume pixels fall back to the global plume-mean wind.

    # WARNING: wind_angles contains atan2(v_slab, u_slab), which is derived
    from the vector-mean of u and v — not from direct averaging of angles.
    Direct angle averaging fails near the 0°/360° wrap boundary (e.g., two
    wind vectors at 350° and 10° average to 0°, but naive mean gives 180°).
    The vector-mean approach used here (average u and v separately, then atan2)
    is always correct.  Do not replace it with np.mean(angles).

    Raises
    ------
    ValueError  if n_transects < 1, if no plume pixels have valid wind,
                or if the global plume-mean wind speed is too weak.
    """
    if n_transects < 1:
        raise ValueError(f"n_transects must be >= 1; got {n_transects}.")

    valid_wind = plume_mask & np.isfinite(u_field) & np.isfinite(v_field)
    if not valid_wind.any():
        raise ValueError(
            "No plume pixels have valid ERA5 wind data.  "
            "Cannot compute local wind directions."
        )

    u_global = float(u_field[valid_wind].mean())
    v_global = float(v_field[valid_wind].mean())
    wind_mag_global = float(np.hypot(u_global, v_global))
    if wind_mag_global < _WIND_SPEED_FLOOR:
        raise ValueError(
            f"Global plume-mean wind {wind_mag_global:.3f} m/s is too weak to "
            "define a reference downwind axis."
        )

    theta_global = float(np.arctan2(v_global, u_global))
    cos_g, sin_g = np.cos(theta_global), np.sin(theta_global)

    # Project all pixels onto global downwind axis
    x_m, y_m, lat_centre, lon_centre = _latlon_to_cartesian(lat, lon)
    x_rot = x_m * cos_g + y_m * sin_g  # (nlat, nlon)

    # Plume extent along global downwind axis
    x_rot_plume = x_rot[plume_mask]
    x_min = float(x_rot_plume.min())
    x_max = float(x_rot_plume.max())

    slab_edges    = np.linspace(x_min, x_max, n_transects + 1)
    slab_centers_m = 0.5 * (slab_edges[:-1] + slab_edges[1:])

    wind_angles = np.full(n_transects, np.nan)
    u_slab      = np.full(n_transects, np.nan)
    v_slab      = np.full(n_transects, np.nan)

    for k in range(n_transects):
        slab_mask = (
            plume_mask
            & (x_rot >= slab_edges[k])
            & (x_rot < slab_edges[k + 1])
            & np.isfinite(u_field)
            & np.isfinite(v_field)
        )

        if not slab_mask.any():
            # No plume pixels in this slab — fall back to global mean
            u_slab[k] = u_global
            v_slab[k] = v_global
            wind_angles[k] = theta_global
            continue

        u_k = float(u_field[slab_mask].mean())
        v_k = float(v_field[slab_mask].mean())
        u_slab[k] = u_k
        v_slab[k] = v_k
        # atan2(mean_v, mean_u) = vector-mean wind angle (no wraparound issue)
        wind_angles[k] = float(np.arctan2(v_k, u_k))

    assert not np.any(np.isnan(wind_angles)), (
        "Internal error: wind_angles contains NaN after slab loop.  "
        "All slabs should have been filled (with fallback to global mean)."
    )

    return slab_centers_m, wind_angles, u_slab, v_slab


def build_dynamic_transects(
    lat: np.ndarray,
    lon: np.ndarray,
    plume_mask: np.ndarray,
    local_wind_angles: np.ndarray,
    u_slab: np.ndarray,
    v_slab: np.ndarray,
    n_transects: int,
    transect_half_width_km: float = 50.0,
) -> list[Transect]:
    """Build transect geometries perpendicular to the local wind direction for each slab.

    For each slab k the transect runs perpendicular to local_wind_angles[k],
    centred on the slab centre position along the reference downwind axis.
    Sample points are spaced at ~0.25° projected at the domain centre latitude
    (≈27.8 km at −37°).

    Parameters
    ----------
    lat                  : (nlat,) ascending latitude centres (°N)
    lon                  : (nlon,) ascending longitude centres (°E)
    plume_mask           : (nlat, nlon) bool — used to locate slab centres
    local_wind_angles    : (n_transects,) local wind angle θ_k (radians, CCW from East)
    u_slab               : (n_transects,) slab-mean eastward wind (m s⁻¹)
    v_slab               : (n_transects,) slab-mean northward wind (m s⁻¹)
    n_transects          : number of transects (must match len(local_wind_angles))
    transect_half_width_km : half-length of each transect on either side of centre (km)

    Returns
    -------
    list of Transect dataclasses, one per slab

    Notes
    -----
    # WARNING: If transect_half_width_km is smaller than the plume cross-section,
    edge pixels are missed and the flux is underestimated.  If larger than the
    plume, background CO pixels (ΔCO ≈ 0) are sampled; this adds noise but not
    systematic bias for a well-estimated background.  The 50 km default covers
    the typical Jan 18 plume cross-section (~40–50 km wide at AI > 2.0).
    Inspect the transect visualisation in the notebook before adjusting.

    Raises
    ------
    ValueError  if n_transects does not match len(local_wind_angles), if no
                plume pixels exist, or if the reference wind is too weak.
    """
    if len(local_wind_angles) != n_transects:
        raise ValueError(
            f"local_wind_angles has {len(local_wind_angles)} entries but "
            f"n_transects = {n_transects}.  They must match."
        )
    if transect_half_width_km <= 0.0:
        raise ValueError(
            f"transect_half_width_km must be positive; got {transect_half_width_km}."
        )

    lat_centre = float(lat.mean())
    lon_centre = float(lon.mean())
    m_per_deg_lat = _M_PER_DEG_LAT
    m_per_deg_lon = _M_PER_DEG_LAT * np.cos(np.radians(lat_centre))

    # Reference direction from vector-mean of slab winds (reproduces the axis
    # used by compute_local_wind_direction so slab centres match)
    u_ref = float(np.nanmean(u_slab))
    v_ref = float(np.nanmean(v_slab))
    wind_mag_ref = float(np.hypot(u_ref, v_ref))
    if wind_mag_ref < _WIND_SPEED_FLOOR:
        raise ValueError(
            f"Vector-mean of slab winds = {wind_mag_ref:.3f} m/s — too weak "
            "to define the reference downwind axis."
        )
    theta_ref = float(np.arctan2(v_ref, u_ref))
    cos_r, sin_r = np.cos(theta_ref), np.sin(theta_ref)

    # Cartesian coordinates of the full domain
    x_m, y_m, _, _ = _latlon_to_cartesian(lat, lon)

    # Locate plume extent along reference axis
    x_rot = x_m * cos_r + y_m * sin_r
    x_rot_plume = x_rot[plume_mask]
    if len(x_rot_plume) == 0:
        raise ValueError("No plume pixels found.  Cannot build transects.")
    x_min = float(x_rot_plume.min())
    x_max = float(x_rot_plume.max())

    # Slab centres (same partitioning as compute_local_wind_direction)
    slab_edges     = np.linspace(x_min, x_max, n_transects + 1)
    slab_centers_m = 0.5 * (slab_edges[:-1] + slab_edges[1:])

    # Transect sample spacing: one TROPOMI pixel at domain centre latitude
    ds_meters   = 0.25 * np.cos(np.radians(lat_centre)) * _M_PER_DEG_LAT
    half_width_m = transect_half_width_km * 1e3
    n_points    = max(int(2 * half_width_m / ds_meters) + 1, 3)
    s_values    = np.linspace(-half_width_m, half_width_m, n_points)
    # Actual spacing may differ slightly from ds_meters due to integer rounding
    ds_actual = float(abs(s_values[1] - s_values[0]))

    transects: list[Transect] = []

    for k in range(n_transects):
        theta_k = float(local_wind_angles[k])
        cos_k, sin_k = np.cos(theta_k), np.sin(theta_k)

        # Slab centre: use the plume pixel centroid for this slab.
        # DO NOT place the centre on the reference axis (slab_centers_m[k] *
        # cos_r, sin_r).  The reference axis passes through the DOMAIN centre
        # along the mean wind direction.  When the wind has a large cross-track
        # component the axis drifts away from the plume in the perpendicular
        # direction, and the transect (which runs perp to the wind) fails to
        # intersect the plume entirely.  Example: with wind at 83° (nearly
        # northward) the transect is nearly east-west; the reference axis moves
        # ~0.113 km north per km of downwind travel, while the plume centroid
        # may be 40–50 km further north — well outside the transect's ±50 km
        # east-west sweep.
        slab_pix = (
            plume_mask
            & (x_rot >= slab_edges[k])
            & (x_rot < slab_edges[k + 1])
        )
        if slab_pix.any():
            xc_m = float(x_m[slab_pix].mean())
            yc_m = float(y_m[slab_pix].mean())
        else:
            # No plume pixels in this slab — fall back to reference axis.
            xc_m = slab_centers_m[k] * cos_r
            yc_m = slab_centers_m[k] * sin_r

        # Transect perpendicular direction: rotate wind direction by +90°
        # Wind direction: (cos_k, sin_k); perpendicular: (−sin_k, cos_k)
        perp_x = -sin_k
        perp_y =  cos_k

        # Sample point positions in domain-relative Cartesian metres
        x_pts = xc_m + s_values * perp_x
        y_pts = yc_m + s_values * perp_y

        # Convert to lat/lon
        pos_lat = lat_centre + y_pts / m_per_deg_lat
        pos_lon = lon_centre + x_pts / m_per_deg_lon

        # Perpendicular wind component: U⊥ = U cos θ_k + V sin θ_k
        u_perp_k = float(u_slab[k] * cos_k + v_slab[k] * sin_k)

        transects.append(Transect(
            positions_lat=pos_lat,
            positions_lon=pos_lon,
            wind_angle=theta_k,
            u_perp=u_perp_k,
            ds_meters=ds_actual,
            slab_pos_m=float(slab_centers_m[k]),
        ))

    return transects

def _apply_aggregation(values: np.ndarray, aggregation: str) -> float:
    """Return mean or median of *values* depending on *aggregation* string."""
    if aggregation == "mean":
        return float(values.mean())
    if aggregation == "median":
        return float(np.median(values))
    raise ValueError(
        f"aggregation must be 'mean' or 'median'; got {aggregation!r}"
    )