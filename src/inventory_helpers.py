"""Bottom-up CAMS GFAS inventory integration for the emission comparison.

Integrates a CAMS GFAS fire-emission field (kg m-2 s-1, instantaneous at a UTC
hour) over a circular region around the fire, to compare against the top-down
CSF/EMG fluxes at a physically-defined (FRP-enclosing) radius.

GFAS conventions handled here:
* longitudes are 0-360 (converted to -180/180);
* latitudes descend (89.95 -> -89.95);
* values are instantaneous emission rates in kg m-2 s-1.

Reuses ``wind_helpers._haversine_km`` for distances. GFED5 is intentionally not
included (top-down vs the hourly, overpass-matched GFAS only).
"""

from __future__ import annotations

import re
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd

from .wind_helpers import _haversine_km

_R_EARTH_M = 6_371_000.0


def parse_gfas_time(fp: Path) -> pd.Timestamp:
    """Parse the UTC valid time from a CAMS GFAS filename (……YYYYMMDDHHMM……)."""
    m = re.search(r"(\d{12}|\d{10})", Path(fp).name)
    if not m:
        raise ValueError(f"Cannot parse a time stamp from {Path(fp).name}")
    ts = m.group(1)
    fmt = "%Y%m%d%H%M" if len(ts) == 12 else "%Y%m%d%H"
    return pd.to_datetime(ts, format=fmt)


def gfas_integral(
    gfas_file: Path,
    varname: str,
    fire_lat: float,
    fire_lon: float,
    radius_km: float,
    crop_deg: float = 12.0,
) -> dict:
    """Integrate a GFAS emission field within ``radius_km`` of the fire.

    Parameters
    ----------
    gfas_file : Path
        A GFAS NetCDF file (e.g. ``*cofire*.nc`` or ``*noxfire*.nc``).
    varname : str
        Variable name inside the file (e.g. ``'cofire'``, ``'noxfire'``).
    fire_lat, fire_lon : float
        Fire centre (degrees, -180/180 longitude).
    radius_km : float
        Integration radius (great-circle).
    crop_deg : float
        Half-width of the lat/lon crop window around the fire (performance).

    Returns
    -------
    dict with keys: total_kg_s, n_cells_nonzero, radius_km, valid_time.

    Notes
    -----
    Assumes the crop window does not straddle the 0/360 longitude seam (true for
    the Chilean domain). Cell area = R^2 . dlat . dlon . cos(lat).
    """
    gfas_file = Path(gfas_file)
    if not gfas_file.exists():
        raise FileNotFoundError(f"GFAS file not found: {gfas_file}")

    with netCDF4.Dataset(str(gfas_file)) as ds:
        if varname not in ds.variables:
            raise KeyError(
                f"{varname!r} not in {gfas_file.name}; vars={list(ds.variables)}")
        lat = np.array(ds.variables["latitude"][:], dtype=float)   # descending
        lon = np.array(ds.variables["longitude"][:], dtype=float)  # 0-360
        fld = np.array(ds.variables[varname][0], dtype=float)      # kg m-2 s-1

    lat_min, lat_max = fire_lat - crop_deg, fire_lat + crop_deg
    lon_min = (fire_lon - crop_deg) % 360.0
    lon_max = (fire_lon + crop_deg) % 360.0
    if lon_min > lon_max:
        raise ValueError(
            "Crop window straddles the 0/360 longitude seam; not supported here.")

    lm = (lat >= lat_min) & (lat <= lat_max)
    om = (lon >= lon_min) & (lon <= lon_max)
    if not lm.any() or not om.any():
        raise ValueError("Empty GFAS crop window — check fire location / crop_deg.")

    lat_c = lat[lm]
    lon_c = lon[om] - 360.0  # -> -180/180
    fld_c = fld[np.ix_(lm, om)]

    dlat_r = abs(float(np.diff(lat).mean())) * np.pi / 180.0
    dlon_r = abs(float(np.diff(lon).mean())) * np.pi / 180.0
    lon_g, lat_g = np.meshgrid(lon_c, lat_c)
    cell_area = (_R_EARTH_M * dlat_r) * (_R_EARTH_M * dlon_r * np.cos(np.radians(lat_g)))

    dist_km = _haversine_km(fire_lat, fire_lon, lat_g.ravel(), lon_g.ravel()).reshape(lat_g.shape)
    mask = dist_km <= radius_km
    total_kg_s = float((fld_c * cell_area)[mask].sum())
    n_cells = int((mask & (fld_c > 0)).sum())

    return dict(
        total_kg_s=total_kg_s,
        n_cells_nonzero=n_cells,
        radius_km=float(radius_km),
        valid_time=parse_gfas_time(gfas_file),
    )


def gfas_radius_sensitivity(
    gfas_file: Path,
    varname: str,
    fire_lat: float,
    fire_lon: float,
    radii_km: list[float],
    crop_deg: float = 12.0,
) -> pd.DataFrame:
    """GFAS integral at several radii (kg s-1). One row per radius."""
    rows = [gfas_integral(gfas_file, varname, fire_lat, fire_lon, r, crop_deg)
            for r in radii_km]
    return pd.DataFrame(rows)
