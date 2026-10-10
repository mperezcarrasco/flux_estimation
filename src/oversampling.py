"""Adams et al. (2019) two-step oversampling of TROPOMI pixels onto a regular grid.

Used for the CO Cross-Sectional Flux retrieval (Griffin et al. 2024, Sect. 2.2:
"a binned upwind–downwind domain of a regular grid size (4 km × 4 km) ...
the satellite observations are weighted by the actual pixel size and
oversampled by 7 km (Adams et al., 2019)").

Adams et al. (2019, ACP 19, 2577, Sect. 3.1) define two steps:

1. **Footprint.** Every grid point whose centre lies inside a pixel's
   4-corner footprint polygon receives that pixel's value. A point covered by
   two overlapping pixels (e.g. two overpasses) receives both.
2. **Radius.** Each grid cell is the weighted mean of all step-1 values within
   an averaging radius of its centre, weight = 1 / pixel area. A large
   (swath-edge) pixel feeds many grid points, so the inverse-area weight keeps
   its total influence equal to that of a small pixel.

Grid convention: a regular lat/lon grid whose cell is ``cell_km`` × ``cell_km``
at a reference latitude (the fire). Centres are ``lat0 + (i + ½)·dlat`` and
``lon0 + (j + ½)·dlon``. Away from the reference latitude the E–W cell width
scales with cos(lat) (4 km at 37.5°S → ≈ 4.4 km at 30°S, ≈ 3.5 km at 45°S).

Validated in ``notebooks/extra/experimental_co_csf_4km_oversampling.ipynb`` and
``tests/test_oversampling.py`` (hand-computed known answers).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

KM_PER_DEG_LAT: float = 111.320   # km per degree of latitude (same as csf_utils._M_PER_DEG_LAT)


@dataclass(frozen=True)
class GridSpec:
    """Regular lat/lon grid with cells of ``cell_km`` at ``ref_lat``.

    Attributes
    ----------
    lat0, lon0 : south / west grid edge (deg)
    dlat, dlon : cell size (deg)
    nlat, nlon : number of rows / columns
    cell_km    : cell size at ref_lat (km)
    ref_lat    : latitude where cells are exactly cell_km × cell_km (deg)
    """

    lat0: float
    lon0: float
    dlat: float
    dlon: float
    nlat: int
    nlon: int
    cell_km: float
    ref_lat: float

    @property
    def lat(self) -> np.ndarray:
        """(nlat,) ascending row-centre latitudes (deg)."""
        return self.lat0 + (np.arange(self.nlat) + 0.5) * self.dlat

    @property
    def lon(self) -> np.ndarray:
        """(nlon,) ascending column-centre longitudes (deg)."""
        return self.lon0 + (np.arange(self.nlon) + 0.5) * self.dlon


def regular_grid_km(lat_min: float, lat_max: float, lon_min: float, lon_max: float,
                    cell_km: float, ref_lat: float) -> GridSpec:
    """Grid covering [lat_min, lat_max] × [lon_min, lon_max] with cell_km cells at ref_lat.

    dlat = cell_km / 111.32 km; dlon = cell_km / (111.32 km · cos(ref_lat)).
    Whole cells only: the grid ends at most one cell short of lat_max / lon_max.
    """
    assert lat_max > lat_min and lon_max > lon_min, "Empty domain."
    assert cell_km > 0, f"cell_km must be > 0; got {cell_km}."
    assert -90.0 < ref_lat < 90.0, f"ref_lat out of range: {ref_lat}."
    dlat = cell_km / KM_PER_DEG_LAT
    dlon = cell_km / (KM_PER_DEG_LAT * np.cos(np.radians(ref_lat)))
    nlat = int(np.floor((lat_max - lat_min) / dlat))
    nlon = int(np.floor((lon_max - lon_min) / dlon))
    assert nlat > 0 and nlon > 0, "Domain smaller than one cell."
    return GridSpec(lat0=float(lat_min), lon0=float(lon_min), dlat=float(dlat),
                    dlon=float(dlon), nlat=nlat, nlon=nlon, cell_km=float(cell_km),
                    ref_lat=float(ref_lat))


def pixel_signed_area_km2(lat_b: np.ndarray, lon_b: np.ndarray) -> np.ndarray:
    """Signed footprint area (km²) of each pixel from its 4 corners (shoelace formula).

    Local equirectangular frame per pixel: x = Δlon·cos(lat0)·111.32 km,
    y = Δlat·111.32 km. The sign gives the corner order (+ counter-clockwise,
    − clockwise). lat_b, lon_b: (n, 4) degrees. Returns (n,) km².
    """
    assert lat_b.ndim == 2 and lat_b.shape[1] == 4 and lat_b.shape == lon_b.shape, \
        f"Corner arrays must be (n, 4); got {lat_b.shape}, {lon_b.shape}."
    lat0 = lat_b.mean(axis=1, keepdims=True)
    lon0 = lon_b.mean(axis=1, keepdims=True)
    x = (lon_b - lon0) * KM_PER_DEG_LAT * np.cos(np.radians(lat0))
    y = (lat_b - lat0) * KM_PER_DEG_LAT
    return 0.5 * np.sum(x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y, axis=1)


def points_in_quads(px: np.ndarray, py: np.ndarray,
                    qx: np.ndarray, qy: np.ndarray) -> np.ndarray:
    """True where point (px, py) lies inside the convex quad (qx, qy) of the same row.

    Edge cross-products must all share one sign (either corner order). The test
    is invariant to scaling lon, so degrees can be used directly.
    Shapes: px, py (m,); qx, qy (m, 4).
    """
    cross = np.empty(qx.shape)
    for k in range(4):
        x0, y0 = qx[:, k], qy[:, k]
        x1, y1 = qx[:, (k + 1) % 4], qy[:, (k + 1) % 4]
        cross[:, k] = (x1 - x0) * (py - y0) - (y1 - y0) * (px - x0)
    return np.all(cross >= 0, axis=1) | np.all(cross <= 0, axis=1)


def footprint_points(lat_b: np.ndarray, lon_b: np.ndarray, lat0: float, dlat: float,
                     nlat: int, lon0: float, dlon: float, nlon: int
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Step 1: every (pixel, grid point) pair with the grid-point centre inside the pixel.

    Grid centres are lat0 + (i + ½)·dlat, lon0 + (j + ½)·dlon. Returns int arrays
    (pix, i, j) of equal length; points outside the grid are not returned.
    """
    i_lo = np.ceil((lat_b.min(1) - lat0) / dlat - 0.5).astype(np.int64)
    i_hi = np.floor((lat_b.max(1) - lat0) / dlat - 0.5).astype(np.int64)
    j_lo = np.ceil((lon_b.min(1) - lon0) / dlon - 0.5).astype(np.int64)
    j_hi = np.floor((lon_b.max(1) - lon0) / dlon - 0.5).astype(np.int64)
    ni = int(max((i_hi - i_lo + 1).max(), 0))
    nj = int(max((j_hi - j_lo + 1).max(), 0))
    out_p, out_i, out_j = [], [], []
    for di in range(ni):
        for dj in range(nj):
            i, j = i_lo + di, j_lo + dj
            idx = np.nonzero((i <= i_hi) & (j <= j_hi) & (i >= 0) & (i < nlat)
                             & (j >= 0) & (j < nlon))[0]
            if idx.size == 0:
                continue
            inside = points_in_quads(lon0 + (j[idx] + 0.5) * dlon, lat0 + (i[idx] + 0.5) * dlat,
                                     lon_b[idx], lat_b[idx])
            out_p.append(idx[inside])
            out_i.append(i[idx][inside])
            out_j.append(j[idx][inside])
    if not out_p:
        return (np.zeros(0, np.int64),) * 3
    return np.concatenate(out_p), np.concatenate(out_i), np.concatenate(out_j)


def _shift(a: np.ndarray, di: int, dj: int) -> np.ndarray:
    """out[i, j] = a[i + di, j + dj], zero where the source is outside the grid."""
    n, m = a.shape
    out = np.zeros_like(a)
    out[max(-di, 0):n - max(di, 0), max(-dj, 0):m - max(dj, 0)] = \
        a[max(di, 0):n - max(-di, 0), max(dj, 0):m - max(-dj, 0)]
    return out


def oversample_adams(vals: np.ndarray, lat_b: np.ndarray, lon_b: np.ndarray,
                     lat_c: np.ndarray, lon0: float, dlat: float, dlon: float, nlon: int,
                     radius_km: float) -> tuple[np.ndarray, dict]:
    """Adams et al. (2019) two-step oversampling of pixel values onto a regular grid.

    Parameters
    ----------
    vals         : (n,) pixel values (any units; output has the same units)
    lat_b, lon_b : (n, 4) pixel corners (deg), one consistent cyclic order
    lat_c        : (nlat,) grid row centres (deg), spacing dlat
    lon0, dlat, dlon, nlon : west grid edge, cell size (deg), number of columns
    radius_km    : step-2 averaging radius (km). Distances use a local flat-Earth
                   approximation at the target row's latitude (< 0.1% error at 7 km).

    Returns
    -------
    field : (nlat, nlon) float64, NaN where no step-1 value lies within radius_km
    diag  : dict — pixel/point counts, pixels containing no grid centre, area percentiles
    """
    assert vals.ndim == 1 and lat_b.shape == (vals.size, 4) and lon_b.shape == (vals.size, 4), \
        f"Shape mismatch: vals {vals.shape}, lat_b {lat_b.shape}, lon_b {lon_b.shape}."
    assert np.all(np.isfinite(vals)), "vals must be finite (filter fills/QA before oversampling)."
    assert radius_km > 0, f"radius_km must be > 0; got {radius_km}."
    nlat = lat_c.size
    lat0 = float(lat_c[0] - 0.5 * dlat)
    area = pixel_signed_area_km2(lat_b, lon_b)
    assert np.all(area != 0), "Degenerate pixel footprint (zero area)."
    assert np.all(area > 0) or np.all(area < 0), \
        "Pixel corners are not in one consistent cyclic order (mixed area signs)."
    area = np.abs(area)
    w = 1.0 / area                                        # km⁻²

    pix, i, j = footprint_points(lat_b, lon_b, lat0, dlat, nlat, lon0, dlon, nlon)
    flat = i * nlon + j
    swv = np.bincount(flat, weights=w[pix] * vals[pix].astype(np.float64),
                      minlength=nlat * nlon).reshape(nlat, nlon)
    sw = np.bincount(flat, weights=w[pix], minlength=nlat * nlon).reshape(nlat, nlon)

    dlat_km = dlat * KM_PER_DEG_LAT
    dlon_km_row = dlon * KM_PER_DEG_LAT * np.cos(np.radians(lat_c))   # (nlat,)
    di_max = int(np.floor(radius_km / dlat_km))
    dj_max = int(np.floor(radius_km / dlon_km_row.min()))
    num = np.zeros_like(swv)
    den = np.zeros_like(sw)
    for di in range(-di_max, di_max + 1):
        for dj in range(-dj_max, dj_max + 1):
            within = (di * dlat_km) ** 2 + (dj * dlon_km_row) ** 2 <= radius_km ** 2
            if not within.any():
                continue
            num += _shift(swv, di, dj) * within[:, None]
            den += _shift(sw, di, dj) * within[:, None]
    field = np.full_like(num, np.nan)
    np.divide(num, den, out=field, where=den > 0)

    n_hit = np.bincount(pix, minlength=vals.size)
    diag = dict(n_pixels=int(vals.size), n_points=int(pix.size),
                n_pixels_no_point=int((n_hit == 0).sum()),
                points_per_pixel_median=float(np.median(n_hit)),
                area_km2_p05=float(np.percentile(area, 5)),
                area_km2_p50=float(np.percentile(area, 50)),
                area_km2_p95=float(np.percentile(area, 95)))
    return field, diag


def oversample_to_grid(vals: np.ndarray, lat_b: np.ndarray, lon_b: np.ndarray,
                       grid: GridSpec, radius_km: float) -> tuple[np.ndarray, dict]:
    """`oversample_adams` on a `GridSpec`. Returns ((nlat, nlon) field, diagnostics)."""
    return oversample_adams(vals, lat_b, lon_b, grid.lat, grid.lon0, grid.dlat,
                            grid.dlon, grid.nlon, radius_km)
