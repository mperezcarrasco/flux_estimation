"""Known-answer tests for src.oversampling and the explicit CSF transect spacing.

Run: /Users/maperezc/Downloads/env/bin/python3 -m pytest tests/ -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.csf_utils import build_dynamic_transects  # noqa: E402
from src.oversampling import (  # noqa: E402
    KM_PER_DEG_LAT, footprint_points, oversample_adams, pixel_signed_area_km2,
    regular_grid_km,
)

FIRE_LAT, FIRE_LON = -37.5484, -72.5545


def _cells(i, j):
    return sorted(zip(i.tolist(), j.tolist()))


def test_area_of_one_degree_pixel_and_sign():
    """1°×1° pixel at lat [0, 1]: area = 111.32² · cos(0.5°) km²; reversed order flips sign."""
    expect = KM_PER_DEG_LAT ** 2 * np.cos(np.radians(0.5))
    ccw = pixel_signed_area_km2(np.array([[0., 0., 1., 1.]]), np.array([[0., 1., 1., 0.]]))[0]
    cw = pixel_signed_area_km2(np.array([[1., 1., 0., 0.]]), np.array([[0., 1., 1., 0.]]))[0]
    assert np.isclose(abs(ccw), expect, rtol=1e-12)
    assert np.isclose(cw, -ccw, rtol=1e-12)


def test_footprint_axis_aligned_pixel():
    """Pixel lat [2.2, 4.1] × lon [3.3, 6.2] on a unit grid holds exactly 6 centres."""
    _, i, j = footprint_points(np.array([[2.2, 2.2, 4.1, 4.1]]), np.array([[3.3, 6.2, 6.2, 3.3]]),
                               0.0, 1.0, 10, 0.0, 1.0, 10)
    assert _cells(i, j) == [(2, 3), (2, 4), (2, 5), (3, 3), (3, 4), (3, 5)]


@pytest.mark.parametrize("order", [slice(None), slice(None, None, -1)], ids=["ccw", "cw"])
def test_footprint_rotated_pixel_both_orders(order):
    """Diamond on (5.5, 5.5), half-diagonal 1.2: centre + 4 edge neighbours, either corner order."""
    lat_b = np.array([[4.3, 5.5, 6.7, 5.5]])[:, order]
    lon_b = np.array([[5.5, 6.7, 5.5, 4.3]])[:, order]
    _, i, j = footprint_points(lat_b, lon_b, 0.0, 1.0, 10, 0.0, 1.0, 10)
    assert _cells(i, j) == [(4, 5), (5, 4), (5, 5), (5, 6), (6, 5)]


def test_inverse_area_weights():
    """Pixels A = [0,2]² (10) and B = [0,1]² (20); radius < 1 cell, so only the point itself.

    Cell (0,0) is in both: (10/aA + 20/aB)/(1/aA + 1/aB); cells (0,1),(1,0),(1,1) only in A.
    """
    a_a = (2 * KM_PER_DEG_LAT) * (2 * KM_PER_DEG_LAT * np.cos(np.radians(1.0)))
    a_b = KM_PER_DEG_LAT ** 2 * np.cos(np.radians(0.5))
    f, _ = oversample_adams(np.array([10.0, 20.0]),
                            np.array([[0., 0., 2., 2.], [0., 0., 1., 1.]]),
                            np.array([[0., 2., 2., 0.], [0., 1., 1., 0.]]),
                            np.arange(4) + 0.5, 0.0, 1.0, 1.0, 4, 0.5)
    assert np.isclose(f[0, 0], (10 / a_a + 20 / a_b) / (1 / a_a + 1 / a_b), rtol=1e-12)
    assert np.allclose([f[0, 1], f[1, 0], f[1, 1]], 10.0, rtol=1e-12)
    assert int(np.isfinite(f).sum()) == 4


@pytest.mark.parametrize("radius_km,want", [
    (7.0, [(a, b) for a in range(-1, 2) for b in range(-1, 2)]),
    (9.0, [(a, b) for a in range(-2, 3) for b in range(-2, 3) if not (abs(a) == 2 and abs(b) == 2)]),
])
def test_radius_neighbourhood_on_4km_grid(radius_km, want):
    """One tiny pixel in cell (5,5) of a 4 km grid at the fire latitude.

    7 km reaches the 3×3 block (0, 4, 5.7 km); 9 km adds the 8.0 / 8.9 km cells
    but not the (±2, ±2) corners (11.3 km).
    """
    # 11.5-cell span (whole-cell floor -> 11; an exact 11-cell span can round to 10.999)
    g = regular_grid_km(FIRE_LAT - 5.75 * 4 / KM_PER_DEG_LAT, FIRE_LAT + 5.75 * 4 / KM_PER_DEG_LAT,
                        FIRE_LON - 5.75 * 4 / (KM_PER_DEG_LAT * np.cos(np.radians(FIRE_LAT))),
                        FIRE_LON + 5.75 * 4 / (KM_PER_DEG_LAT * np.cos(np.radians(FIRE_LAT))),
                        cell_km=4.0, ref_lat=FIRE_LAT)
    assert (g.nlat, g.nlon) == (11, 11)
    lat_b = g.lat[5] + 0.1 * g.dlat * np.array([[-1., -1., 1., 1.]])
    lon_b = g.lon[5] + 0.1 * g.dlon * np.array([[-1., 1., 1., -1.]])
    f, _ = oversample_adams(np.array([1.0]), lat_b, lon_b, g.lat, g.lon0, g.dlat, g.dlon,
                            g.nlon, radius_km)
    fi, fj = np.nonzero(np.isfinite(f))
    assert _cells(fi - 5, fj - 5) == sorted(want)
    assert np.allclose(f[np.isfinite(f)], 1.0, rtol=1e-12)


def test_constant_field_across_latitudes():
    """A constant field stays constant on a grid spanning 66°S–6°S.

    The E–W cell shrinks to ~2 km at 66°S, so the 7 km neighbourhood is wider in
    columns there than at 35°S: numerator and denominator must use the same cells.
    """
    g = regular_grid_km(-66.83, -6.83, -76.0, -70.0, cell_km=4.0, ref_lat=FIRE_LAT)
    rng = np.random.default_rng(0)
    lat_c = rng.uniform(-65.0, -8.0, 200)
    lon_c = rng.uniform(-75.0, -71.0, 200)
    hl, hw = 2.75 / KM_PER_DEG_LAT, 3.5 / (KM_PER_DEG_LAT * np.cos(np.radians(lat_c)))
    lat_b = lat_c[:, None] + hl * np.array([-1., -1., 1., 1.])
    lon_b = lon_c[:, None] + hw[:, None] * np.array([-1., 1., 1., -1.])
    f, _ = oversample_adams(np.full(200, 3.7), lat_b, lon_b, g.lat, g.lon0, g.dlat, g.dlon,
                            g.nlon, 7.0)
    assert np.isfinite(f).any()
    assert np.allclose(f[np.isfinite(f)], 3.7, rtol=1e-12)


def test_transect_sample_spacing_is_explicit():
    """100 km half-width at 4 km spacing -> 51 samples exactly 4 km apart."""
    lat = np.linspace(-40.0, -30.0, 50)
    lon = np.linspace(-78.0, -68.0, 50)
    plume = np.zeros((50, 50), bool)
    plume[10:40, 20:25] = True
    tr = build_dynamic_transects(lat, lon, plume, np.full(3, np.pi / 2), np.zeros(3),
                                 np.full(3, 5.0), n_transects=3, transect_half_width_km=100.0,
                                 sample_spacing_m=4000.0)
    assert all(len(t.positions_lat) == 51 for t in tr)
    assert all(t.ds_meters == 4000.0 for t in tr)
    with pytest.raises(TypeError):
        build_dynamic_transects(lat, lon, plume, np.full(3, np.pi / 2), np.zeros(3),
                                np.full(3, 5.0), n_transects=3, transect_half_width_km=100.0)
