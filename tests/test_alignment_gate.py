"""Tests for the MC plume-alignment gate (src.error_estimation.plume_axis_pca / angular_diff_deg).

Unit tests use synthetic plumes with known axes. The regression tests recompute
θ_plume for every converged member of the stored nb07/nb08 ensembles and require
exact agreement with the values the ensembles logged.

Run: /Users/maperezc/Downloads/env/bin/python3 -m pytest tests/ -q
"""
import glob
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import error_estimation as ee  # noqa: E402

GATE = ee.load_alignment_gate_config(ROOT / "config" / "mc_alignment_gate.yaml")
GATE_KW = dict(radius_km=GATE["pca_radius_km"], far_field_min_km=GATE["far_field_min_km"],
               min_pixels=GATE["min_pca_pixels"], min_far_pixels=GATE["min_far_field_pixels"])


def test_gate_config_values():
    """The manuscript gate: 450 km PCA radius, 150 km far field, 15° threshold."""
    assert GATE == {"pca_radius_km": 450.0, "far_field_min_km": 150.0,
                    "max_wind_plume_angle_deg": 15.0, "min_pca_pixels": 20,
                    "min_far_field_pixels": 3}


@pytest.mark.parametrize("a,b,want", [(10, 350, 20.0), (350, 10, 20.0), (0, 180, 180.0),
                                      (90, 90, 0.0), (-170, 170, 20.0)])
def test_angular_diff_deg(a, b, want):
    assert np.isclose(ee.angular_diff_deg(np.radians(a), np.radians(b)), want, atol=1e-12)


def _line_plume(theta_deg: float, lat0: float, lon0: float, n: int = 200, length_km: float = 400.0):
    """Pixels on a straight line from the source along θ (CCW from east), unit AI."""
    s = np.linspace(5.0, length_km, n)
    th = np.radians(theta_deg)
    lat = lat0 + s * np.sin(th) / ee.KM_PER_DEG
    lon = lon0 + s * np.cos(th) / (ee.KM_PER_DEG * np.cos(np.radians(lat0)))
    return lat, lon, np.ones(n)


@pytest.mark.parametrize("theta", [123.3, 60.0, -30.0, 179.0])
def test_pca_recovers_line_axis_and_sign(theta):
    """A straight plume along θ gives θ (not θ + 180°): sign points to the far field."""
    lat, lon, ai = _line_plume(theta, -37.5, -72.5)
    got = np.degrees(ee.plume_axis_pca(-37.5, -72.5, lat, lon, ai, **GATE_KW))
    assert ee.angular_diff_deg(np.radians(got), np.radians(theta)) < 1e-6


def test_pca_too_few_pixels_raises():
    lat, lon, ai = _line_plume(60.0, -37.5, -72.5, n=10)
    with pytest.raises(ValueError, match="^PCA: only 10 plume pixels"):
        ee.plume_axis_pca(-37.5, -72.5, lat, lon, ai, **GATE_KW)


def test_pca_no_far_field_raises():
    lat, lon, ai = _line_plume(60.0, -37.5, -72.5, n=50, length_km=140.0)   # all < 150 km
    with pytest.raises(ValueError, match="^PCA: only 0 far-field pixels"):
        ee.plume_axis_pca(-37.5, -72.5, lat, lon, ai, **GATE_KW)


def test_pca_far_field_edge_exactly_min_pixels():
    """Exactly 3 pixels beyond 150 km (150.5, 152, 154 km) is enough; 2 is not."""
    s_near = np.linspace(5.0, 140.0, 40)
    for s_far, ok in [(np.array([150.5, 152.0, 154.0]), True), (np.array([150.5, 152.0]), False)]:
        s = np.concatenate([s_near, s_far])
        lat = -37.5 + s * np.sin(np.radians(60.0)) / ee.KM_PER_DEG
        lon = -72.5 + s * np.cos(np.radians(60.0)) / (ee.KM_PER_DEG * np.cos(np.radians(-37.5)))
        if ok:
            got = np.degrees(ee.plume_axis_pca(-37.5, -72.5, lat, lon, np.ones(s.size), **GATE_KW))
            assert abs(got - 60.0) < 1e-6
        else:
            with pytest.raises(ValueError, match="^PCA: only 2 far-field pixels"):
                ee.plume_axis_pca(-37.5, -72.5, lat, lon, np.ones(s.size), **GATE_KW)


# ── Regression against the stored ensembles (real data) ──────────────────────
@pytest.fixture(scope="module")
def plume_pixels():
    """AER_AI plume pixels exactly as nb07/nb08 build them (QA 50, boundary clip, AI > 2.0)."""
    import shapely
    from src import plume_helpers as ph
    from src.data_helpers import load_aer_ai
    roi = dict(lat_min=-66.83, lat_max=-6.83, lon_min=-103.05, lon_max=-43.05, qa_threshold=50)
    ch = defaultdict(list)
    for p in sorted(glob.glob(str(ROOT / "data" / "0118" / "S5P_OFFL_L2__AER_AI_2026*"))):
        for k, v in load_aer_ai(Path(p), **roi).items():
            ch[k].append(v)
    ai = {k: np.concatenate(v) for k, v in ch.items()}
    pb = ph.load_plume_boundary(ROOT / "data" / "0118" / "plume_boundary.geojson")
    inb = shapely.contains_xy(pb.boundary, ai["lon"], ai["lat"])
    ai["ai"] = np.where(inb, ai["ai"], np.minimum(ai["ai"], pb.threshold - 1e-3))
    keep = ai["ai"] > 2.0
    return ai["lat"][keep], ai["lon"][keep], ai["ai"][keep]


@pytest.mark.parametrize("members_csv", ["co_error_members.csv", "nox_error_members.csv"])
def test_gate_reproduces_stored_members(plume_pixels, members_csv):
    """θ_plume and Δθ recomputed with src equal the values logged by nb07 / nb08."""
    df = pd.read_csv(ROOT / "outputs" / "error_mc" / "0118" / members_csv)
    lat, lon, ai = plume_pixels
    theta = np.array([np.degrees(ee.plume_axis_pca(r.fire_lat, r.fire_lon, lat, lon, ai, **GATE_KW))
                      for r in df.itertuples()])
    np.testing.assert_allclose(theta, df["theta_plume_deg"].to_numpy(), rtol=0, atol=1e-9)
    dth = np.array([ee.angular_diff_deg(np.radians(a), np.radians(b))
                    for a, b in zip(df["theta_wind_deg"], theta)])
    np.testing.assert_allclose(dth, df["delta_theta_deg"].to_numpy(), rtol=0, atol=1e-9)
    assert (dth <= GATE["max_wind_plume_angle_deg"]).all()
