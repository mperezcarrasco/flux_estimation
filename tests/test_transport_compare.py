"""Tests for src.transport_compare (HYSPLIT model vs observed TROPOMI CO).

Unit tests use small synthetic grids with hand-computed answers. The regression
tests rebuild model_obs_comparison.csv from the cdump and the L2 files and require
exact agreement with the table nb06 wrote.

Run: /Users/maperezc/Downloads/env/bin/python3 -m pytest tests/ -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import transport_compare as tc  # noqa: E402

CFG = tc.load_comparison_config(ROOT / "config" / "hysplit_comparison.yaml")
CMP_CSV = ROOT / "outputs" / "hysplit_co" / "0118" / "model_obs_comparison.csv"
CDUMP = ROOT / "outputs" / "hysplit_co" / "run_0118" / "cdump"
MOL_M2_TO_MOLEC_CM2 = 6.02214076e23 / 1.0e4


def test_config_values():
    """Settings used for the manuscript comparison (nb06)."""
    assert (CFG.lat_min, CFG.lat_max, CFG.lon_min, CFG.lon_max, CFG.res_deg) == (-45.0, -17.0, -90.0, -66.0, 0.25)
    assert (CFG.qa_uint8, CFG.bg_percentile, CFG.threshold_mol_m2, CFG.near_source_km) == (50, 10.0, 1.0e-3, 50.0)
    assert [o["offset_d"] for o in CFG.overpasses] == [1, 2, 3, 4]
    assert np.isclose(CFG.threshold_mol_m2 * MOL_M2_TO_MOLEC_CM2, 6.02e16, rtol=1e-3)


def test_index_regrid_containing_cell_and_outside_zero():
    """Model cells 1°, centres 0.5..3.5; analysis centres inside take the containing cell, outside 0."""
    mlat, mlon = np.meshgrid(np.arange(4) + 0.5, np.arange(4) + 0.5, indexing="ij")
    field = np.arange(16, dtype=float).reshape(4, 4)
    grid = tc.AnalysisGrid(lat_edges=None, lon_edges=None,
                           lat_ctrs=np.array([0.2, 1.9, 3.99, 4.6]), lon_ctrs=np.array([-0.3, 0.7, 2.5]))
    out = tc.IndexRegrid(mlat, mlon, grid)(field)
    want = np.array([[0, 0, 2], [0, 4, 6], [0, 12, 14], [0, 0, 0]], float)   # lat 4.6 and lon -0.3 outside
    np.testing.assert_array_equal(out, want)


def test_footprint_comparison_hand_values():
    """3×3 field; centre cell is near source and excluded; one footprint cell lacks obs."""
    mod = np.array([[2e-3, 5e-4, 4e-3], [3e-3, 9e-3, 1e-3], [6e-3, 2e-3, 0.0]])
    enh = np.array([[4e-3, 1e-2, np.nan], [9e-3, 5e-2, 3e-3], [6e-3, 1e-3, 2e-3]])
    dist = np.array([[100., 200., 300.], [400., 10., 600.], [700., 800., 900.]])
    r = tc.footprint_comparison(mod, enh, dist, threshold_mol_m2=1e-3, near_source_km=50.0)
    # footprint: mod > 1e-3 and dist > 50 -> (0,0) 2e-3, (0,2) 4e-3, (1,0) 3e-3, (2,0) 6e-3, (2,1) 2e-3
    assert r["n_region_cells"] == 5 and r["n_obs_valid_in_region"] == 4       # (0,2) has no obs
    assert r["mod_median_molm2"] == round(float(np.median([2e-3, 3e-3, 6e-3, 2e-3])), 5)
    assert r["obs_median_molm2"] == round(float(np.median([4e-3, 9e-3, 6e-3, 1e-3])), 5)
    assert r["mod_footprint_median_molm2"] == 3e-3
    assert r["mod_reach_p99_km"] == round(float(np.percentile([100., 300., 400., 700., 800.], 99)), 0)
    # observed footprint: enh > 1e-3 and far -> 100, 200, 400, 600, 700, 900 km
    assert r["obs_reach_p95_km"] == round(float(np.percentile([100., 200., 400., 600., 700., 900.], 95)), 0)


def test_footprint_comparison_empty_raises():
    with pytest.raises(AssertionError, match="empty modeled footprint"):
        tc.footprint_comparison(np.zeros((2, 2)), np.ones((2, 2)), np.full((2, 2), 100.0),
                                threshold_mol_m2=1e-3, near_source_km=50.0)


# ── Regression against nb06's table (real data) ──────────────────────────────
@pytest.fixture(scope="module")
def model():
    from src import hysplit_helpers as H
    ds = H.read_cdump(CDUMP)
    col = tc.model_column_mol_m2(ds, CFG)
    grid = tc.analysis_grid(CFG)
    regrid = tc.IndexRegrid(ds.latitude.values, ds.longitude.values, grid)
    return col, pd.to_datetime(ds.time.values), regrid, grid


def test_comparison_table_reproduces_nb06(model):
    col, times, regrid, grid = model
    got = tc.comparison_table(col, times, regrid, ROOT / "data", CFG, grid)
    want = pd.read_csv(CMP_CSV)
    pd.testing.assert_frame_equal(got.reset_index(drop=True), want, check_dtype=False, rtol=0, atol=1e-12)


def test_series_matches_table_and_paper(model):
    """Footprint-median series equals the table at the overpasses; peak/+48 h are the paper's values."""
    col, times, regrid, grid = model
    dist = tc.source_distance_km(CFG, grid)
    s = tc.footprint_median_series(col, regrid, dist, threshold_mol_m2=CFG.threshold_mol_m2,
                                   near_source_km=CFG.near_source_km)
    table = pd.read_csv(CMP_CSV)
    for _, row in table.iterrows():
        k = int(np.argmin(np.abs(times - pd.Timestamp(row["model_time"]))))
        assert round(s[k], 5) == row["mod_footprint_median_molm2"]
    k48 = int(np.argmin(np.abs((times - pd.Timestamp("2026-01-18T10:00")).total_seconds() / 3600 - 48.0)))
    assert round(np.nanmax(s) * MOL_M2_TO_MOLEC_CM2 / 1e17, 1) == 9.3   # manuscript: ~9.3e17 molec cm-2
    assert round(s[k48] * MOL_M2_TO_MOLEC_CM2 / 1e17, 1) == 1.4         # manuscript: ~1.4e17 by +48 h
