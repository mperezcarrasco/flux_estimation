"""Manuscript numbers vs the pipeline outputs that produce them.

Each expected value is the number printed in the manuscript (Overleaf agusample.tex,
commit d17f265, 2026-10-10), rounded as printed; the test reads the output file
written by the notebook named in the comment. A failure means the outputs and the
paper disagree — re-run the notebook or correct the manuscript; do not edit the
expected value without checking which one is wrong.

GFAS-based numbers are not tested: the GFAS input files are 24-h averages, not the
hourly field the paper describes (open item, Oct 2026).

Run: /Users/maperezc/Downloads/env/bin/python3 -m pytest tests/ -q
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RETR = ROOT / "outputs" / "retrievals" / "0118"


def _series(path: Path) -> pd.Series:
    return pd.read_csv(path, index_col=0)["value"]


def test_co_point_estimate():                        # nb02; manuscript L273 (see note)
    r = _series(RETR / "co_retrieval.csv")
    assert round(float(r["co_flux_variant_kg_s"])) == 1363
    assert int(r["n_transects_valid_variant"]) == 6   # Fig co_pipeline caption: six valid transects
    assert float(r["grid_cell_km"]) == 4.0            # L188: 4 km x 4 km grid
    assert round(float(r["injection_pressure_hpa"])) == 882          # L182
    assert round(float(r["wind_speed_alh_ms"]), 2) == 8.72           # L182


def test_co_monte_carlo():                           # nb07; manuscript L129, L273, L301
    d = _series(RETR / "co_distribution.csv")
    assert int(d["n"]) == 287
    assert round(float(d["median"])) == 1273
    assert round(float(d["p2_5"])) == 676
    assert round(float(d["p97_5"])) == 1843


def test_nox_point_estimate():                       # nb03; manuscript L275, Table emg_params
    r = _series(RETR / "nox_retrieval.csv")
    assert round(float(r["nox_flux_g_s"])) == 8415
    assert round(float(r["nox_tau_h"]), 2) == 0.75
    assert round(float(r["emg_r2"]), 3) == 0.846
    assert round(float(r["emg_mu_peak_km"]), 1) == 79.0


def test_nox_monte_carlo():                          # nb08; manuscript L129, L275, L283
    d = _series(RETR / "nox_distribution.csv")
    assert int(d["n"]) == 285
    assert round(float(d["median"])) == 7505
    # Manuscript prints 3,621; the output is 3,620.46 -> 3,620. Expected value is the
    # output (the manuscript rounding is the error); fix the paper, not this line.
    assert round(float(d["p2_5"])) == 3620
    assert round(float(d["p97_5"])) == 9655


def test_co_to_nox_mass_ratio():                     # manuscript L305: ~260 (NOx on the NO basis)
    co = float(_series(RETR / "co_distribution.csv")["median"])
    nox_no = float(_series(RETR / "nox_distribution.csv")["median"]) * 30.006 / 46.005
    assert round(co / (nox_no / 1000.0), -1) == 260


def test_hysplit_comparison():                       # nb06; manuscript L314, L449, L451
    t = pd.read_csv(ROOT / "outputs" / "hysplit_co" / "0118" / "model_obs_comparison.csv")
    assert t["offset_d"].tolist() == [1, 2, 3, 4]
    assert round(float(t["obs_over_mod"].min()), 1) == 2.3 and round(float(t["obs_over_mod"].max()), 1) == 4.0
    assert round(float(t.loc[0, "mod_reach_p99_km"]), -1) == 1080          # +1 d footprint reach
    assert round(float(t.loc[3, "mod_reach_p99_km"]), -1) == 2220          # +4 d footprint reach
    cov = t["n_obs_valid_in_region"] / t["n_region_cells"]
    assert round(100 * float(cov.min())) == 53 and round(100 * float(cov.max())) == 81
    assert round(t.loc[0, "obs_median_molm2"] * 6.02214076e19 / 1e17, 1) == 5.4   # L449: 5.4e17 vs 2.3e17
    assert round(t.loc[0, "mod_median_molm2"] * 6.02214076e19 / 1e17, 1) == 2.3
