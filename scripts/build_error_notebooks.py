#!/usr/bin/env python3
"""Generate the two Monte-Carlo retrieval-error notebooks (CO + NOx).

Run with the project env:
    /Users/maperezc/Downloads/env/bin/python3 scripts/build_error_notebooks.py

Emits:
    notebooks/07_co_emission_error_mc.ipynb
    notebooks/08_nox_emission_error_mc.ipynb
"""
from __future__ import annotations
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"

# --- shared header for both notebooks -------------------------------------
COMMON_PARAMS = '''\
# ── Monte-Carlo error-estimation parameters (sampled ranges agreed in brainstorm) ─
N_SAMPLES: int = 500
SEED: int      = 42

FIRE_DATE  = "2026-01-18"
TARGET_DAY = "".join(FIRE_DATE.split("-")[1:])  # "0118"

# Analyst-choice sample ranges
AI_THRESHOLDS    = [1.5, 1.75, 2.0, 2.25, 2.5]   # discrete; also selects the pressure KDE
QA_LEVELS_UINT8  = [50, 70]                        # qa_value cut (loader uses uint8 0-100)

# Injection-pressure KDE (matched AER_LH pixels)
PRESS_RADIUS_KM    = 300.0
PRESS_MIN_PIXELS   = 15

# Centroid KDE (FRP-weighted MODIS), recentered on the retrieval centroid
CENTROID_LOCAL_RADIUS_KM = 200.0
CENTROID_MIN_CONF        = 75
CENTROID_SAT             = "Aqua"
CENTROID_DAYNIGHT        = "D"

# Fail-loud control: abort if too many members fail.
MAX_FAIL_FRACTION = 0.25
'''

LOAD_PRIORS = '''\
import sys, glob, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")  # KDE/curve_fit chatter; failures are still raised explicitly

ROOT = Path(".").resolve()
while not ((ROOT / "src").exists() and (ROOT / "data").exists()):
    if ROOT == ROOT.parent:
        raise RuntimeError("Could not locate project root (need src/ and data/).")
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from src import error_estimation as ee
from src.data_helpers import (load_co_l2, load_no2_l2, load_aer_ai,
                              load_aer_lh, get_modis_hotspot)

DATA = ROOT / "data"
ERA5_PL = DATA / "ERA5_winds_pressure_levels.nc"
MODIS_CSV = DATA / "fire_nrt_M-C61_723973.csv"
HOTSPOT_CSV = DATA / "daily_dominant_hotspot_summary.csv"
QA_BASE = 50

# Wide loader domain (matches the science notebooks)
LAT_MIN, LAT_MAX = -66.83, -6.83
LON_MIN, LON_MAX = -103.05, -43.05

modis = get_modis_hotspot(HOTSPOT_CSV, FIRE_DATE,
                          lat_min=LAT_MIN, lat_max=LAT_MAX, lon_min=LON_MIN, lon_max=LON_MAX)
FIRE_LAT = float(modis["dominant_lat"].values[0])
FIRE_LON = float(modis["dominant_lon"].values[0])
print(f"Retrieval centroid: ({FIRE_LAT:.4f}, {FIRE_LON:.4f})  FRP={modis['dominant_frp_mw'].values[0]} MW")

ai_base  = load_aer_ai(glob.glob(str(DATA/TARGET_DAY/"S5P_OFFL_L2__AER_AI_2026*"))[0],
                       lat_min=LAT_MIN, lat_max=LAT_MAX, lon_min=LON_MIN, lon_max=LON_MAX, qa_threshold=QA_BASE)
alh_base = load_aer_lh(glob.glob(str(DATA/TARGET_DAY/"S5P_OFFL_L2__AER_LH_2026*"))[0],
                       lat_min=LAT_MIN, lat_max=LAT_MAX, lon_min=LON_MIN, lon_max=LON_MAX, qa_threshold=QA_BASE)

# ── Injection-pressure KDE per AI threshold (so AI threshold is a live parameter) ─
pressure_kdes = {}
for thr in AI_THRESHOLDS:
    try:
        pk = ee.build_pressure_kde(ai_base["ai"], ai_base["lat"], ai_base["lon"],
                                   alh_base["alh_pres"], alh_base["lat"], alh_base["lon"],
                                   FIRE_LAT, FIRE_LON, ai_threshold=thr,
                                   radius_km=PRESS_RADIUS_KM, min_pixels=PRESS_MIN_PIXELS)
        pressure_kdes[thr] = pk
    except ValueError as e:
        print(f"  AI thr {thr}: pressure KDE skipped ({e})")
AVAILABLE_AI = sorted(pressure_kdes)
assert AVAILABLE_AI, "No AI threshold yielded a valid pressure KDE."
print(f"AI thresholds with a pressure KDE: {AVAILABLE_AI}")
for thr in AVAILABLE_AI:
    pk = pressure_kdes[thr]
    print(f"  thr {thr}: n={pk['n']:3d}  median={pk['median_hpa']:.0f} hPa  IQR={pk['iqr_hpa']:.0f}")

# ── Centroid KDE (recentered on the retrieval centroid, keep KDE spread) ──────
ck = ee.build_centroid_kde(MODIS_CSV, FIRE_DATE, FIRE_LAT, FIRE_LON,
                           local_radius_km=CENTROID_LOCAL_RADIUS_KM,
                           min_confidence=CENTROID_MIN_CONF,
                           satellite=CENTROID_SAT, daynight=CENTROID_DAYNIGHT)
_cs = ck["kde"].resample(5000, seed=SEED)
CENTROID_DLAT = FIRE_LAT - float(_cs[0].mean())
CENTROID_DLON = FIRE_LON - float(_cs[1].mean())
print(f"Centroid KDE: n={ck['n']}, recenter shift = ({CENTROID_DLAT:+.3f}, {CENTROID_DLON:+.3f})")

# ── ERA5 pressure-level wind slice at overpass ───────────────────────────────
OVERPASS = pd.Timestamp(FIRE_DATE) + pd.Timedelta(hours=13.0 - FIRE_LON/15.0)
era5 = ee.load_era5_pl_slice(ERA5_PL, OVERPASS)
print(f"ERA5 slice: {era5.selected_time}  ({len(era5.p_sorted)} levels)")
'''

SAMPLE_COMMON = '''\
# ── Draw the parameter samples (one row per member) ──────────────────────────
rng = np.random.default_rng(SEED)

# AI threshold (discrete, among those with a valid pressure KDE)
samp_ai = rng.choice(AVAILABLE_AI, size=N_SAMPLES)

# Injection pressure: sample from the AI-threshold-matched KDE, clip to ERA5 range
p_lo, p_hi = float(era5.p_sorted.min()), float(era5.p_sorted.max())
samp_pressure = np.empty(N_SAMPLES)
for thr in AVAILABLE_AI:
    m = samp_ai == thr
    if m.any():
        draw = pressure_kdes[thr]["kde"].resample(int(m.sum()), seed=int(rng.integers(1e9))).flatten()
        samp_pressure[m] = np.clip(draw, p_lo, p_hi)

# Centroid: recentered KDE draws
_cdraw = ck["kde"].resample(N_SAMPLES, seed=SEED + 1)
samp_lat = _cdraw[0] + CENTROID_DLAT
samp_lon = _cdraw[1] + CENTROID_DLON

# QA level
samp_qa = rng.choice(QA_LEVELS_UINT8, size=N_SAMPLES)
'''

# ===========================================================================
# CO notebook
# ===========================================================================
def build_co():
    nb = nbf.v4.new_notebook()
    c = []
    md = lambda s: c.append(nbf.v4.new_markdown_cell(s))
    co = lambda s: c.append(nbf.v4.new_code_cell(s))

    md("# 07 — CO (CSF) retrieval error via Monte-Carlo parameter sampling\n\n"
       "Propagates injection-pressure, fire-centroid, and analyst-choice uncertainty through the "
       "Cross-Sectional Flux CO retrieval (the AER_LH wind variant — the manuscript headline CO number).\n\n"
       "**Reported as a single combined distribution** (median + 95% CI). Per-member inputs are logged "
       "for the tornado / variance decomposition. See `src/error_estimation.py` for the design notes.")
    co(COMMON_PARAMS + '''
# CO-specific ranges
BG_START_RANGE   = (-350.0, -250.0)   # far-upwind window start (km)
BG_WIDTH_RANGE   = (80.0, 120.0)      # window width (km)
THW_RANGE        = (75.0, 150.0)      # transect half-width (km)
NTRANSECT_CHOICES = list(range(8, 15))
GRID_CHOICES     = [0.20, 0.25, 0.30]
NORTH_CLIP_LAT   = -26.83             # clip AER_AI artefacts north of this latitude
MIN_BG_PIXELS    = 10
OUT_DIR = None  # set after ROOT is known''')
    co(LOAD_PRIORS)
    co('''\
# ── Load CO L2 (all orbits) and AER_AI at both QA levels; precompute grids ────
from collections import defaultdict

def _load_co_concat(qa):
    chunks, scal = defaultdict(list), {}
    for p in glob.glob(str(DATA/TARGET_DAY/"S5P_OFFL_L2__CO_____*.nc")):
        d = load_co_l2(Path(p), lat_min=LAT_MIN, lat_max=LAT_MAX,
                       lon_min=LON_MIN, lon_max=LON_MAX, qa_threshold=qa)
        if d is None:
            continue
        for k, v in d.items():
            (chunks[k].append(v) if isinstance(v, np.ndarray) else scal.__setitem__(k, v))
    out = {k: np.concatenate(v) for k, v in chunks.items()}; out.update(scal)
    return out

OUT_DIR = ROOT / "outputs" / "error_mc" / TARGET_DAY
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Precompute (qa, grid_spacing) -> (lat, lon, co_g, ai_g). Grid is centroid-independent.
grids = {}
for qa in QA_LEVELS_UINT8:
    co_flat = _load_co_concat(qa)
    ai_flat = load_aer_ai(glob.glob(str(DATA/TARGET_DAY/"S5P_OFFL_L2__AER_AI_2026*"))[0],
                          lat_min=LAT_MIN, lat_max=LAT_MAX, lon_min=LON_MIN, lon_max=LON_MAX, qa_threshold=qa)
    for g in GRID_CHOICES:
        lat, lon, co_g, ai_g = ee.bin_co_ai_grid(
            co_flat["co"], co_flat["lat"], co_flat["lon"],
            ai_flat["ai"], ai_flat["lat"], ai_flat["lon"],
            LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, g)
        grids[(qa, g)] = (lat, lon, co_g, ai_g)
    print(f"qa={qa}: grids built for spacings {GRID_CHOICES}")
print(f"Total precomputed grids: {len(grids)}")''')
    co(SAMPLE_COMMON + '''
# CO analyst-choice draws
samp_grid   = rng.choice(GRID_CHOICES, size=N_SAMPLES)
samp_bgstart = rng.uniform(*BG_START_RANGE, size=N_SAMPLES)
samp_bgwidth = rng.uniform(*BG_WIDTH_RANGE, size=N_SAMPLES)
samp_thw    = rng.uniform(*THW_RANGE, size=N_SAMPLES)
samp_nt     = rng.choice(NTRANSECT_CHOICES, size=N_SAMPLES)
print(f"Drew {N_SAMPLES} parameter sets.")''')
    co('''\
# ── Monte-Carlo loop ─────────────────────────────────────────────────────────
records, fails = [], []
for i in range(N_SAMPLES):
    lat, lon, co_g, ai_g = grids[(int(samp_qa[i]), float(samp_grid[i]))]
    u, v, spd, theta = ee.wind_from_pressure(era5, float(samp_lat[i]), float(samp_lon[i]), float(samp_pressure[i]))
    bg_min = float(samp_bgstart[i]); bg_max = bg_min + float(samp_bgwidth[i])
    row = dict(member=i, pressure_hpa=float(samp_pressure[i]),
               fire_lat=float(samp_lat[i]), fire_lon=float(samp_lon[i]),
               ai_threshold=float(samp_ai[i]), qa=int(samp_qa[i]), grid_deg=float(samp_grid[i]),
               bg_x_min_km=bg_min, bg_width_km=float(samp_bgwidth[i]),
               transect_half_width_km=float(samp_thw[i]), n_transects=int(samp_nt[i]),
               wind_speed=spd)
    try:
        r = ee.co_flux_member(lat, lon, co_g, ai_g, u, v, theta,
                              ai_threshold=float(samp_ai[i]), north_clip_lat=NORTH_CLIP_LAT,
                              bg_x_min_km=bg_min, bg_x_max_km=bg_max,
                              n_transects=int(samp_nt[i]), transect_half_width_km=float(samp_thw[i]),
                              min_bg_pixels=MIN_BG_PIXELS)
        row.update(r); records.append(row)
    except ValueError as e:
        row["error"] = str(e); fails.append(row)

df = pd.DataFrame(records)
fail_frac = len(fails) / N_SAMPLES
print(f"Converged: {len(df)}/{N_SAMPLES}  (failures: {len(fails)}, {fail_frac:.1%})")
assert fail_frac <= MAX_FAIL_FRACTION, (
    f"Failure fraction {fail_frac:.1%} exceeds MAX_FAIL_FRACTION={MAX_FAIL_FRACTION:.0%}; "
    "inspect `fails` before trusting the distribution.")''')
    co('''\
# ── Combined distribution summary ────────────────────────────────────────────
summ = ee.summarize(df["flux_kgs"].to_numpy(), "CO flux", "kg/s")
print("CO flux (CSF, AER_LH wind) — combined Monte-Carlo distribution")
print(f"  median   : {summ['median']:.1f} kg/s")
print(f"  95% CI   : [{summ['p2_5']:.1f}, {summ['p97_5']:.1f}] kg/s")
print(f"  rel. err : ±{summ['rel_err_pct']:.1f}%   (n={summ['n']})")
print(f"  t/day    : {summ['median']*86.4:.0f}  [{summ['p2_5']*86.4:.0f}, {summ['p97_5']*86.4:.0f}]")
print(f"\\nReference (nb02 AER_LH variant): 945 kg/s")''')
    co('''\
# ── Tornado / variance decomposition ─────────────────────────────────────────
input_cols = ["pressure_hpa", "fire_lat", "fire_lon", "ai_threshold", "qa",
              "grid_deg", "bg_x_min_km", "bg_width_km", "transect_half_width_km", "n_transects"]
tor = ee.tornado_sensitivity(df[input_cols], df["flux_kgs"].to_numpy(), "CO flux")
print(tor.to_string(index=False))''')
    co('''\
# ── Figures: distribution + tornado ──────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
ax = axes[0]
ax.hist(df["flux_kgs"], bins=40, color="#2c7be5", alpha=0.8, edgecolor="white")
ax.axvline(summ["median"], color="k", lw=2, label=f"median {summ['median']:.0f}")
ax.axvline(summ["p2_5"], color="k", ls="--", lw=1)
ax.axvline(summ["p97_5"], color="k", ls="--", lw=1, label="95% CI")
ax.axvline(945, color="#e05c00", ls=":", lw=2, label="nb02 point (945)")
ax.set(xlabel="CO flux (kg/s)", ylabel="members", title="CO CSF flux — combined MC distribution")
ax.legend(fontsize=8)

ax = axes[1]
t = tor.iloc[::-1]
ax.barh(t["parameter"], t["std_beta"], color=np.where(t["std_beta"] >= 0, "#2c7be5", "#e05c00"))
ax.axvline(0, color="k", lw=0.6)
ax.set(xlabel="standardised regression coeff (β)", title="Sensitivity (tornado)")
fig.tight_layout()
fig.savefig(OUT_DIR / "co_error_mc.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved: {OUT_DIR/'co_error_mc.png'}")''')
    co('''\
# ── Persist the per-member log + summary ─────────────────────────────────────
import json
df.to_csv(OUT_DIR / "co_error_members.csv", index=False)
tor.to_csv(OUT_DIR / "co_error_tornado.csv", index=False)
with open(OUT_DIR / "co_error_summary.json", "w") as f:
    json.dump({"summary": summ, "n_samples": N_SAMPLES, "n_fail": len(fails),
               "reference_kgs": 945.0}, f, indent=2)
print("Wrote co_error_members.csv, co_error_tornado.csv, co_error_summary.json")''')
    nb.cells = c
    nbf.write(nb, NB_DIR / "07_co_emission_error_mc.ipynb")
    print("wrote 07_co_emission_error_mc.ipynb")


# ===========================================================================
# NOx notebook
# ===========================================================================
def build_nox():
    nb = nbf.v4.new_notebook()
    c = []
    md = lambda s: c.append(nbf.v4.new_markdown_cell(s))
    co = lambda s: c.append(nbf.v4.new_code_cell(s))

    md("# 08 — NOx (EMG) retrieval error via Monte-Carlo parameter sampling\n\n"
       "Propagates injection-pressure, fire-centroid, analyst-choice, and **pixel-bootstrap** "
       "uncertainty through the EMG NOx retrieval. Reports a single combined distribution for "
       "**E(NOx)** and **τ**.\n\n"
       "**Raw ensemble (no fit-quality gate)** per the agreed scope — but every fit diagnostic "
       "(`x0`, `r2`, `cv_E`, `peak_x_km`, `fit_stable`) is logged so degenerate fits can be filtered "
       "post-hoc. A subset of draws collapse `x0`→0 (τ→0, E→∞); the diagnostic cell below quantifies them.")
    co(COMMON_PARAMS + '''
# NOx-specific ranges
NOX_NO2_RATIO_GAMMA      = 1.32     # Jin et al. (2021), held fixed
NO2_MOLAR_MASS_G_PER_MOL = 46.005
DOWNWIND_MIN_KM   = -100.0
DOWNWIND_MAX_RANGE = (350.0, 550.0)
CROSSWIND_RANGE    = (80.0, 120.0)
GRID_DY_CHOICES    = [0.04, 0.05, 0.06]   # deg; converted to km below
BG_START_RANGE     = (-120.0, -80.0)
BG_END_RANGE       = (-30.0, -15.0)
EXTRACTION_RADIUS_RANGE = (400.0, 500.0)
BOOTSTRAP_ITERS    = 20
MIN_BG_PIXELS      = 10
DEG_TO_KM          = 111.32''')
    co(LOAD_PRIORS)
    co('''\
# ── Load NO2 L2 flat pixels at both QA levels ────────────────────────────────
no2_by_qa = {}
for qa in QA_LEVELS_UINT8:
    no2_by_qa[qa] = load_no2_l2(glob.glob(str(DATA/TARGET_DAY/"S5P_OFFL_L2__NO2____2026*"))[0],
                                lat_min=LAT_MIN, lat_max=LAT_MAX, lon_min=LON_MIN, lon_max=LON_MAX, qa_threshold=qa)
    print(f"qa={qa}: {len(no2_by_qa[qa]['no2']):,} NO2 pixels")
OUT_DIR = ROOT / "outputs" / "error_mc" / TARGET_DAY
OUT_DIR.mkdir(parents=True, exist_ok=True)''')
    co(SAMPLE_COMMON + '''
# NOx analyst-choice draws
samp_dwmax   = rng.uniform(*DOWNWIND_MAX_RANGE, size=N_SAMPLES)
samp_cross   = rng.uniform(*CROSSWIND_RANGE, size=N_SAMPLES)
samp_gridd   = rng.choice(GRID_DY_CHOICES, size=N_SAMPLES)
samp_bgstart = rng.uniform(*BG_START_RANGE, size=N_SAMPLES)
samp_bgend   = rng.uniform(*BG_END_RANGE, size=N_SAMPLES)
samp_radius  = rng.uniform(*EXTRACTION_RADIUS_RANGE, size=N_SAMPLES)
print(f"Drew {N_SAMPLES} parameter sets.")''')
    co('''\
# ── Monte-Carlo loop (pixel bootstrap inside each member) ────────────────────
records, fails = [], []
for i in range(N_SAMPLES):
    no2 = no2_by_qa[int(samp_qa[i])]
    u, v, spd, theta = ee.wind_from_pressure(era5, float(samp_lat[i]), float(samp_lon[i]), float(samp_pressure[i]))
    grid_dy_km = float(samp_gridd[i]) * DEG_TO_KM
    row = dict(member=i, pressure_hpa=float(samp_pressure[i]),
               fire_lat=float(samp_lat[i]), fire_lon=float(samp_lon[i]),
               ai_threshold=float(samp_ai[i]), qa=int(samp_qa[i]),
               grid_dy_deg=float(samp_gridd[i]), downwind_max_km=float(samp_dwmax[i]),
               crosswind_km=float(samp_cross[i]), extraction_radius_km=float(samp_radius[i]),
               bg_x_min_km=float(samp_bgstart[i]), bg_x_max_km=float(samp_bgend[i]),
               wind_speed=spd)
    try:
        r = ee.nox_flux_member(no2["lat"], no2["lon"], no2["no2"],
                               float(samp_lat[i]), float(samp_lon[i]), u, v, spd,
                               extraction_radius_km=float(samp_radius[i]),
                               downwind_min_km=DOWNWIND_MIN_KM, downwind_max_km=float(samp_dwmax[i]),
                               crosswind_km=float(samp_cross[i]), grid_dy_km=grid_dy_km,
                               bg_x_min_km=float(samp_bgstart[i]), bg_x_max_km=float(samp_bgend[i]),
                               nox_no2_ratio_gamma=NOX_NO2_RATIO_GAMMA,
                               no2_molar_mass_g_per_mol=NO2_MOLAR_MASS_G_PER_MOL,
                               rng=rng, bootstrap_iters=BOOTSTRAP_ITERS, min_bg_pixels=MIN_BG_PIXELS)
        row.update(r); records.append(row)
    except (ValueError, RuntimeError) as e:
        row["error"] = str(e); fails.append(row)

df = pd.DataFrame(records)
fail_frac = len(fails) / N_SAMPLES
print(f"Converged: {len(df)}/{N_SAMPLES}  (failures: {len(fails)}, {fail_frac:.1%})")
assert fail_frac <= MAX_FAIL_FRACTION, (
    f"Failure fraction {fail_frac:.1%} exceeds MAX_FAIL_FRACTION={MAX_FAIL_FRACTION:.0%}.")''')
    co('''\
# ── Combined distributions: E(NOx) and tau ───────────────────────────────────
summ_E = ee.summarize(df["E_nox_g_per_s"].to_numpy(), "E(NOx)", "g/s")
summ_T = ee.summarize(df["tau_h"].to_numpy(), "tau", "h")
for s, ref in [(summ_E, "7788-8853 g/s"), (summ_T, "0.67-0.83 h")]:
    print(f"{s['name']:8s}: median={s['median']:.3g} {s['unit']}  "
          f"95% CI=[{s['p2_5']:.3g}, {s['p97_5']:.3g}]  ±{s['rel_err_pct']:.0f}%   (ref {ref})")
print(f"\\nE(NOx) t/day: median={summ_E['median']*86400/1e6:.1f}  "
      f"[{summ_E['p2_5']*86400/1e6:.1f}, {summ_E['p97_5']*86400/1e6:.1f}]")''')
    co('''\
# ── Fit-quality diagnostic (logged, NOT gated) ───────────────────────────────
# Quantifies the degenerate-fit tail so it can be filtered during reanalysis.
deg = df["x0_km"] < 5.0
print(f"Members with collapsed x0 (<5 km): {int(deg.sum())} ({deg.mean():.1%})")
print(f"fit_stable (cv_E<=0.5): {df['fit_stable'].mean():.1%} | peak_within_30km: {df['peak_within_30km'].mean():.1%}")
print(f"r2 percentiles [5,50,95]: {np.round(np.percentile(df['r2'], [5,50,95]), 3)}")
gated = df[(df["x0_km"].between(5, 200)) & (df["r2"] >= 0.7)]
if len(gated) > 10:
    gE = ee.summarize(gated["E_nox_g_per_s"].to_numpy(), "E(NOx) gated", "g/s")
    gT = ee.summarize(gated["tau_h"].to_numpy(), "tau gated", "h")
    print(f"\\n[post-hoc gated, n={len(gated)}] E median={gE['median']:.0f} CI=[{gE['p2_5']:.0f},{gE['p97_5']:.0f}]"
          f" | tau median={gT['median']:.2f} CI=[{gT['p2_5']:.2f},{gT['p97_5']:.2f}]")''')
    co('''\
# ── Tornado for E(NOx) and tau ───────────────────────────────────────────────
input_cols = ["pressure_hpa", "fire_lat", "fire_lon", "ai_threshold", "qa",
              "grid_dy_deg", "downwind_max_km", "crosswind_km", "extraction_radius_km",
              "bg_x_min_km", "bg_x_max_km"]
tor_E = ee.tornado_sensitivity(df[input_cols], df["E_nox_g_per_s"].to_numpy(), "E(NOx)")
tor_T = ee.tornado_sensitivity(df[input_cols], df["tau_h"].to_numpy(), "tau")
print("E(NOx) tornado:\\n", tor_E.to_string(index=False))
print("\\ntau tornado:\\n", tor_T.to_string(index=False))''')
    co('''\
# ── Figures ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
for ax, col, s, ref, c0 in [
    (axes[0,0], "E_nox_g_per_s", summ_E, 8853, "#2c7be5"),
    (axes[0,1], "tau_h", summ_T, 0.67, "#7a42f4")]:
    d = df[col]
    q99 = np.percentile(d, 99)
    ax.hist(d[d <= q99], bins=40, color=c0, alpha=0.8, edgecolor="white")
    ax.axvline(s["median"], color="k", lw=2, label=f"median {s['median']:.3g}")
    ax.axvline(s["p2_5"], color="k", ls="--", lw=1); ax.axvline(s["p97_5"], color="k", ls="--", lw=1)
    ax.axvline(ref, color="#e05c00", ls=":", lw=2, label=f"nb03 ({ref})")
    ax.set(xlabel=f"{s['name']} ({s['unit']})", ylabel="members",
           title=f"{s['name']} — combined MC (≤99th pct shown)")
    ax.legend(fontsize=8)
for ax, tor in [(axes[1,0], tor_E), (axes[1,1], tor_T)]:
    t = tor.iloc[::-1]
    ax.barh(t["parameter"], t["std_beta"], color=np.where(t["std_beta"] >= 0, "#2c7be5", "#e05c00"))
    ax.axvline(0, color="k", lw=0.6)
    ax.set(xlabel="standardised β", title=f"Sensitivity — {tor.attrs['output_name']}")
fig.tight_layout()
fig.savefig(OUT_DIR / "nox_error_mc.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved: {OUT_DIR/'nox_error_mc.png'}")''')
    co('''\
# ── Persist ──────────────────────────────────────────────────────────────────
import json
df.to_csv(OUT_DIR / "nox_error_members.csv", index=False)
tor_E.to_csv(OUT_DIR / "nox_error_tornado_E.csv", index=False)
tor_T.to_csv(OUT_DIR / "nox_error_tornado_tau.csv", index=False)
with open(OUT_DIR / "nox_error_summary.json", "w") as f:
    json.dump({"E_nox": summ_E, "tau": summ_T, "n_samples": N_SAMPLES, "n_fail": len(fails),
               "reference_E_gs": [7788, 8853], "reference_tau_h": [0.67, 0.83]}, f, indent=2)
print("Wrote nox_error_members.csv, nox_error_tornado_{E,tau}.csv, nox_error_summary.json")''')
    nb.cells = c
    nbf.write(nb, NB_DIR / "08_nox_emission_error_mc.ipynb")
    print("wrote 08_nox_emission_error_mc.ipynb")


if __name__ == "__main__":
    build_co()
    build_nox()
