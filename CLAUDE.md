# Project Memory — Wildfires (TROPOMI top-down fire emissions → GRL paper)

## Current stage: writing the paper

The science is **complete**. We are now **writing the GRL manuscript from the
six main notebooks** and reconciling the paper to the code. The working
manuscript lives in **`paper_staging/`** (`main.tex` + `figures/`), not yet in
the journal repo. If you are an agent joining now, your job is almost certainly
manuscript-editing or notebook-reconciliation, **not** new method development.

**Scope changed from earlier versions of this file.** Three pivots happened and
the paper reflects them:

1. **Inventory benchmark: GFED5 → CAMS GFAS.** We compare top-down fluxes to the
   *hourly, overpass-matched* GFAS inventory, integrated over an *FRP-defined*
   radius. The old daily-GFED5 comparison and its "temporal mismatch" narrative
   are retired.
2. **Plume ensemble: discrete grid → 500-member Monte-Carlo.** Trajectories are
   now sampled from observed KDEs (injection pressure, FRP-weighted source).
3. **HYSPLIT is now IN scope** as a *forward dispersion validation* of the CO
   flux (not the old backward-trajectory cross-check). The emission-factor (EF)
   / AMF analysis has been **cut** from the paper.

---

## The manuscript (`paper_staging/`)

- `main.tex` — revised, notebook-consistent GRL draft (AGU `agujournal2019`
  class, on Overleaf). Abstract + intro foreground **fluxes + plume**; GFAS
  comparison is explicitly **secondary**. EF/AMF/GFED5/molar-ratio content removed.
- `figures/` — `fig2_emg_fit`, `fig3_csf_transects`, `fig4_pathway_ensemble`,
  `fig5_gfas_comparison`, `fig6_hysplit_forward`, `figS1_injection_pressure`
  (copied from `outputs/…`; see the figure→notebook map below).
- **Overleaf transport:** there is **no Overleaf MCP in the registry**. Options:
  (a) user uploads `main.tex` + `figures/`; (b) Overleaf Git push with a user
  token (project ID `6a0f7af9e2e5c9e2559e5d7d`); (c) the `@mjyoo2/overleaf-mcp`
  server (`npx -y @mjyoo2/overleaf-mcp`, env `OVERLEAF_PROJECT_ID` +
  `OVERLEAF_GIT_TOKEN`) — note its `write_file` is **text-only** (no binary
  figures). The git token is a secret the user must supply.

**Open manuscript items (need the user):** authors/affiliations; the `Jin2021`
bibliography entry (still a TODO); **Figure 1** (event overview AER_AI+NO₂) not
yet generated.

---

## Main pipeline — six notebooks (`notebooks/`, run order 01 → 06)

Notebooks are thin; logic lives in `src/`. Each finds the project root by walking
up for `data/` + `src/`, so they run from any CWD under the repo.

| # | Notebook | Purpose | Key output | src used |
|---|---|---|---|---|
| 01 | `01_fire_source_characterization.ipynb` | MODIS clustering → dominant fire cluster + FRP | fire centroid (37.55°S, 72.55°W; 954 MW) | `fire_clustering` |
| 02 | `02_co_emission_csf.ipynb` | **CO flux** via Cross-Sectional Flux | CSF CO flux + diagnostic | `csf_utils`, `data_helpers`, `wind_helpers` |
| 03 | `03_nox_emission_emg.ipynb` | **NOₓ flux + lifetime** via EMG; AER_LH injection height | EMG fit, E, τ, injection pressure | `emg_utils`, `data_helpers`, `wind_helpers` |
| 04 | `04_emission_inventory_comparison.ipynb` | **Secondary:** top-down vs CAMS GFAS at FRP radius (GFED5 removed) | ratios, radius sweep, `comparison.png` | `inventory_helpers`, `fire_clustering` |
| 05 | `05_plume_pathway_ensemble.ipynb` | 500-member ERA5 MC trajectory corridor + emission-independent plume mask | `corridor_map.png`, `outputs/pathway/` | `trajectory`, `data_helpers`, `fire_clustering` |
| 06 | `06_hysplit_forward_validation.ipynb` | HYSPLIT-CONC forward run driven by the CO flux vs observed CO column | `model_obs_overlay.png`, comparison table | `hysplit_helpers`, `inventory_helpers` |
| 07 | `07_co_emission_error_mc.ipynb` | 500-member MC error propagation for CO (CSF) | `co_distribution.csv`, `co_error_members.csv`, tornado plot | `error_estimation` |
| 08 | `08_nox_emission_error_mc.ipynb` | 500-member MC error propagation for NOₓ (EMG) + plume-alignment gate | `nox_distribution.csv`, `nox_error_members.csv`, tornado plot | `error_estimation` |

01/02/03 are **byte-for-byte copies** of the originals
(`temporal_fire_hotspot_detection`, `co_estimation_csf`, `nox_estimation_emg`),
which remain in `notebooks/` untouched. Prototypes (GFED5 comparisons, STILT,
pre-pivot ensembles, background analyses) remain in `notebooks/extra/`.

**Figure → notebook map (manuscript):** Fig 2 ← nb03 `stage4_emg_fit`; Fig 3 ←
nb02 `co_csf_diagnostic`; Fig 4 ← nb05 `corridor_map`; Fig 5 ← nb04
`comparison`; Fig 6 ← nb06 `model_obs_overlay`; Fig S1 ← nb03
`injection_pressure_analysis`. Fig 1 still to be made.

---

## `src/` modules and their purpose

| Module | Purpose | Status |
|---|---|---|
| `data_helpers.py` | TROPOMI L2 loaders (CO, NO₂, AER_AI, AER_LH) + MODIS hotspot | original |
| `fire_clustering.py` | MODIS load/filter, hierarchical clustering, dominant cluster, **+ `frp_enclosing_radius`** (FRP-defined integration radius, R99) | extended |
| `wind_helpers.py` | AER_LH→plume matching → injection pressure; haversine, lat/lon→xyz. **Refactored:** `match_alh_plume_pressures` is the shared match path used by both `select_injection_pressure` (scalar) and the trajectory KDE | extended |
| `emg_utils.py` | EMG function, gridding, bootstrap fit (NOₓ) | original |
| `csf_utils.py` | Dynamic-transect CSF machinery (CO) | original |
| `trajectory.py` | **NEW.** ERA5 wind field, RK4 isobaric advection, observation-informed priors (pressure KDE, FRP-weighted spatial KDE), ensemble run, corridor probability | new |
| `hysplit_helpers.py` | **NEW.** HYSPLIT CONTROL/SETUP writers, run wrapper (Rosetta), cdump reader (NumPy-2-patched monetio), column integration | new |
| `inventory_helpers.py` | **NEW.** CAMS GFAS integration (0–360 lon, cell area, haversine sum) at a radius; radius sweep. GFED5 deliberately excluded | new |
| `error_estimation.py` | **NEW.** MC ensembles for CO and NOₓ: centroid KDE, pressure KDE, analyst-choice sampling, plume-alignment gate (PCA), pixel bootstrap, tornado decomposition | new |
| `plotting.py`, `pipeline.py` | Diagnostic plots; `main.py` orchestrators | original |

All new code follows the fail-loud rules: explicit params, asserts at
boundaries, no silent fallbacks (see global `~/.claude/CLAUDE.md`).

---

## Methods and their purpose (for agents new to the codebase)

- **EMG (NOₓ, nb03 / `emg_utils`).** Rotate NO₂ into a wind-aligned frame, build a
  downwind line-density profile, fit the Exponentially Modified Gaussian → the
  NOₓ emission flux `E` and effective lifetime `τ` from one overpass.
- **CSF (CO, nb02 / `csf_utils`).** Integrate background-subtracted CO columns
  along wind-perpendicular transects → the instantaneous CO flux. Flux
  consistency across transects is the internal check (CO is long-lived).
- **Injection pressure (`wind_helpers`).** Match AER_LH pixels to the AER_AI
  plume on the unit sphere; the median pressure sets the ERA5 wind level and the
  trajectory/HYSPLIT release height. Purpose: get transport at the layer holding
  the plume mass, not the surface.
- **FRP-enclosing radius (`fire_clustering.frp_enclosing_radius`).** The radius
  enclosing 99% of the cumulative daytime MODIS FRP of the complex (R99=142 km).
  Purpose: a *physically-defined*, reproducible integration radius for the
  inventory comparison instead of an arbitrary distance.
- **MC trajectory ensemble (nb05 / `trajectory`).** 500 ERA5-driven isobaric
  trajectories sampled from observed KDEs → a transport-pathway corridor and an
  emission-independent plume mask. Purpose: the pathway (and the background mask
  for the retrievals); it samples *release* uncertainty only, so it bounds the
  pathway, **not** the physical plume width.
- **HYSPLIT-CONC forward validation (nb06 / `hysplit_helpers`).** Drive a HYSPLIT
  concentration run with the *retrieved* CO flux; compare the modeled column to
  the observed multi-day TROPOMI CO. Purpose: an independent physical check that
  the retrieved flux reproduces the observed plume (it does, to order of
  magnitude, with leading-edge convergence by day 3–4).
- **GFAS comparison (nb04 / `inventory_helpers`).** Benchmark top-down vs the
  hourly, overpass-matched GFAS inventory at R99. Purpose: a *sampling-matched*
  bottom-up cross-check (removes the diurnal mismatch of daily inventories).

---

## Canonical numbers (the paper lineage — Jan 18 2026)

All numbers inspected and confirmed. MC medians are the headline values; point
estimates from nb02/03 are the central inputs to the MC ensembles.

| Quantity | Value |
|---|---|
| Fire centroid / FRP | 37.5484°S, 72.5545°E / 954 MW |
| Injection pressure | **891 hPa** (97 matched pixels, IQR 131 hPa) → U = **8.88 m/s** |
| NOₓ (EMG, MC median) | **7,357 g/s**; 95% CI [3,579–9,337] g/s; R² = 0.907 |
| CO (CSF, MC median) | **793.6 kg/s**; 95% CI [202–1,289] kg/s (AER_LH wind variant) |
| FRP radius R99 | **142.3 km** (99% daytime FRP; stable for caps ≥150 km) |
| GFAS @142 km | CO 332 kg/s; NOₓ 10,871 g/s |
| Top-down vs GFAS | **CO ×2.4 above; NOₓ 0.67× (33% below)** |
| HYSPLIT forward | model reproduces obs CO column to order of magnitude; leading edge converges day 3–4 |
| Observed plume reach | AER_AI/CO > 2,000 km north then offshore in 96 h |

---

## Environment (read before running anything)

**Python:** `/Users/maperezc/Downloads/env/bin/python3` (3.13; scipy, xarray,
netCDF4, matplotlib, monetio). The Homebrew `/opt/homebrew/bin/python3` lacks
these and will fail.

**Run a notebook:**
```bash
PY=/Users/maperezc/Downloads/env/bin/python3
$PY -m nbconvert --to notebook --execute notebooks/05_plume_pathway_ensemble.ipynb \
  --ExecutePreprocessor.timeout=900 --output /tmp/out.ipynb
# (kernel may start anywhere under the repo; the root-finder handles CWD)
```

**HYSPLIT-CONC toolchain (for nb06):**
- Binary: `outputs/pysplit_trajectories/run_00h/hycs_std` — **x86_64**, runs under
  **Rosetta** (`arch -x86_64`). HYSPLIT v5.1.0.
- Its x86_64 runtime libs (`libgfortran.5`, `libquadmath.0`, `libgcc_s.1`) live at
  `stilt_project/lib_x86_64/` (restored from a scipy wheel; the binary's rpath is
  absolute to that path).
- Meteorology: GDAS1 ARL weeks 3–4 at `stilt_met/gdas1.jan26.w3` / `.w4`
  (~571 MB each; re-downloadable from the NOAA ARL archive).
- **`monetio`** (cdump reader) is installed from GitHub and is **incompatible with
  NumPy 2** out of the box; `src/hysplit_helpers.patch_monetio_numpy2()` applies a
  runtime monkeypatch (do **not** edit site-packages). Always read cdump via
  `hysplit_helpers.read_cdump`.

---

## `main.py` CLI (still valid for the 02/03 retrievals)

```bash
PY=/Users/maperezc/Downloads/env/bin/python3
$PY main.py cluster --modis-csv data/fire_nrt_M-C61_723973.csv \
   --start 2026-01-18 --end 2026-01-18 --output /tmp/centroids.csv
$PY main.py nox --date 2026-01-18 --centroid-csv /tmp/centroids.csv   # EMG NOx
$PY main.py co  --date 2026-01-18 --centroid-csv /tmp/centroids.csv   # CSF CO
```
Centroid precedence: explicit `--fire-lat/--fire-lon` > `--centroid-csv` lookup >
`click.UsageError`. Each run writes `results/run_metadata.json` with the full
resolved config and every derived number.

---

## Conventions and gotchas

- **TROPOMI fill = `9.9692e+36`** is finite to `np.isfinite`; check `>= FILL`
  (handled in `data_helpers.py`).
- **QA**: `qa_value` is uint8 0–100; configs use 0.5, loaders ×100 internally.
- **Data dirs** are `data/{MMDD}/` (e.g. `data/0118/`), month-day only.
- **Fail-fast** (global `~/.claude/CLAUDE.md`): no silent fallbacks for critical
  params; assert at boundaries. The one designed-in soft fallback is
  `select_injection_pressure` → `fallback_hpa` with a `warnings.warn`.
- **GFAS is overpass-matched**, so — unlike the retired GFED5 comparison — the
  CO ×2.4 / NOₓ −33% departures are **not** diurnal-sampling artefacts. Do not
  reintroduce "temporal mismatch" language into the GFAS comparison.
- **Don't push to the user's live Overleaf** without showing the diff and getting
  the git token; the Overleaf Git bridge is single-branch (commits land live).
