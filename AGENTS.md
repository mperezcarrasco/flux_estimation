# Project Memory — Wildfires (TROPOMI top-down fire emissions → ACS ES&T Air paper)

## Current stage: writing the paper

The science is **complete**. We are now **writing the ACS ES&T Air manuscript**
and reconciling the paper to the updated probabilistic results. The canonical
manuscript lives **on Overleaf** (project `6a0f7af9e2e5c9e2559e5d7d`), mirrored
via the Git bridge to the `overleaf` remote in this repo. If you are an agent
joining now, your job is almost certainly manuscript-editing, **not** new method
development.

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

## The manuscript (Overleaf + Git bridge)

- **Live file:** `agusample.tex` on the `overleaf/main` branch (Git remote
`overleaf`, token already configured, project `6a0f7af9e2e5c9e2559e5d7d`).
- **Read the latest version:** `git show overleaf/main:agusample.tex` (after
`git fetch overleaf`). Always fetch before reading to get Overleaf edits.
- `**paper_staging/`** is a local snapshot only — it may be stale. The Overleaf
branch is the source of truth.
- `figures/` on Overleaf: `fig01_data_overview.png` (placeholder — not yet
generated), `fig2_emg_fit`, `fig3_csf_transects`, `fig4_pathway_ensemble`,
`fig5_gfas_comparison`, `fig6_hysplit_forward`, `figS1_injection_pressure`.
- **Bibliography:** `agusample.bib` on Overleaf already contains all real
references (not the AGU sample). Do not overwrite it.

### Overleaf editing rule — READ CAREFULLY

**By default, all manuscript additions and edits are proposed in the chat only.**
Only push to Overleaf (`git push overleaf HEAD:main`) when the user explicitly
says to push or says "push it". Never push speculatively or as part of a
"let me just try" action. Showing the user the diff in chat is the correct
workflow; the user decides when it goes live.

**Open manuscript items (need the user):** authors/affiliations; **Figure 1**
(event overview AERAI+NO₂) not yet generated; keywords (5–8); abstract trim
to ≤200 words; updated numbers from probabilistic results.

---

## Main pipeline — nine notebooks (`notebooks/`, run order 01 → 09)

Notebooks are thin; logic lives in `src/`. Each finds the project root by walking
up for `data/` + `src/`, so they run from any CWD under the repo.


| #   | Notebook                                    | Purpose                                                                                                                                                                                                                                                                              | Key output                                                                                 | src used                                        |
| --- | ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------ | ----------------------------------------------- |
| 01  | `01_fire_source_characterization.ipynb`     | MODIS clustering → dominant fire cluster + FRP                                                                                                                                                                                                                                       | fire centroid (37.55°S, 72.55°W; 954 MW)                                                   | `fire_clustering`                               |
| 02  | `02_co_emission_csf.ipynb`                  | **CO flux** via Cross-Sectional Flux                                                                                                                                                                                                                                                 | CSF CO flux + diagnostic                                                                   | `csf_utils`, `data_helpers`, `wind_helpers`     |
| 03  | `03_nox_emission_emg.ipynb`                 | **NOₓ flux + lifetime** via EMG; AER_LH injection height                                                                                                                                                                                                                             | EMG fit, E, τ, injection pressure                                                          | `emg_utils`, `data_helpers`, `wind_helpers`     |
| 04  | `04_emission_inventory_comparison.ipynb`    | **Secondary:** top-down vs CAMS GFAS at FRP radius (GFED5 removed)                                                                                                                                                                                                                   | ratios, radius sweep, `comparison.png`                                                     | `inventory_helpers`, `fire_clustering`          |
| 05  | `05_plume_pathway_ensemble.ipynb`           | 500-member ERA5 MC trajectory corridor + emission-independent plume mask                                                                                                                                                                                                             | `corridor_map.png`, `outputs/pathway/`                                                     | `trajectory`, `data_helpers`, `fire_clustering` |
| 06  | `06_hysplit_forward_validation.ipynb`       | HYSPLIT-CONC forward run driven by the CO flux vs observed CO column                                                                                                                                                                                                                 | `model_obs_overlay.png`, comparison table                                                  | `hysplit_helpers`, `inventory_helpers`          |
| 07  | `07_co_emission_error_mc.ipynb`             | 500-member MC error propagation for CO (CSF). Samples: injection pressure KDE, centroid KDE, analyst choices (AI threshold, QA, n_transects, transect width, background window). Plume mask now uses the saved plume boundary (see Conventions — "Plume boundary + min_valid_pixels") instead of an ad hoc latitude clip; CSF transects also gated by `min_valid_pixels=5`. **Plume-alignment gate (15°, per-member AI-weighted PCA at 450 km radius) rejects ~43% of draws. 286 converged.** Dominant drivers: pressure (78%), fire_lon (17%). | `co_distribution.csv`, `co_error_members.csv`, tornado plot                                | `error_estimation`                              |
| 08  | `08_nox_emission_error_mc.ipynb`            | 500-member MC error propagation for NOₓ (EMG). Plume boundary applied to the AI pressure-KDE/PCA inputs (see Conventions). Plume-alignment gate (15°, per-member AI-weighted PCA at 450 km radius) rejects ~43% of draws. Remaining 284 converged. Dominant drivers: fire_lat (59%), fire_lon (25%). **No pixel bootstrap** (removed — see Conventions). | `nox_distribution.csv`, `nox_error_members.csv`, tornado plot                              | `error_estimation`                              |
| 09  | `09_emission_inventory_comparison_mc.ipynb` | Top-down MC distributions vs GFAS at R99. Reports percentile of GFAS within the top-down ensemble.                                                                                                                                                                                   | `comparison_mc_histograms.png`, `comparison_mc_errorbars.png`, `comparison_mc_summary.csv` | `error_estimation`, `inventory_helpers`         |


**01/02/03 are no longer byte-for-byte copies of the originals** (that was
true through June 2026, before the plume-boundary/`min_valid_pixels`/overpass
fixes below — the *pre-pivot* originals, `temporal_fire_hotspot_detection`,
`co_estimation_csf`, `nox_estimation_emg`, still sit untouched in
`notebooks/` for reference only, but nb02/03 have diverged from them and are
now the actively-maintained canonical notebooks). Prototypes (GFED5
comparisons, STILT, pre-pivot ensembles, background analyses) remain in
`notebooks/extra/`. Two "investigation" copies of nb02/03
(`co_emission_csf-investigation.ipynb`, `nox_emission_emg-investigation.ipynb`)
were the testbed where the plume-boundary/`min_valid_pixels` fixes were
first prototyped before being propagated everywhere else — keep them if you
need to re-derive a fix in isolation, but they are not consumed by anything
downstream.

**Figure → notebook map (manuscript):** Fig 2 ← nb03 `stage4_emg_fit`; Fig 3 ←
nb02 `co_csf_diagnostic`; Fig 4 ← nb05 `corridor_map`; Fig 5 ← nb04
`comparison`; Fig 6 ← nb06 `model_obs_overlay`; Fig S1 ← nb03
`injection_pressure_analysis`; MC uncertainty figures ← nb07/08 `*_error_mc.png`;
GFAS vs MC figures ← nb09 `comparison_mc_*.png`. Fig 1 still to be made.

### Supporting notebooks (not part of the numbered 01→09 pipeline)

| Notebook | Purpose | Key output |
| --- | --- | --- |
| `plume_boundary_inspection.ipynb` | **Authors** `data/{MMDD}/plume_boundary.geojson` — histogram + background-stats AI threshold, DBSCAN blob-clustering of AI>threshold pixels, manual `KEEP_BLOB_IDS` review, concave-hull boundary. Run this **first** for any new date before nb02/03/05/07/08 or the CLI will fail loudly. | `data/{MMDD}/plume_boundary.geojson` |
| `00_data_overview_figure.ipynb` | Manuscript Fig 1 (data overview): full AER_AI swath, saved plume boundary overlay, injection-pressure histogram, wind-aligned frame. | `outputs/figures/fig01_data_overview.png` |
| `00b_mc_parameter_overview.ipynb` | Manuscript Fig 1b (MC priors): injection-pressure KDE, AER_LH coverage, plume-PCA/wind-alignment-gate diagram. | `outputs/figures/fig01b_mc_parameter_overview*.png` |
| `paper_figures.ipynb` | Regenerates **all** publication figures (Fig 2, 3, 4, 5, S1, S2, S3, plus the CO/NOx retrieval-pipeline figures) in one pass. Recomputes EMG/CSF locally (not by importing nb02/03) — after the July 2026 fixes this now numerically agrees with nb02/03 exactly (see Conventions "Overpass-time bug fix"), which is what surfaced that bug. Panel titles/annotations show both the point estimate and the MC posterior explicitly. | `paper_staging/figures/*.png` |

---

## `src/` modules and their purpose


| Module                       | Purpose                                                                                                                                                                                                                                                | Status   |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------- |
| `data_helpers.py`            | TROPOMI L2 loaders (CO, NO₂, AER_AI, AER_LH) + MODIS hotspot                                                                                                                                                                                           | original |
| `fire_clustering.py`         | MODIS load/filter, hierarchical clustering, dominant cluster, **+ `frp_enclosing_radius`** (FRP-defined integration radius, R99)                                                                                                                       | extended |
| `wind_helpers.py`            | AER_LH→plume matching → injection pressure; haversine, lat/lon→xyz. **Refactored:** `match_alh_plume_pressures` is the shared match path used by both `select_injection_pressure` (scalar) and the trajectory KDE                                      | extended |
| `emg_utils.py`               | EMG function, gridding, bootstrap fit (NOₓ)                                                                                                                                                                                                            | original |
| `csf_utils.py`               | Dynamic-transect CSF machinery (CO)                                                                                                                                                                                                                    | original |
| `trajectory.py`              | **NEW.** ERA5 wind field, RK4 isobaric advection, observation-informed priors (pressure KDE, FRP-weighted spatial KDE), ensemble run, corridor probability                                                                                             | new      |
| `hysplit_helpers.py`         | **NEW.** HYSPLIT CONTROL/SETUP writers, run wrapper (Rosetta), cdump reader (NumPy-2-patched monetio), column integration                                                                                                                              | new      |
| `inventory_helpers.py`       | **NEW.** CAMS GFAS integration (0–360 lon, cell area, haversine sum) at a radius; radius sweep. GFED5 deliberately excluded                                                                                                                            | new      |
| `error_estimation.py`        | **NEW.** MC ensembles for CO and NOₓ: centroid KDE, pressure KDE, analyst-choice sampling, **plume-alignment gate (PCA, both CO and NOₓ)**, tornado decomposition. **No pixel bootstrap** (removed; caused systematic downward bias via sparse-peak cell dropout). `co_flux_member` takes `min_valid_pixels` (CO/CSF transect gate) | new      |
| `plume_helpers.py`           | **NEW (July 2026).** Plume boundary definition/masking: `robust_ai_threshold` (median + k·MAD background stats), `cluster_ai_blobs` (DBSCAN on AI>threshold pixels), `build_boundary_from_blobs` (concave hull of kept blobs, `shapely.concave_hull`), `save_plume_boundary`/`load_plume_boundary` (GeoJSON I/O + provenance), `plume_mask` (AI>threshold AND inside boundary). Authored interactively via `plume_boundary_inspection.ipynb`, consumed by every other retrieval entry point. | new      |
| `plotting.py`, `pipeline.py` | Diagnostic plots; `main.py` orchestrators. `pipeline.py` now requires a saved plume boundary for the requested date (fails loudly if missing) and concatenates all matched L2 overpasses instead of using the first match                            | original |


All new code follows the fail-loud rules: explicit params, asserts at
boundaries, no silent fallbacks (see global `~/.Codex/AGENTS.md`).

---

## Methods and their purpose (for agents new to the codebase)

- **EMG (NOₓ, nb03 / `emg_utils`).** Rotate NO₂ into a wind-aligned frame, build a
downwind line-density profile, fit the Exponentially Modified Gaussian → the
NOₓ emission flux `E` and effective lifetime `τ` from one overpass.
- **CSF (CO, nb02 / `csf_utils`).** Integrate background-subtracted CO columns
along wind-perpendicular transects → the instantaneous CO flux. Flux
consistency across transects is the internal check (CO is long-lived).
- **Injection pressure (`wind_helpers`).** Match AER_LH pixels to the AER_AI
plume on the unit sphere; the **mean** pressure (default `aggregation="mean"` in
`select_injection_pressure`) sets the ERA5 wind level and the trajectory/HYSPLIT
release height. Purpose: get transport at the layer holding the plume mass, not
the surface.
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
- **Plume boundary via visual inspection (`plume_helpers`, decided July 2026).**
The AER_AI plume is identified by DBSCAN-clustering all AI>threshold pixels
into spatial blobs, then a human picks which blobs are the real plume
(`KEEP_BLOB_IDS`) vs. isolated noise; the kept blobs' concave hulls become
the saved boundary polygon. **Why manual over fully automatic:** a fixed
AI=2.0 threshold alone lets through sparse, spatially disconnected noise
pixels (continental artefacts, unrelated aerosol) that a single global
threshold can't distinguish from the true plume — clustering narrows the
candidates to spatially coherent blobs, but *which* blob is the fire's
plume is still a judgment call the pipeline cannot safely automate yet.
**Why not the earlier ad hoc clips** (`plume[160:, :] = False`,
`lon > -72`, `north_clip_lat`): those were hardcoded to the exact geometry
of one date's plume and silently wrong for any other date or even a
different plume orientation. **Limitation:** only two dates have a
reviewed boundary so far (0118, 0119) — every entry point fails loudly
(no silent fallback) if a boundary is missing for the requested date.
**Future work:** automating blob selection (e.g. nearest-to-fire-centroid
heuristic, or a learned classifier) is the natural next step once more
dates need boundaries; noted here so it doesn't get silently reinvented
as "just lower the threshold" again.
- **`min_valid_pixels` transect gate (`csf_utils.compute_csf_dynamic`,
decided July 2026).** A CSF transect is only accepted if it has at least
`min_valid_pixels` finite ΔCO samples (of ~9 possible at the current
0.25°/100 km-half-width geometry); transects below that are treated
exactly like transects with zero data (rejected, not just down-weighted).
**Why:** even inside the plume boundary, individual transects near the
fire source can sit mostly in a genuine TROPOMI pixel-coverage gap
(cloud/swath gaps) rather than a real low-CO region — those transects
were previously accepted anyway and dragged the flux estimate down.
**Why 5 (of ~9 possible), not automatic/adaptive:** chosen by inspection
in `co_emission_csf-investigation.ipynb` Section 4b (printed per-transect
valid-pixel counts against the histogram of what's achievable at this
grid spacing) rather than fit to the data — it is a data-quality floor,
not a free retrieval parameter, so it should not be tuned to move the
answer. **Backward-compatible default is 1** (i.e. off) in the function
signature; every call site that wants the gate must pass it explicitly
— this was a deliberate fail-loud choice so a caller can't silently
inherit stricter filtering than it asked for.

---

## Canonical numbers (the paper lineage — Jan 18 2026)

MC medians are the headline values; point estimates from nb02/03 are the
central inputs to the MC ensembles. **Numbers last fully verified July 2
2026**, after three fixes were propagated through the whole pipeline (nb02,
nb03, nb04, nb05, nb07, nb08, `src/pipeline.py`, `src/error_estimation.py`,
`paper_figures.ipynb`) — full detail in Conventions below:

1. **Plume boundary** (`src/plume_helpers.py`, `notebooks/plume_boundary_inspection.ipynb`,
   see Methods above) replaces every ad hoc AI-threshold spatial clip
   (`plume[160:, :] = False`, `lon > -72` clip, `north_clip_lat`) with a
   hand-reviewed, DBSCAN-blob-clustered inclusion polygon per date, saved to
   `data/{MMDD}/plume_boundary.geojson`. Only exists for **0118 and 0119** so
   far — everything below is 0118.
2. **`min_valid_pixels=5`** in `compute_csf_dynamic` (CO/CSF only, see
   Methods above) rejects transects with too few finite ΔCO samples.
3. **Overpass-time bug fix** (`data_helpers.load_co_l2`/`load_no2_l2`) — an
   index-space bug in the NOx overpass-time lookup that got much worse once
   overpasses started being concatenated for fix #1/#2; see Conventions
   "Overpass-time scanline-index bug fix" for the full explanation. Only
   changes the **NOx point estimate**.

Fixes #1 and #2 also required loading **all AER_AI/AER_LH/CO/NO2 overpasses**
for a date (previously several call sites used only the first matched L2
file) — this is what made the pre-existing overpass-time bug (#3) large
enough to notice.


| Quantity                        | Value                                                                                                      | Source  |
| ------------------------------- | ---------------------------------------------------------------------------------------------------------- | ------- |
| Fire centroid / FRP             | 37.5484°S, 72.5545°W / 954 MW                                                                              | nb01    |
| Injection pressure (NOx, r=450km) | **883 hPa** (mean of 99 matched pixels, IQR 73 hPa) → U ≈ **8.73 m/s** (overpass 18:21 UTC — see Conventions "Overpass-time bug fix") | nb03    |
| NOₓ point estimate (EMG)        | **8,642.4 g/s**; τ = 0.74 h; R² = 0.851; x0 = 23.3 km                                                     | nb03    |
| NOₓ (MC median, 284 members)    | **7,450.0 g/s**; 95% CI [3,722–9,630] g/s; ±39.7%  *(216 members rejected by alignment gate; unaffected by the overpass-time bug — nb08 uses the solar-time formula, not per-pixel lookup)* | nb08    |
| NOₓ (MC post-hoc gated, n≈282)  | E median ≈ **7,475 g/s**; CI [3,769–9,635] g/s                                                            | nb08    |
| CO point estimate (CSF, AER_LH) | **1,233.6 kg/s** (6/10 valid transects); ERA5 10m variant 1,572.5 kg/s (5/10 valid)                       | nb02    |
| CO (MC median, 286 members)     | **1,255.1 kg/s**; 95% CI [610.7–1,779.4] kg/s; ±46.6%  *(214 members rejected by alignment gate)*         | nb07    |
| FRP radius R99                  | **142.3 km** (99% daytime FRP; stable for caps ≥150 km)                                                    | nb01    |
| GFAS @142 km (overpass-matched) | CO **331.9 kg/s**; NOₓ **16,667.9 g/s** (as NO₂; see "GFAS NOₓ mass-basis fix" below — this is *not* the same 10,871 g/s quoted in older notes, which was the unconverted NO-basis value) | nb04    |
| Top-down vs GFAS (point est.)   | **CO ×3.72 above** (AER_LH variant); **NOₓ ×0.52 below** *(nb04 re-run July 2 after the overpass-time fix — was ×0.49 against the stale 8,112.7 g/s point estimate)* | nb04    |
| Top-down vs GFAS (MC)           | **CO**: median/GFAS ×3.78, GFAS at p0.3 of top-down posterior (below ~all members); **NOₓ**: median/GFAS ×0.45, GFAS at p99.6 (above ~all members) | nb09    |
| HYSPLIT forward                 | model reproduces obs CO column to order of magnitude; leading edge converges day 3–4  *(not re-verified against the new CO flux — nb06 not re-run this pass)* | nb06    |
| Observed plume reach            | AER_AI/CO > 2,000 km north then offshore in 96 h                                                           | nb05    |


**MC tornado — dominant uncertainty sources (post alignment gate, post-fix):**

- CO: injection pressure 78.1%, fire_lon 17.3%, bg_width_km 2.0%, transect_half_width_km 1.3%, n_transects 1.1%
- NOₓ: fire_lat 58.5%, fire_lon 25.4%, pressure 8.9%, downwind_max_km 6.8%

Note the NOₓ tornado ranking **flipped** vs the previous verification (was
pressure 64% / fire_lat 28%; now fire_lat 58.5% / fire_lon 25.4% / pressure
8.9%) — a direct consequence of the boundary-consistent AI/pressure-KDE input
no longer being dominated by the ad hoc clip's residual noise.

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
- `**monetio`** (cdump reader) is installed from GitHub and is **incompatible with
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

**Requires a plume boundary.** Both `nox` and `co` now call
`_load_plume_boundary_or_raise` and fail loudly (`FileNotFoundError`) if
`data/{MMDD}/plume_boundary.geojson` doesn't exist for `--date`. Run
`notebooks/plume_boundary_inspection.ipynb` for that date first — the
example above works out of the box because `data/0118/plume_boundary.geojson`
already exists.

---

## Conventions and gotchas

- **TROPOMI fill = `9.9692e+36`** is finite to `np.isfinite`; check `>= FILL`
(handled in `data_helpers.py`).
- **QA**: `qa_value` is uint8 0–100; configs use 0.5, loaders ×100 internally.
- **Data dirs** are `data/{MMDD}/` (e.g. `data/0118/`), month-day only.
- **Fail-fast** (global `~/.Codex/AGENTS.md`): no silent fallbacks for critical
params; assert at boundaries. The one designed-in soft fallback is
`select_injection_pressure` → `fallback_hpa` with a `warnings.warn`.
- **GFAS is overpass-matched**, so — unlike the retired GFED5 comparison — the
top-down/GFAS departures are **not** diurnal-sampling artefacts. Do not
reintroduce "temporal mismatch" language into the GFAS comparison.
- **NOₓ pixel bootstrap was removed from `error_estimation.nox_flux_member`.**
A single bootstrap draw caused systematic downward bias: sparse plume-peak
cells (1–3 pixels each) have ~37% probability of going empty per draw, which
deflates the line density peak → smaller `a` → lower E. Diagnostic nb12 showed
that all MC members give similar x0 (22–25 km) and a (350–410 mol/m) when
re-run on full pixels; the old stored E values were 1.5–2.5× too low.
CO has no pixel bootstrap and is unchanged.
- **Plume-alignment gate applies to both CO and NOₓ (June 2026).** Both nb07
and nb08 run a per-member AI-weighted PCA on AER_AI pixels within 450 km of
the sampled centroid; members where |θ_wind − θ_plume| > 15° are rejected as
physically inconsistent. ~214–216/500 members are rejected in both notebooks
(same pressure samples, same PCA geometry — exact count shifts slightly with
the boundary/min_valid_pixels fixes below). CO: 286 converged; NOₓ: 284
converged.
- **Plume boundary + min_valid_pixels (July 2026).** Two related fixes were
propagated through the entire pipeline (nb02, nb03, nb05, nb07, nb08,
`src/pipeline.py`, `src/error_estimation.py`):
  1. **Plume boundary** (`src/plume_helpers.py`, authored via
     `notebooks/plume_boundary_inspection.ipynb`): AER_AI pixels above
     threshold are DBSCAN-clustered into spatial blobs, hand-reviewed, and the
     kept blobs' concave hulls become a saved inclusion polygon
     (`data/{MMDD}/plume_boundary.geojson`). This replaces every ad hoc
     AI-threshold spatial clip (`plume[160:, :] = False` in nb02,
     `np.where(lon > -72, clip(ai, 0, 1.99), ai)` in nb03/nb07/nb08,
     `north_clip_lat` in `error_estimation.co_flux_member`,
     `plume_mask.clip_north_idx` in `pipeline.py`). Boundaries only exist for
     **0118 and 0119** — AI=2.0 does not produce a consistent, reviewable
     plume on other dates. Any pipeline entry point now **fails loudly**
     (`FileNotFoundError`) if no boundary file exists for the requested date;
     there is no automatic fallback threshold.
  2. **`min_valid_pixels=5`** (`csf_utils.compute_csf_dynamic`, CO/CSF only):
     rejects individual CSF transects with fewer than 5 finite ΔCO samples (of
     ~9 possible at the 0.25° grid / 100 km half-width) — these transects sat
     mostly in a pixel-coverage gap near the fire source and pulled the flux
     estimate down. Backward-compatible default (`min_valid_pixels=1`); only
     activated where explicitly passed.
  Combined effect: CO point estimate 779.7 → 1,233.6 kg/s, CO MC median 837.9
  → 1,255.1 kg/s (now in much closer agreement with each other); NOₓ point
  estimate 7,929 → 8,112.7 g/s (superseded again below), NOₓ MC median 7,331
  → 7,450.0 g/s (smaller shift — NOₓ/EMG does not grid-mask by AI, so it only
  benefits from the boundary-consistent AI feeding the pressure-KDE/PCA
  machinery, not a transect-count gate). Also fixed a
  `shapely.vectorized.contains` deprecation warning globally (→
  `shapely.contains_xy`).
- **Overpass-time scanline-index bug fix (July 2 2026).** `find_overpass_time`
  (nb03), `_find_overpass_time` (`src/pipeline.py`), and paper_figures.ipynb's
  `_find_overpass` all located the NO2/CO pixel closest to the fire, then did
  `scanline_idx = closest_pixel_idx // 450` to look up that scanline's
  timestamp in `data["time"]`. This is invalid: `closest_pixel_idx` indexes
  the QA/domain-**filtered** pixel array, while the `// 450` arithmetic only
  makes sense against the **raw, unfiltered** swath layout — two different
  index spaces. It happened to land close enough with a single unfiltered
  orbit, but broke badly once overpasses started being concatenated (the
  bogus index can land anywhere across ~3 h of combined scan time). Root
  fix: `data_helpers.load_co_l2`/`load_no2_l2` now expand `time_utc` to
  per-pixel *before* filtering, so `data["time"]` stays positionally aligned
  with `lat`/`lon` — consumers now just do `times[closest_pixel_idx]`, no
  scanline arithmetic at all. This only affects the **NOₓ point estimate**
  (nb03, `pipeline.py`'s EMG path, paper_figures.ipynb): overpass shifted to
  18:21 UTC, wind 8.73 m/s (was 8.22 m/s), giving **E(NOₓ) = 8,642.4 g/s, τ =
  0.74 h, R² = 0.851** (see table above — supersedes the 8,112.7 g/s number
  from the plume-boundary pass above). CO (solar-time overpass formula, not
  per-pixel lookup) and the NOₓ MC (nb08, also solar-time formula) are
  **unaffected**. `paper_figures.ipynb`'s independently-computed EMG fit now
  matches nb03 exactly, which is what surfaced the bug (they used to
  disagree because each computed a different wrong overpass time).
- **NOₓ MC interpretation (nb09, re-run July 2 2026).** GFAS NOₓ (16,667.9 g/s,
  NO₂ basis — see "GFAS NOₓ mass-basis fix" below) falls above ~all 284 MC
  members (p99.6). CO GFAS (331.9 kg/s) falls below ~all 286 MC members (p0.3);
  the top-down CO estimate is now robustly *higher* than GFAS across nearly the
  entire posterior, more so than before the fix (previously nb09 was stale and
  had not been re-run against the gated CO distribution).
- **GFAS NOₓ mass-basis fix (`inventory_helpers`, nb04/nb09/`paper_figures.ipynb`).**
  CAMS GFAS `noxfire` reports NOₓ **as NO** (30.006 g/mol) — confirmed against
  the GFAS/Kaiser 2012 emission-factor convention (AMT 18/3669/2025 states GFAS
  "provides daily mean emission data for NOₓ as NO but not for NO₂"). The
  top-down EMG flux is NOₓ **as NO₂** (`γ · M_NO2`, `emg_utils.py`,
  γ = 1.32, M(NO₂) = 46.005 g/mol; see EMG Methods above). Comparing the two
  directly — as an earlier pass did — silently mixed mass bases: GFAS NOₓ was
  quoted as 10,871 g/s (NO basis) against a top-down flux expressed as NO₂,
  understating the true gap by ~35%. **Fix:** `GFAS_NO_TO_NO2 = 46.005/30.006 =
  1.5332` is now applied explicitly (named constant, documented inline, not a
  silent multiply) wherever GFAS `noxfire` is loaded, so both sides are NO₂
  before any ratio or percentile is computed. GFAS NOₓ @ R99 moves
  **10,871 → 16,667.9 g/s**; CO (`cofire`) is unambiguous and unaffected.
  Also found and fixed in the same pass: nb04 and nb09 both had a stray
  output-directory target of `0119` (no complete inputs) instead of the
  canonical `0118`, which had been silently orphaning `radius_sensitivity.csv`
  and the MC comparison summary — `paper_figures.ipynb`'s Fig 3 and Fig S1
  regenerate from these corrected notebooks. Manuscript-facing ratio: top-down
  NOₓ MC median (7,331 g/s) is now **56% below** GFAS (was reported as "33%
  below" pre-fix); still the 100th percentile (GFAS above the entire posterior)
  — the *direction* of the discrepancy is unchanged, only its magnitude.
- **Don't push to the user's live Overleaf** without showing the diff and getting
the git token; the Overleaf Git bridge is single-branch (commits land live).

