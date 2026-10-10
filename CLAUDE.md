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
| 02  | `02_co_emission_csf.ipynb`                  | **CO flux** via Cross-Sectional Flux on the **4 km × 4 km Adams-oversampled grid** (Griffin 2024; since Oct 2026, was 0.25° binning) — see Methods "CO grid"                                                                                                                       | CSF CO flux + diagnostic, `outputs/retrievals/0118/co_retrieval.csv`                       | `csf_utils`, `oversampling`, `data_helpers`, `wind_helpers` |
| 03  | `03_nox_emission_emg.ipynb`                 | **NOₓ flux + lifetime** via EMG; AER_LH injection height                                                                                                                                                                                                                             | EMG fit, E, τ, injection pressure                                                          | `emg_utils`, `data_helpers`, `wind_helpers`     |
| 04  | `04_emission_inventory_comparison.ipynb`    | **Secondary:** top-down vs CAMS GFAS at FRP radius (GFED5 removed)                                                                                                                                                                                                                   | ratios, radius sweep, `comparison.png`                                                     | `inventory_helpers`, `fire_clustering`          |
| 05  | `05_plume_pathway_ensemble.ipynb`           | 500-member ERA5 MC trajectory corridor + emission-independent plume mask                                                                                                                                                                                                             | `corridor_map.png`, `outputs/pathway/`                                                     | `trajectory`, `data_helpers`, `fire_clustering` |
| 06  | `06_hysplit_forward_validation.ipynb`       | HYSPLIT-CONC forward run driven by the CO flux vs observed CO column                                                                                                                                                                                                                 | `model_obs_overlay.png`, comparison table                                                  | `hysplit_helpers`, `inventory_helpers`          |
| 07  | `07_co_emission_error_mc.ipynb`             | 500-member MC error propagation for CO (CSF). Samples: injection pressure KDE, centroid KDE, analyst choices (AI threshold, QA, n_transects, transect width, background window). Plume mask uses the saved plume boundary (see Conventions — "Plume boundary + min_valid_pixels"). CO on the 4 km oversampled grid (same as nb02); CSF transects gated by `min_valid_fraction=0.5` (> 50% of samples finite). **Plume-alignment gate (15°, per-member AI-weighted PCA at 450 km radius) rejects 213/500 (42.6%). 287 converged.** Dominant drivers: pressure (57%), transect half-width (24%), fire_lon (12%). | `co_distribution.csv`, `co_error_members.csv`, tornado plot                                | `error_estimation`                              |
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

**Prototypes broken by the Oct 2026 CSF signature change (expected, fail loudly):**
`csf_utils.build_dynamic_transects`/`compute_csf_dynamic` now *require*
`sample_spacing_m` and `min_valid_fraction` (the old `min_valid_pixels` is gone),
so these `notebooks/extra/` notebooks raise `TypeError` until updated:
`co_emission_csf-investigation`, `co_emissions_comparison`,
`co_emissions_comparison_gfas`, `co_estimation_csf`,
`co_grid_resolution_sensitivity` (the latter is also superseded — its July
numbers predate the plume boundary and its oversampling was radius-only).
`scripts/build_error_notebooks.py` (the old generator of nb07/nb08) is stale
(AI thresholds 1.5–2.5, 300 km radius, `bin_co_ai_grid`); nb07/nb08 are
hand-maintained now — do not regenerate them from it.

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
| `extra/experimental_co_csf_4km_oversampling.ipynb` | Record of the 0.25° → 4 km grid comparison (Oct 2026). Self-contained; known-answer checks: 0.25° path = frozen pre-migration nb02 (1,275.01 kg/s, 6/10) and `src.compute_csf_dynamic` exactly; 4 km path = `src.oversampling` + `compute_csf_dynamic` exactly = current nb02 export. Oversampling unit tests T0–T5. | `outputs/experimental/co_4km_oversampled/0118/` (both `fig_co_pipeline` versions, comparison CSV, `run_config.json`) |
| `paper_figures.ipynb` | Regenerates **all** publication figures (Fig 2, 3, 4, 5, S1, S2, S3, plus the CO/NOx retrieval-pipeline figures) in one pass. Recomputes EMG/CSF locally (not by importing nb02/03) — after the July 2026 fixes this now numerically agrees with nb02/03 exactly (see Conventions "Overpass-time bug fix"), which is what surfaced that bug. Panel titles/annotations show both the point estimate and the MC posterior explicitly. | `paper_staging/figures/*.png` |

---

## `src/` modules and their purpose


| Module                       | Purpose                                                                                                                                                                                                                                                | Status   |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------- |
| `data_helpers.py`            | TROPOMI L2 loaders (CO, NO₂, AER_AI, AER_LH) + MODIS hotspot. `load_co_l2`/`load_aer_ai` take keyword-only `with_corners=True` to also return pixel corners `lat_b`/`lon_b` (n, 4) for oversampling | original |
| `fire_clustering.py`         | MODIS load/filter, hierarchical clustering, dominant cluster, **+ `frp_enclosing_radius`** (FRP-defined integration radius, R99)                                                                                                                       | extended |
| `wind_helpers.py`            | AER_LH→plume matching → injection pressure; haversine, lat/lon→xyz. **Refactored:** `match_alh_plume_pressures` is the shared match path used by both `select_injection_pressure` (scalar) and the trajectory KDE                                      | extended |
| `emg_utils.py`               | EMG function, gridding, bootstrap fit (NOₓ)                                                                                                                                                                                                            | original |
| `csf_utils.py`               | Dynamic-transect CSF machinery (CO). Since Oct 2026 `build_dynamic_transects`/`compute_csf_dynamic` require keyword-only `sample_spacing_m` (was hardcoded 0.25°) and `compute_csf_dynamic` requires `min_valid_fraction` (replaces `min_valid_pixels`) | original |
| `oversampling.py`            | **NEW (Oct 2026).** Adams et al. (2019) two-step oversampling (footprint points + radius mean, weight 1/pixel area) and `regular_grid_km` (cells of `cell_km` at a reference latitude). Tests: `tests/test_oversampling.py` (pytest, hand-computed known answers) | new      |
| `trajectory.py`              | **NEW.** ERA5 wind field, RK4 isobaric advection, observation-informed priors (pressure KDE, FRP-weighted spatial KDE), ensemble run, corridor probability                                                                                             | new      |
| `hysplit_helpers.py`         | **NEW.** HYSPLIT CONTROL/SETUP writers, run wrapper (Rosetta), cdump reader (NumPy-2-patched monetio), column integration (the model–obs comparison is in `transport_compare.py`)                                                                                                                              | new      |
| `inventory_helpers.py`       | **NEW.** CAMS GFAS integration (0–360 lon, cell area, haversine sum) at a radius; radius sweep. GFED5 deliberately excluded                                                                                                                            | new      |
| `error_estimation.py`        | **NEW.** MC ensembles for CO and NOₓ: centroid KDE, pressure KDE, analyst-choice sampling, **plume-alignment gate (PCA, both CO and NOₓ)**, tornado decomposition. **No pixel bootstrap** (removed; caused systematic downward bias via sparse-peak cell dropout). `oversample_co_ai_grid` builds the 4 km CO/AI grids; `co_flux_member` takes `min_valid_fraction` and `sample_spacing_m`. **Plume-alignment gate** `plume_axis_pca` / `angular_diff_deg` (moved from nb07/nb08 Oct 2026, numerically identical; parameters in `config/mc_alignment_gate.yaml`, loaded with `load_alignment_gate_config`) | new      |
| `transport_compare.py`       | **NEW (Oct 2026).** HYSPLIT model vs observed TROPOMI CO (moved from nb06; also used by `co_transport_temporal_coherence`): `bin_observed_co` (0.25° analysis grid), `observed_enhancement` (p10 background), `IndexRegrid` (containing-cell lookup, no griddata ties), `footprint_comparison` (T, near-source exclusion, same-cell medians, p95/p99 reach), `footprint_median_series`, `comparison_table`. Settings in `config/hysplit_comparison.yaml` | new      |
| `plume_helpers.py`           | **NEW (July 2026).** Plume boundary definition/masking: `robust_ai_threshold` (median + k·MAD background stats), `cluster_ai_blobs` (DBSCAN on AI>threshold pixels), `build_boundary_from_blobs` (concave hull of kept blobs, `shapely.concave_hull`), `save_plume_boundary`/`load_plume_boundary` (GeoJSON I/O + provenance), `plume_mask` (AI>threshold AND inside boundary). Authored interactively via `plume_boundary_inspection.ipynb`, consumed by every other retrieval entry point. | new      |
| `plotting.py`, `pipeline.py` | Diagnostic plots; `main.py` orchestrators. `pipeline.py` now requires a saved plume boundary for the requested date (fails loudly if missing) and concatenates all matched L2 overpasses instead of using the first match                            | original |


All new code follows the fail-loud rules: explicit params, asserts at
boundaries, no silent fallbacks (see global `~/.claude/CLAUDE.md`).

**Tests (`tests/`, pytest; run `$PY -m pytest tests/ -q`, ~2 s, needs `data/` and
`outputs/`).** 36 tests as of 2026-10-11:
- `test_oversampling.py` — Adams oversampling known answers (area, footprint
  points both corner orders, 1/area weights, 7/9 km neighbourhoods, constant
  field over 66°S–6°S) and the explicit transect spacing.
- `test_alignment_gate.py` — synthetic-plume PCA axis/sign, far-field edge,
  and a regression that recomputes θ_plume/Δθ for every stored nb07 (287) and
  nb08 (285) member exactly.
- `test_transport_compare.py` — index regrid and footprint maths on synthetic
  grids, plus a regression that rebuilds `model_obs_comparison.csv` from the
  cdump + L2 files exactly, and the series peak / +48 h values in the paper.
- `test_paper_numbers.py` — every non-GFAS manuscript number (Overleaf
  d17f265) against the output file that produces it. Note: the paper prints the
  NOx MC lower bound as 3,621 g/s; the output is 3,620.46 → 3,620 (paper to fix).
Each test was checked to fail when the code it covers is deliberately broken.

**Shared configs (Oct 2026):** `config/mc_alignment_gate.yaml` (nb07, nb08,
tests) and `config/hysplit_comparison.yaml` (nb06, `co_transport_temporal_coherence`,
tests). nb06 asserts that the comparison config's source location and column
conversion equal its HYSPLIT run settings.

---

## Methods and their purpose (for agents new to the codebase)

- **EMG (NOₓ, nb03 / `emg_utils`).** Rotate NO₂ into a wind-aligned frame, build a
downwind line-density profile, fit the Exponentially Modified Gaussian → the
NOₓ emission flux `E` and effective lifetime `τ` from one overpass.
- **CSF (CO, nb02 / `csf_utils`).** Integrate background-subtracted CO columns
along wind-perpendicular transects → the instantaneous CO flux. Flux
consistency across transects is the internal check (CO is long-lived).
- **CO grid: 4 km × 4 km Adams-oversampled (`oversampling`, decided Oct 2026).**
Regular lat/lon grid with 4 km cells at the fire latitude (Δlat 0.0359°,
Δlon 0.0453°; E–W cell ≈ 4.4 km at 30°S). Step 1: every grid point inside a
pixel footprint (L2 `latitude_bounds`/`longitude_bounds`) takes the pixel
value; step 2: each cell = mean of step-1 values within 7 km, weight 1/pixel
area. Applied to CO and AER_AI. Transect samples every 4 km (51 per 200 km
transect). **Why:** matches Griffin et al. (2024, Sect. 2.2: 4 km grid,
"oversampled by 7 km", citing Adams et al. 2019 Sect. 3.1). Plain binning
cannot be used at 4 km: a CO pixel (~5.5 × 7 km) is larger than a cell, so
most cells would be empty. Config: `config/default_csf.yaml` `grid.cell_km`,
`grid.oversample_radius_km`, `csf.sample_spacing_km`. Effect on 18 Jan:
point estimate 1,275.0 → 1,363.3 kg/s (same 6 transects; the change is
mostly the near-source transect, 749 → 1,693 kg/s, whose narrow plume core
the 0.25° cells averaged out).
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
- **Transect coverage gate (`csf_utils.compute_csf_dynamic`, July 2026;
now `min_valid_fraction`, Oct 2026).** A CSF transect is only accepted if
**more than** `min_valid_fraction = 0.5` of its samples have a finite ΔCO
(≥ 26 of 51 at 4 km / 100 km half-width; 38–76 samples across the MC's
75–150 km half-widths); transects below that are treated exactly like
transects with zero data (rejected, not just down-weighted). The fraction
was chosen by the user (Oct 2026) so the gate means the same coverage at
every transect width. *History below is the original count-based gate:*
`min_valid_pixels=5` of ~9 samples at the old 0.25° geometry.
**Why:** even inside the plume boundary, individual transects near the
fire source can sit mostly in a genuine TROPOMI pixel-coverage gap
(cloud/swath gaps) rather than a real low-CO region — those transects
were previously accepted anyway and dragged the flux estimate down.
**Why 5 (of ~9 possible), not automatic/adaptive:** chosen by inspection
in `co_emission_csf-investigation.ipynb` Section 4b (printed per-transect
valid-pixel counts against the histogram of what's achievable at this
grid spacing) rather than fit to the data — it is a data-quality floor,
not a free retrieval parameter, so it should not be tuned to move the
answer. Since Oct 2026 the argument has **no default** — every call site
must pass it.

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
| Injection pressure (r=450km)    | **882 hPa** (mean of 99 matched pixels, median 898, IQR 73 hPa) → U = **8.72 m/s** at 123.3° (same for CO and NOx; nb02/nb03 and `main.py co`/`nox`, verified 2026-10-11). *Previously listed as 883 hPa / 8.73 m/s.* | nb02/nb03 |
| NOₓ point estimate (EMG)        | **8,414.7 g/s**; τ = 0.75 h; R² = 0.846; x0 = 23.6 km; μx = 79.0 km (nb03 run 2026-07-05 = `outputs/retrievals/0118/nox_retrieval.csv`; `main.py nox` gives the same E/τ/R² on 2026-10-11). *The 8,642.4 g/s / 0.74 h / 0.851 previously listed here was stale.* ⚠ `main.py nox` reports bootstrap CV(a)=CV(E)=0.000 and "peak offset +85.9 km" vs nb03 cv_E=2.66, cv_a=0.048 — not investigated. | nb03    |
| NOₓ (MC median, 285 members)    | **7,505.0 g/s**; 95% CI [3,620.5–9,655.4] g/s; ±40%  *(215 members rejected by alignment gate; re-run 2026-10-11 reproduces the 2026-07-05 run: same members, median equal to 1e-8, 8 members differ ≤ 1e-4 relative — EMG fit numerical noise). Member 455 has E ≈ 69,900 g/s (≈9× median), outside the CI; inflates std only.* | nb08    |
| NOₓ (MC post-hoc gated, n=283)  | E median **7,510 g/s**; CI [3,644–9,659] g/s                                                              | nb08    |
| CO point estimate (CSF, AER_LH) | **1,363.3 kg/s** (6/10 valid transects; 4 km oversampled grid, injection 882 hPa, U 8.72 m/s, bg 16.18 mmol/m²); ERA5 10m variant 1,929.4 kg/s (4/10 valid). *Re-run 2026-10-06; `main.py co` gives the same 1,363.35. Was 1,275.0 on the 0.25° grid (nb02 run of 2026-07-05 — the 1,233.6 previously listed here was already stale).* | nb02    |
| CO (MC median, 287 members)     | **1,273.3 kg/s**; 95% CI [675.6–1,842.7] kg/s; ±45.8%  *(213 members rejected by alignment gate; re-run 2026-10-06. On 0.25°: 1,249.7 [605.0–1,731.0], n=287 — the 1,255.1/286 previously listed here was already stale)* | nb07    |
| FRP radius R99                  | **142.3 km** (99% daytime FRP; stable for caps ≥150 km)                                                    | nb01    |
| GFAS @142 km (overpass-matched) | ⚠ **STALE — the GFAS files used are 24-h rolling averages, not hourly** (found 2026-10-06; hourly 18–19 UTC gives CO ≈ 60.6 kg/s). Old values: CO **331.9 kg/s**; NOₓ **16,667.9 g/s** (as NO₂; see "GFAS NOₓ mass-basis fix" below — this is *not* the same 10,871 g/s quoted in older notes, which was the unconverted NO-basis value) | nb04    |
| Top-down vs GFAS (point est.)   | ⚠ STALE (24-h GFAS, see above). Against that same stale field, nb04 re-run 2026-10-06 gives CO ×4.11. Old: **CO ×3.72 above** (AER_LH variant); **NOₓ ×0.52 below** *(nb04 re-run July 2 after the overpass-time fix — was ×0.49 against the stale 8,112.7 g/s point estimate)* | nb04    |
| Top-down vs GFAS (MC)           | ⚠ STALE (24-h GFAS). Against that same stale field, nb09 re-run 2026-10-06 gives CO median/GFAS ×3.84, GFAS at p0.0. Old: **CO**: median/GFAS ×3.78, GFAS at p0.3 of top-down posterior (below ~all members); **NOₓ**: median/GFAS ×0.45, GFAS at p99.6 (above ~all members) | nb09    |
| HYSPLIT forward                 | model reproduces obs CO column to order of magnitude; obs/model 2.3–4.0 at +1…+4 d; footprint reach (p99) 1,082 → 2,218 km. **Re-run 2026-10-11 driven by the 4 km MC median 1,273.3 kg/s** (CONTROL 4.584056e6 kg/h). Previous run (1,249.7 kg/s) kept in `outputs/hysplit_co/prev_run_1249p7kgs_0p25deg/`. nb06 now asserts the cached `cdump` was driven by the current source term. GDAS1 files re-downloaded to `stilt_met/` and x86_64 Fortran runtime re-extracted to `stilt_project/lib_x86_64/` (scipy-1.13.1 macOS x86_64 wheel; `libgcc_s.1.dylib` → `libgcc_s.1.1.dylib` symlink); both git-ignored. | nb06    |
| Observed plume reach            | AER_AI/CO > 2,000 km north then offshore in 96 h                                                           | nb05    |


**MC tornado — dominant uncertainty sources (post alignment gate, post-fix):**

- CO (4 km grid, 2026-10-06): injection pressure 57.1%, transect_half_width_km 23.8%, fire_lon 12.4%, n_transects 4.1%, bg_width_km 1.4%. *(0.25° grid: pressure 78.1%, fire_lon 17.3%, transect_half_width 1.3%.)* ⚠ Unexplained: wider transects now give **larger** fluxes; once a transect spans the plume the flux should not depend on width, so this suggests positive ΔCO off-plume (background slightly low). Not investigated.
- NOₓ (2026-10-11): fire_lat 59.5%, fire_lon 26.6%, pressure 6.9%, downwind_max_km 6.4%

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
- **Fail-fast** (global `~/.claude/CLAUDE.md`): no silent fallbacks for critical
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
the boundary/min_valid_pixels fixes below). CO: 287 converged (Oct 2026); NOₓ: 284
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
- **HYSPLIT comparison unified (2026-10-11).** `co_transport_temporal_coherence`
  used to rebuild the modeled footprint-median curve itself with
  `griddata('nearest')` on a grid to −65°E (nb06 and the paper map use −66°E),
  and its check against nb06 compared the wrong column with a 5e-4 mol/m²
  tolerance, so its +3 d value (2.075 mmol/m²) silently disagreed with nb06's
  table (2.11). Both now call `src.transport_compare` with
  `config/hysplit_comparison.yaml`; the check is exact. Effect: the
  `fig_transport_combined` curve moves slightly (+48 h 1.399 → 1.423×10¹⁷
  molec cm⁻²); paper values unchanged at the printed precision (peak 9.3e17,
  ~1.4e17 at +48 h). All other outputs byte-identical after the move.
- **Don't push to the user's live Overleaf** without showing the diff and getting
the git token; the Overleaf Git bridge is single-branch (commits land live).

