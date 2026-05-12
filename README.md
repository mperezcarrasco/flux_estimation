# Wildfire Flux Estimation — TROPOMI EMG (NO\_x) and CSF (CO)

Top-down emission flux estimation for individual fire events from
**Sentinel-5P / TROPOMI** Level-2 observations and **ERA5** reanalysis winds.

Two satellite retrieval methods are implemented:

| Method | Species | Reference |
|---|---|---|
| **Exponentially Modified Gaussian (EMG)** | NO\_x | Beirle et al. (2011); Jin et al. (2021) |
| **Cross-Sectional Flux (CSF)** | CO | Mebust et al. (2011); Griffin et al. (2021) |

Both methods use the **TROPOMI AER\_LH** product to choose the aerosol injection
pressure and **ERA5 pressure-level winds** at that pressure to drive plume
transport.

The library reproduces the analysis in
`notebooks/nox_estimation_emg.ipynb` and `notebooks/co_estimation_csf.ipynb`.

---

## 1. Install

Tested on Python 3.13. Required packages:

```bash
pip install numpy scipy pandas xarray h5py netCDF4 \
            matplotlib click pyyaml
```

No installation step — clone the repo and run `python main.py`.

---

## 2. Required input data

Three independent data sources must be downloaded by the user and placed
inside the `data/` directory. The CLI **fails loudly** if any file is missing.

### 2.1 TROPOMI Level-2 swaths (OFFL)

Source: [Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu/) (free, registration required).

For every fire date `YYYY-MM-DD`, place **four** OFFL netCDF files into
`data/{MMDD}/`:

```
data/
└── 0118/                                                    # MMDD of fire date
    ├── S5P_OFFL_L2__NO2____20260118T173523_*.nc            # NO2 (needed by `nox`)
    ├── S5P_OFFL_L2__CO_____20260118T173523_*.nc            # CO  (needed by `co`)
    ├── S5P_OFFL_L2__AER_AI_20260118T173523_*.nc            # AER_AI (both)
    └── S5P_OFFL_L2__AER_LH_20260118T173523_*.nc            # AER_LH (both)
```

* Use the **OFFL** processor stream (offline, fully processed) — not NRTI.
* All four files must cover the same overpass (same `42837` orbit number in
  the filename, for example).
* QA filtering (`qa_value >= 0.5`) is applied automatically.

### 2.2 ERA5 pressure-level winds

Source: [Copernicus Climate Data Store — ERA5 hourly on pressure levels](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels).

Download a single netCDF spanning the analysis period at the variables and
levels listed below, and place it at:

```
data/ERA5_winds_pressure_levels.nc
```

Requested fields:
* Variables: `u_component_of_wind`, `v_component_of_wind`
* Pressure levels (hPa): 125, 175, 225, 300, 400, 500, 600, 700, 775, 825, 875, 925, 975
* Temporal sampling: hourly
* Spatial extent: must encompass the fire centroid and surrounding ERA5 grid points

A second optional surface-wind file (`ERA5_winds.nc`) is referenced by the
original notebooks but **not** used by `main.py co` since this library uses
only the AER\_LH pressure-level wind variant.

### 2.3 MODIS active-fire detections (FIRMS MCD14DL)

Source: [NASA FIRMS Archive Download](https://firms.modaps.eosdis.nasa.gov/download/).

Request the **MODIS Aqua** product over the spatial/temporal window of
interest. The download is a single CSV named `fire_nrt_M-C61_*.csv` with
the standard MCD14DL columns. Place it in `data/`:

```
data/fire_nrt_M-C61_<request_id>.csv
```

---

## 3. Usage

The CLI has three subcommands, intended to be run in this order.

### 3.1 `cluster` — build a daily fire-centroid CSV from MODIS

`cluster` performs the same fire-source identification as
`notebooks/temporal_fire_hotspot_detection.ipynb`:

1. Filter MCD14DL detections by date window, domain bounding box,
   confidence, daytime flag and satellite.
2. Hierarchical-linkage clustering of pixel coordinates with a
   20 km cut distance (Jin et al. 2021 convention).
3. FRP-weighted centroid for each cluster.
4. Dominant cluster = highest mean-FRP cluster within
   `search_radius_km` of an anchor point (Concepción by default)
   and above `min_frp_mw`.

```bash
python main.py cluster \
  --modis-csv data/fire_nrt_M-C61_723973.csv \
  --start 2026-01-18 --end 2026-01-18 \
  --output data/fire_centroids.csv
```

Output CSV schema:
```csv
date,fire_lat,fire_lon,frp_mw,n_pixels,n_clusters_total,dist_to_anchor_km
2026-01-18,-37.5484,-72.5545,953.6838,8,24,91.1449
```

This file is the centroid-CSV input expected by `nox` and `co`.

### 3.2 `nox` — EMG NO\_x emission flux

```bash
python main.py nox \
  --date 2026-01-18 \
  --centroid-csv data/fire_centroids.csv
```

Or supply the centroid directly (skips the cluster step entirely):

```bash
python main.py nox \
  --date 2026-01-18 \
  --fire-lat -37.5484 --fire-lon -72.5545
```

Console output is a summary block; full per-run metadata is written to
`outputs/20260118/nox/results/run_metadata.json`.

### 3.3 `co` — CSF CO emission flux

```bash
python main.py co \
  --date 2026-01-18 \
  --centroid-csv data/fire_centroids.csv
```

Only the AER\_LH-pressure-level wind variant is implemented.

---

## 4. Outputs

Every run writes to `outputs/{YYYYMMDD}/{nox|co}/`:

```
outputs/
└── 20260118/
    ├── nox/
    │   ├── plots/
    │   │   ├── 01_nox_scene.png             # NO2 column + fire + 300 km circle
    │   │   ├── 02_nox_line_density.png      # cross-wind sum * dy profile
    │   │   └── 03_emg_fit.png               # EMG fit with R^2, tau, E
    │   └── results/
    │       └── run_metadata.json            # full results + config snapshot
    └── co/
        ├── plots/
        │   ├── 01_co_scene.png              # CO column + fire + radius
        │   └── 02_csf_diagnostic.png        # 2x2 panel: CO map, transects, fluxes
        └── results/
            └── run_metadata.json
```

The `run_metadata.json` files contain every parameter used (the full
resolved config), the selected injection pressure with IQR, the wind
vector, intermediate fit metrics, and the final flux estimates in three
units (g/s, t/day, Gg/yr where applicable).

---

## 5. Configuration

Defaults live in `config/`:

| File | Purpose |
|---|---|
| `default_cluster.yaml` | MODIS pixel filtering, linkage method, dominant-cluster anchor |
| `default_emg.yaml` | TROPOMI QA, AER\_LH injection-pressure aggregation, EMG bootstrap parameters |
| `default_csf.yaml` | TROPOMI QA, gridding resolution, Change-D background window, CSF transect geometry |

Override any default by editing a copy and passing `--config path/to/custom.yaml`:

```bash
python main.py nox --date 2026-01-18 \
  --centroid-csv data/fire_centroids.csv \
  --config config/my_high_resolution.yaml
```

Each YAML is heavily commented — read the file to know what every key does.

---

## 6. Method summary

### EMG (NO\_x — `main.py nox`)

The along-wind line density of a single instantaneous point source diffusing
laterally as a Gaussian while decaying exponentially along the wind axis
follows an **exponentially modified Gaussian**:

```
ρ(x) = (a / x0) · exp[(σ²/2x0²) + μ_x/x0 − x/x0]
        · erfc{(σ/x0 − (x−μ_x)/σ)/√2} + B
```

* `a` (mol m⁻¹) — integrated line density
* `x0` (km) — apparent source offset / e-folding decay length
* `μ_x`, `σ` (km) — plume centre and Gaussian width
* `B` (mol m⁻¹) — background line density

Fit with `scipy.optimize.curve_fit`; bootstrap with 50 perturbed initial
guesses; CV of derived flux *E* is reported as a stability metric.

Emission flux: `E = a · U / x0`. NO\_x lifetime: `τ = x0 / U`.
Conversion to mass: multiplied by γ = 1.32 (NO\_x/NO\_2 ratio, Jin et al.
2021 Table S1) and by molar mass of NO\_2 (46.005 g mol⁻¹).

### CSF (CO — `main.py co`)

CO total column is integrated across **n transects** perpendicular to the
local plume-mean wind direction; per-transect flux is

```
F_k = U⊥,k · ∫ ΔCO(y) dy
```

with `ΔCO = CO − bg_CO`, and `bg_CO` taken as the median far-upwind
(200–300 km) CO column ("Change D" approach).

CO molar mass = 0.02801 kg mol⁻¹. Output flux is the mean (or median,
configurable) of valid per-transect fluxes.

---

## 7. Repository layout

```
wildfires/
├── main.py                        # Click CLI entry point (cluster / nox / co)
├── README.md
├── config/
│   ├── default_cluster.yaml
│   ├── default_emg.yaml
│   └── default_csf.yaml
├── src/
│   ├── data_helpers.py            # TROPOMI L2 loaders (NO2, CO, AER_AI, AER_LH)
│   ├── wind_helpers.py            # AER_LH → injection-pressure selection
│   ├── fire_clustering.py         # MODIS clustering + dominant-cluster picker
│   ├── emg_utils.py               # EMG function + bootstrap fit + gridding
│   ├── csf_utils.py               # CSF dynamic-transect machinery
│   ├── plotting.py                # All diagnostic plots
│   └── pipeline.py                # End-to-end orchestrators (NOx and CO)
├── data/
│   ├── {MMDD}/*.nc                # TROPOMI L2 swaths (user-supplied)
│   ├── ERA5_winds_pressure_levels.nc
│   └── fire_nrt_M-C61_*.csv
├── outputs/                       # Generated at run time
│   └── {YYYYMMDD}/{nox|co}/{plots,results}/
└── notebooks/                     # Reference implementations (read-only)
```

---

## 8. Caveats

* **Single overpass per day.** TROPOMI provides one observation per fire
  day at ~13:30 local time. Daily-integrated emission rates cannot be
  derived without an external diurnal-cycle constraint.
* **AMF bias over smoke.** The standard TROPOMI NO\_2 product uses a 1°
  TM5 a-priori profile that underestimates NO\_2 in dense fire plumes by
  a factor of ~1.8 (Jin et al. 2021, Table S6). The library reports the
  **uncorrected** flux; downstream AMF recalculation is not implemented.
* **Injection-pressure IQR > 100 hPa.** A vertically diffuse plume is
  flagged via `UserWarning`; results should be interpreted with the
  knowledge that a single-level wind is an approximation.
* **EMG quality criteria.** Stable fits require `R² > 0.5`,
  `CV(E) <= 0.5`, and the fitted peak position within 30 km of the
  origin. The CLI prints these flags but does not refuse to write the
  result — inspect them before reporting.

---

## 9. Reference

The reference notebooks are kept in `notebooks/` and document the
exact derivation of every numerical choice in the library:

* `notebooks/temporal_fire_hotspot_detection.ipynb` — MODIS clustering pipeline
* `notebooks/nox_estimation_emg.ipynb` — EMG NO\_x flux
* `notebooks/co_estimation_csf.ipynb` — CSF CO flux
