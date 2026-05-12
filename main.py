#!/usr/bin/env python3
"""Wildfire flux estimation CLI.

Three subcommands reproduce the analysis notebooks end-to-end:

    main.py cluster   MODIS active-fire detections -> daily fire centroid CSV
    main.py nox       TROPOMI NO2 + ERA5 -> EMG NOx emission flux
    main.py co        TROPOMI CO  + ERA5 -> CSF CO  emission flux

All three commands consume YAML configs from ``config/`` (defaults provided)
and write outputs under ``outputs/{YYYYMMDD}/{nox|co}/{plots,results}/``.

See README.md for required input data layout.
"""

from __future__ import annotations

from pathlib import Path

import click
import pandas as pd
import yaml

from src.fire_clustering import lookup_centroid, run_clustering
from src.pipeline import run_csf_pipeline, run_emg_pipeline


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DEFAULT = PROJECT_ROOT / "data"
OUTPUT_DEFAULT = PROJECT_ROOT / "outputs"


# --------------------------------------------------------------------- helpers

def _load_config(path: Path) -> dict:
    """Load a YAML config and fail loudly if missing or malformed."""
    if not path.exists():
        raise click.ClickException(f"Config file not found: {path}")
    with open(path, "r") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise click.ClickException(
            f"Config file {path} did not parse to a dict; got {type(cfg)}."
        )
    return cfg


def _resolve_centroid(
    fire_lat: float | None,
    fire_lon: float | None,
    centroid_csv: Path | None,
    fire_date: str,
) -> tuple[float, float, str]:
    """Resolve the fire centroid for `nox`/`co`.

    Precedence: explicit lat+lon > centroid CSV lookup. If neither is
    given (or only one of lat/lon is given) -> ClickException.

    Returns
    -------
    (fire_lat, fire_lon, source)  where source describes the resolution path.
    """
    if (fire_lat is None) != (fire_lon is None):
        raise click.UsageError(
            "Both --fire-lat and --fire-lon must be supplied together."
        )
    if fire_lat is not None:
        return float(fire_lat), float(fire_lon), "cli flags"
    if centroid_csv is not None:
        try:
            lat, lon = lookup_centroid(centroid_csv, pd.Timestamp(fire_date))
        except (FileNotFoundError, KeyError, ValueError) as exc:
            raise click.ClickException(str(exc))
        return float(lat), float(lon), f"centroid CSV ({centroid_csv.name})"
    raise click.UsageError(
        "Must provide either (--fire-lat AND --fire-lon) or --centroid-csv."
    )


def _print_emg_summary(res: dict) -> None:
    fit = res["emg_fit"]
    inj = res["injection_pressure"]
    wind = res["wind_at_injection"]
    click.echo("")
    click.echo("=" * 64)
    click.echo(f"NOx EMG result  --  {res['fire_date']}  "
               f"(fire {res['fire_lat']:.4f}, {res['fire_lon']:.4f})")
    click.echo("=" * 64)
    click.echo(f"  Injection pressure   : {inj['selected_hpa']:.0f} hPa "
               f"(IQR {inj['iqr_hpa']:.0f} hPa, n_alh={inj['n_alh_pixels']})")
    click.echo(f"  Wind                 : {wind['speed_m_s']:.2f} m/s @ "
               f"{wind['theta_deg_ccw_east']:.1f} deg CCW East")
    click.echo(f"  EMG bootstrap        : {fit['n_converged']}/"
               f"{fit['n_attempted']} fits converged")
    click.echo(f"  R^2                  : {fit['r2']:.3f}")
    click.echo(f"  tau (NOx lifetime)   : {fit['tau_h']:.2f} h")
    click.echo(f"  E (NOx)              : {fit['E_nox_g_per_s']:.0f} g/s "
               f"({fit['E_nox_t_per_day']:.0f} t/day)")
    click.echo(f"  CV(a)={fit['cv_a']:.3f}  CV(E)={fit['cv_E']:.3f}  "
               f"stable={fit['fit_stable']}")
    click.echo(f"  Peak offset          : {fit['peak_x_km']:+.1f} km  "
               f"(<=30 km: {fit['peak_within_30km']})")
    click.echo("=" * 64)


def _print_csf_summary(res: dict) -> None:
    csf = res["csf"]
    inj = res["injection_pressure"]
    wind = res["wind_at_injection"]
    bg = res["background"]
    click.echo("")
    click.echo("=" * 64)
    click.echo(f"CO CSF result  --  {res['fire_date']}  "
               f"(fire {res['fire_lat']:.4f}, {res['fire_lon']:.4f})")
    click.echo("=" * 64)
    click.echo(f"  Injection pressure   : {inj['selected_hpa']:.0f} hPa "
               f"(IQR {inj['iqr_hpa']:.0f} hPa, n_alh={inj['n_alh_pixels']})")
    click.echo(f"  Wind (AER_LH)        : {wind['speed_m_s']:.2f} m/s @ "
               f"{wind['theta_deg_ccw_east']:.1f} deg CCW East")
    click.echo(f"  Background CO        : {bg['bg_co_mmol_m2']:.4f} mmol/m^2 "
               f"({bg['source']})")
    click.echo(f"  Valid transects      : {csf['n_valid_transects']}/"
               f"{csf['n_transects']}  (half-width {csf['transect_half_width_km']:.0f} km)")
    click.echo(f"  Mean CO flux         : {csf['flux_mean_kg_s']:.2f} kg/s  "
               f"({csf['flux_mean_t_per_day']:.0f} t/day, "
               f"{csf['flux_mean_Gg_per_yr']:.1f} Gg/yr)")
    click.echo("=" * 64)


# --------------------------------------------------------------------- CLI

@click.group()
@click.version_option("0.1.0", prog_name="wildfire-flux")
def cli() -> None:
    """TROPOMI-based fire flux estimation (EMG for NOx, CSF for CO)."""


# ----- cluster ------------------------------------------------------------

@cli.command("cluster")
@click.option("--modis-csv", required=True, type=click.Path(
    exists=True, dir_okay=False, path_type=Path),
    help="Raw FIRMS MCD14DL CSV download (e.g. fire_nrt_M-C61_*.csv).")
@click.option("--start", "date_start", required=True, type=str,
    help="First date (YYYY-MM-DD).")
@click.option("--end", "date_end", required=True, type=str,
    help="Last date (YYYY-MM-DD), inclusive.")
@click.option("--output", "output_csv", required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Path to write the centroid CSV (consumed by nox / co commands).")
@click.option("--config", "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=str(CONFIG_DIR / "default_cluster.yaml"), show_default=True,
    help="YAML config (override CLUSTER defaults).")
def cluster_cmd(
    modis_csv: Path, date_start: str, date_end: str,
    output_csv: Path, config_path: Path,
) -> None:
    """Cluster MODIS detections and produce a daily centroid CSV."""
    cfg = _load_config(config_path)
    out = run_clustering(
        modis_csv,
        pd.Timestamp(date_start), pd.Timestamp(date_end),
        cfg,
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False, float_format="%.4f")
    click.echo(f"\nWrote {len(out)} centroid rows -> {output_csv}")
    click.echo(out.to_string(index=False))


# ----- nox ----------------------------------------------------------------

@cli.command("nox")
@click.option("--date", "fire_date", required=True, type=str,
    help="Fire date YYYY-MM-DD (matches data/{MMDD}/ subdirectory).")
@click.option("--centroid-csv", type=click.Path(
    exists=True, dir_okay=False, path_type=Path),
    help="CSV from `main.py cluster`. Mutually substitutable with --fire-lat/--fire-lon.")
@click.option("--fire-lat", type=float,
    help="Fire centroid latitude (deg N). Use INSTEAD of --centroid-csv.")
@click.option("--fire-lon", type=float,
    help="Fire centroid longitude (deg E). Use INSTEAD of --centroid-csv.")
@click.option("--config", "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=str(CONFIG_DIR / "default_emg.yaml"), show_default=True)
@click.option("--data-root", type=click.Path(
    exists=True, file_okay=False, path_type=Path),
    default=str(DATA_DEFAULT), show_default=True,
    help="Root of TROPOMI + ERA5 data (expects data/{MMDD}/ and ERA5_*.nc).")
@click.option("--output-root", type=click.Path(
    file_okay=False, path_type=Path),
    default=str(OUTPUT_DEFAULT), show_default=True,
    help="Root of outputs (per-date subdirs created automatically).")
def nox_cmd(
    fire_date: str,
    centroid_csv: Path | None,
    fire_lat: float | None, fire_lon: float | None,
    config_path: Path, data_root: Path, output_root: Path,
) -> None:
    """Run the EMG NOx pipeline for one fire date."""
    cfg = _load_config(config_path)
    lat, lon, src = _resolve_centroid(fire_lat, fire_lon, centroid_csv, fire_date)
    click.echo(f"Fire centroid resolved from {src}: ({lat:.4f}, {lon:.4f})")
    yyyymmdd = "".join(fire_date.split("-"))
    out_dir = output_root / yyyymmdd / "nox"
    res = run_emg_pipeline(
        fire_date=fire_date, fire_lat=lat, fire_lon=lon,
        config=cfg, data_root=data_root, output_root=out_dir,
    )
    _print_emg_summary(res)
    click.echo(f"\nOutputs -> {out_dir}")


# ----- co -----------------------------------------------------------------

@cli.command("co")
@click.option("--date", "fire_date", required=True, type=str,
    help="Fire date YYYY-MM-DD (matches data/{MMDD}/ subdirectory).")
@click.option("--centroid-csv", type=click.Path(
    exists=True, dir_okay=False, path_type=Path),
    help="CSV from `main.py cluster`. Mutually substitutable with --fire-lat/--fire-lon.")
@click.option("--fire-lat", type=float,
    help="Fire centroid latitude (deg N). Use INSTEAD of --centroid-csv.")
@click.option("--fire-lon", type=float,
    help="Fire centroid longitude (deg E). Use INSTEAD of --centroid-csv.")
@click.option("--config", "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=str(CONFIG_DIR / "default_csf.yaml"), show_default=True)
@click.option("--data-root", type=click.Path(
    exists=True, file_okay=False, path_type=Path),
    default=str(DATA_DEFAULT), show_default=True)
@click.option("--output-root", type=click.Path(
    file_okay=False, path_type=Path),
    default=str(OUTPUT_DEFAULT), show_default=True)
def co_cmd(
    fire_date: str,
    centroid_csv: Path | None,
    fire_lat: float | None, fire_lon: float | None,
    config_path: Path, data_root: Path, output_root: Path,
) -> None:
    """Run the CSF CO pipeline for one fire date (AER_LH wind variant)."""
    cfg = _load_config(config_path)
    lat, lon, src = _resolve_centroid(fire_lat, fire_lon, centroid_csv, fire_date)
    click.echo(f"Fire centroid resolved from {src}: ({lat:.4f}, {lon:.4f})")
    yyyymmdd = "".join(fire_date.split("-"))
    out_dir = output_root / yyyymmdd / "co"
    res = run_csf_pipeline(
        fire_date=fire_date, fire_lat=lat, fire_lon=lon,
        config=cfg, data_root=data_root, output_root=out_dir,
    )
    _print_csf_summary(res)
    click.echo(f"\nOutputs -> {out_dir}")


if __name__ == "__main__":
    cli()
