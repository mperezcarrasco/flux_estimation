"""Diagnostic plots for the EMG (NOx) and CSF (CO) pipelines.

All functions take an explicit output Path and write a 150 dpi PNG.
Each returns the Path that was written (for downstream logging).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


# --------------------------------------------------------------------- NOx

def plot_nox_scene(
    lat_domain: np.ndarray,
    lon_domain: np.ndarray,
    no2_domain: np.ndarray,
    fire_lat: float,
    fire_lon: float,
    extraction_radius_km: float,
    output_path: Path,
    m_per_deg_lat: float = 111_320.0,
) -> Path:
    """Scene scatter: NO2 column + fire centroid + extraction radius."""
    fig, ax = plt.subplots(figsize=(8, 7))
    pos = no2_domain > 0
    if pos.any():
        vmin, vmax = np.nanpercentile(no2_domain[pos] * 1e6, (1, 99))
    else:
        vmin, vmax = 0, 25
    ax.scatter(lon_domain, lat_domain, c=no2_domain * 1e6,
               vmin=vmin, vmax=vmax, cmap="YlOrRd",
               alpha=0.4, s=0.3, rasterized=True)
    theta_c = np.linspace(0, 2 * np.pi, 360)
    r_lat = extraction_radius_km / (m_per_deg_lat / 1000.0)
    r_lon = extraction_radius_km / (
        (m_per_deg_lat / 1000.0) * np.cos(np.radians(fire_lat))
    )
    ax.plot(fire_lon + r_lon * np.cos(theta_c),
            fire_lat + r_lat * np.sin(theta_c),
            "b--", alpha=0.4, lw=1,
            label=f"{extraction_radius_km:.0f} km radius")
    ax.scatter(fire_lon, fire_lat, marker="x", s=200, c="green",
               linewidths=2,
               label=f"Fire: {fire_lat:.3f}N, {fire_lon:.3f}E")
    ax.set(xlabel="Longitude (deg E)", ylabel="Latitude (deg N)",
           title="TROPOMI NO2 + fire centroid")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_nox_line_density(
    x_centres: np.ndarray,
    profile_mol_per_m: np.ndarray,
    output_path: Path,
) -> Path:
    """Cross-wind-integrated NO2 vs downwind distance."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x_centres, profile_mol_per_m * 1e6, "C0-")
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5, ls="--")
    ax.set(xlabel="Downwind distance (km)",
           ylabel="NO2 line density (umol m^-1)",
           title="NO2 line density profile")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_emg_fit(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    y_pred: np.ndarray,
    r2: float,
    tau_h: float,
    E_nox_g_s: float,
    output_path: Path,
) -> Path:
    """Per-bootstrap-median EMG fit overlaid on observed line density."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x_fit, y_fit, "o", ms=3, color="darkred", alpha=0.7,
            label="Observed (cross-wind sum * dy)")
    ax.plot(x_fit, y_pred, "-", lw=2, color="navy",
            label=f"EMG fit  (R^2 = {r2:.3f})")
    ax.axvline(0, color="blue", lw=0.8, ls=":", label="Fire centre")
    ax.set(
        title=f"EMG fit  --  tau = {tau_h:.2f} h, "
              f"E(NOx) = {E_nox_g_s:.0f} g s^-1",
        xlabel="Downwind x_rot (km)",
        ylabel="NO2 line density (mol m^-1)",
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


# --------------------------------------------------------------------- CO

def plot_co_scene(
    lat_domain: np.ndarray,
    lon_domain: np.ndarray,
    co_domain: np.ndarray,
    fire_lat: float,
    fire_lon: float,
    extraction_radius_km: float,
    output_path: Path,
    m_per_deg_lat: float = 111_320.0,
) -> Path:
    """Scene scatter: CO column + fire centroid + extraction radius."""
    fig, ax = plt.subplots(figsize=(8, 7))
    pos = co_domain > 0
    if pos.any():
        vmin, vmax = np.nanpercentile(co_domain[pos] * 1e6, (1, 99))
    else:
        vmin, vmax = 0, 25
    ax.scatter(lon_domain, lat_domain, c=co_domain * 1e6,
               vmin=vmin, vmax=vmax, cmap="YlOrRd",
               alpha=0.4, s=0.3, rasterized=True)
    theta_c = np.linspace(0, 2 * np.pi, 360)
    r_lat = extraction_radius_km / (m_per_deg_lat / 1000.0)
    r_lon = extraction_radius_km / (
        (m_per_deg_lat / 1000.0) * np.cos(np.radians(fire_lat))
    )
    ax.plot(fire_lon + r_lon * np.cos(theta_c),
            fire_lat + r_lat * np.sin(theta_c),
            "b--", alpha=0.4, lw=1,
            label=f"{extraction_radius_km:.0f} km radius")
    ax.scatter(fire_lon, fire_lat, marker="x", s=200, c="green",
               linewidths=2,
               label=f"Fire: {fire_lat:.3f}N, {fire_lon:.3f}E")
    ax.set(xlabel="Longitude (deg E)", ylabel="Latitude (deg N)",
           title="TROPOMI CO + fire centroid")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_csf_diagnostic(
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    co_grid_mol_m2: np.ndarray,
    plume_mask: np.ndarray,
    transects: list,
    transect_fluxes_kg_s: np.ndarray,
    slab_pos_m: np.ndarray,
    flux_mean_kg_s: float,
    fire_lat: float,
    fire_lon: float,
    inj_pres_hpa: float,
    wind_speed_m_s: float,
    bg_co_mol_m2: float,
    fire_date: str,
    n_transects: int,
    output_path: Path,
) -> Path:
    """2x2 panel diagnostic for the AER_LH-wind CSF run.

    Panels:
      [0,0] CO column + plume mask + transect lines
      [0,1] (empty / metadata text panel)
      [1,0] Per-transect flux vs downwind position
      [1,1] Mean flux bar chart
    """
    slab_cmap = plt.cm.plasma
    slab_colors = [slab_cmap(k / max(n_transects - 1, 1))
                   for k in range(n_transects)]

    co_mmol = np.where(np.isfinite(co_grid_mol_m2), co_grid_mol_m2 * 1e3, np.nan)
    finite = np.isfinite(co_mmol)
    vmax_co = float(np.nanpercentile(co_mmol[finite], 98)) if finite.any() else 1.0

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ── [0, 0]: CO column + AI plume + transects ──────────────────────
    ax = axes[0, 0]
    im = ax.pcolormesh(
        lon_grid, lat_grid, co_mmol,
        cmap="inferno", norm=mcolors.Normalize(0, vmax_co), shading="auto",
    )
    if plume_mask.any():
        ax.contour(
            lon_grid, lat_grid, plume_mask.astype(np.float32),
            levels=[0.5], colors="lime", linewidths=1.5,
        )
    fig.colorbar(im, ax=ax, extend="max", pad=0.02, label="CO (mmol m^-2)")
    for k, t in enumerate(transects):
        ax.plot(t.positions_lon, t.positions_lat,
                "-", color=slab_colors[k], lw=2.2, alpha=0.9)
        mid = len(t.positions_lat) // 2
        ax.plot(t.positions_lon[mid], t.positions_lat[mid],
                "o", color=slab_colors[k], ms=6, zorder=5)
    ax.plot(fire_lon, fire_lat, "r*", ms=10, zorder=6, label="Fire centre")
    ax.legend(loc="upper right", fontsize=8)
    ax.set(
        xlabel="Longitude (deg)", ylabel="Latitude (deg)",
        title=f"CO + plume mask + {n_transects} transects "
              f"(AER_LH wind {inj_pres_hpa:.0f} hPa, {wind_speed_m_s:.2f} m/s)",
    )

    # ── [0, 1]: metadata text ─────────────────────────────────────────
    ax_meta = axes[0, 1]
    ax_meta.axis("off")
    text = (
        f"Fire date           {fire_date}\n"
        f"Fire centroid       {fire_lat:.4f} N, {fire_lon:.4f} E\n"
        f"Injection pressure  {inj_pres_hpa:.0f} hPa\n"
        f"Wind speed          {wind_speed_m_s:.2f} m s^-1\n"
        f"Background CO       {bg_co_mol_m2 * 1e3:.4f} mmol m^-2\n"
        f"\n"
        f"Mean CO flux        {flux_mean_kg_s:.2f} kg s^-1\n"
        f"  ({flux_mean_kg_s * 86400 / 1000:.0f} t day^-1)\n"
        f"\n"
        f"Valid transects     "
        f"{int(np.sum(np.isfinite(transect_fluxes_kg_s)))}/{n_transects}"
    )
    ax_meta.text(
        0.02, 0.98, text, transform=ax_meta.transAxes,
        fontsize=11, family="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", fc="#f6f6f6", ec="#cccccc"),
    )

    # ── [1, 0]: per-transect flux ─────────────────────────────────────
    ax2 = axes[1, 0]
    pos_km = slab_pos_m / 1e3
    valid = np.isfinite(transect_fluxes_kg_s)
    ax2.scatter(pos_km[valid], transect_fluxes_kg_s[valid],
                color="#e05c00", s=80, zorder=4, alpha=0.85)
    ax2.plot(pos_km[valid], transect_fluxes_kg_s[valid],
             color="#e05c00", lw=0.9, alpha=0.5, zorder=3)
    for k in np.where(~valid)[0]:
        ax2.axvline(pos_km[k], color="#cccccc", lw=0.8, ls=":")
    ax2.axhline(flux_mean_kg_s, color="#e05c00", lw=1.5, ls="--",
                label=f"Mean = {flux_mean_kg_s:.2f} kg/s "
                      f"(n={int(valid.sum())}/{n_transects})")
    ax2.axhline(0, color="k", lw=0.5, ls=":")
    ax2.set(xlabel="Downwind position (km)", ylabel="Flux (kg s^-1)",
            title="Per-transect CO flux")
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    # ── [1, 1]: bar chart ─────────────────────────────────────────────
    ax_bar = axes[1, 1]
    bar = ax_bar.bar(["CO flux\n(AER_LH wind)"], [flux_mean_kg_s],
                     color="#e05c00", alpha=0.85, width=0.4)
    for b in bar:
        ax_bar.text(b.get_x() + b.get_width() / 2.0,
                    b.get_height() + flux_mean_kg_s * 0.02,
                    f"{flux_mean_kg_s:.1f} kg/s",
                    ha="center", va="bottom",
                    fontweight="bold", fontsize=12)
    ax_bar.set(ylabel="CO flux (kg s^-1)", title="Mean CO emission rate")
    ax_bar.grid(axis="y", alpha=0.3)
    ax_bar.set_ylim(bottom=0)

    fig.suptitle(
        f"CO Cross-Sectional Flux  --  {fire_date}\n"
        f"Change D background, AER_LH wind ({inj_pres_hpa:.0f} hPa)",
        fontweight="bold", fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
