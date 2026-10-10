"""Generate Fig 7 — Monte Carlo retrieval uncertainty (2×2 panel)."""

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "outputs/error_mc/0118"
OUT_DIR = REPO / "paper_staging/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CO_MEMBERS  = DATA_DIR / "co_error_members.csv"
NOX_MEMBERS = DATA_DIR / "nox_error_members.csv"
CO_TORNADO  = DATA_DIR / "co_error_tornado.csv"
NOX_TORNADO = DATA_DIR / "nox_error_tornado_E.csv"

for p in [CO_MEMBERS, NOX_MEMBERS, CO_TORNADO, NOX_TORNADO]:
    assert p.exists(), f"Missing file: {p}"

# ---------------------------------------------------------------------------
# Publication-style rcParams (ACS ES&T Air — two-column, ~7 in wide)
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    "font.family":        "sans-serif",
    "font.sans-serif":    ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":          8,
    "axes.labelsize":     9,
    "axes.titlesize":     9,
    "xtick.labelsize":    7.5,
    "ytick.labelsize":    7.5,
    "legend.fontsize":    7.5,
    "axes.linewidth":     0.8,
    "xtick.major.width":  0.8,
    "ytick.major.width":  0.8,
    "xtick.minor.width":  0.5,
    "ytick.minor.width":  0.5,
    "xtick.direction":    "out",
    "ytick.direction":    "out",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.dpi":         150,
    "savefig.dpi":        300,
})

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
co_df  = pd.read_csv(CO_MEMBERS)
nox_df = pd.read_csv(NOX_MEMBERS)
co_tor = pd.read_csv(CO_TORNADO)
nx_tor = pd.read_csv(NOX_TORNADO)

# Filter only converged members (non-NaN flux)
co_flux  = co_df["flux_kgs"].dropna().values
nox_flux = nox_df["E_nox_g_per_s"].dropna().values

assert len(co_flux)  == 283, f"Expected 283 CO members, got {len(co_flux)}"
assert len(nox_flux) == 279, f"Expected 279 NOx members, got {len(nox_flux)}"

# ---------------------------------------------------------------------------
# Tornado: keep top-4, nice labels
# ---------------------------------------------------------------------------
LABEL_MAP = {
    "pressure_hpa":           "Injection\npressure",
    "fire_lon":               "Fire longitude",
    "fire_lat":               "Fire latitude",
    "n_transects":            "N transects",
    "bg_x_min_km":            "BG start",
    "transect_half_width_km": "Transect width",
    "bg_width_km":            "BG width",
    "extraction_radius_km":   "Extraction\nradius",
    "downwind_max_km":        "Downwind\nmax",
    "crosswind_km":           "Crosswind",
    "grid_dy_deg":            "Grid Δy",
}

def prep_tornado(df: pd.DataFrame, top_n: int = 4) -> pd.DataFrame:
    df = df.copy()
    df["label"] = df["parameter"].map(LABEL_MAP).fillna(df["parameter"])
    df = df.sort_values("var_share_pct", ascending=False).head(top_n)
    return df.reset_index(drop=True)

co_top  = prep_tornado(co_tor,  top_n=4)
nox_top = prep_tornado(nx_tor,  top_n=4)

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
HIST_FACE  = "#c6cdd4"   # light blue-grey fill
HIST_EDGE  = "white"
C_PE       = "#2166ac"   # blue  — point estimate
C_MED      = "#000000"   # black — MC median
C_CI       = "#d73027"   # red   — 95 % CI bounds
C_BAR_CO   = "#4393c3"   # bar colour CO tornado
C_BAR_NOX  = "#d6604d"   # bar colour NOₓ tornado

# ---------------------------------------------------------------------------
# Point estimates and CI values (from CLAUDE.md canonical numbers)
# ---------------------------------------------------------------------------
CO_PE   = 779.7    # kg/s
CO_MED  = 837.9    # kg/s
CO_CI_L = 376.0    # kg/s
CO_CI_U = 1207.0   # kg/s

NOX_PE   = 7929.0  # g/s
NOX_MED  = 7331.0  # g/s
NOX_CI_L = 2878.0  # g/s  (using nb08 value)
NOX_CI_U = 9330.0  # g/s

# ---------------------------------------------------------------------------
# Helper: draw histogram panel
# ---------------------------------------------------------------------------
def hist_panel(ax, flux, unit, pe, med, ci_l, ci_u, n):
    ax.hist(flux, bins=30, color=HIST_FACE, edgecolor=HIST_EDGE,
            linewidth=0.4, zorder=2)
    ylim = ax.get_ylim()

    # Lines
    lw = 1.3
    ax.axvline(pe,   color=C_PE,  linestyle="--", linewidth=lw,
               label=f"Point est. ({pe:,.0f})", zorder=4)
    ax.axvline(med,  color=C_MED, linestyle="-",  linewidth=lw,
               label=f"MC median ({med:,.0f})", zorder=5)
    ax.axvline(ci_l, color=C_CI,  linestyle=":",  linewidth=lw,
               label=f"95% CI ({ci_l:,.0f}–{ci_u:,.0f})", zorder=4)
    ax.axvline(ci_u, color=C_CI,  linestyle=":",  linewidth=lw, zorder=4)

    ax.set_xlabel(f"Emission flux ({unit})", labelpad=3)
    ax.set_ylabel("Members (count)")

    # Annotation box (top-right, clear of the median line for NOx)
    ci_str = f"n = {n}\nMedian = {med:,.0f} {unit}\n95% CI [{ci_l:,.0f}–{ci_u:,.0f}]"
    ax.text(0.98, 0.97, ci_str, transform=ax.transAxes,
            fontsize=7, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.40", fc="white",
                      ec="#aaaaaa", lw=0.6, alpha=0.92))

    leg = ax.legend(frameon=True, framealpha=0.90, edgecolor="#aaaaaa",
                    loc="upper left", fontsize=6.8,
                    handlelength=1.8, labelspacing=0.25,
                    borderpad=0.5)
    leg.get_frame().set_linewidth(0.5)

# ---------------------------------------------------------------------------
# Helper: tornado panel
# ---------------------------------------------------------------------------
def tornado_panel(ax, df, color, species_label):
    bars = ax.barh(df["label"], df["var_share_pct"],
                   color=color, edgecolor="white", linewidth=0.4,
                   height=0.55, zorder=2)

    # Value labels inside / outside bars
    for bar, val in zip(bars, df["var_share_pct"]):
        xpos = bar.get_width()
        ha   = "left"
        xoff = 0.8
        ax.text(xpos + xoff, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", ha=ha, fontsize=7)

    ax.set_xlabel("Variance contribution (%)")
    ax.set_xlim(0, df["var_share_pct"].max() * 1.22)
    ax.invert_yaxis()
    ax.xaxis.grid(True, color="#dddddd", linewidth=0.5, zorder=1)
    ax.set_axisbelow(True)

# ---------------------------------------------------------------------------
# Build figure
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8))
(ax_co_hist, ax_nox_hist), (ax_co_tor, ax_nox_tor) = axes

# (a) CO histogram
hist_panel(ax_co_hist, co_flux, "kg/s",
           CO_PE, CO_MED, CO_CI_L, CO_CI_U, n=283)
ax_co_hist.set_title("CO flux (CSF)", pad=4)

# (b) NOₓ histogram
hist_panel(ax_nox_hist, nox_flux, "g/s",
           NOX_PE, NOX_MED, NOX_CI_L, NOX_CI_U, n=279)
ax_nox_hist.set_title("NO$_x$ flux (EMG)", pad=4)

# (c) CO tornado
tornado_panel(ax_co_tor, co_top, C_BAR_CO, "CO")
ax_co_tor.set_title("CO — sensitivity", pad=4)

# (d) NOₓ tornado
tornado_panel(ax_nox_tor, nox_top, C_BAR_NOX, "NOₓ")
ax_nox_tor.set_title("NO$_x$ — sensitivity", pad=4)

# Panel labels (a–d)
panel_labels = ["(a)", "(b)", "(c)", "(d)"]
for ax, label in zip([ax_co_hist, ax_nox_hist, ax_co_tor, ax_nox_tor],
                     panel_labels):
    ax.text(-0.13, 1.04, label, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="bottom")

fig.subplots_adjust(left=0.13, right=0.97, bottom=0.10,
                    top=0.93, hspace=0.48, wspace=0.38)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_path = OUT_DIR / "fig7_mc_uncertainty.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"Saved → {out_path}")
plt.close(fig)
