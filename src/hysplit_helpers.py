"""HYSPLIT-CONC forward-run helpers for the emission forward-validation notebook.

Stages a HYSPLIT concentration run driven by a retrieved emission rate, runs the
(x86_64-under-Rosetta) ``hycs_std`` binary, and reads the binary ``cdump`` into
xarray via ``monetio``.

Environment notes
-----------------
* ``hycs_std`` is HYSPLIT v5.1.0 (x86_64); on Apple Silicon it runs under
  Rosetta. Its x86_64 gfortran runtime lives at
  ``<root>/stilt_project/lib_x86_64`` (absolute rpath in the binary).
* ``monetio`` 0.3.2 predates NumPy 2.0 and calls the removed
  ``ndarray.newbyteorder``. ``patch_monetio_numpy2`` swaps in a NumPy-2-correct
  ``ModelBin.parse_hdata8`` at runtime (no site-packages mutation), so the patch
  travels with this repo and is reproducible.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Static HYSPLIT support files required in every run directory.
_STATIC_FILES = ("ASCDATA.CFG", "LANDUSE.ASC", "ROUGLEN.ASC", "TERRAIN.ASC")


# ──────────────────────────────────────────────────────────────────────────────
# monetio NumPy-2.0 compatibility patch
# ──────────────────────────────────────────────────────────────────────────────
_PATCHED = False


def patch_monetio_numpy2() -> None:
    """Make monetio 0.3.2 read cdump under NumPy >= 2.0 (idempotent).

    Replaces ``ModelBin.parse_hdata8`` with a copy that uses
    ``arr.view(arr.dtype.newbyteorder())`` instead of the removed
    ``arr.newbyteorder()``. Vendored verbatim from monetio 0.3.2 except for that
    one line.
    """
    global _PATCHED
    if _PATCHED:
        return
    import datetime as _dt
    import monetio.models.hysplit as hy

    def _parse_hdata6and7_np2(self, hdata6, hdata7, century):
        # NumPy 2.0: empty structured arrays are not falsy -> test .size explicitly.
        if hdata6 is None or np.size(hdata6) == 0:
            return False, None, None
        pdate1 = _dt.datetime(century + int(hdata6["oyear"][0]), int(hdata6["omonth"][0]),
                              int(hdata6["oday"][0]), int(hdata6["ohr"][0]), int(hdata6["omin"][0]))
        pdate2 = _dt.datetime(century + int(hdata7["oyear"][0]), int(hdata7["omonth"][0]),
                              int(hdata7["oday"][0]), int(hdata7["ohr"][0]), int(hdata7["omin"][0]))
        dt = pdate2 - pdate1
        self.atthash["sample time hours"] = dt.days * 24 + dt.seconds / 3600.0
        self.atthash["time description"] = (
            "End of sampling time period" if self.sample_time_stamp == "end"
            else "start of sampling time period")
        return True, pdate1, pdate2

    @staticmethod
    def _parse_hdata8_np2(hdata8a, hdata8b, pdate1):
        lev_name = hdata8a["lev"][0]
        col_name = hdata8a["poll"][0].decode("UTF-8")
        edata = hdata8b.byteswap().view(hdata8b.dtype.newbyteorder())  # NumPy 2.0 API
        concframe = pd.DataFrame.from_records(edata)
        concframe["levels"] = lev_name
        concframe["time"] = pdate1
        names = concframe.columns.values
        names = ["y" if x == "jndx" else x for x in names]
        names = ["x" if x == "indx" else x for x in names]
        names = ["z" if x == "levels" else x for x in names]
        concframe.columns = names
        concframe.set_index(["time", "z", "y", "x"], inplace=True)
        concframe.rename(columns={"conc": col_name}, inplace=True)
        return concframe

    hy.ModelBin.parse_hdata6and7 = _parse_hdata6and7_np2
    hy.ModelBin.parse_hdata8 = _parse_hdata8_np2
    _PATCHED = True


def read_cdump(cdump_path: Path):
    """Read a HYSPLIT binary ``cdump`` into an xarray Dataset (via monetio)."""
    cdump_path = Path(cdump_path)
    if not cdump_path.exists():
        raise FileNotFoundError(f"cdump not found: {cdump_path}")
    patch_monetio_numpy2()
    import monetio.models.hysplit as hy
    return hy.open_dataset(str(cdump_path))


# ──────────────────────────────────────────────────────────────────────────────
# CONTROL / SETUP writers
# ──────────────────────────────────────────────────────────────────────────────
def write_control(
    run_dir: Path,
    start: datetime,
    run_hours: int,
    src_lat: float,
    src_lon: float,
    src_height_m: float,
    pollutant_id: str,
    emission_rate_per_hr: float,
    emission_hours: float,
    met_dir: Path,
    met_files: list[str],
    grid_spacing: tuple[float, float],
    grid_span: tuple[float, float],
    out_name: str,
    level_top_m: float,
    sample_interval_h: int,
    top_of_model_m: float = 25000.0,
) -> Path:
    """Write a HYSPLIT CONTROL file for a single-source concentration run.

    Emission is constant ``emission_rate_per_hr`` (mass/hr) for the first
    ``emission_hours`` from ``start``. Concentration is accumulated on one
    source-centred output grid (HYSPLIT centres the grid on the release point
    when the centre line is ``0.0 0.0``), one vertical layer (0..level_top_m),
    sampled every ``sample_interval_h`` hours (averaged).
    """
    met_dir = Path(met_dir)
    for mf in met_files:
        if not (met_dir / mf).exists():
            raise FileNotFoundError(f"Met file missing: {met_dir / mf}")

    L = []
    L.append(f"{start.year % 100:02d} {start.month:02d} {start.day:02d} {start.hour:02d} {start.minute:02d}")
    L.append("1")
    L.append(f"{src_lat:.4f} {src_lon:.4f} {src_height_m:.1f}")
    L.append(f"{int(run_hours)}")
    L.append("0")                       # vertical motion: 0 = use met data
    L.append(f"{top_of_model_m:.1f}")
    L.append(f"{len(met_files)}")
    for mf in met_files:
        L.append(f"{met_dir}/")
        L.append(mf)
    # pollutant
    L.append("1")
    L.append(pollutant_id[:4])
    L.append(f"{emission_rate_per_hr:.6e}")
    L.append(f"{emission_hours:.1f}")
    L.append("00 00 00 00 00")          # release start = run start
    # concentration grid (source-centred: HYSPLIT centres on the release point)
    L.append("1")
    L.append("0.0 0.0")
    L.append(f"{grid_spacing[0]:.3f} {grid_spacing[1]:.3f}")
    L.append(f"{grid_span[0]:.2f} {grid_span[1]:.2f}")
    L.append("./")
    L.append(out_name)
    L.append("1")                       # number of vertical levels
    L.append(f"{int(level_top_m)}")
    L.append("00 00 00 00 00")          # sample start
    L.append("00 00 00 00 00")          # sample stop
    L.append(f"00 {int(sample_interval_h):02d} 00")  # sample interval (type hr min) -> averaged
    # deposition (none: passive tracer)
    L.append("1")
    L.append("0.0 0.0 0.0")
    L.append("0.0 0.0 0.0 0.0 0.0")
    L.append("0.0 0.0 0.0")
    L.append("0.0")
    L.append("0.0")

    ctrl = run_dir / "CONTROL"
    ctrl.write_text("\n".join(L) + "\n")
    return ctrl


def write_setup_cfg(run_dir: Path, numpar: int = 5000, maxpar: int = 500000) -> Path:
    """Write a minimal SETUP.CFG (passive tracer, no chemistry)."""
    cfg = (
        " &SETUP\n"
        f" numpar = {numpar},\n"
        f" maxpar = {maxpar},\n"
        " initd = 0,\n"
        " kpuff = 0,\n"
        " ichem = 0,\n"
        " kmsl = 0,\n"
        " /\n"
    )
    p = run_dir / "SETUP.CFG"
    p.write_text(cfg)
    return p


# ──────────────────────────────────────────────────────────────────────────────
# Stage + run
# ──────────────────────────────────────────────────────────────────────────────
def stage_run_dir(run_dir: Path, hysplit_bin: Path, static_src: Path) -> None:
    """Create ``run_dir`` and copy the binary + static support files into it."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    hysplit_bin = Path(hysplit_bin)
    if not hysplit_bin.exists():
        raise FileNotFoundError(f"hycs_std not found: {hysplit_bin}")
    shutil.copy2(hysplit_bin, run_dir / "hycs_std")
    os.chmod(run_dir / "hycs_std", 0o755)
    for f in _STATIC_FILES:
        src = Path(static_src) / f
        if not src.exists():
            raise FileNotFoundError(f"Static HYSPLIT file missing: {src}")
        shutil.copy2(src, run_dir / f)


def run_hysplit(run_dir: Path, x86_64: bool = True, timeout_s: int = 1800) -> str:
    """Run ``hycs_std`` in ``run_dir`` (Rosetta on Apple Silicon). Returns MESSAGE tail.

    Raises RuntimeError if the run fails or no cdump is produced.
    """
    run_dir = Path(run_dir)
    cmd = (["arch", "-x86_64"] if x86_64 else []) + ["./hycs_std"]
    proc = subprocess.run(cmd, cwd=run_dir, stdin=subprocess.DEVNULL,
                          capture_output=True, text=True, timeout=timeout_s)
    msg = (run_dir / "MESSAGE").read_text()[-2000:] if (run_dir / "MESSAGE").exists() else ""
    cdumps = list(run_dir.glob("cdump*"))
    if not cdumps:
        raise RuntimeError(
            f"HYSPLIT produced no cdump.\nstdout:\n{proc.stdout[-1500:]}\n"
            f"MESSAGE tail:\n{msg[-1500:]}"
        )
    return msg


# ──────────────────────────────────────────────────────────────────────────────
# cdump -> column
# ──────────────────────────────────────────────────────────────────────────────
def cdump_to_column(ds, species: str, level_thickness_m: float):
    """Convert a single-layer cdump concentration (mass/m3) to a column burden.

    Returns an xarray DataArray (time, y, x) of column mass per m2
    (= layer-mean concentration x layer thickness).
    """
    # HYSPLIT pads pollutant IDs to 4 chars (e.g. 'CO  '); match on stripped name.
    match = [v for v in ds.data_vars if str(v).strip() == species.strip()]
    if not match:
        raise KeyError(
            f"Species {species!r} not in cdump vars {[str(v) for v in ds.data_vars]}")
    da = ds[match[0]]
    # collapse the single vertical level if present
    for zdim in ("z", "levels"):
        if zdim in da.dims:
            da = da.sum(dim=zdim)
    return da * float(level_thickness_m)
