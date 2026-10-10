# TROPOMI L2 loading helpers
import h5py
import numpy as np
import pandas as pd

TROPOMI_FILL: float    = 9.96e36

def _read_corners(f: h5py.File, n_pixels: int) -> tuple[np.ndarray, np.ndarray]:
    """Pixel footprint corners ((n_pixels, 4) float64 deg) from an open TROPOMI L2 file."""
    geo = f["PRODUCT/SUPPORT_DATA/GEOLOCATIONS"]
    lat_b = geo["latitude_bounds"][0].astype(np.float64).reshape(-1, 4)
    lon_b = geo["longitude_bounds"][0].astype(np.float64).reshape(-1, 4)
    assert lat_b.shape[0] == n_pixels, f"corner/centre size mismatch: {lat_b.shape[0]} vs {n_pixels}"
    return lat_b, lon_b

def load_co_l2(filepath,
                lat_min: float = -47.0,
                lat_max: float = -26.0,
                lon_min: float = -82.0,
                lon_max: float = -62.0,
                qa_threshold: float = 50,
                *, with_corners: bool = False):
    """Load TROPOMI L2 CO (mol m⁻²), filtered to the domain and qa_value >= qa_threshold.

    with_corners=True also returns the pixel footprint corners ``lat_b``/``lon_b``
    ((n, 4) deg, PRODUCT/SUPPORT_DATA/GEOLOCATIONS), needed by src.oversampling.
    """
    with h5py.File(filepath, "r") as f:
        co_raw_2d = f["PRODUCT/carbonmonoxide_total_column"][0].astype(np.float64)
        qa_raw = f["PRODUCT/qa_value"][0].astype(np.int32).ravel()
        lat_f  = f["PRODUCT/latitude"][0].astype(np.float32).ravel()
        lon_f  = f["PRODUCT/longitude"][0].astype(np.float32).ravel()
        time_utc_scan = f["PRODUCT/time_utc"][0][:]  # (n_scan,) one timestamp per scanline
        if with_corners:
            lat_b, lon_b = _read_corners(f, lat_f.size)
    n_scan, n_pix = co_raw_2d.shape
    co_raw = co_raw_2d.ravel()
    # Expand per-scanline time to per-pixel *before* filtering, so it stays
    # positionally aligned with lat/lon/co through the `valid` mask below —
    # downstream code can then index `time` directly by pixel, with no
    # scanline arithmetic (which breaks once orbits are concatenated).
    time_utc_px = np.repeat(time_utc_scan, n_pix)
    co_raw[co_raw >= TROPOMI_FILL] = np.nan
    valid = ((lat_f >= lat_min) & (lat_f <= lat_max)
             & (lon_f >= lon_min) & (lon_f <= lon_max)
             & (qa_raw >= qa_threshold) & np.isfinite(co_raw))
    assert valid.any(), f"No valid L2 CO pixels in domain (QA >= {qa_threshold})."
    out = dict(co=co_raw[valid].astype(np.float32),
               lat=lat_f[valid], lon=lon_f[valid], source="l2_swath",
               time=time_utc_px[valid])
    if with_corners:
        out.update(lat_b=lat_b[valid], lon_b=lon_b[valid])
    return out

def load_no2_l2(filepath,
                lat_min: float = -47.0,
                lat_max: float = -26.0,
                lon_min: float = -82.0,
                lon_max: float = -62.0,
                qa_threshold: float = 50):
    with h5py.File(filepath, "r") as f:
        no2_raw_2d = f["PRODUCT/nitrogendioxide_tropospheric_column"][0].astype(np.float64)
        qa_raw = f["PRODUCT/qa_value"][0].astype(np.int32).ravel()
        lat_f  = f["PRODUCT/latitude"][0].astype(np.float32).ravel()
        lon_f  = f["PRODUCT/longitude"][0].astype(np.float32).ravel()
        time_utc_scan = f["PRODUCT/time_utc"][0][:]  # (n_scan,) one timestamp per scanline
    n_scan, n_pix = no2_raw_2d.shape
    no2_raw = no2_raw_2d.ravel()
    # Expand per-scanline time to per-pixel *before* filtering — see load_co_l2.
    time_utc_px = np.repeat(time_utc_scan, n_pix)
    no2_raw[no2_raw >= TROPOMI_FILL] = np.nan
    valid = ((lat_f >= lat_min) & (lat_f <= lat_max)
             & (lon_f >= lon_min) & (lon_f <= lon_max)
             & (qa_raw >= qa_threshold) & np.isfinite(no2_raw) & (no2_raw < 1.0))
    assert valid.any(), f"No valid L2 NO2 pixels in domain (QA >= {qa_threshold})."
    return dict(no2=no2_raw[valid].astype(np.float32),
                lat=lat_f[valid], lon=lon_f[valid], source="l2_swath",
                time=time_utc_px[valid])

def load_aer_ai(filepath,
                lat_min: float = -47.0,
                lat_max: float = -26.0,
                lon_min: float = -82.0,
                lon_max: float = -62.0,
                qa_threshold: float = 50,
                *, with_corners: bool = False):
    """Load TROPOMI L2 AER_AI (354/388 nm), filtered to the domain and qa_value >= qa_threshold.

    with_corners=True also returns the pixel footprint corners ``lat_b``/``lon_b``
    ((n, 4) deg, PRODUCT/SUPPORT_DATA/GEOLOCATIONS), needed by src.oversampling.
    """
    with h5py.File(filepath, "r") as f:
        ai_raw = f["PRODUCT/aerosol_index_354_388"][0].astype(np.float32).ravel()
        qa_raw = f["PRODUCT/qa_value"][0].astype(np.int32).ravel()
        lat_f  = f["PRODUCT/latitude"][0].astype(np.float32).ravel()
        lon_f  = f["PRODUCT/longitude"][0].astype(np.float32).ravel()
        if with_corners:
            lat_b, lon_b = _read_corners(f, lat_f.size)
    ai_raw[ai_raw >= TROPOMI_FILL] = np.nan
    valid = ((lat_f >= lat_min) & (lat_f <= lat_max)
             & (lon_f >= lon_min) & (lon_f <= lon_max)
             & (qa_raw >= qa_threshold) & np.isfinite(ai_raw))
    assert valid.any(), "No valid AER_AI pixels in domain."
    out = dict(ai=ai_raw[valid], lat=lat_f[valid], lon=lon_f[valid])
    if with_corners:
        out.update(lat_b=lat_b[valid], lon_b=lon_b[valid])
    return out

def load_aer_lh(filepath,
                lat_min: float = -47.0,
                lat_max: float = -26.0,
                lon_min: float = -82.0,
                lon_max: float = -62.0,
                qa_threshold: float = 50):
    with h5py.File(filepath, "r") as f:
        alh_pres_raw = f["PRODUCT/aerosol_mid_pressure"][0].astype(np.float32).ravel()  # (4172, 448) Pa
        alh_ht_raw   = f["PRODUCT/aerosol_mid_height"][0].astype(np.float32).ravel()    # (4172, 448) m
        qa_raw = f["PRODUCT/qa_value"][0].astype(np.float32).ravel()    # (4172, 448)
        lat_f  = f["PRODUCT/latitude"][0].astype(np.float32).ravel()               # (4172, 448)
        lon_f  = f["PRODUCT/longitude"][0].astype(np.float32).ravel()              # (4172, 448)
    alh_pres_raw[alh_pres_raw >= TROPOMI_FILL] = np.nan
    alh_ht_raw[alh_ht_raw >= TROPOMI_FILL] = np.nan
    valid = ((lat_f >= lat_min) & (lat_f <= lat_max)
             & (lon_f >= lon_min) & (lon_f <= lon_max)
             & (qa_raw >= qa_threshold) & np.isfinite(alh_pres_raw) & np.isfinite(alh_ht_raw))
    assert valid.any(), "No valid AER_LH pixels in domain."
    return dict(alh_pres=alh_pres_raw[valid], alh_ht=alh_ht_raw[valid], lat=lat_f[valid], lon=lon_f[valid])

def get_modis_hotspot(hotspot_csv,
                    fire_date,
                    lat_min: float = -47.0,
                    lat_max: float = -26.0,
                    lon_min: float = -82.0,
                    lon_max: float = -62.0,                   
                   ) -> pd.DataFrame:
    modis_all = pd.read_csv(hotspot_csv, parse_dates=["date"])
    modis = modis_all[
           (modis_all["date"]   == pd.Timestamp(fire_date))
        &  (modis_all["dominant_lat"]   >= lat_min) & (modis_all["dominant_lat"]  <= lat_max)
        &  (modis_all["dominant_lon"]  >= lon_min) & (modis_all["dominant_lon"] <= lon_max)
    ].copy().reset_index(drop=True)
    return modis

