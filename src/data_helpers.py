# TROPOMI L2 loading helpers
import h5py
import numpy as np
import pandas as pd

TROPOMI_FILL: float    = 9.96e36

def load_co_l2(filepath,
                lat_min: float = -47.0,
                lat_max: float = -26.0,
                lon_min: float = -82.0,
                lon_max: float = -62.0,
                qa_threshold: float = 50):
    with h5py.File(filepath, "r") as f:
        co_raw = f["PRODUCT/carbonmonoxide_total_column"][0].astype(np.float64).ravel()
        qa_raw = f["PRODUCT/qa_value"][0].astype(np.int32).ravel()
        lat_f  = f["PRODUCT/latitude"][0].astype(np.float32).ravel()
        lon_f  = f["PRODUCT/longitude"][0].astype(np.float32).ravel()
        time_utc_str = f["PRODUCT/time_utc"][0]  
    co_raw[co_raw >= TROPOMI_FILL] = np.nan
    valid = ((lat_f >= lat_min) & (lat_f <= lat_max)
             & (lon_f >= lon_min) & (lon_f <= lon_max)
             & (qa_raw >= qa_threshold) & np.isfinite(co_raw))
    assert valid.any(), f"No valid L2 CO pixels in domain (QA >= {qa_threshold})."
    return dict(co=co_raw[valid].astype(np.float32),
                lat=lat_f[valid], lon=lon_f[valid], source="l2_swath", time=time_utc_str)

def load_no2_l2(filepath,
                lat_min: float = -47.0,
                lat_max: float = -26.0,
                lon_min: float = -82.0,
                lon_max: float = -62.0,
                qa_threshold: float = 50):
    with h5py.File(filepath, "r") as f:
        no2_raw = f["PRODUCT/nitrogendioxide_tropospheric_column"][0].astype(np.float64).ravel()
        qa_raw = f["PRODUCT/qa_value"][0].astype(np.int32).ravel()
        lat_f  = f["PRODUCT/latitude"][0].astype(np.float32).ravel()
        lon_f  = f["PRODUCT/longitude"][0].astype(np.float32).ravel()
        time_utc_str = f["PRODUCT/time_utc"][0][:]
    no2_raw[no2_raw >= TROPOMI_FILL] = np.nan
    valid = ((lat_f >= lat_min) & (lat_f <= lat_max)
             & (lon_f >= lon_min) & (lon_f <= lon_max)
             & (qa_raw >= qa_threshold) & np.isfinite(no2_raw) & (no2_raw < 1.0))
    assert valid.any(), f"No valid L2 NO2 pixels in domain (QA >= {qa_threshold})."
    return dict(no2=no2_raw[valid].astype(np.float32),
                lat=lat_f[valid], lon=lon_f[valid], source="l2_swath", time=time_utc_str)

def load_aer_ai(filepath,
                lat_min: float = -47.0,
                lat_max: float = -26.0,
                lon_min: float = -82.0,
                lon_max: float = -62.0,
                qa_threshold: float = 50):
    with h5py.File(filepath, "r") as f:
        ai_raw = f["PRODUCT/aerosol_index_354_388"][0].astype(np.float32).ravel()
        qa_raw = f["PRODUCT/qa_value"][0].astype(np.int32).ravel()
        lat_f  = f["PRODUCT/latitude"][0].astype(np.float32).ravel()
        lon_f  = f["PRODUCT/longitude"][0].astype(np.float32).ravel()
    ai_raw[ai_raw >= TROPOMI_FILL] = np.nan
    valid = ((lat_f >= lat_min) & (lat_f <= lat_max)
             & (lon_f >= lon_min) & (lon_f <= lon_max)
             & (qa_raw >= qa_threshold) & np.isfinite(ai_raw))
    assert valid.any(), "No valid AER_AI pixels in domain."
    return dict(ai=ai_raw[valid], lat=lat_f[valid], lon=lon_f[valid])

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

