import os

# Directories
ROOT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(ROOT_DIR, "data")
 
CYGNSS_DIR     = os.path.join(DATA_DIR, "cygnss")
CYGNSS_QC_DIR  = os.path.join(DATA_DIR, "cygnss_qc")
CYGNSS_VEG_DIR = os.path.join(DATA_DIR, "cygnss_veg")   # 3 km veg-corrected SR
CYGNSS_ROUGH_DIR = os.path.join(DATA_DIR, "cygnss_rough")  # SMAP-cell roughness-corrected SR
CYGNSS_SM_DIR  = os.path.join(DATA_DIR, "cygnss_sm")     # retrieved soil moisture
SMAP_DIR      = os.path.join(DATA_DIR, "smap")
ERA5_DIR      = os.path.join(DATA_DIR, "era5")
MODIS_DIR     = os.path.join(DATA_DIR, "modis")     # 0.1° gridded NDVI
MODISKM_DIR   = os.path.join(DATA_DIR, "modiskm")   # 1 km NDVI
RESULTS_DIR   = os.path.join(ROOT_DIR, "results")
FIGURES_DIR   = os.path.join(ROOT_DIR, "figures")
 
# Study regions
REGIONS = {
    "guatemala_honduras": {
        "label":   "Guatemala & Honduras",
        "lat_min": 12,
        "lat_max": 18.5,
        "lon_min": -93.0,
        "lon_max": -82,
    },
   "pakistan": {
        "label":   "Pakistan (Sindh)",
        "lat_min": 25.0,
        "lat_max": 28.5,
        "lon_min": 67.0,
        "lon_max": 73.0,
    },
    "mosambique": {
        "label":   "Mosambique",
        "lat_min": -16.5,
        "lat_max": -12.0,
        "lon_min": 36.0,
        "lon_max": 40.0,
    },
    "central_african_republic": {
        "label":   "Central African Republic",
        "lat_min": 6.0,
        "lat_max": 10.0,
        "lon_min": 20.0,
        "lon_max": 24.5,
    },
    "brasil": {
        "label":   "Brasil",
        "lat_min": -25.0,
        "lat_max": -19.0,
        "lon_min": -55.0,
        "lon_max": -50.0,   
    }, 
    "mississippi": {
        "label":   "Mississippi, US",
        "lat_min": 31.0,
        "lat_max": 35.0,
        "lon_min": -93.0,
        "lon_max": -88.0,
    },



}

# Download time period
START_DATE = "20210101"   # we dont really need 2022 but its just easiest to let it run
END_DATE   = "20251231"

#CYGNSS info
SATELLITES = [f"cyg0{i}" for i in range(1, 9)]

CYGNSS_COLLECTION_ID = "C2832195379-POCLOUD"

CYGNSS_VARIABLES = [
    "ddm_snr",
    "sp_lat",
    "sp_lon",
    "sp_rx_gain",
    "tx_to_sp_range",
    "rx_to_sp_range",
    "gps_tx_power_db_w",
    "gps_ant_gain_db_i",
    "quality_flags",
    "prn_code",
    "sp_inc_angle",
    "modis_land_cover",
    "sp_land_valid",
]


# ============================================================
# ANALYSIS PARAMETERS
# ============================================================

GRID_RESOLUTION  = 0.1    # degrees — analysis grid cell size
GAUSSIAN_SIGMA   = 1.5    # Gaussian smoothing sigma
MIN_OBS_PER_CELL = 3      # minimum observations to compute mean SR
L1_WAVELENGTH    = 0.1903 # GPS L1 wavelength in metres