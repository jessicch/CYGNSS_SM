import os
import sys
from datetime import datetime

# make the repo root importable so config resolves (three levels up)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.settings import CYGNSS_QC_DIR, SMAP_DIR, ERA5_DIR, RESULTS_DIR, REGIONS

# analysis period
START_DATE = datetime(2022, 1, 1)
END_DATE   = datetime(2025, 12, 31)

# processing settings
WINDOW_DAYS    = 7   # weekly averages, every day of the week counts
SIGMA_TEMPORAL = 4

OUTPUT_DIR = os.path.join(RESULTS_DIR, "temporal")
os.makedirs(os.path.join(OUTPUT_DIR, "data"), exist_ok=True)

# runtime config, filled in by main() from the CLI args
SR_COL    = "sr"
CACHE_DIR = CYGNSS_QC_DIR
REGION    = "pakistan"
LAT_MIN   = None
LAT_MAX   = None
LON_MIN   = None
LON_MAX   = None
