import os
import sys

# make the repo root importable so config resolves (three levels up)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.settings import CYGNSS_QC_DIR, SMAP_DIR, ERA5_DIR, RESULTS_DIR, REGIONS

# default scene
YEAR  = 2022
MONTH = 1          # January

# grid settings — the analysis grid is the SMAP EASE-2 9 km grid
GAUSSIAN_SIGMA = 1.5     # 2-D Gaussian smoothing σ in grid cells (9 km each)

OUTPUT_DIR = os.path.join(RESULTS_DIR, "spatial")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# runtime config, filled in by main() from the CLI args
SR_COL    = "sr"
CACHE_DIR = CYGNSS_QC_DIR
REGION    = "pakistan"
LAT_MIN   = 25.0
LAT_MAX   = 28.5
LON_MIN   = 67.0
LON_MAX   = 73.0
