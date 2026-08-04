# Build QC cache for CYGNSS data
# run: python processing/build_qc_cache.py --region pakistan --year 2022

import os
import sys
import glob
import argparse
import time
import warnings
from calendar import monthrange
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import CYGNSS_DIR, CYGNSS_QC_DIR, REGIONS


# QC constants

# gps l1 wavelength (m)
L1_WAVELENGTH = 0.1903

# quality flag bitmask, bits to reject
QF_BITMASK = (
    0x00000002 |   # S-band powered up
    0x00000008 |   # spacecraft attitude error
    0x00000010 |   # black body DDM
    0x00000080 |   # DDM is test pattern
    0x00008000 |   # direct signal in DDM
    0x00010000     # low confidence GPS EIRP
)

# per prn db biases, index = prn - 1. prn 4 has none -> nan -> dropped later
PRN_BIASES = np.array([
     1.017,  0.004,  1.636,    np.nan, -0.610,  0.240, -0.709,  0.605,
     1.498, -0.783, -0.230,  -1.021,   0.007, -0.730, -0.376, -0.481,
     0.256, -0.206, -0.206,   0.345,  -0.909, -0.838, -0.858,  1.140,
     0.880,  0.163,  0.409,  -0.712,  -1.032,  0.877, -0.562, -0.819,
], dtype=float)

# qc thresholds
INC_ANGLE_MAX  = 65.0
RX_GAIN_MIN    = 0.0    # strict >
RX_GAIN_MAX    = 13.0   # inclusive
DDM_SNR_MIN    = 2.0    # strict >
# sp_land_valid: 1 = land, 0 = low confidence land, -99 = fill (ocean)
SP_LAND_FILL   = -99    # only the fill gets dropped

# columns needed from the raw files, optionals kept if present
REQUIRED_VARS = (
    "sp_lat", "sp_lon",
    "ddm_snr", "gps_tx_power_db_w", "gps_ant_gain_db_i",
    "sp_rx_gain", "tx_to_sp_range", "rx_to_sp_range",
    "sp_inc_angle", "prn_code", "quality_flags",
)
OPTIONAL_VARS = (
    "ddm_timestamp_utc",
    "sp_land_valid",
    "modis_land_cover",
)

# columns kept in the qc output. date is the source-file day, set in load_raw_day
OUTPUT_COLUMNS = (
    "date",
    "sp_lat", "sp_lon", "sr",
    "sp_inc_angle", "prn_code",
    "ddm_timestamp_utc",
    "sp_land_valid",
    "modis_land_cover",
)


# Core functions

# Calculate SR with PRN bias
def compute_sr(df: pd.DataFrame) -> pd.Series:
    prn  = df["prn_code"].values.astype(int)
    bias = np.where(
        (prn >= 1) & (prn <= 32),
        PRN_BIASES[np.clip(prn - 1, 0, 31)],
        np.nan,
    )
    sr = (
        df["ddm_snr"].values
        - df["gps_tx_power_db_w"].values
        - df["gps_ant_gain_db_i"].values
        - df["sp_rx_gain"].values
        - 20 * np.log10(L1_WAVELENGTH)
        + 20 * np.log10(df["tx_to_sp_range"].values + df["rx_to_sp_range"].values)
        + 20 * np.log10(4 * np.pi)
        + bias
    )
    return pd.Series(sr, index=df.index)


# Apply QC filters and return passing observations
def apply_qc(
    df: pd.DataFrame,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> pd.DataFrame:
    if df.empty:
        return df.iloc[0:0]

    # wrap 0-360 longitudes to ±180
    df = df.copy()
    df["sp_lon"] = np.where(df["sp_lon"] > 180, df["sp_lon"] - 360, df["sp_lon"])

    # step 1: bounding box
    df = df[
        (df["sp_lat"] >= lat_min) & (df["sp_lat"] <= lat_max) &
        (df["sp_lon"] >= lon_min) & (df["sp_lon"] <= lon_max)
    ]
    if df.empty:
        return df

    # step 2: quality flag bitmask reject
    df = df[(df["quality_flags"].astype(np.int64) & QF_BITMASK) == 0]

    # step 3: threshold filters
    df = df[df["sp_inc_angle"] <= INC_ANGLE_MAX]
    df = df[(df["sp_rx_gain"] > RX_GAIN_MIN) & (df["sp_rx_gain"] <= RX_GAIN_MAX)]
    df = df[df["ddm_snr"] > DDM_SNR_MIN]
    df = df[(df["tx_to_sp_range"] > 0) & (df["rx_to_sp_range"] > 0)]

    # Drop ocean rows (fill value -99); keep inland water
    if "sp_land_valid" in df.columns:
        df = df[df["sp_land_valid"].astype(np.int32) != SP_LAND_FILL]
    else:
        print("  WARN: sp_land_valid column missing - ocean filter skipped")
    if df.empty:
        return df

    # step 5: keep rows where all sr inputs are finite
    finite_cols = ["ddm_snr", "gps_tx_power_db_w", "gps_ant_gain_db_i",
                   "sp_rx_gain", "tx_to_sp_range", "rx_to_sp_range"]
    df = df[np.isfinite(df[finite_cols].values).all(axis=1)]
    if df.empty:
        return df

    # step 6: compute sr, drop nan rows
    df = df.reset_index(drop=True)
    df["sr"] = compute_sr(df)
    df = df[np.isfinite(df["sr"])]

    # keep only the output columns
    keep = [c for c in OUTPUT_COLUMNS if c in df.columns]
    return df[keep].reset_index(drop=True)


# Loaders

# Load one raw NetCDF file, return empty DataFrame on error
def _load_one_nc(fpath: str) -> pd.DataFrame:
    try:
        ds = xr.open_dataset(fpath)
    except Exception:
        return pd.DataFrame()

    try:
        col = {}
        for var in REQUIRED_VARS:
            if var not in ds:
                ds.close()
                return pd.DataFrame()
            col[var] = ds[var].values.ravel()
        for var in OPTIONAL_VARS:
            if var in ds:
                col[var] = ds[var].values.ravel()
        df = pd.DataFrame(col)
    except Exception:
        df = pd.DataFrame()
    finally:
        ds.close()

    return df


# Load all raw files for one day and tag with date
def load_raw_day(cygnss_dir: str, region: str, date: datetime) -> pd.DataFrame:
    date_str = date.strftime("%Y%m%d")
    yyyy, mm = date.strftime("%Y"), date.strftime("%m")
    pattern  = os.path.join(cygnss_dir, region, yyyy, mm, f"*_{date_str}.nc")
    files    = sorted(glob.glob(pattern))

    if not files:
        return pd.DataFrame()

    dfs = [_load_one_nc(f) for f in files]
    dfs = [d for d in dfs if not d.empty]
    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
    df["date"] = pd.Timestamp(date.year, date.month, date.day)
    return df


# Load one month from QC cache
def load_qc_cache_month(qc_dir: str, region: str, year: int, month: int) -> pd.DataFrame:
    import geopandas as gpd
    fpath = os.path.join(qc_dir, region, str(year),
                         f"cygnss_qc_{region}_{year}{month:02d}.parquet")
    if not os.path.exists(fpath):
        return pd.DataFrame()
    return gpd.read_parquet(fpath)


# Cache builder CLI

def _bbox_for(region: str) -> tuple[float, float, float, float]:
    r = REGIONS[region]
    return r["lat_min"], r["lat_max"], r["lon_min"], r["lon_max"]


def _out_path(region: str, year: int, month: int) -> str:
    folder = os.path.join(CYGNSS_QC_DIR, region, str(year))
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f"cygnss_qc_{region}_{year}{month:02d}.parquet")


# Build one month-file (skip if exists unless force=True)
def build_month(region: str, year: int, month: int, force: bool = False) -> dict:
    out_path = _out_path(region, year, month)

    if os.path.exists(out_path) and not force:
        return {"status": "skipped", "n_rows": None, "n_days_with_data": None,
                "elapsed_s": 0.0, "out_path": out_path}

    lat_min, lat_max, lon_min, lon_max = _bbox_for(region)

    t0 = time.time()
    n_days = monthrange(year, month)[1]
    parts = []
    n_days_with_data = 0

    for day in range(1, n_days + 1):
        date = datetime(year, month, day)
        raw  = load_raw_day(CYGNSS_DIR, region, date)
        if raw.empty:
            continue
        qc = apply_qc(raw, lat_min, lat_max, lon_min, lon_max)
        if qc.empty:
            continue
        n_days_with_data += 1
        parts.append(qc)

    elapsed = time.time() - t0

    if not parts:
        return {"status": "empty", "n_rows": 0, "n_days_with_data": 0,
                "elapsed_s": elapsed, "out_path": out_path}

    df = pd.concat(parts, ignore_index=True)

    # Save as GeoParquet with point geometry
    import geopandas as gpd
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["sp_lon"].values, df["sp_lat"].values),
        crs="EPSG:4326",
    )
    gdf.to_parquet(out_path, compression="snappy", index=False)

    return {"status": "written", "n_rows": len(df),
            "n_days_with_data": n_days_with_data,
            "elapsed_s": elapsed, "out_path": out_path}


# Build all month-files for a region
def build_region(region: str, year_start: int, year_end: int,
                 month_start: int = 1, month_end: int = 12,
                 force: bool = False) -> None:
    if region not in REGIONS:
        print(f"ERROR: region '{region}' not in REGIONS. "
              f"Available: {list(REGIONS.keys())}")
        return

    print("\n" + "=" * 70)
    print(f"  Building QC cache — region={region}  "
          f"years={year_start}-{year_end}  months={month_start}-{month_end}")
    print(f"  Source : {os.path.relpath(os.path.join(CYGNSS_DIR, region))}")
    print(f"  Target : {os.path.relpath(os.path.join(CYGNSS_QC_DIR, region))}")
    print(f"  Force  : {force}")
    print("=" * 70)

    total_rows = 0
    written = skipped = empty = 0

    # list of (year, month) targets
    targets: list[tuple[int, int]] = []
    for year in range(year_start, year_end + 1):
        m_lo = month_start if year == year_start else 1
        m_hi = month_end   if year == year_end   else 12
        for month in range(m_lo, m_hi + 1):
            targets.append((year, month))

    bar = tqdm(targets, desc=f"{region}", unit="mo", dynamic_ncols=True)
    for year, month in bar:
        bar.set_postfix_str(f"{year}-{month:02d}")
        stats = build_month(region, year, month, force=force)
        tag = f"{year}-{month:02d}"
        if stats["status"] == "written":
            written += 1
            total_rows += stats["n_rows"]
            tqdm.write(
                f"  [WRITE] {tag}  rows={stats['n_rows']:>9,}  "
                f"days={stats['n_days_with_data']:>2}  "
                f"time={stats['elapsed_s']:>5.1f}s  "
                f"-> {os.path.relpath(stats['out_path'])}"
            )
        elif stats["status"] == "skipped":
            skipped += 1
            tqdm.write(f"  [SKIP ] {tag}  (exists; use --force to rebuild)")
        else:
            empty += 1
            tqdm.write(f"  [EMPTY] {tag}  no QC-passing observations in this month")

    bar.close()
    print("\n" + "-" * 70)
    print(f"  Done.  written={written}  skipped={skipped}  empty={empty}  "
          f"total_rows={total_rows:,}")
    print("-" * 70 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build CYGNSS QC'd GeoParquet cache (data/cygnss_qc/...)"
    )
    parser.add_argument("--region", type=str, default="pakistan",
                        help="Region key from config.settings.REGIONS (default: pakistan)")
    parser.add_argument("--all-regions", action="store_true",
                        help="Build cache for every region in REGIONS")
    parser.add_argument("--year", type=int, default=None,
                        help="Single year to build (shortcut for --year-start/--year-end)")
    parser.add_argument("--year-start", type=int, default=2021)
    parser.add_argument("--year-end",   type=int, default=2025)
    parser.add_argument("--month-start", type=int, default=1)
    parser.add_argument("--month-end",   type=int, default=12)
    parser.add_argument("--force", action="store_true",
                        help="Rebuild even if the month-file already exists")
    args = parser.parse_args()

    if args.year is not None:
        year_start = args.year
        year_end   = args.year
    else:
        year_start = args.year_start
        year_end   = args.year_end

    regions = list(REGIONS.keys()) if args.all_regions else [args.region]

    for region in regions:
        build_region(region, year_start, year_end,
                     month_start=args.month_start, month_end=args.month_end,
                     force=args.force)


if __name__ == "__main__":
    main()
