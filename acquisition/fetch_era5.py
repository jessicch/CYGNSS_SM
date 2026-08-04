# downloads era5-land soil moisture and the land-sea mask from the cds api.
import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import ERA5_DIR, REGIONS, START_DATE, END_DATE

try:
    import cdsapi
except ImportError:
    print("cdsapi not installed. Run:  pip install cdsapi")
    sys.exit(1)


# ERA5-Land dataset on CDS
DATASET = "reanalysis-era5-land"

# hours to average for daily mean — all 24 hours
HOURS = [f"{h:02d}:00" for h in range(24)]


# 1 file per month per region
def output_path(region_name: str, year: int, month: int) -> str:
    folder = os.path.join(ERA5_DIR, region_name, str(year))
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f"era5_{region_name}_{year}{month:02d}.nc")


# land sea mask path
def lsm_path(region_name: str) -> str:
    folder = os.path.join(ERA5_DIR, region_name)
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f"era5_{region_name}_lsm.nc")


# List of (year, month) from start to end
def build_months(start: str, end: str) -> list:
    cur    = datetime.strptime(start, "%Y%m%d").replace(day=1)
    end_dt = datetime.strptime(end,   "%Y%m%d")
    months = []
    while cur <= end_dt:
        months.append((cur.year, cur.month))
        # advance one month
        if cur.month == 12:
            cur = cur.replace(year=cur.year + 1, month=1)
        else:
            cur = cur.replace(month=cur.month + 1)
    return months


# List of day strings for a given month
def days_in_month(year: int, month: int) -> list:
    import calendar
    n = calendar.monthrange(year, month)[1]
    return [f"{d:02d}" for d in range(1, n + 1)]


# download land-sea mask for region
def download_lsm(client: cdsapi.Client, region_name: str, bounds: dict) -> bool:
    out = lsm_path(region_name)
    if os.path.exists(out):
        print(f"  LSM already on disk: {out}")
        return False

    tmp = out + ".tmp.nc"
    try:
        client.retrieve(
            DATASET,
            {
                "variable"       : ["land_sea_mask"],
                "product_type"   : "reanalysis",
                "year"           : "2020",
                "month"          : "01",
                "day"            : "01",
                "time"           : "00:00",
                "data_format"    : "netcdf",
                "download_format": "unarchived",
                "area"           : [
                    bounds["lat_max"],
                    bounds["lon_min"],
                    bounds["lat_min"],
                    bounds["lon_max"],
                ],
            },
            tmp,
        )

        ds = xr.open_dataset(tmp)

        # variable name varies by CDS version
        var_name = None
        for candidate in ["lsm", "land_sea_mask"]:
            if candidate in ds:
                var_name = candidate
                break

        if var_name is None:
            print(f"  WARNING: LSM variable not found in downloaded file for {region_name}")
            ds.close()
            os.remove(tmp)
            return False

        # drop time dimension
        time_dim = next((d for d in ["valid_time", "time"] if d in ds.dims), None)
        ds_lsm = ds[[var_name]].squeeze(time_dim, drop=True) if time_dim else ds[[var_name]]
        ds_lsm = ds_lsm.rename({var_name: "lsm"})

        # normalize longitudes to -180:180
        lon_name = next((c for c in ["longitude", "lon"] if c in ds_lsm.coords), None)
        if lon_name is not None and float(ds_lsm[lon_name].max()) > 180:
            ds_lsm = ds_lsm.assign_coords(
                {lon_name: (((ds_lsm[lon_name] + 180) % 360) - 180)}
            ).sortby(lon_name)
        ds_lsm["lsm"].attrs["long_name"]    = "Land-sea mask"
        ds_lsm["lsm"].attrs["units"]        = "0-1"
        ds_lsm["lsm"].attrs["description"]  = (
            "ERA5-Land land-sea mask: 1 = land, 0 = sea. Static field."
        )
        ds_lsm.to_netcdf(out)
        ds.close()
        os.remove(tmp)
        print(f"  ✓ LSM saved: {out}")
        return True

    except Exception as e:
        print(f"\n  Error downloading LSM for {region_name}: {e}")
        for p in [tmp, out]:
            if os.path.exists(p):
                try: os.remove(p)
                except: pass
        return False


# download monthly soil moisture for region
def download_month(client: cdsapi.Client, region_name: str,
                   bounds: dict, year: int, month: int) -> bool:
    out = output_path(region_name, year, month)
    if os.path.exists(out):
        return False   # already on disk

    tmp = out + ".tmp.nc"

    try:
        client.retrieve(
            DATASET,
            {
                "variable"    : ["volumetric_soil_water_layer_1"],
                "product_type": "reanalysis",
                "year"        : str(year),
                "month"       : f"{month:02d}",
                "day"         : days_in_month(year, month),
                "time"        : HOURS,
                "data_format" : "netcdf",
                "download_format": "unarchived",
                # server-side clip to region bbox
                "area"        : [
                    bounds["lat_max"],   # N
                    bounds["lon_min"],   # W
                    bounds["lat_min"],   # S
                    bounds["lon_max"],   # E
                ],
            },
            tmp,
        )

        # resample hourly to daily mean
        ds = xr.open_dataset(tmp)

        # variable name differs by ERA5 version
        var_name = None
        for candidate in ["swvl1", "volumetric_soil_water_layer_1"]:
            if candidate in ds:
                var_name = candidate
                break

        if var_name is None:
            print(f"    WARNING: soil moisture variable not found in {out}")
            ds.close()
            os.remove(tmp)
            return False

        # resample to daily mean
        time_dim  = next((d for d in ["valid_time", "time"] if d in ds.dims), "valid_time")
        ds_daily  = ds[[var_name]].resample({time_dim: "1D"}).mean()
        ds_daily = ds_daily.rename({var_name: "soil_moisture"})
        ds_daily.attrs["units"]       = "m3 m-3"
        ds_daily.attrs["description"] = (
            "ERA5-Land volumetric soil water layer 1 (0-7 cm), daily mean"
        )

        ds_daily.to_netcdf(out)
        ds.close()
        os.remove(tmp)
        return True

    except Exception as e:
        print(f"\n    Error ({year}-{month:02d}): {e}")
        for p in [tmp, out]:
            if os.path.exists(p):
                try: os.remove(p)
                except: pass
        return False


# download lsm and monthly data for region
def download_region(region_name: str, bounds: dict,
                    months: list, client: cdsapi.Client):
    print(f"\n{'='*60}")
    print(f"  {bounds['label']}")
    print(f"  {months[0][0]}-{months[0][1]:02d} → "
          f"{months[-1][0]}-{months[-1][1]:02d}  "
          f"({len(months)} months)")
    print(f"{'='*60}")

    download_lsm(client, region_name, bounds)

    # figure out what's already on disk
    missing = [
        (y, m) for y, m in months
        if not os.path.exists(output_path(region_name, y, m))
    ]
    already = len(months) - len(missing)
    print(f"  {already}/{len(months)} months already on disk"
          f" — downloading {len(missing)} remaining\n")

    if not missing:
        print("  Nothing to do.")
        return

    saved = failed = 0

    for year, month in tqdm(missing, desc=f"  {region_name}", unit="month"):
        ok = download_month(client, region_name, bounds, year, month)
        if ok:
            saved += 1
            tqdm.write(f"  ✓ {year}-{month:02d}")
        else:
            failed += 1

    print(f"\n  Saved: {saved}  |  Failed/skipped: {failed}")


def main():
    parser = argparse.ArgumentParser(
        description="Download ERA5-Land soil moisture via CDS API."
    )
    parser.add_argument("--region", default=None,
                        help="Only download this region (e.g. pakistan)")
    parser.add_argument("--year",   type=int, default=None,
                        help="Only download this year (e.g. 2022)")
    args = parser.parse_args()

    start = f"{args.year}0101" if args.year else START_DATE
    end   = f"{args.year}1231" if args.year else END_DATE

    print("\n" + "="*60)
    print("  ERA5-Land Soil Moisture Download")
    print(f"  Dataset  : {DATASET}")
    print(f"  Variable : volumetric_soil_water_layer_1 (0–7 cm)")
    print(f"  Period   : {start} → {end}")
    if args.region:
        print(f"  Region   : {args.region}")
    print("="*60)

    # reads cdsapirc automatically
    client = cdsapi.Client()

    months = build_months(start, end)

    for region_name, bounds in REGIONS.items():
        if args.region and region_name != args.region:
            print(f"\n  Skipping {bounds['label']}")
            continue
        download_region(region_name, bounds, months, client)

    print("\n  Done.\n")


if __name__ == "__main__":
    main()