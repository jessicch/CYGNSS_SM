import os
import sys
import time
import numpy as np
import pandas as pd
import xarray as xr
from datetime import timedelta

# make validation/ importable so the shared common module resolves
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import shared

import context as ctx


# split [start, end) into consecutive window_days-day chunks
def build_windows(start, end, window_days=3):
    windows = []
    cur = start
    while cur < end:
        win_end = min(cur + timedelta(days=window_days), end)
        windows.append((cur, win_end))
        cur = win_end
    return windows


# every calendar date in [win_start, win_end)
def dates_in_window(win_start, win_end):
    dates = []
    cur = win_start
    while cur < win_end:
        dates.append(cur)
        cur += timedelta(days=1)
    return dates


# cache months in memory
_QC_MONTH_CACHE = {}


def _load_qc_month_cached(year, month):
    key = (year, month)
    if key in _QC_MONTH_CACHE:
        return _QC_MONTH_CACHE[key]

    # pick the loader + column matching the SR_COL switch
    gdf = shared.load_cygnss_cache_month(
        ctx.SR_COL, ctx.CACHE_DIR, ctx.REGION, year, month
    )

    if gdf.empty:
        df = pd.DataFrame(columns=["date", "sp_lat", "sp_lon", "sr"])
    elif "date" not in gdf.columns:
        # old cache without the date column, needs a rebuild
        raise RuntimeError(
            f"QC cache for {ctx.REGION} {year}-{month:02d} is missing the `date` "
            "column. Rebuild it with:\n  python processing/build_qc_cache.py "
            f"--region {ctx.REGION} --year {year} --month-start {month} "
            f"--month-end {month} --force"
        )
    else:
        df = pd.DataFrame({
            "date":   gdf["date"].values,
            "sp_lat": gdf["sp_lat"].values,
            "sp_lon": gdf["sp_lon"].values,
            "sr":     gdf[shared.value_column(ctx.SR_COL)].values,  # stays named 'sr' downstream
        })
        # clip to analysis bbox
        df = df[
            (df["sp_lat"] >= ctx.LAT_MIN) & (df["sp_lat"] <= ctx.LAT_MAX) &
            (df["sp_lon"] >= ctx.LON_MIN) & (df["sp_lon"] <= ctx.LON_MAX)
        ].reset_index(drop=True)

    _QC_MONTH_CACHE[key] = df
    return df


def load_cygnss_day(date):
    month_df = _load_qc_month_cached(date.year, date.month)
    if month_df.empty:
        return pd.DataFrame(columns=["sp_lat", "sp_lon", "sr"])

    target = pd.Timestamp(date.year, date.month, date.day)
    mask = (month_df["date"].values == np.datetime64(target))
    day_df = month_df.loc[mask, ["sp_lat", "sp_lon", "sr"]].reset_index(drop=True)
    return day_df[np.isfinite(day_df["sr"].values)].reset_index(drop=True)


# cygnss series cached to disk
def build_cygnss_series(windows, cache_path, force=False):
    if not force and os.path.exists(cache_path):
        print(f"  Loading cached CYGNSS time series from\n  {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return data["cygnss_raw"]

    print(f"  Processing {len(windows)} windows of {ctx.WINDOW_DAYS} days ...")
    print("  (First run only — results will be cached)")

    sr_series = np.full(len(windows), np.nan)
    t0 = time.time()

    for wi, (win_start, win_end) in enumerate(windows):
        # mean of daily means so each day weighs equally
        daily_means = []
        for date in dates_in_window(win_start, win_end):
            df = load_cygnss_day(date)
            if not df.empty:
                daily_means.append(float(df["sr"].mean()))

        if daily_means:
            sr_series[wi] = float(np.mean(daily_means))

        # progress every 10 windows
        if (wi + 1) % 10 == 0 or wi == len(windows) - 1:
            elapsed  = time.time() - t0
            frac     = (wi + 1) / len(windows)
            eta_sec  = (elapsed / frac) * (1 - frac) if frac > 0 else 0
            eta_str  = f"{eta_sec/60:.0f} min" if eta_sec > 90 else f"{eta_sec:.0f} s"
            pct      = 100 * frac
            v   = shared.variant_label(ctx.SR_COL)
            val = (f"{v['short']} = {sr_series[wi]:.2f} {v['unit']}"
                   if np.isfinite(sr_series[wi]) else "no data")
            print(f"  CYGNSS: window {wi+1}/{len(windows)}  ({pct:.0f}%)  "
                  f"ETA ≈ {eta_str}   {val}")

    np.savez_compressed(cache_path, cygnss_raw=sr_series)
    print(f"\n  Cache saved → {os.path.relpath(cache_path)}")
    return sr_series


# build smap time series
def build_smap_series(windows):
    sm_series = np.full(len(windows), np.nan)

    print(f"\n  Processing {len(windows)} SMAP windows ...")
    t0 = time.time()

    for wi, (win_start, win_end) in enumerate(windows):
        daily_means = []
        for date in dates_in_window(win_start, win_end):
            date_str = date.strftime("%Y%m%d")
            yyyy, mm = date.strftime("%Y"), date.strftime("%m")
            fpath = os.path.join(
                ctx.SMAP_DIR, ctx.REGION, yyyy, mm,
                f"smap_{ctx.REGION}_{date_str}.h5"
            )
            if not os.path.exists(fpath):
                continue
            try:
                df = pd.read_hdf(fpath, key="smap")
            except Exception:
                continue

            # pool am + pm into one column
            sm = shared.pool_soil_moisture(df)
            if sm is None:
                continue
            df = df.assign(soil_moisture=sm)

            df = df.dropna(subset=["soil_moisture"])
            df = df[df["soil_moisture"] > 0]
            df = df[
                (df["latitude"]  >= ctx.LAT_MIN) & (df["latitude"]  <= ctx.LAT_MAX) &
                (df["longitude"] >= ctx.LON_MIN) & (df["longitude"] <= ctx.LON_MAX)
            ]
            if not df.empty:
                daily_means.append(df["soil_moisture"].mean())

        if daily_means:
            sm_series[wi] = float(np.mean(daily_means))

        if (wi + 1) % 50 == 0 or wi == len(windows) - 1:
            elapsed = time.time() - t0
            frac    = (wi + 1) / len(windows)
            eta     = (elapsed / frac) * (1 - frac) if frac > 0 else 0
            print(f"  SMAP: window {wi+1}/{len(windows)}  "
                  f"({100*frac:.0f}%)  ETA ≈ {eta:.0f} s")

    return sm_series


# build era5 time series
def build_era5_series(windows):
    era5_series = np.full(len(windows), np.nan)

    # cache datasets to avoid reopening
    _ds_cache = {}

    def _get_ds(year, month):
        key = (year, month)
        if key not in _ds_cache:
            fpath = os.path.join(
                ctx.ERA5_DIR, ctx.REGION, str(year),
                f"era5_{ctx.REGION}_{year}{month:02d}.nc"
            )
            if not os.path.exists(fpath):
                _ds_cache[key] = None
            else:
                try:
                    _ds_cache[key] = xr.open_dataset(fpath)
                except Exception:
                    _ds_cache[key] = None
        return _ds_cache[key]

    # mask land cells, use natural earth fallback
    lsm_native = shared.load_era5_lsm_native(ctx.REGION)
    if lsm_native is not None:
        n_land = int((lsm_native[2] >= 0.5).sum())
        print(f"  ERA5 LSM: {n_land}/{lsm_native[2].size} native cells on land "
              f"(applied to regional mean)")
    else:
        print("  ERA5 LSM: none found — falling back to the Natural Earth "
              "coastline mask")

    # fallback masks, built once per grid shape
    _ne_masks = {}

    print(f"\n  Processing {len(windows)} ERA5 windows ...")

    for wi, (win_start, win_end) in enumerate(windows):
        daily_means = []
        for date in dates_in_window(win_start, win_end):
            ds = _get_ds(date.year, date.month)
            if ds is None:
                continue

            # find the time dimension name
            time_dim = next(
                (d for d in ("valid_time", "time") if d in ds.dims), None
            )
            if time_dim is None:
                continue

            # grab this day's data
            try:
                day_data = ds["soil_moisture"].sel(
                    {time_dim: date.strftime("%Y-%m-%d")}, method="nearest"
                )
            except Exception:
                continue

            # clip to bbox
            lat_var = next((v for v in ("latitude", "lat") if v in ds.coords), None)
            lon_var = next((v for v in ("longitude", "lon") if v in ds.coords), None)

            if lat_var and lon_var:
                lat_vals = ds[lat_var].values
                lon_vals = ds[lon_var].values
                lat_mask = (lat_vals >= ctx.LAT_MIN) & (lat_vals <= ctx.LAT_MAX)
                lon_mask = (lon_vals >= ctx.LON_MIN) & (lon_vals <= ctx.LON_MAX)

                if lat_mask.any() and lon_mask.any():
                    lat_idx = np.where(lat_mask)[0]
                    lon_idx = np.where(lon_mask)[0]
                    vals = day_data.isel({lat_var: lat_idx, lon_var: lon_idx}).values
                    # drop ocean cells
                    if (lsm_native is not None
                            and lsm_native[2].shape == day_data.shape):
                        land = lsm_native[2] >= 0.5
                    else:
                        key = day_data.shape
                        if key not in _ne_masks:
                            _ne_masks[key] = shared.natural_earth_land_mask(
                                lat_vals, lon_vals
                            )
                        land = _ne_masks[key]
                    vals = np.where(land[np.ix_(lat_idx, lon_idx)], vals, np.nan)
                else:
                    vals = day_data.values
            else:
                vals = day_data.values

            val = float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan
            if np.isfinite(val):
                daily_means.append(val)

        if daily_means:
            era5_series[wi] = float(np.mean(daily_means))

    for ds in _ds_cache.values():
        if ds is not None:
            ds.close()

    valid = np.isfinite(era5_series).sum()
    print(f"  ERA5: {valid}/{len(windows)} windows with valid data")
    return era5_series
