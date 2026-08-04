import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime
from calendar import monthrange

# make validation/ importable so the shared common module resolves
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import shared

import context as ctx
from grids import bin_points_to_ease2


# load cygnss for month
def load_cygnss_month(year, month):
    gdf = shared.load_cygnss_cache_month(
        ctx.SR_COL, ctx.CACHE_DIR, ctx.REGION, year, month
    )

    if gdf.empty:
        print(f"  ERROR: {ctx.SR_COL} cache missing for {ctx.REGION} {year}-{month:02d}")
        return pd.DataFrame(columns=["sp_lat", "sp_lon", "sr"])

    df = pd.DataFrame({
        "sp_lat": gdf["sp_lat"].values,
        "sp_lon": gdf["sp_lon"].values,
        "sr":     gdf[shared.value_column(ctx.SR_COL)].values,
    })
    # final bbox clip
    df = df[
        (df["sp_lat"] >= ctx.LAT_MIN) & (df["sp_lat"] <= ctx.LAT_MAX) &
        (df["sp_lon"] >= ctx.LON_MIN) & (df["sp_lon"] <= ctx.LON_MAX)
    ].reset_index(drop=True)
    return df


# load smap month as grid
def load_smap_month(year, month, nlat, nlon, row_south, col_west):
    n_days = monthrange(year, month)[1]

    lats, lons, sms = [], [], []
    files_found = 0
    for day in range(1, n_days + 1):
        date     = datetime(year, month, day)
        date_str = date.strftime("%Y%m%d")
        yyyy, mm = date.strftime("%Y"), date.strftime("%m")
        fpath    = os.path.join(ctx.SMAP_DIR, ctx.REGION, yyyy, mm,
                                f"smap_{ctx.REGION}_{date_str}.h5")
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
        if df.empty:
            continue

        files_found += 1
        lats.append(df["latitude"].values)
        lons.append(df["longitude"].values)
        sms.append(df["soil_moisture"].values)

    print(f"  SMAP: {files_found} daily files loaded")
    if not lats:
        return np.full((nlat, nlon), np.nan)

    grid, _ = bin_points_to_ease2(
        np.concatenate(lats), np.concatenate(lons), np.concatenate(sms),
        nlat, nlon, row_south, col_west,
    )
    return grid


# load era5 month on grid
def load_era5_month(year, month, lat_centres, lon_centres):
    fpath = os.path.join(ctx.ERA5_DIR, ctx.REGION, str(year),
                         f"era5_{ctx.REGION}_{year}{month:02d}.nc")

    empty = np.full((len(lat_centres), len(lon_centres)), np.nan)

    if not os.path.exists(fpath):
        print(f"  ERA5 file not found: {os.path.relpath(fpath)}")
        return empty

    try:
        ds = xr.open_dataset(fpath)
    except Exception as exc:
        print(f"  ERA5 open error: {exc}")
        return empty

    # find the soil moisture variable
    var_name = None
    for candidate in ("soil_moisture", "swvl1", "volumetric_soil_water_layer_1", "sm"):
        if candidate in ds:
            var_name = candidate
            break
    if var_name is None:
        print(f"  ERA5: soil moisture variable not found. Available: {list(ds.data_vars)}")
        ds.close()
        return empty

    # average over the time dimension
    time_dims = [d for d in ds[var_name].dims if d in ("time", "valid_time")]
    da = ds[var_name].mean(dim=time_dims, skipna=True) if time_dims else ds[var_name]

    # coordinate names
    lat_var = next((v for v in ("latitude", "lat") if v in ds.coords), None)
    lon_var = next((v for v in ("longitude", "lon") if v in ds.coords), None)
    if lat_var is None or lon_var is None:
        print("  ERA5: lat/lon coordinates not found")
        ds.close()
        return empty

    era5_lats = ds[lat_var].values
    era5_lons = ds[lon_var].values

    # crop to region
    lat_mask = (era5_lats >= ctx.LAT_MIN - 0.5) & (era5_lats <= ctx.LAT_MAX + 0.5)
    lon_mask = (era5_lons >= ctx.LON_MIN - 0.5) & (era5_lons <= ctx.LON_MAX + 0.5)
    era5_lats_r = era5_lats[lat_mask]
    era5_lons_r = era5_lons[lon_mask]

    vals = da.values
    # ensure 2-D [lat × lon]
    if vals.ndim == 1:
        vals = vals.reshape(len(era5_lats), len(era5_lons))
    lat_idx_arr = np.where(lat_mask)[0]
    lon_idx_arr = np.where(lon_mask)[0]
    vals_r = vals[np.ix_(lat_idx_arr, lon_idx_arr)]

    ds.close()

    # interpolate ERA5 onto the target grid
    era5_lon_2d, era5_lat_2d = np.meshgrid(era5_lons_r, era5_lats_r)

    finite = np.isfinite(vals_r)
    if finite.sum() < 4:
        return empty

    return shared.interp_points_to_grid(
        era5_lat_2d[finite], era5_lon_2d[finite], vals_r[finite],
        lat_centres, lon_centres
    )
