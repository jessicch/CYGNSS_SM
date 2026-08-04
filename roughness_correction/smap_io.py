from __future__ import annotations
import os
from datetime import date as _date, timedelta

import numpy as np
import pandas as pd

from .ease2_grid import Ease2Grid


# path to a daily smap file
def smap_day_path(smap_dir: str, region: str, d: _date) -> str:
    return os.path.join(
        smap_dir, region,
        f"{d.year:04d}", f"{d.month:02d}",
        f"smap_{region}_{d.strftime('%Y%m%d')}.h5",
    )

# path to the old store, used as fallback
def _legacy_rough_path(smap_dir: str, region: str, d: _date) -> str:
    return os.path.join(
        smap_dir, "smap_roughness_estimation", region,
        f"{d.year:04d}", f"{d.month:02d}",
        f"smap_rough_{region}_{d.strftime('%Y%m%d')}.h5",
    )

# reads one day, legacy store as fallback. None if nothing exists.
def _read_smap_day(smap_dir: str, region: str, d: _date):
    for path in (smap_day_path(smap_dir, region, d),
                 _legacy_rough_path(smap_dir, region, d)):
        if os.path.exists(path):
            try:
                return pd.read_hdf(path, key="smap")
            except Exception:
                return None
    return None

# one soil_moisture column pooling am and pm
def _pooled_soil_moisture(df: pd.DataFrame):
    sm = df["soil_moisture"] if "soil_moisture" in df.columns else None
    if "soil_moisture_pm" in df.columns:
        pm = df["soil_moisture_pm"]
        sm = pm if sm is None else sm.fillna(pm)
    return sm

# Needed info from SMAP
_SM_COLS = ("latitude", "longitude", "soil_moisture")

# Loads a day and extracts needed information. saves to dataframe.
def load_smap_day(smap_dir: str, region: str, date: _date):

    df = _read_smap_day(smap_dir, region, date)
    if df is None:
        return None

    sm = _pooled_soil_moisture(df)
    if sm is None or "latitude" not in df.columns:
        return None
    df = df.assign(soil_moisture=sm)

    # Keep only the needed columns, that arent nan values
    df = df[list(_SM_COLS)].dropna(subset=list(_SM_COLS))
    if df.empty:
        return None

    grid = Ease2Grid()

    cell_id = grid.cell_id(df["latitude"].values, df["longitude"].values)
    df = df.assign(cell_id=cell_id)
    df = df[df["cell_id"] >= 0]

    # Calculate mean of soil moisture
    soil_moisture_mean = df.groupby("cell_id", sort=False)["soil_moisture"].mean()

    return pd.DataFrame({
        "cell_id":  soil_moisture_mean.index.values.astype(np.int64),
        "soil_moisture": soil_moisture_mean.values.astype(np.float64),
    })

# Finds the first value for clay fraction in each cell, since the value is constant.
def load_clay_index(smap_dir: str, region: str, year: int):

    grid = Ease2Grid()
    clay_by_cell_id: dict[int, float] = {}

    # Goes through each file one by one until we find one with clay fraction
    datestart = _date(year, 1, 1)
    end = _date(year, 12, 31)
    while datestart <= end:
        df = _read_smap_day(smap_dir, region, datestart)
        datestart += timedelta(days=1)
        if df is None or "clay_fraction" not in df.columns:
            continue

        clay_fraction = df["clay_fraction"].values
        lat_all = df["latitude"].values
        lon_all = df["longitude"].values
        valid_clay = (
            np.isfinite(clay_fraction) & (clay_fraction >= 0.0) & (clay_fraction <= 1.0)
            & np.isfinite(lat_all) & np.isfinite(lon_all)
        )
        if not valid_clay.any():
            continue

        lat = lat_all[valid_clay]
        lon = lon_all[valid_clay]
        clay_fraction  = clay_fraction[valid_clay]
        cell_id = grid.cell_id(lat, lon)
        for cellid, clayvalue in zip(cell_id, clay_fraction):
            id_int = int(cellid)
            if id_int < 0:
                continue
            if id_int not in clay_by_cell_id: #only adds the first time we get clay fraction for that cell, since its static.
                clay_by_cell_id[id_int] = float(clayvalue)

    return clay_by_cell_id
