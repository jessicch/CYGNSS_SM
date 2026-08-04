from __future__ import annotations
import os
import time

import numpy as np
import pandas as pd
import geopandas as gpd

from .constants import INC_ANGLE_TRAIN_MIN, INC_ANGLE_TRAIN_MAX
from .smap_io import load_smap_day, load_clay_index
from .ease2_grid import Ease2Grid
from .mironov import mironov_epsilon
from .fresnel import rl_reflectivity


# afs table path, tag is '2021' or '2019-2021'
def afs_table_path(rough_dir: str, region: str, tag) -> str:
    folder = os.path.join(rough_dir, region)
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f"afs_{region}_{tag}.parquet")


def _veg_month_path(veg_dir: str, region: str, year: int, month: int):
    return os.path.join(
        veg_dir, region, str(year),
        f"cygnss_veg_{region}_{year}{month:02d}.parquet",
    )



# Loads each cygnss month using only the given degree limits
def _load_veg_month_train(veg_dir: str, region: str, year: int, month: int):
    path = _veg_month_path(veg_dir, region, year, month)
    if not os.path.exists(path):
        return None
    veg_month_df = gpd.read_parquet(path)
    keep = ( #filters all obs
        np.isfinite(veg_month_df["sr_veg"].values)
        & np.isfinite(veg_month_df["sp_inc_angle"].values)
        & (veg_month_df["sp_inc_angle"].values >= INC_ANGLE_TRAIN_MIN)
        & (veg_month_df["sp_inc_angle"].values <= INC_ANGLE_TRAIN_MAX)
    )
    veg_month_df = veg_month_df.loc[keep, ["date", "sp_lat", "sp_lon", "sp_inc_angle", "sr_veg"]]
    if veg_month_df.empty:
        return None

    return pd.DataFrame({
        "date": pd.to_datetime(veg_month_df["date"].values),
        "sp_lat": veg_month_df["sp_lat"].values.astype(np.float64),
        "sp_lon": veg_month_df["sp_lon"].values.astype(np.float64),
        "sp_inc_angle": veg_month_df["sp_inc_angle"].values.astype(np.float64),
        "sr_veg": veg_month_df["sr_veg"].values.astype(np.float64),
    })


# accumulates one year of cygnss x smap matchups into the per-cell dicts
def _accumulate_year(region: str,
                     year: int,
                     veg_dir: str,
                     smap_dir: str,
                     grid: Ease2Grid,
                     clay_by_cell_id: dict,
                     sum_afs_db: dict,
                     n_obs: dict,
                     days_set: dict) -> int:

    days_with_match = 0

    for month in range(1, 13):
        df_cyg = _load_veg_month_train(veg_dir, region, year, month)
        if df_cyg is None:
            continue

        # From latlon to ease2
        cell_id_obs = grid.cell_id(df_cyg["sp_lat"].values, df_cyg["sp_lon"].values)
        df_cyg = df_cyg.assign(_cell_id=cell_id_obs, _day=df_cyg["date"].dt.date)

        for day, obs in df_cyg.groupby("_day", sort=False):
            smap_df = load_smap_day(smap_dir, region, day)
            if smap_df is None:
                continue

            # Dictionary to look up soil moisture by cell id
            soil_moisture_by_cell_id = dict(zip(smap_df["cell_id"].astype(int),
                                     smap_df["soil_moisture"].astype(float)))

            day_matched = False
            for cell_id, theta, sr_db in zip(obs["_cell_id"].values.astype(np.int64),
                                             obs["sp_inc_angle"].values,
                                             obs["sr_veg"].values):
                int_id = int(cell_id)
                if int_id < 0:
                    continue

                # Get soil moisture and clay for a specific cell
                soil_moisture = soil_moisture_by_cell_id.get(int_id)
                if soil_moisture is None:
                    continue
                clay = clay_by_cell_id.get(int_id)
                if clay is None:
                    continue

                # Calculates final reflectivity based on Mironov smap reflectivity
                eps_m = mironov_epsilon(soil_moisture, clay)
                r2 = float(rl_reflectivity(eps_m, theta))
                if not (np.isfinite(r2) and r2 > 0.0):
                    continue

                # Uses cygnss SR with Mironov smap reflectivity to calculate the AFS
                afs_db = sr_db - 10.0*np.log10(r2)

                # afs is accumulated and averaged in db per cell
                sum_afs_db[int_id] = sum_afs_db.get(int_id, 0.0) + afs_db
                n_obs[int_id] = n_obs.get(int_id, 0) + 1
                days_set.setdefault(int_id, set()).add(day)
                day_matched = True

            if day_matched:
                days_with_match += 1

    return days_with_match


# calculates afs, pooling one or more years into a per-cell table
def train_years(region: str,
                years,
                veg_dir: str,
                smap_dir: str,
                rough_dir: str,
                force: bool = False):

    years = sorted({int(y) for y in years})
    tag = f"{years[0]}-{years[-1]}" if len(years) > 1 else str(years[0])

    out_path = afs_table_path(rough_dir, region, tag)
    if os.path.exists(out_path) and not force: # no need to run if file alr exists. use force when rewriting.
        return {"status": "skipped", "n_cells": 0, "n_obs": 0,
                "n_days_used": 0, "elapsed_s": 0.0, "out_path": out_path}

    t0 = time.time()
    grid = Ease2Grid()

    # Clay is static per cell; merge the per-year lookups so a cell keeps clay
    # even if it only had SMAP data in some of the years.
    clay_by_cell_id: dict[int, float] = {}
    for year in years:
        for cid, cv in load_clay_index(smap_dir, region, year).items():
            clay_by_cell_id.setdefault(cid, cv)

    sum_afs_db: dict[int, float] = {}
    n_obs: dict[int, int] = {}
    days_set: dict[int, set]  = {}

    days_with_match = 0
    for year in years:
        days_with_match += _accumulate_year(
            region, year, veg_dir, smap_dir, grid,
            clay_by_cell_id, sum_afs_db, n_obs, days_set,
        )

    # Array of cell_ids with at least 1 observation
    cell_ids = np.array(sorted(n_obs.keys()), dtype=np.int64)
    # Converted to row,col
    row = (cell_ids // grid.N_COLS).astype(np.int32)
    col = (cell_ids %  grid.N_COLS).astype(np.int32)
    # Lat lon of each cells center point
    lat_cell, lon_cell = grid.cell_center(row, col)
    n_obs_cell = np.array([n_obs[c] for c in cell_ids], dtype=np.int64)
    sum_afs_db_arr = np.array([sum_afs_db[c] for c in cell_ids], dtype=np.float64)
    days_w_obs = np.array([len(days_set[c]) for c in cell_ids], dtype=np.int64)
    db_a = sum_afs_db_arr / n_obs_cell

    # Saved for future years correction
    out = pd.DataFrame({
        "cell_id": cell_ids,
        "row": row,
        "col": col,
        "lat": lat_cell,
        "lon": lon_cell,
        "afs_db": db_a,
        "n_obs": n_obs_cell,
        "n_days": days_w_obs,
        "train_tag": tag,
        "region": region,
    })
    out.to_parquet(out_path, compression="snappy", index=False)

    return {
        "status": "written",
        "n_cells": int(len(out)),
        "n_obs": int(n_obs_cell.sum()),
        "n_days_used": int(days_with_match),  #debug
        "elapsed_s": time.time() - t0,
        "out_path": out_path,
    }


# Single-year convenience wrapper (backward compatible).
def train_year(region: str,
               year: int,
               veg_dir: str,
               smap_dir: str,
               rough_dir: str,
               force: bool = False):
    return train_years(region, [year], veg_dir, smap_dir, rough_dir, force=force)
