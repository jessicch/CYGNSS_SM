from __future__ import annotations ## for not always having to use strings
import os
import time

import numpy as np
import pandas as pd


from roughness_correction.ease2_grid import Ease2Grid
from .ndvi import build_cell_ndvi
from .vwc import vwc_hunt
from .correction import gamma_db, tau_from_M, correct_sr_db
from .constants import YUEH_B

# Paths
def _qc_path(qc_dir: str, region: str, year: int, month: int) -> str:
    return os.path.join(
        qc_dir, region, str(year),
        f"cygnss_qc_{region}_{year}{month:02d}.parquet",
    )


def _veg_path(veg_dir: str, region: str, year: int, month: int) -> str:
    folder = os.path.join(veg_dir, region, str(year))
    os.makedirs(folder, exist_ok=True)
    return os.path.join(
        folder, f"cygnss_veg_{region}_{year}{month:02d}.parquet",
    )


# Runs the vegetation correction

def vegetation_correction(
    *,
    region:    str,
    year:      int,
    month:     int,
    qc_dir:    str,
    veg_dir:   str,
    modiskm_dir: str,
    force:     bool = False,
) -> dict:


    out_path = _veg_path(veg_dir, region, year, month)
    if os.path.exists(out_path) and not force:
        return {"status": "skipped", "n_rows": None,
                "elapsed_s": 0.0, "out_path": out_path}  # added for debug

    qc_path = _qc_path(qc_dir, region, year, month)
    if not os.path.exists(qc_path):
        return {"status": "no_qc", "n_rows": 0, "elapsed_s": 0.0,
                "out_path": out_path}

    t0 = time.time()

    # Load QC observations for the month 
    import geopandas as gpd
    cygnssqc_df = gpd.read_parquet(qc_path)
    if cygnssqc_df.empty:
        return {"status": "empty", "n_rows": 0,
                "elapsed_s": time.time() - t0, "out_path": out_path} #debug

    # Per-observation b needs the CYGNSS land-cover class; bail clearly if absent.
    if "modis_land_cover" not in cygnssqc_df.columns:
        return {"status": "no_modis_lc", "n_rows": 0,
                "elapsed_s": time.time() - t0, "out_path": out_path}

    # Build the EASE-2 9 km global grid, this is what SMAP uses
    grid = Ease2Grid()

    # For each CYGNSS observation, find which cell it belongs to
    cell_id = grid.cell_id(cygnssqc_df["sp_lat"].values, cygnssqc_df["sp_lon"].values)
    cygnssqc_df = cygnssqc_df.assign(_cell_id=cell_id)
    in_grid = cygnssqc_df["_cell_id"].values >= 0 # true if obs is inside the grid
    if not in_grid.any():
        return {"status": "empty", "n_rows": 0,
                "elapsed_s": time.time() - t0, "out_path": out_path} #debug 

    # cell-mean ndvi time series for the year
    cells_w_cygnssobs = set(int(c) for c in np.unique(cygnssqc_df.loc[in_grid, "_cell_id"].values))
    ndvi_dir = os.path.join(modiskm_dir, region, "ndvi")
    cell_ndvi = build_cell_ndvi(
        region_dir=ndvi_dir,
        year=year,
        grid=grid,
        cells_w_cygnssobs=cells_w_cygnssobs,
    )

    # correction for each row. vwc uses the cell-mean ndvi, b and stem factor
    # come from the observation's own land cover.
    n = len(cygnssqc_df)
    sr_veg      = np.full(n, np.nan, dtype=np.float32)
    tau_valid   = np.full(n, np.nan, dtype=np.float32)
    Mcell_array = np.full(n, np.nan, dtype=np.float32)     # per-obs M = b_obs · VWC_obs
    obs_b_array = np.full(n, np.nan, dtype=np.float32)     # b from the obs's land cover
    vwc_array   = np.full(n, np.nan, dtype=np.float32)     # per-obs VWC (cell NDVI + obs stem)
    gamma_array = np.full(n, 0.0, dtype=np.float32)        # default 0 = no gamma correction

    # Dataframe to arrays
    cell_id_array   = cygnssqc_df["_cell_id"].values.astype(np.int64)
    date_values = pd.to_datetime(cygnssqc_df["date"].values).to_numpy() # datetime64[ns]
    date_in_number  = date_values.astype("datetime64[D]").astype(np.int64)  # days since epoch
    dates_dt  = pd.DatetimeIndex(date_values).to_pydatetime()
    inc_angle = cygnssqc_df["sp_inc_angle"].values.astype(np.float64)
    sr_db     = cygnssqc_df["sr"].values.astype(np.float64)
    # Per-observation MODIS land-cover class means per-observation b AND stem factor
    modis_lc  = cygnssqc_df["modis_land_cover"].values.astype(np.float64)

    # (cell_id, day_ordinal) gives cell-mean NDVI interpolated to that day
    ndvi_cache: dict[tuple[int, int], float] = {}
    # cell_id to (ndvi_max, ndvi_min) for the year
    maxmin_cache: dict[int, tuple[float, float]] = {}

    for i in range(n):
        cell_id = int(cell_id_array[i])
        if cell_id < 0:
            continue

        key = (cell_id, int(date_in_number[i]))
        ndvi_now = ndvi_cache.get(key)
        if ndvi_now is None:
            ndvi_now = cell_ndvi.interpolate_day(cell_id, dates_dt[i])
            ndvi_cache[key] = ndvi_now
        mm = maxmin_cache.get(cell_id)
        if mm is None:
            mm = cell_ndvi.annual_maxmin(cell_id, year)
            maxmin_cache[cell_id] = mm
        ndvi_max, ndvi_min = mm

        # land cover for this observation, classes without a b get b = 0
        lc_int = int(modis_lc[i]) if np.isfinite(modis_lc[i]) else -1
        vwc   = vwc_hunt(ndvi_now, ndvi_min, ndvi_max, lc_int)
        b_obs = YUEH_B.get(lc_int, 0.0)

        Mcell_array[i] = b_obs * vwc
        obs_b_array[i] = b_obs
        vwc_array[i]   = vwc
        gamma_array[i] = gamma_db(vwc)

    # Calculate tau and SR_veg over all valid rows
    M_valid = np.isfinite(Mcell_array)
    tau_valid[M_valid] = tau_from_M(Mcell_array[M_valid], inc_angle[M_valid]).astype(np.float32)
    sr_veg[M_valid]   = correct_sr_db(sr_db[M_valid], tau_valid[M_valid],
                                       gamma_array[M_valid]).astype(np.float32)

    out = cygnssqc_df.drop(columns=["_cell_id"]).copy()
    out["sr_veg"] = sr_veg
    out["tau_valid"] = tau_valid
    out["M_cell_day"] = Mcell_array      # per-observation M = obs_b · VWC_obs
    out["obs_b"] = obs_b_array
    out["vwc_cell"] = vwc_array          # per-obs VWC (cell-mean NDVI, obs's stem factor)
    out["gamma_db"] = gamma_array

    # compress w snappy for smaller file sizes
    out.to_parquet(out_path, compression="snappy", index=False)

    return {
        "status":     "written",
        "n_rows":     int(len(out)),
        "elapsed_s":  time.time() - t0,
        "out_path":   out_path,
    }
