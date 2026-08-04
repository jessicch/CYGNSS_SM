from __future__ import annotations
import os
import time

import numpy as np
import pandas as pd

from .ease2_grid import Ease2Grid

# Paths
def _veg_month_path(veg_dir: str, region: str, year: int, month: int) -> str:
    return os.path.join(
        veg_dir, region, str(year),
        f"cygnss_veg_{region}_{year}{month:02d}.parquet",
    )


def _rough_month_path(rough_dir: str, region: str, year: int, month: int) -> str:
    folder = os.path.join(rough_dir, region, str(year))
    os.makedirs(folder, exist_ok=True)
    return os.path.join(
        folder, f"cygnss_rough_{region}_{year}{month:02d}.parquet",
    )


def afs_table_path(rough_dir: str, region: str, train_year: int) -> str:
    return os.path.join(rough_dir, region, f"afs_{region}_{train_year}.parquet")


# Loads AFS table frmo parquet
def load_afs_table(rough_dir: str, region: str, train_year: int):
    path = afs_table_path(rough_dir, region, train_year)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"AFS calibration not found: {path}.  Run train_year() first." #debug
        )
    afs_df = pd.read_parquet(path)
    afs_lookup = dict(zip(afs_df["cell_id"].astype(int),
                          afs_df["afs_db"].astype(float)))
    return afs_df, afs_lookup, Ease2Grid()


# applies the afs correction to one monthly file, clay is stamped onto every
# observation for the retrieval step
def apply_month(*,
                region:          str,
                year:            int,
                month:           int,
                veg_dir:         str,
                rough_dir:       str,
                afs_lookup:      dict,
                grid:            Ease2Grid,
                clay_by_cell_id: dict,
                force:           bool = False):
    out_path = _rough_month_path(rough_dir, region, year, month)
    if os.path.exists(out_path) and not force:
        return {"status": "skipped", "n_rows": None, "n_matched": 0,
                "elapsed_s": 0.0, "out_path": out_path}

    veg_path = _veg_month_path(veg_dir, region, year, month)
    if not os.path.exists(veg_path):
        return {"status": "no_veg", "n_rows": 0, "n_matched": 0,
                "elapsed_s": 0.0, "out_path": out_path}

    t0 = time.time()

    import geopandas as gpd
    veg_month_df = gpd.read_parquet(veg_path)
    if veg_month_df.empty:
        return {"status": "empty", "n_rows": 0, "n_matched": 0,
                "elapsed_s": time.time() - t0, "out_path": out_path}

    n = len(veg_month_df)
    sp_lat = veg_month_df["sp_lat"].values.astype(np.float64)
    sp_lon = veg_month_df["sp_lon"].values.astype(np.float64)
    sr_veg = veg_month_df["sr_veg"].values.astype(np.float64)

    cell_id = grid.cell_id(sp_lat, sp_lon)
    row, col = grid.cell_of(sp_lat, sp_lon)

    sr_rough        = np.full(n, np.nan, dtype=np.float32)
    afs_db_used     = np.full(n, np.nan, dtype=np.float32)
    clay_fraction   = np.full(n, np.nan, dtype=np.float32)
    smap_cell_lat   = np.full(n, np.nan, dtype=np.float64)
    smap_cell_lon   = np.full(n, np.nan, dtype=np.float64)
    smap_cell_id    = cell_id.astype(np.int64)

    # Compute centre lat/lon for in-grid points 
    in_grid = cell_id >= 0
    if in_grid.any():
        clat, clon = grid.cell_center(row[in_grid], col[in_grid])
        smap_cell_lat[in_grid] = clat
        smap_cell_lon[in_grid] = clon

    # Per-obs clay lookup (static per cell), same dict-vectorisation as AFS.
    if in_grid.any() and clay_by_cell_id:
        cell_id_in = cell_id[in_grid]
        clay_arr = np.fromiter(
            (clay_by_cell_id.get(int(c), np.nan) for c in cell_id_in),
            dtype=np.float64, count=cell_id_in.size,
        )
        clay_fraction[in_grid] = clay_arr.astype(np.float32)

    # Per-obs AFS lookup. 
    if in_grid.any() and afs_lookup:
        cell_id_in = cell_id[in_grid]
        afs_arr = np.fromiter(
            (afs_lookup.get(int(c), np.nan) for c in cell_id_in),
            dtype=np.float64, count=cell_id_in.size,
        )
        afs_db_used[in_grid] = afs_arr.astype(np.float32)

        good = in_grid.copy()
        good[in_grid] = np.isfinite(afs_arr) & np.isfinite(sr_veg[in_grid])
        if good.any():
            # sr_rough = sr_veg - afs_db.
            sr_rough[good] = (sr_veg[good] - afs_db_used[good]).astype(np.float32)

    out = veg_month_df.copy()
    out["sr_rough"]      = sr_rough
    out["afs_db_used"]   = afs_db_used
    out["clay_fraction"] = clay_fraction
    out["smap_cell_id"]  = smap_cell_id
    out["smap_cell_lat"] = smap_cell_lat
    out["smap_cell_lon"] = smap_cell_lon
    out.to_parquet(out_path, compression="snappy", index=False)

    return {
        "status":    "written",
        "n_rows":    int(n),
        "n_matched": int(np.isfinite(sr_rough).sum()),
        "elapsed_s": time.time() - t0,
        "out_path":  out_path,
    }
