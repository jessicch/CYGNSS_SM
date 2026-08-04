from __future__ import annotations
import os
import time

import numpy as np
import pandas as pd

from .retrieval import retrieve_soil_moisture


# Paths
def _rough_month_path(rough_dir: str, region: str, year: int, month: int) -> str:
    return os.path.join(
        rough_dir, region, str(year),
        f"cygnss_rough_{region}_{year}{month:02d}.parquet",
    )


def _sm_month_path(sm_dir: str, region: str, year: int, month: int) -> str:
    folder = os.path.join(sm_dir, region, str(year))
    os.makedirs(folder, exist_ok=True)
    return os.path.join(
        folder, f"cygnss_sm_{region}_{year}{month:02d}.parquet",
    )


# retrieves soil moisture for one monthly file by inverting sr_rough,
# clay comes from the rough file itself
def retrieve_month(*,
                   region:    str,
                   year:      int,
                   month:     int,
                   rough_dir: str,
                   sm_dir:    str,
                   force:     bool = False):

    out_path = _sm_month_path(sm_dir, region, year, month)
    if os.path.exists(out_path) and not force:
        return {"status": "skipped", "n_rows": None, "n_retrieved": 0,
                "elapsed_s": 0.0, "out_path": out_path}

    rough_path = _rough_month_path(rough_dir, region, year, month)
    if not os.path.exists(rough_path):
        return {"status": "no_rough", "n_rows": 0, "n_retrieved": 0,
                "elapsed_s": 0.0, "out_path": out_path}

    t0 = time.time()

    import geopandas as gpd
    rough_df = gpd.read_parquet(rough_path)
    if rough_df.empty:
        return {"status": "empty", "n_rows": 0, "n_retrieved": 0,
                "elapsed_s": time.time() - t0, "out_path": out_path}

    # Clay was stamped per observation by apply_month; older rough files
    # predate the column and must be rebuilt (build_rough_apply --force).
    if "clay_fraction" not in rough_df.columns:
        return {"status": "no_clay", "n_rows": int(len(rough_df)),
                "n_retrieved": 0, "elapsed_s": time.time() - t0,
                "out_path": out_path}

    n = len(rough_df)
    sr_rough = rough_df["sr_rough"].values.astype(np.float64)   # smooth-surface reflectivity (dB)
    theta    = rough_df["sp_inc_angle"].values.astype(np.float64)
    clay     = rough_df["clay_fraction"].values.astype(np.float64)

    # Only invert rows where every input is finite.
    good = np.isfinite(sr_rough) & np.isfinite(clay) & np.isfinite(theta)

    sm_ret = np.full(n, np.nan, dtype=np.float64)
    if good.any():
        sm_ret[good] = retrieve_soil_moisture(sr_rough[good], clay[good], theta[good])

    out = rough_df.copy()
    out["sm_retrieved"] = sm_ret.astype(np.float32)
    out.to_parquet(out_path, compression="snappy", index=False)

    return {
        "status":      "written",
        "n_rows":      int(n),
        "n_retrieved": int(np.isfinite(sm_ret).sum()),
        "elapsed_s":   time.time() - t0,
        "out_path":    out_path,
    }
