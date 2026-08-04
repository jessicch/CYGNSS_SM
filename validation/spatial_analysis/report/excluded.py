# map of observations removed by SM retrieval (too wet or too dry)

import os
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import base
import context as ctx
from plots import _map_aspect, _add_grid_labels

from config.settings import CYGNSS_SM_DIR
from processing.cygnss_sm import load_sm_cache_month
from roughness_correction.retrieval import (
    retrieve_soil_moisture, SM_RETRIEVAL_MIN, SM_RETRIEVAL_MAX,
)


def plot_excluded_obs(region, year, month):
    base.set_region(region)
    df = load_sm_cache_month(CYGNSS_SM_DIR, region, year, month)
    if df.empty:
        print(f"  {region}: no sm month-file — skipped")
        return
    df = df[df["sp_lat"].between(ctx.LAT_MIN, ctx.LAT_MAX) &
            df["sp_lon"].between(ctx.LON_MIN, ctx.LON_MAX)]

    sr_rough = df["sr_rough"].values.astype(float)
    clay     = df["clay_fraction"].values.astype(float)
    theta    = df["sp_inc_angle"].values.astype(float)
    sm       = df["sm_retrieved"].values.astype(float)

    have = np.isfinite(sr_rough) & np.isfinite(clay) & np.isfinite(theta)
    lost = have & ~np.isfinite(sm)

    # clamp=True pins out-of-range obs to the bounds, so wet and dry separate
    clamped = retrieve_soil_moisture(sr_rough[lost], clay[lost], theta[lost], clamp=True)
    wet = clamped >= SM_RETRIEVAL_MAX
    dry = clamped <= SM_RETRIEVAL_MIN
    n_lost = int(lost.sum())
    n_wet, n_dry = int(wet.sum()), int(dry.sum())
    print(f"  {region}: {n_lost:,} of {int(have.sum()):,} obs not retrievable "
          f"({n_wet:,} too wet, {n_dry:,} too dry)")
    if n_lost == 0:
        print(f"  {region}: nothing removed — no figure")
        return

    label  = ctx.REGIONS[region]["label"]
    long_m = datetime(year, month, 1).strftime("%B")
    lats = df["sp_lat"].values[lost]
    lons = df["sp_lon"].values[lost]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(lons, lats, color="tab:orange", s=8, linewidths=0)
    ax.set_xlim(ctx.LON_MIN, ctx.LON_MAX)
    ax.set_ylim(ctx.LAT_MIN, ctx.LAT_MAX)
    ax.set_aspect(_map_aspect())
    _add_grid_labels(ax)
    ax.grid(True, alpha=0.25, linewidth=0.5, color="0.7")   # white grid is invisible here
    ax.set_title(
        f"Observations Removed in Soil Moisture Inversion\n"
        f"{label}, {long_m} {year} — n={n_lost:,}", fontsize=11
    )
    plt.tight_layout()
    fpath = base.out_path("excluded", f"{region}_{year}{month:02d}_excluded_obs.png")
    plt.savefig(fpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Excluded-obs map saved → {os.path.relpath(fpath)}")
