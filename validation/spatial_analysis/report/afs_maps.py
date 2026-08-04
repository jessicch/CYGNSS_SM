# map of trained afs per cell, greyscale (rougher = lighter)

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import base
import context as ctx
from grids import make_ease2_grid
from plots import _map_aspect, _add_grid_labels

from config.settings import CYGNSS_ROUGH_DIR
from roughness_correction.apply_pipeline import afs_table_path


def plot_afs_map(region, train_tag):
    base.set_region(region)
    path = afs_table_path(CYGNSS_ROUGH_DIR, region, train_tag)
    if not os.path.exists(path):
        print(f"  {region}: no afs table for {train_tag} — skipped")
        return
    afs = pd.read_parquet(path)

    lat_c, lon_c, row_south, col_west = make_ease2_grid(
        ctx.LAT_MIN, ctx.LAT_MAX, ctx.LON_MIN, ctx.LON_MAX
    )
    nlat, nlon = len(lat_c), len(lon_c)
    grid = np.full((nlat, nlon), np.nan)
    i = row_south - afs["row"].values
    j = afs["col"].values - col_west
    ok = (i >= 0) & (i < nlat) & (j >= 0) & (j < nlon)
    grid[i[ok], j[ok]] = afs["afs_db"].values[ok]

    vals = grid[np.isfinite(grid)]
    vmin = float(np.percentile(vals, 2))
    vmax = float(np.percentile(vals, 98))
    label  = ctx.REGIONS[region]["label"]
    extent = [ctx.LON_MIN, ctx.LON_MAX, ctx.LAT_MIN, ctx.LAT_MAX]

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(grid, origin="lower", extent=extent, aspect=_map_aspect(),
                   cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
    plt.colorbar(im, ax=ax, label="AFS (dB)", fraction=0.046, pad=0.04)
    ax.set_title(f"Roughness Calibration (AFS)\n{label}, trained {train_tag}", fontsize=11)
    _add_grid_labels(ax)
    plt.tight_layout()
    fpath = base.out_path("afs_maps", f"{region}_afs_{train_tag}.png")
    plt.savefig(fpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  AFS map saved → {os.path.relpath(fpath)}  ({int(ok.sum())} cells)")
