# six-panel montage per region (sr, sr_veg, sr_rough, sm, smap, era5)

import os
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import base
import context as ctx
from common import shared
from plots import _map_aspect, _add_grid_labels


def plot_region_montage(region, year, month, sigma):
    base.set_region(region)
    res = {v: base.load_results(region, v, year, month, sigma)
           for v in base.VARIANTS}
    missing = [v for v, r in res.items() if r is None]
    if missing:
        print(f"  {region}: npz missing for {missing} — montage skipped")
        return

    smap = res["sr"]["smap_smooth"]
    era5 = res["sr"]["era5_smooth"]
    label   = ctx.REGIONS[region]["label"]
    short_m = datetime(year, month, 1).strftime("%b")
    long_m  = datetime(year, month, 1).strftime("%B")
    extent  = [ctx.LON_MIN, ctx.LON_MAX, ctx.LAT_MIN, ctx.LAT_MAX]

    def pct(grid):
        vals = grid[np.isfinite(grid)]
        if not vals.size:
            return None, None
        return float(np.percentile(vals, 2)), float(np.percentile(vals, 98))

    # smap, era5 and retrieved sm share one colour scale, sr panels get their own
    sm_vals = np.concatenate([smap[np.isfinite(smap)], era5[np.isfinite(era5)]])
    sm_scale = ((float(np.percentile(sm_vals, 2)), float(np.percentile(sm_vals, 98)))
                if sm_vals.size else (0.0, 0.5))

    panels = [
        (res["sr"]["cygnss_smooth"],       "sr",                       None),
        (res["sr_rough"]["cygnss_smooth"], "sr_rough",                 None),
        (smap,                             "SMAP Soil Moisture",       sm_scale),
        (res["sr_veg"]["cygnss_smooth"],   "sr_veg",                   None),
        (res["sm"]["cygnss_smooth"],       "sm",                       sm_scale),
        (era5,                             "ERA5-Land Soil Moisture",  sm_scale),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for ax, (grid, name, scale) in zip(axes.ravel(), panels):
        if name in shared.VARIANT_LABELS:
            lab = shared.variant_label(name)
            title, cbar_label = lab["short"], lab["cbar"]
        else:
            title, cbar_label = name, "SM (m³ m⁻³)"
        lo, hi = scale if scale else pct(grid)
        im = ax.imshow(grid, origin="lower", extent=extent, aspect=_map_aspect(),
                       cmap="YlGnBu", vmin=lo, vmax=hi, interpolation="nearest")
        plt.colorbar(im, ax=ax, label=cbar_label, fraction=0.046, pad=0.04)
        ax.set_title(f"{title}\n({short_m} {year}, σ_G={sigma})", fontsize=10)
        _add_grid_labels(ax)

    plt.suptitle(
        f"Spatial Distribution — {label}, {long_m} {year}  "
        f"(grid=EASE-2 9 km, σ_G={sigma})",
        fontsize=12, y=1.0
    )
    plt.tight_layout()
    fpath = base.out_path("montage",
                          f"{region}_{year}{month:02d}_ease9km_sigma{sigma}_montage.png")
    plt.savefig(fpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Montage saved → {os.path.relpath(fpath)}")
