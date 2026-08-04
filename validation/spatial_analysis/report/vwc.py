# delta r per correction stratified by cell vwc, mean over regions

import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import base
from bundles import delta_r_in_mask


def plot_delta_vs_vwc(bundles, year, month, sigma):
    centers = np.arange(0.0, 9.0, 1.0)                      # whole-number VWC (kg/m²): 0,1,…,8
    edges   = np.append(centers - 0.5, centers[-1] + 0.5)   # 1.0-wide bins centred on the integers
    refs    = [("smap", "SMAP"), ("era5", "ERA5-Land")]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    rows = []
    for ax, (ref, ref_name) in zip(axes, refs):
        stack = []
        for b in bundles.values():
            arr = np.full((len(centers), len(base.CORR_VARIANTS)), np.nan)
            for k in range(len(centers)):
                m = np.isfinite(b["vwc"]) & (b["vwc"] >= edges[k]) & (b["vwc"] < edges[k + 1])
                d = delta_r_in_mask(b, m, ref)
                if d:
                    arr[k] = [d[v] for v in base.CORR_VARIANTS]
            stack.append(arr)
        stack = np.stack(stack)
        mean  = np.nanmean(stack, axis=0)
        n_reg = np.isfinite(stack[:, :, 0]).sum(axis=0)

        for iv, v in enumerate(base.CORR_VARIANTS):
            ax.plot(centers, mean[:, iv], marker="o", ms=4,
                    color=base.VARIANT_COLORS[v], label=base.short_variant(v))
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_xlabel("Cell VWC (kg m⁻²)", fontsize=9)
        ax.set_ylabel("Δr vs SR", fontsize=9)
        ax.set_title(f"Δr by VWC — vs {ref_name}", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        for k in range(len(centers)):
            rows.append({"reference": ref_name, "vwc_bin": centers[k],
                         "n_regions": int(n_reg[k]),
                         **{f"dr_{base.short_variant(v).lower()}": mean[k, iv]
                            for iv, v in enumerate(base.CORR_VARIANTS)}})

    plt.suptitle(
        f"Change in correlation (Δr) vs VWC — mean over regions, "
        f"{datetime(year, month, 1).strftime('%B')} {year}",
        fontsize=11
    )
    plt.tight_layout()
    fpath = base.out_path("stratified", f"delta_r_vs_vwc_{year}{month:02d}_sigma{sigma}.png")
    plt.savefig(fpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(fpath.replace(".png", ".csv"), index=False)
    print(f"  Δr vs VWC saved → {os.path.relpath(fpath)} (+ .csv)")
