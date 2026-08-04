# delta r per correction stratified by dominant igbp class, mean over regions

import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import base
from bundles import delta_r_in_mask
from vegetation_correction.constants import IGBP_NAMES


def plot_delta_vs_igbp(bundles, year, month, sigma):
    refs = [("smap", "SMAP"), ("era5", "ERA5-Land")]
    # class 0 is modis fill, not a real igbp class
    classes = sorted({int(c) for b in bundles.values()
                      for c in np.unique(b["igbp"][np.isfinite(b["igbp"])])
                      if int(c) in IGBP_NAMES})

    # compute first, so classes no region has enough cells for can be dropped
    results = {}
    for ref, _ in refs:
        mean  = np.full((len(classes), len(base.CORR_VARIANTS)), np.nan)
        n_reg = np.zeros(len(classes), dtype=int)
        for k, c in enumerate(classes):
            per = []
            for b in bundles.values():
                d = delta_r_in_mask(b, b["igbp"] == c, ref)
                if d:
                    per.append([d[v] for v in base.CORR_VARIANTS])
            if per:
                mean[k]  = np.nanmean(np.array(per), axis=0)
                n_reg[k] = len(per)
        results[ref] = (mean, n_reg)

    keep = [k for k in range(len(classes))
            if any(results[ref][1][k] > 0 for ref, _ in refs)]
    classes = [classes[k] for k in keep]

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    rows = []
    for ax, (ref, ref_name) in zip(axes, refs):
        mean, n_reg = results[ref]
        mean, n_reg = mean[keep], n_reg[keep]

        x = np.arange(len(classes))
        w = 0.26
        for iv, v in enumerate(base.CORR_VARIANTS):
            ax.bar(x + (iv - 1) * w, mean[:, iv], width=w,
                   color=base.VARIANT_COLORS[v], label=base.short_variant(v))
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([IGBP_NAMES.get(c, str(c)) for c in classes],
                           rotation=40, ha="right", fontsize=8)
        ax.set_ylabel("Δr vs SR", fontsize=9)
        ax.set_title(f"Δr by IGBP class — vs {ref_name}", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

        for k, c in enumerate(classes):
            rows.append({"reference": ref_name, "igbp": c,
                         "class": IGBP_NAMES.get(c, str(c)), "n_regions": int(n_reg[k]),
                         **{f"dr_{base.short_variant(v).lower()}": mean[k, iv]
                            for iv, v in enumerate(base.CORR_VARIANTS)}})

    plt.suptitle(
        f"Change in correlation (Δr) vs land cover — mean over regions, "
        f"{datetime(year, month, 1).strftime('%B')} {year}",
        fontsize=11
    )
    plt.tight_layout()
    fpath = base.out_path("stratified", f"delta_r_vs_igbp_{year}{month:02d}_sigma{sigma}.png")
    plt.savefig(fpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(fpath.replace(".png", ".csv"), index=False)
    print(f"  Δr vs IGBP saved → {os.path.relpath(fpath)} (+ .csv)")
