# top-3 dominant igbp classes per region by cell share

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import base                       # owns the sys.path setup, import first
import context as ctx
from bundles import veg_grids
from vegetation_correction.constants import IGBP_NAMES

YEAR, MONTH, TOP_N = 2023, 8, 3


def main():
    rows = []
    for region in base.STUDY_REGIONS:
        base.set_region(region)
        _, igbp = veg_grids(region, YEAR, MONTH)
        label = ctx.REGIONS[region]["label"]
        if igbp is None:
            print(f"  {region}: no sm month-file — skipped")
            continue

        cls = igbp[np.isfinite(igbp)].astype(int)
        cls = cls[np.isin(cls, list(IGBP_NAMES))]   # drop modis fill / non-igbp
        total = cls.size
        counts = pd.Series(cls).value_counts()

        for rank, (c, n) in enumerate(counts.head(TOP_N).items(), start=1):
            rows.append({
                "region": label,
                "rank": rank,
                "igbp": int(c),
                "class": IGBP_NAMES.get(int(c), str(c)),
                "cells": int(n),
                "pct": 100.0 * n / total,
            })

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200, "display.max_rows", None)
    print("\nTop-3 land-cover classes per region "
          f"({YEAR}-{MONTH:02d}, share of land cells on the EASE-2 9 km grid)\n")
    for label, g in df.groupby("region", sort=False):
        print(f"{label}")
        for _, r in g.iterrows():
            print(f"  {r['rank']}. {r['class']:<28} {r['pct']:5.1f}%  ({r['cells']} cells)")
        print()

    base.save_table(
        df[["region", "rank", "class", "pct"]].rename(columns={"pct": "pct_cells"}),
        "land_cover_top3", fmt="{:.1f}",
    )


if __name__ == "__main__":
    main()
