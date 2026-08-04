# delta r per correction relative to plain sr, raw and smoothed

import numpy as np
import pandas as pd

import base
import context as ctx


def write_delta_tables(skill):
    names = {"sr_veg": "veg", "sr_rough": "rough", "sm": "sm"}
    for mode, suffix in (("raw", "raw"), ("smooth", "smoothed")):
        rows = []
        for region, per in skill.items():
            if "sr" not in per:
                continue
            row = {"region": ctx.REGIONS[region]["label"]}
            for v in base.CORR_VARIANTS:
                for ref, key in (("smap", "r_cs"), ("era5", "r_ce")):
                    col = f"dr_{names[v]}_{ref}"
                    row[col] = (per[v][f"{key}_{mode}"] - per["sr"][f"{key}_{mode}"]
                                if v in per else np.nan)
            rows.append(row)
        df = pd.DataFrame(rows)
        mean = {c: df[c].mean() for c in df.columns if c != "region"}
        mean["region"] = "Mean over regions"
        df = pd.concat([df, pd.DataFrame([mean])], ignore_index=True)
        base.save_table(df, f"delta_r_{suffix}", fmt="{:+.3f}")
