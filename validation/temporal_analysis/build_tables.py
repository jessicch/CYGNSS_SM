# temporal skill tables and delta-r tables from npz results

import os
import sys
import argparse

import numpy as np
import pandas as pd

# make sibling modules + validation/ (for common) importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import context as ctx
from common import shared

# the six study regions, in the order they appear in the report
STUDY_REGIONS = ["pakistan", "mosambique", "central_african_republic",
                 "guatemala_honduras", "brasil", "mississippi"]

VARIANTS      = ["sr", "sr_veg", "sr_rough", "sm"]
CORR_VARIANTS = ["sr_veg", "sr_rough", "sm"]   # everything compared against plain sr

TABLES_DIR = os.path.join(ctx.OUTPUT_DIR, "tables")


def npz_path(region, sr_col, year_start, year_end):
    return os.path.join(
        ctx.OUTPUT_DIR, "data",
        f"{region}_temporal_{sr_col}_{year_start}_{year_end}.npz",
    )


# the scalar stats of one saved run, None if that run does not exist yet
def load_stats(region, sr_col, year_start, year_end):
    path = npz_path(region, sr_col, year_start, year_end)
    if not os.path.exists(path):
        return None
    res = np.load(path)
    return {k: float(res[k]) for k in (
        "r_cs_raw", "r_ce_raw", "nrmse_cs_raw", "nrmse_ce_raw",
        "r_cs_smooth", "r_ce_smooth", "nrmse_cs_smooth", "nrmse_ce_smooth",
    )}


# "sr_veg" -> "SR_veg" etc, keeps the table headers short
def short_variant(v):
    return shared.variant_label(v)["short"].replace("CYGNSS ", "")


# csv result table, same writer as the spatial report
def save_table(df, stem, fmt="{:.3f}"):
    os.makedirs(TABLES_DIR, exist_ok=True)
    csv_path = os.path.join(TABLES_DIR, f"{stem}.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Table saved → {os.path.relpath(csv_path)}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect temporal-analysis npz results into skill and delta-r tables"
    )
    parser.add_argument("--year-start", type=int, default=2023)
    parser.add_argument("--year-end",   type=int, default=2025)
    args = parser.parse_args()

    skill = {}
    for region in STUDY_REGIONS:
        per = {}
        for v in VARIANTS:
            s = load_stats(region, v, args.year_start, args.year_end)
            if s is None:
                print(f"  {region}/{v}: npz missing — run main.py for this combination")
                continue
            per[v] = s
        if per:
            skill[region] = per

    # skill tables, raw and smoothed
    for mode, suffix in (("raw", "raw"), ("smooth", "smoothed")):
        rows = []
        for region, per in skill.items():
            for k, v in enumerate(VARIANTS):
                if v not in per:
                    continue
                s = per[v]
                rows.append({
                    "region":     ctx.REGIONS[region]["label"] if k == 0 else "",
                    "variant":    short_variant(v),
                    "r_smap":     s[f"r_cs_{mode}"],
                    "nrmse_smap": s[f"nrmse_cs_{mode}"],
                    "r_era5":     s[f"r_ce_{mode}"],
                    "nrmse_era5": s[f"nrmse_ce_{mode}"],
                })
        save_table(pd.DataFrame(rows), f"skill_{suffix}")

    # delta-r tables: each corrected variant against plain sr, per reference
    for mode, suffix in (("raw", "raw"), ("smooth", "smoothed")):
        rows = []
        for region, per in skill.items():
            if "sr" not in per:
                continue
            row = {"region": ctx.REGIONS[region]["label"]}
            for v in CORR_VARIANTS:
                tag = v.replace("sr_", "")
                if v in per:
                    row[f"dr_{tag}_smap"] = per[v][f"r_cs_{mode}"] - per["sr"][f"r_cs_{mode}"]
                    row[f"dr_{tag}_era5"] = per[v][f"r_ce_{mode}"] - per["sr"][f"r_ce_{mode}"]
                else:
                    row[f"dr_{tag}_smap"] = np.nan
                    row[f"dr_{tag}_era5"] = np.nan
            rows.append(row)
        df = pd.DataFrame(rows)
        mean_row = {"region": "Mean over regions"}
        for c in df.columns[1:]:
            mean_row[c] = float(df[c].mean())
        df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
        save_table(df, f"delta_r_{suffix}")


if __name__ == "__main__":
    main()
