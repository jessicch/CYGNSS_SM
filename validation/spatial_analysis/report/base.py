# shared report utilities: region list, npz loading, table writer

import os
import sys

import numpy as np

# make report/, spatial_analysis/, validation/ (for common) + repo root importable
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(_HERE))))

import context as ctx
from common import shared

# the six study regions, in the order they appear in the report
STUDY_REGIONS = ["pakistan", "mosambique", "central_african_republic",
                 "guatemala_honduras", "brasil", "mississippi"]

VARIANTS      = ["sr", "sr_veg", "sr_rough", "sm"]
CORR_VARIANTS = ["sr_veg", "sr_rough", "sm"]   # everything compared against plain sr

# one colour per correction, shared by the stratified figures
VARIANT_COLORS = {"sr_veg": "seagreen", "sr_rough": "lightblue", "sm": "darkblue"}

# bins thinner than this give meaningless correlations
MIN_CELLS_PER_BIN = 30

REPORT_DIR = os.path.join(ctx.OUTPUT_DIR, "report")


def out_path(sub, fname):
    d = os.path.join(REPORT_DIR, sub)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, fname)


# route a region into the shared context, same as main.py does
def set_region(region):
    ctx.REGION = region
    _r = ctx.REGIONS[region]
    ctx.LAT_MIN, ctx.LAT_MAX = _r["lat_min"], _r["lat_max"]
    ctx.LON_MIN, ctx.LON_MAX = _r["lon_min"], _r["lon_max"]


def npz_path(region, sr_col, year, month, sigma):
    return os.path.join(
        ctx.OUTPUT_DIR, "npz",
        f"{region}_{sr_col}_{year}{month:02d}_ease9km_sigma{sigma}_results.npz",
    )


# one saved npz, None if that scene has not been run yet
def load_results(region, sr_col, year, month, sigma):
    path = npz_path(region, sr_col, year, month, sigma)
    if not os.path.exists(path):
        return None
    return np.load(path)


# "sr_veg" to "SR_veg" etc, keeps the table headers short
def short_variant(v):
    return shared.variant_label(v)["short"].replace("CYGNSS ", "")


# full-scene r / nrmse scalars for every region and variant, straight from the npz
def collect_skill(year, month, sigma):
    skill = {}
    for region in STUDY_REGIONS:
        per = {}
        for v in VARIANTS:
            res = load_results(region, v, year, month, sigma)
            if res is None:
                print(f"  {region}/{v}: npz missing — run main.py for this scene")
                continue
            per[v] = {k: float(res[k]) for k in (
                "r_cs_raw", "r_ce_raw", "nrmse_cs_raw", "nrmse_ce_raw",
                "r_cs_smooth", "r_ce_smooth", "nrmse_cs_smooth", "nrmse_ce_smooth",
            )}
        if per:
            skill[region] = per
    return skill


# csv result table for the report — one file per stem
def save_table(df, stem, fmt="{:.3f}"):
    csv_path = out_path("tables", f"{stem}.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Table saved → {os.path.relpath(csv_path)}")
