# per-region vwc table, stratified igbp delta-r, and verification numbers

import os

import numpy as np
import pandas as pd

import base
import bundles
from common import shared

YEAR, MONTH, SIGMA = 2023, 8, 1.5

REGION_NAMES = {
    "pakistan": "Pakistan (Sindh)",
    "mosambique": "Mozambique",
    "central_african_republic": "Central African Republic",
    "guatemala_honduras": "Guatemala--Honduras",
    "brasil": "Brasil",
    "mississippi": "Mississippi, US",
}

IGBP_NAMES = {
    2: "Evergreen Broadleaf Forest", 5: "Mixed Forest", 7: "Open Shrublands",
    8: "Woody Savannas", 9: "Savannas", 10: "Grasslands",
    11: "Permanent Wetlands", 12: "Croplands",
    14: "Cropland/Natural Vegetation Mosaic", 16: "Barren or Sparsely Vegetated",
}


# same as bundles.region_bundle but keeps the raw grids next to the smoothed
def bundle_both(region):
    base.set_region(region)
    res = {v: base.load_results(region, v, YEAR, MONTH, SIGMA) for v in base.VARIANTS}
    if any(r is None for r in res.values()):
        return None
    vwc_grid, igbp_grid = bundles.veg_grids(region, YEAR, MONTH)
    if vwc_grid is None or vwc_grid.shape != res["sr"]["smap_smooth"].shape:
        return None
    out = {"vwc": vwc_grid, "igbp": igbp_grid}
    for kind, key in (("smooth", "smooth"), ("raw", "unsmooth")):
        b = {v: res[v][f"cygnss_{key}"] for v in base.VARIANTS}
        b.update(smap=res["sr"][f"smap_{key}"], era5=res["sr"][f"era5_{key}"],
                 vwc=vwc_grid)
        out[kind] = b
    return out


def main():
    first = base.load_results("pakistan", "sr", YEAR, MONTH, SIGMA)
    print("npz keys:", sorted(first.files))

    packs = {}
    for region in base.STUDY_REGIONS:
        p = bundle_both(region)
        if p is None:
            print(f"  {region}: incomplete — skipped")
            continue
        packs[region] = p

    # ---- per-region mean vwc ---------------------------------------------
    rows = []
    for region, p in packs.items():
        v = p["vwc"][np.isfinite(p["vwc"])]
        rows.append({"Region": REGION_NAMES[region],
                     "Mean VWC (kg/m2)": float(np.mean(v)),
                     "Share of cells VWC > 5 (%)": 100.0 * float(np.mean(v > 5.0))})
    df = pd.DataFrame(rows)
    base.save_table(df, "vwc_regions", fmt="{:.2f}")
    print(df.to_string(index=False))

    # ---- stratified igbp delta r, raw vs smoothed ------------------------
    strat_rows = []
    for ref in ("smap", "era5"):
        for cls, name in IGBP_NAMES.items():
            row = {"reference": ref, "class": name}
            for kind in ("smooth", "raw"):
                stack = []
                for p in packs.values():
                    m = p["igbp"] == cls
                    d = bundles.delta_r_in_mask(p[kind], m, ref)
                    stack.append([d[v] for v in base.CORR_VARIANTS] if d
                                 else [np.nan] * len(base.CORR_VARIANTS))
                mean = np.nanmean(np.array(stack, dtype=float), axis=0)
                for iv, v in enumerate(base.CORR_VARIANTS):
                    row[f"dr_{v}_{kind}"] = mean[iv]
            strat_rows.append(row)
    df = pd.DataFrame(strat_rows)
    base.save_table(df, "igbp_deltar_smoothed_vs_raw", fmt="{:.3f}")
    print(df.to_string(index=False))

    # ---- verification numbers -------------------------------------------
    tdir = os.path.join(os.path.dirname(base.REPORT_DIR), "..", "temporal")
    skill_raw = pd.read_csv(base.out_path("tables", "skill_raw.csv")).ffill()
    skill_smo = pd.read_csv(base.out_path("tables", "skill_smoothed.csv")).ffill()
    merged = skill_raw.merge(skill_smo, on=["region", "variant"], suffixes=("_raw", "_smo"))
    for sub, tag in ((merged, "all regions"),
                     (merged[merged.region != "Central African Republic"], "excl CAR")):
        print(f"spatial smoothing gain ({tag}): "
              f"dr_smap={ (sub.r_smap_smo - sub.r_smap_raw).mean():+.3f} "
              f"dr_era5={ (sub.r_era5_smo - sub.r_era5_raw).mean():+.3f} "
              f"dnrmse_smap={ (sub.nrmse_smap_smo - sub.nrmse_smap_raw).mean():+.3f} "
              f"dnrmse_era5={ (sub.nrmse_era5_smo - sub.nrmse_era5_raw).mean():+.3f}")

    tr = pd.read_csv(os.path.join(tdir, "tables", "skill_raw.csv")).ffill()
    sr = tr[tr.variant == "SR"]
    print(f"temporal raw SR mean r: smap={sr.r_smap.mean():.3f} era5={sr.r_era5.mean():.3f}")
    nb = sr[sr.region != "Brasil"]
    print(f"  excl Brasil:          smap={nb.r_smap.mean():.3f} era5={nb.r_era5.mean():.3f}")

    # smap vs era5 with each other, spatially and temporally
    for region, p in packs.items():
        r, _, n = shared.corr_nrmse(p["smooth"]["smap"], p["smooth"]["era5"])
        print(f"spatial smap-era5 (smoothed) {region}: r={r:+.3f} (n={n})")
    for region in base.STUDY_REGIONS:
        path = os.path.join(tdir, "data", f"{region}_temporal_sr_2023_2025.npz")
        if not os.path.exists(path):
            continue
        d = np.load(path)
        if region == "pakistan":
            print("temporal npz keys:", sorted(d.files))
        for kind in ("raw", "smooth"):
            s, e = d[f"smap_{kind}"], d[f"era5_{kind}"]
            r, _, n = shared.corr_nrmse(s, e)
            print(f"temporal smap-era5 ({kind}) {region}: r={r:+.3f} (n={n})")


if __name__ == "__main__":
    main()
