# bias and ubRMSE of SM series against references (m³/m³ units)

import os
import sys
import argparse

import numpy as np
import pandas as pd

# make sibling modules + validation/ (for common) importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import context as ctx
from build_tables import STUDY_REGIONS, npz_path, save_table


# bias = mean(sm − ref); ubRMSE = std(sm − ref), i.e. the RMSE after the
# bias is removed — the convention the CYGNSS/SMAP literature reports
def bias_ubrmse(sm, ref):
    m = np.isfinite(sm) & np.isfinite(ref)
    d = sm[m] - ref[m]
    return float(np.mean(d)), float(np.std(d))


def main():
    parser = argparse.ArgumentParser(
        description="Bias and ubRMSE (m³/m³) of the weekly SM series per region"
    )
    parser.add_argument("--year-start", type=int, default=2023)
    parser.add_argument("--year-end",   type=int, default=2025)
    args = parser.parse_args()

    # presentable headers, in the order pandas will render them
    C_REGION = "Region"
    C_BS, C_US = "Bias (SMAP)", "ubRMSE (SMAP)"
    C_BE, C_UE = "Bias (ERA5)", "ubRMSE (ERA5)"

    rows = []
    for region in STUDY_REGIONS:
        path = npz_path(region, "sm", args.year_start, args.year_end)
        if not os.path.exists(path):
            print(f"  {region}/sm: npz missing — run main.py for this combination")
            continue
        res = np.load(path)
        sm  = res["cygnss_sr_raw"]   # for the sm variant this is SM in m³/m³
        b_s, u_s = bias_ubrmse(sm, res["smap_raw"])
        b_e, u_e = bias_ubrmse(sm, res["era5_raw"])
        rows.append({
            C_REGION: ctx.REGIONS[region]["label"],
            C_BS: b_s, C_US: u_s,
            C_BE: b_e, C_UE: u_e,
        })

    df = pd.DataFrame(rows)
    mean_row = {C_REGION: "Mean over regions"}
    for c in df.columns[1:]:
        mean_row[c] = float(df[c].mean())
    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
    save_table(df, "sm_bias_ubrmse")


if __name__ == "__main__":
    main()
