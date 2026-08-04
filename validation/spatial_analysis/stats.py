import os
import sys
import numpy as np

# make validation/ importable so the shared common module resolves
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import shared


# full-scene Pearson r and nRMSE between CYGNSS and reference grids
def compute_stats(cygnss_grid, smap_grid, era5_grid):
    c = cygnss_grid.ravel()
    s = smap_grid.ravel()
    e = era5_grid.ravel()

    r_cs, nrmse_cs, n_cs = shared.corr_nrmse(c, s)
    r_ce, nrmse_ce, n_ce = shared.corr_nrmse(c, e)

    print("\n  ── Full-Scene Spatial Statistics ──")
    print(f"  {'Pair':<20}  {'r':>7}  {'nRMSE':>8}  {'n':>6}")
    print(f"  {'CYGNSS–SMAP':<20}  {r_cs:>7.4f}  {nrmse_cs:>8.4f}  {n_cs:>6}")
    print(f"  {'CYGNSS–ERA5':<20}  {r_ce:>7.4f}  {nrmse_ce:>8.4f}  {n_ce:>6}")

    return {
        "r_cygnss_smap":     r_cs,   "nrmse_cygnss_smap": nrmse_cs, "n_cs": n_cs,
        "r_cygnss_era5":     r_ce,   "nrmse_cygnss_era5": nrmse_ce, "n_ce": n_ce,
    }
