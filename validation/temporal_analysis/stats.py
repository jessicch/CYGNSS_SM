import os
import sys
import numpy as np
from scipy.ndimage import gaussian_filter1d

# make validation/ importable so the shared common module resolves
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import shared


# 1-D Gaussian smoothing; the weekly series are gap-free (verified for
# all regions and variants), so a NaN means that assumption broke and
# gaussian_filter1d would silently smear it across the series
def smooth_series(data, sigma):
    if sigma == 0 or np.isnan(sigma):
        return data.copy()
    if not np.isfinite(data).all():
        raise ValueError(
            "temporal series contains empty windows — the pipeline "
            "assumes every window has data (holds for 7-day windows)"
        )
    return gaussian_filter1d(data, sigma=sigma)


# Pearson r and nRMSE between CYGNSS, SMAP and ERA5
def compute_stats(cygnss, smap, era5, label=""):
    results = {}

    for series, name in [(cygnss, "CYGNSS SR"), (smap, "SMAP SM"), (era5, "ERA5 SM")]:
        fin = series[np.isfinite(series)]
        mean_v = float(np.mean(fin)) if fin.size else np.nan
        std_v  = float(np.std(fin))  if fin.size else np.nan
        cv_v   = std_v / abs(mean_v) if mean_v != 0 else np.nan
        results[name] = {"mean": mean_v, "sd": std_v, "cv": cv_v}

    r_cs, nrmse_cs, n_cs = shared.corr_nrmse(cygnss, smap)
    r_ce, nrmse_ce, n_ce = shared.corr_nrmse(cygnss, era5)
    results["r_cygnss_smap"]     = r_cs
    results["nrmse_cygnss_smap"] = nrmse_cs
    results["r_cygnss_era5"]     = r_ce
    results["nrmse_cygnss_era5"] = nrmse_ce
    results["n_cs"]              = n_cs
    results["n_ce"]              = n_ce

    print(f"\n  ── Statistics {label} ──")
    print(f"  {'Dataset':<14} {'Mean':>10}  {'SD':>8}  {'CV':>8}")
    for name in ("CYGNSS SR", "SMAP SM", "ERA5 SM"):
        s = results[name]
        print(f"  {name:<14} {s['mean']:>10.4f}  {s['sd']:>8.4f}  {s['cv']:>8.4f}")
    print()
    print(f"  {'Pair':<20}  {'r':>7}  {'nRMSE':>8}  {'n':>6}")
    print(f"  {'CYGNSS–SMAP':<20}  {r_cs:>7.4f}  {nrmse_cs:>8.4f}  {n_cs:>6}")
    print(f"  {'CYGNSS–ERA5':<20}  {r_ce:>7.4f}  {nrmse_ce:>8.4f}  {n_ce:>6}")

    return results
