import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from datetime import datetime, timedelta

# make sibling modules + validation/ (for common) importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import context as ctx
from common import shared
from loaders import build_windows, build_cygnss_series, build_smap_series, build_era5_series
from stats import smooth_series, compute_stats
from plots import plot_triple_timeseries, plot_scatter


def main():
    parser = argparse.ArgumentParser(
        description="Temporal analysis: CYGNSS SR vs SMAP/ERA5 SM"
    )
    parser.add_argument(
        "--sigma-cygnss", type=float, default=float(ctx.SIGMA_TEMPORAL),
        help=f"Smoothing sigma for CYGNSS SR in window steps (default: {ctx.SIGMA_TEMPORAL})"
    )
    parser.add_argument(
        "--sigma-sm", type=float, default=float(ctx.SIGMA_TEMPORAL),
        help=f"Smoothing sigma for SMAP/ERA5 SM in window steps (default: {ctx.SIGMA_TEMPORAL})"
    )
    parser.add_argument(
        "--force-reprocess", action="store_true",
        help="Ignore cached CYGNSS time series and reprocess from raw files"
    )
    parser.add_argument(
        "--window-days", type=int, default=ctx.WINDOW_DAYS,
        help=f"Window length in days for the averages (default: {ctx.WINDOW_DAYS})"
    )
    parser.add_argument(
        "--year-start", type=int, default=ctx.START_DATE.year,
        help=f"Start year (default: {ctx.START_DATE.year})"
    )
    parser.add_argument(
        "--year-end", type=int, default=ctx.END_DATE.year,
        help=f"End year inclusive (default: {ctx.END_DATE.year})"
    )
    parser.add_argument(
        "--sr-col", choices=["sr", "sr_veg", "sr_rough", "sm"], default="sr",
        help="'sr' = QC cache (uncorrected); "
             "'sr_veg' = vegetation-corrected cache; "
             "'sr_rough' = roughness-corrected cache; "
             "'sm' = retrieved soil moisture (cygnss_sm cache)."
    )
    parser.add_argument(
        "--region", type=str, default="pakistan",
        choices=list(ctx.REGIONS.keys()),
        help="Which REGIONS entry to analyse (sets the bbox for clipping)."
    )
    args = parser.parse_args()

    # route loader + region into the shared context
    ctx.SR_COL      = args.sr_col
    ctx.CACHE_DIR   = shared.select_cache_dir(args.sr_col)
    ctx.REGION      = args.region
    ctx.WINDOW_DAYS = args.window_days
    _r            = ctx.REGIONS[ctx.REGION]
    ctx.LAT_MIN, ctx.LAT_MAX = _r["lat_min"], _r["lat_max"]
    ctx.LON_MIN, ctx.LON_MAX = _r["lon_min"], _r["lon_max"]

    sigma_cygnss = args.sigma_cygnss
    sigma_sm     = args.sigma_sm
    force        = args.force_reprocess
    # --year-end is inclusive, windows run to jan 1 of the next year
    start_date   = datetime(args.year_start, 1, 1)
    end_date     = datetime(args.year_end + 1, 1, 1)
    start_year   = start_date.year
    end_year     = args.year_end

    print("\n" + "=" * 70)
    print(f"  {ctx.REGIONS[ctx.REGION]['label']} Temporal Analysis")
    print(f"  Period  : {start_date.date()} → {(end_date - timedelta(days=1)).date()}  (inclusive)")
    print(f"  Region  : {ctx.LAT_MIN}°–{ctx.LAT_MAX}°N, {ctx.LON_MIN}°–{ctx.LON_MAX}°E")
    print(f"  Window  : {ctx.WINDOW_DAYS}-day averages")
    print(f"  σ_G(CYGNSS) : {sigma_cygnss} steps = {sigma_cygnss * ctx.WINDOW_DAYS:.0f} days")
    print(f"  σ_G(SM)     : {sigma_sm} steps = {sigma_sm * ctx.WINDOW_DAYS:.0f} days")
    print("=" * 70)

    # step 1: build windows
    windows = build_windows(start_date, end_date, ctx.WINDOW_DAYS)
    # centre dates for the x-axis
    dates = np.array([
        w[0] + timedelta(days=ctx.WINDOW_DAYS / 2) for w in windows
    ])
    print(f"\n  {len(windows)} windows built  "
          f"({dates[0].strftime('%Y-%m-%d')} → {dates[-1].strftime('%Y-%m-%d')})")

    # step 2: CYGNSS
    # window length is part of the name so 5-day and weekly caches never mix
    cache_path = os.path.join(
        ctx.OUTPUT_DIR, "data",
        f"{ctx.REGION}_cygnss_cache_{ctx.SR_COL}_{start_year}_{end_year}_"
        f"win{ctx.WINDOW_DAYS}d_dailymean.npz"
    )
    variant = shared.variant_label(ctx.SR_COL)
    print(f"\n[Step 2]  {variant['short']} time series ...")
    cygnss_raw = build_cygnss_series(windows, cache_path, force=force)

    # sr stays on its absolute db scale, r and nrmse ignore the offset anyway
    print(f"  {variant['short']}: ", end="")
    print(f"min={np.nanmin(cygnss_raw):.2f} {variant['unit']}  "
          f"max={np.nanmax(cygnss_raw):.2f} {variant['unit']}  "
          f"valid windows={np.isfinite(cygnss_raw).sum()}/{len(windows)}")

    # step 3: SMAP
    print(f"\n[Step 3]  SMAP SM time series ...")
    smap_raw = build_smap_series(windows)
    print(f"  SMAP SM: mean={np.nanmean(smap_raw):.4f}  "
          f"valid={np.isfinite(smap_raw).sum()}/{len(windows)}")

    # step 4: ERA5
    print(f"\n[Step 4]  ERA5-Land SM time series ...")
    era5_raw = build_era5_series(windows)
    print(f"  ERA5 SM: mean={np.nanmean(era5_raw):.4f}  "
          f"valid={np.isfinite(era5_raw).sum()}/{len(windows)}")

    # step 5: smooth
    print(f"\n[Step 5]  Gaussian temporal smoothing "
          f"(σ_G CYGNSS={sigma_cygnss}, SM={sigma_sm}) ...")
    cygnss_smooth = smooth_series(cygnss_raw, sigma_cygnss)
    smap_smooth   = smooth_series(smap_raw,   sigma_sm)
    era5_smooth   = smooth_series(era5_raw,   sigma_sm)

    # step 6: stats
    print(f"\n[Step 6]  Statistics")
    stats_raw    = compute_stats(cygnss_raw,    smap_raw,   era5_raw,
                                 label=f"unsmoothed (σ_G=0)")
    stats_smooth = compute_stats(cygnss_smooth, smap_smooth, era5_smooth,
                                 label=f"smoothed (CYGNSS σ_G={sigma_cygnss}, SM σ_G={sigma_sm})")

    # step 7: plots
    print(f"\n[Step 7]  Generating plots ...")
    plot_triple_timeseries(
        dates, cygnss_smooth, smap_smooth, era5_smooth,
        cygnss_raw, smap_raw, era5_raw,
        sigma_cygnss, sigma_sm,
        os.path.join(ctx.OUTPUT_DIR, f"{ctx.REGION}_temporal_triple_{ctx.SR_COL}_{start_year}_{end_year}.png")
    )
    plot_scatter(
        cygnss_smooth, smap_smooth, era5_smooth,
        os.path.join(ctx.OUTPUT_DIR, f"{ctx.REGION}_temporal_scatter_{ctx.SR_COL}_{start_year}_{end_year}.png"),
        start_date=start_date, end_date=end_date - timedelta(days=1)
    )

    # step 8: save results
    npz_path = os.path.join(
        ctx.OUTPUT_DIR, "data",
        f"{ctx.REGION}_temporal_{ctx.SR_COL}_{start_year}_{end_year}.npz"
    )
    np.savez_compressed(
        npz_path,
        dates            = np.array([str(d.date()) for d in dates]),
        cygnss_sr_raw    = cygnss_raw,
        cygnss_sr_smooth = cygnss_smooth,
        smap_raw         = smap_raw,
        smap_smooth      = smap_smooth,
        era5_raw         = era5_raw,
        era5_smooth      = era5_smooth,
        r_cs_raw         = np.array(stats_raw["r_cygnss_smap"]),
        r_ce_raw         = np.array(stats_raw["r_cygnss_era5"]),
        r_cs_smooth      = np.array(stats_smooth["r_cygnss_smap"]),
        r_ce_smooth      = np.array(stats_smooth["r_cygnss_era5"]),
        nrmse_cs_raw     = np.array(stats_raw["nrmse_cygnss_smap"]),
        nrmse_ce_raw     = np.array(stats_raw["nrmse_cygnss_era5"]),
        nrmse_cs_smooth  = np.array(stats_smooth["nrmse_cygnss_smap"]),
        nrmse_ce_smooth  = np.array(stats_smooth["nrmse_cygnss_era5"]),
    )
    print(f"\n  Results saved → {os.path.relpath(npz_path)}")

    print("\n" + "=" * 70)
    print(f"  COMPLETE")
    print(f"  σ=0   CYGNSS–SMAP r = {stats_raw['r_cygnss_smap']:.4f}   "
          f"CYGNSS–ERA5 r = {stats_raw['r_cygnss_era5']:.4f}")
    print(f"  σ_C={sigma_cygnss:.0f}/σ_S={sigma_sm:.0f} "
          f"CYGNSS–SMAP r = {stats_smooth['r_cygnss_smap']:.4f}   "
          f"CYGNSS–ERA5 r = {stats_smooth['r_cygnss_era5']:.4f}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
