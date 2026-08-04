import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from datetime import datetime

# make sibling modules + validation/ (for common) importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import context as ctx
from common import shared
from grids import make_ease2_grid, bin_points_to_ease2, interpolate_grid_2d, smooth_2d, land_mask
from loaders import load_cygnss_month, load_smap_month, load_era5_month
from stats import compute_stats
from plots import plot_scatter_grid


def main():
    parser = argparse.ArgumentParser(
        description="Spatial analysis: CYGNSS SR vs SMAP/ERA5 SM"
    )
    parser.add_argument("--sigma", type=float, default=ctx.GAUSSIAN_SIGMA,
                        help=f"2-D Gaussian smoothing σ in grid cells (default: {ctx.GAUSSIAN_SIGMA})")
    parser.add_argument("--no-interp", action="store_true",
                        help="Skip 2-D interpolation; use binned grid only")
    parser.add_argument("--year",  type=int, default=ctx.YEAR)
    parser.add_argument("--month", type=int, default=ctx.MONTH)
    parser.add_argument("--sr-col", choices=["sr", "sr_veg", "sr_rough", "sm"],
                        default="sr",
                        help="'sr' (QC), 'sr_veg' (vegetation-corrected), "
                             "'sr_rough' (roughness-corrected), "
                             "'sm' (retrieved soil moisture).")
    parser.add_argument("--region", choices=list(ctx.REGIONS.keys()),
                        default=ctx.REGION,
                        help="Region from config.REGIONS (default: pakistan).")
    args = parser.parse_args()

    # route loader + region into the shared context
    ctx.SR_COL    = args.sr_col
    ctx.CACHE_DIR = shared.select_cache_dir(args.sr_col)
    ctx.REGION    = args.region
    _r            = ctx.REGIONS[ctx.REGION]
    ctx.LAT_MIN, ctx.LAT_MAX = _r["lat_min"], _r["lat_max"]
    ctx.LON_MIN, ctx.LON_MAX = _r["lon_min"], _r["lon_max"]

    sigma     = args.sigma
    year      = args.year
    month     = args.month
    do_interp = not args.no_interp

    # let the plots reflect the actual scene
    ctx.YEAR  = year
    ctx.MONTH = month

    month_str = datetime(year, month, 1).strftime("%B %Y")
    tag = f"{ctx.REGION}_{ctx.SR_COL}_{year}{month:02d}_ease9km_sigma{sigma}"

    # build the ease-2 9 km grid over the region
    lat_c, lon_c, row_south, col_west = make_ease2_grid(
        ctx.LAT_MIN, ctx.LAT_MAX, ctx.LON_MIN, ctx.LON_MAX
    )
    nlat = len(lat_c)
    nlon = len(lon_c)
    approx_km = sigma * 9.0

    print("\n" + "=" * 70)
    print("  Spatial Analysis — CYGNSS SR vs SMAP/ERA5 SM")
    print(f"  Region  : {ctx.REGIONS[ctx.REGION]['label']}  {ctx.LAT_MIN}°–{ctx.LAT_MAX}°N, {ctx.LON_MIN}°–{ctx.LON_MAX}°E")
    print(f"  Period  : {month_str}")
    print(f"  Grid    : EASE-2 9 km  ({nlat} × {nlon} cells)")
    print(f"  σ_G     : {sigma} cells ≈ {approx_km:.0f} km")
    print(f"  Interp  : {'2-D linear (Delaunay)' if do_interp else 'none'}")
    print(f"  Output  : {os.path.relpath(ctx.OUTPUT_DIR)}")
    print("=" * 70)

    # land/ocean mask so interpolated fields stop at the coastline
    lmask = land_mask(lat_c, lon_c)
    print(f"\n  Land mask: {int(lmask.sum())}/{lmask.size} cells on land")

    # prefer era5's own land-sea mask, fall back to natural earth
    era5_lmask = shared.era5_land_mask_on_grid(ctx.REGION, lat_c, lon_c)
    if era5_lmask is None:
        era5_lmask = lmask
        print("  ERA5 LSM : none found — using Natural Earth mask for ERA5")
    else:
        print(f"  ERA5 LSM : {int(era5_lmask.sum())}/{era5_lmask.size} cells on land")

    # ── 1. CYGNSS ──
    print(f"\n[Step 1]  Loading CYGNSS for {month_str} ...")
    cygnss_df = load_cygnss_month(year, month)
    print(f"  Observations after QC  : {len(cygnss_df):,}")

    if cygnss_df.empty:
        print("  ERROR: No CYGNSS data found — check the cache and file naming.")
        sys.exit(1)

    cygnss_binned, _ = bin_points_to_ease2(
        cygnss_df["sp_lat"].values, cygnss_df["sp_lon"].values,
        cygnss_df["sr"].values,
        nlat, nlon, row_south, col_west
    )
    filled = int(np.isfinite(cygnss_binned).sum())
    total  = cygnss_binned.size
    print(f"  Grid cells with data   : {filled}/{total}  ({100*filled/total:.1f}%)")

    if do_interp:
        print("  Applying 2-D linear interpolation (land cells only) ...")
        cygnss_grid = interpolate_grid_2d(cygnss_binned, lat_c, lon_c)
        cygnss_grid[~lmask] = np.nan   # clip the interpolation to the coastline
        after = int(np.isfinite(cygnss_grid).sum())
        print(f"  Grid cells after interp: {after}/{total}  ({100*after/total:.1f}%)")
    else:
        cygnss_grid = cygnss_binned.copy()

    # ── 2. SMAP ──
    print(f"\n[Step 2]  Loading SMAP for {month_str} ...")
    smap_grid_raw = load_smap_month(year, month, nlat, nlon, row_south, col_west)
    smap_grid_raw[~lmask] = np.nan   # clip to the coastline
    print(f"  Grid cells with data   : {np.isfinite(smap_grid_raw).sum()}/{smap_grid_raw.size}")

    # ── 3. ERA5 ──
    print(f"\n[Step 3]  Loading ERA5-Land for {month_str} ...")
    era5_grid_raw = load_era5_month(year, month, lat_c, lon_c)
    era5_grid_raw[~(era5_lmask & lmask)] = np.nan   # clip to ERA5's own LSM and the coastline
    print(f"  Grid cells with data   : {np.isfinite(era5_grid_raw).sum()}/{era5_grid_raw.size}")

    # ── 4. Gaussian smoothing ──
    print(f"\n[Step 4]  Applying 2-D Gaussian smoothing (σ={sigma}) ...")
    cygnss_smooth = smooth_2d(cygnss_grid,  sigma)
    smap_smooth   = smooth_2d(smap_grid_raw, sigma)
    era5_smooth   = smooth_2d(era5_grid_raw, sigma)

    # ── 5. Full-scene statistics ──
    print(f"\n[Step 5]  Computing full-scene statistics ...")
    stats_raw    = compute_stats(cygnss_grid,  smap_grid_raw, era5_grid_raw)
    print("  (above: unsmoothed)")
    stats_smooth = compute_stats(cygnss_smooth, smap_smooth,  era5_smooth)
    print(f"  (above: smoothed, σ={sigma})")

    # each output type gets its own subfolder under OUTPUT_DIR
    def out_path(sub: str, fname: str) -> str:
        d = os.path.join(ctx.OUTPUT_DIR, sub)
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, fname)

    # ── 6. Save results ──
    npz_path = out_path("npz", f"{tag}_results.npz")
    np.savez_compressed(
        npz_path,
        lat_centres      = lat_c,
        lon_centres      = lon_c,
        cygnss_smooth    = cygnss_smooth,
        smap_smooth      = smap_smooth,
        era5_smooth      = era5_smooth,
        cygnss_unsmooth  = cygnss_grid,
        smap_unsmooth    = smap_grid_raw,
        era5_unsmooth    = era5_grid_raw,
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

    # ── 7. All-variant scatter grid (reads the npz of every variant) ──
    plot_scatter_grid(
        year, month, sigma,
        out_path("scatter", f"{ctx.REGION}_{year}{month:02d}_ease9km_sigma{sigma}_scatter_grid.png")
    )

    print("\n" + "=" * 70)
    print("  COMPLETE")
    print(f"  Unsmoothed  CYGNSS–SMAP r={stats_raw['r_cygnss_smap']:.4f}  "
          f"nRMSE={stats_raw['nrmse_cygnss_smap']:.4f}  |  "
          f"CYGNSS–ERA5 r={stats_raw['r_cygnss_era5']:.4f}  "
          f"nRMSE={stats_raw['nrmse_cygnss_era5']:.4f}")
    print(f"  σ_G={sigma}      CYGNSS–SMAP r={stats_smooth['r_cygnss_smap']:.4f}  "
          f"nRMSE={stats_smooth['nrmse_cygnss_smap']:.4f}  |  "
          f"CYGNSS–ERA5 r={stats_smooth['r_cygnss_era5']:.4f}  "
          f"nRMSE={stats_smooth['nrmse_cygnss_era5']:.4f}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
