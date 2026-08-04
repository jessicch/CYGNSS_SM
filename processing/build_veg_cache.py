# runs the vegetation correction on every qc month, writes sr_veg

import os
import sys
import argparse
import warnings
from datetime import datetime

from tqdm import tqdm

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    CYGNSS_QC_DIR, CYGNSS_VEG_DIR, MODISKM_DIR, REGIONS,
)
from vegetation_correction.pipeline import vegetation_correction


def build_region(region: str, year_start: int, year_end: int,
                 month_start: int = 1, month_end: int = 12,
                 force: bool = False) -> None:
    if region not in REGIONS:
        print(f"ERROR: region '{region}' not in REGIONS. "
              f"Available: {list(REGIONS.keys())}")
        return

    print("\n" + "=" * 70)
    print(f"  Vegetation correction — region={region}  "
          f"years={year_start}-{year_end}  months={month_start}-{month_end}")
    print(f"  QC src  : {os.path.relpath(os.path.join(CYGNSS_QC_DIR, region))}")
    print(f"  Veg dst : {os.path.relpath(os.path.join(CYGNSS_VEG_DIR, region))}")
    print(f"  MODIS km: {os.path.relpath(os.path.join(MODISKM_DIR, region))}")
    print(f"  Force   : {force}")
    print("=" * 70)

    targets: list[tuple[int, int]] = []
    for year in range(year_start, year_end + 1):
        m_lo = month_start if year == year_start else 1
        m_hi = month_end   if year == year_end   else 12
        for month in range(m_lo, m_hi + 1):
            targets.append((year, month))

    written = skipped = empty = missing_qc = no_igbp = 0
    total_rows = 0

    bar = tqdm(targets, desc=f"{region}", unit="mo", dynamic_ncols=True)
    for year, month in bar:
        bar.set_postfix_str(f"{year}-{month:02d}")
        stats = vegetation_correction(
            region=region, year=year, month=month,
            qc_dir=CYGNSS_QC_DIR, veg_dir=CYGNSS_VEG_DIR,
            modiskm_dir=MODISKM_DIR, force=force,
        )
        tag = f"{year}-{month:02d}"
        s = stats["status"]
        if s == "written":
            written += 1
            total_rows += stats["n_rows"]
            tqdm.write(
                f"  [WRITE] {tag}  rows={stats['n_rows']:>9,}  "
                f"time={stats['elapsed_s']:>6.1f}s  "
                f"-> {os.path.relpath(stats['out_path'])}"
            )
        elif s == "skipped":
            skipped += 1
            tqdm.write(f"  [SKIP ] {tag}  (exists; use --force to rebuild)")
        elif s == "no_qc":
            missing_qc += 1
            tqdm.write(f"  [NO QC] {tag}  no QC cache file -> nothing to correct")
        elif s == "no_igbp":
            no_igbp += 1
            tqdm.write(f"  [NOIGBP]{tag}  no MCD12Q1 raster in {MODISKM_DIR}/{region}/igbp/")
        else:
            empty += 1
            tqdm.write(f"  [EMPTY] {tag}  no observations to correct")

    bar.close()
    print("\n" + "-" * 70)
    print(f"  Done. written={written} skipped={skipped} empty={empty} "
          f"no_qc={missing_qc} no_igbp={no_igbp}  total_rows={total_rows:,}")
    print("-" * 70 + "\n")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build CYGNSS vegetation-corrected GeoParquet cache.",
    )
    p.add_argument("--region", type=str, default="pakistan",
                   help="Region key from config.settings.REGIONS (default: pakistan)")
    p.add_argument("--all-regions", action="store_true",
                   help="Build cache for every region in REGIONS")
    p.add_argument("--year", type=int, default=None,
                   help="Single year (shortcut for --year-start/--year-end)")
    p.add_argument("--year-start", type=int, default=2023)
    p.add_argument("--year-end",   type=int, default=2025)
    p.add_argument("--month-start", type=int, default=1)
    p.add_argument("--month-end",   type=int, default=12)
    p.add_argument("--force", action="store_true",
                   help="Rebuild even if a month-file already exists")
    args = p.parse_args()

    if args.year is not None:
        ys, ye = args.year, args.year
    else:
        ys, ye = args.year_start, args.year_end

    regions = list(REGIONS.keys()) if args.all_regions else [args.region]
    for r in regions:
        build_region(r, ys, ye,
                     month_start=args.month_start,
                     month_end=args.month_end,
                     force=args.force)


if __name__ == "__main__":
    main()
