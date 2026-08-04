# roughness training step. pools years of cygnss x smap matchups into per-cell afs table. 

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    CYGNSS_VEG_DIR, CYGNSS_ROUGH_DIR, SMAP_DIR, REGIONS,
)
from roughness_correction.train_pipeline import train_years


def main() -> None:
    p = argparse.ArgumentParser(
        description="Train per-cell AFS_dB roughness calibration "
                    "from one or more years of CYGNSS + SMAP.",
    )
    p.add_argument("--region", required=True,
                   help=f"Region key (choices: {list(REGIONS.keys())})")
    p.add_argument("--year", type=int, default=None,
                   help="Single calibration year (e.g. 2023)")
    p.add_argument("--year-start", type=int, default=None,
                   help="First calibration year (use with --year-end)")
    p.add_argument("--year-end", type=int, default=None,
                   help="Last calibration year (inclusive)")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing AFS parquet")
    args = p.parse_args()

    if args.region not in REGIONS:
        p.error(f"unknown region {args.region!r}. Available: {list(REGIONS.keys())}")

    if args.year is not None:
        years = [args.year]
    elif args.year_start is not None and args.year_end is not None:
        years = list(range(args.year_start, args.year_end + 1))
    else:
        p.error("Provide either --year or both --year-start and --year-end.")

    year_label = f"{years[0]}-{years[-1]}" if len(years) > 1 else str(years[0])

    print("\n" + "=" * 70)
    print(f"  Roughness training - region={args.region}  years={year_label}")
    print(f"  CYGNSS veg src : {os.path.relpath(os.path.join(CYGNSS_VEG_DIR, args.region))}")
    print(f"  SMAP  src      : {os.path.relpath(os.path.join(SMAP_DIR, args.region))}")
    print(f"  AFS dst        : {os.path.relpath(os.path.join(CYGNSS_ROUGH_DIR, args.region))}")
    print(f"  Force          : {args.force}")
    print("=" * 70)

    stats = train_years(
        region=args.region,
        years=years,
        veg_dir=CYGNSS_VEG_DIR,
        smap_dir=SMAP_DIR,
        rough_dir=CYGNSS_ROUGH_DIR,
        force=args.force,
    )

    print("\n" + "-" * 70)
    if stats["status"] == "written":
        print(f"  [WRITE] cells={stats['n_cells']:,}  obs={stats['n_obs']:,}  "
              f"days_used={stats['n_days_used']:,}  "
              f"time={stats['elapsed_s']:.1f}s")
        print(f"  -> {os.path.relpath(stats['out_path'])}")
    elif stats["status"] == "skipped":
        print(f"  [SKIP ] AFS table already exists (use --force to rebuild):")
        print(f"          {os.path.relpath(stats['out_path'])}")
    else:
        print(f"  [{stats['status'].upper()}] no AFS table written. {stats}")
    print("-" * 70 + "\n")


if __name__ == "__main__":
    main()
