# roughness apply. sr_rough = sr_veg - afs_db for every veg month

import argparse
import os
import sys
import warnings

from tqdm import tqdm

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    CYGNSS_VEG_DIR, CYGNSS_ROUGH_DIR, SMAP_DIR, REGIONS,
)
from roughness_correction.apply_pipeline import apply_month, load_afs_table
from roughness_correction.smap_io import load_clay_index


def main() -> None:
    p = argparse.ArgumentParser(
        description="Apply per-cell AFS_dB roughness calibration to "
                    "vegetation-corrected CYGNSS years.",
    )
    p.add_argument("--region", required=True,
                   help=f"Region key (choices: {list(REGIONS.keys())})")
    p.add_argument("--train-tag", default=None,
                   help="AFS table tag to use as calibration, e.g. '2021' or "
                        "'2019-2021'. Overrides --train-year.")
    p.add_argument("--train-year", type=int, default=None,
                   help="Single-year AFS table to use (shortcut for --train-tag)")
    p.add_argument("--year", type=int, default=None,
                   help="Single year to correct (shortcut)")
    p.add_argument("--year-start", type=int, default=None)
    p.add_argument("--year-end",   type=int, default=None)
    p.add_argument("--month-start", type=int, default=1)
    p.add_argument("--month-end",   type=int, default=12)
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing month files")
    args = p.parse_args()

    if args.region not in REGIONS:
        p.error(f"unknown region {args.region!r}. Available: {list(REGIONS.keys())}")

    if args.train_tag is not None:
        train_tag = args.train_tag
    elif args.train_year is not None:
        train_tag = str(args.train_year)
    else:
        p.error("Provide either --train-tag or --train-year.")

    if args.year is not None:
        ys, ye = args.year, args.year
    elif args.year_start is not None and args.year_end is not None:
        ys, ye = args.year_start, args.year_end
    else:
        p.error("Provide either --year or both --year-start and --year-end.")

    print("\n" + "=" * 70)
    print(f"  Roughness apply - region={args.region}  "
          f"train_tag={train_tag}  years={ys}-{ye}  "
          f"months={args.month_start}-{args.month_end}")
    print(f"  CYGNSS veg src : {os.path.relpath(os.path.join(CYGNSS_VEG_DIR, args.region))}")
    print(f"  AFS table      : {os.path.relpath(os.path.join(CYGNSS_ROUGH_DIR, args.region, f'afs_{args.region}_{train_tag}.parquet'))}")
    print(f"  Rough dst      : {os.path.relpath(os.path.join(CYGNSS_ROUGH_DIR, args.region))}")
    print(f"  Force          : {args.force}")
    print("=" * 70)

    # load the afs table once
    afs_df, afs_lookup, grid = load_afs_table(
        CYGNSS_ROUGH_DIR, args.region, train_tag,
    )
    print(f"  Loaded AFS for {len(afs_df):,} EASE-2 cells "
          f"(median AFS_dB={float(afs_df['afs_db'].median()):.2f})")

    # clay lookup per cell, falls back to the training years if the run years
    # have no smap files
    clay_by_cell_id: dict[int, float] = {}
    clay_years = list(range(ys, ye + 1))
    for year in clay_years:
        for cid, cv in load_clay_index(SMAP_DIR, args.region, year).items():
            clay_by_cell_id.setdefault(cid, cv)
    if not clay_by_cell_id:
        fallback_years = [int(y) for y in train_tag.split("-")]
        fallback_years = list(range(fallback_years[0], fallback_years[-1] + 1))
        print(f"  No clay in {clay_years}; falling back to training years "
              f"{fallback_years}")
        for year in fallback_years:
            for cid, cv in load_clay_index(SMAP_DIR, args.region, year).items():
                clay_by_cell_id.setdefault(cid, cv)
    print(f"  Loaded clay for {len(clay_by_cell_id):,} EASE-2 cells")

    targets = []
    for year in range(ys, ye + 1):
        m_lo = args.month_start if year == ys else 1
        m_hi = args.month_end   if year == ye else 12
        for month in range(m_lo, m_hi + 1):
            targets.append((year, month))

    written = skipped = empty = no_veg = 0
    total_rows = 0
    total_matched = 0

    bar = tqdm(targets, desc=args.region, unit="mo", dynamic_ncols=True)
    for year, month in bar:
        bar.set_postfix_str(f"{year}-{month:02d}")
        stats = apply_month(
            region=args.region, year=year, month=month,
            veg_dir=CYGNSS_VEG_DIR, rough_dir=CYGNSS_ROUGH_DIR,
            afs_lookup=afs_lookup, grid=grid,
            clay_by_cell_id=clay_by_cell_id, force=args.force,
        )
        tag = f"{year}-{month:02d}"
        s = stats["status"]
        if s == "written":
            written += 1
            total_rows    += stats["n_rows"]
            total_matched += stats["n_matched"]
            tqdm.write(
                f"  [WRITE] {tag}  rows={stats['n_rows']:>9,}  "
                f"matched={stats['n_matched']:>9,}  "
                f"time={stats['elapsed_s']:>6.1f}s  "
                f"-> {os.path.relpath(stats['out_path'])}"
            )
        elif s == "skipped":
            skipped += 1
            tqdm.write(f"  [SKIP ] {tag}  (exists; use --force to rebuild)")
        elif s == "no_veg":
            no_veg += 1
            tqdm.write(f"  [NOVEG] {tag}  no cygnss_veg month-file")
        else:
            empty += 1
            tqdm.write(f"  [EMPTY] {tag}  empty input")

    bar.close()
    print("\n" + "-" * 70)
    print(f"  Done. written={written} skipped={skipped} empty={empty} "
          f"no_veg={no_veg}  total_rows={total_rows:,}  "
          f"total_matched={total_matched:,}")
    print("-" * 70 + "\n")


if __name__ == "__main__":
    main()
