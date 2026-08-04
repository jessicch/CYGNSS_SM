# SM retrieval from roughness-corrected CYGNSS 
import argparse
import os
import sys
import warnings

from tqdm import tqdm

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    CYGNSS_ROUGH_DIR, CYGNSS_SM_DIR, REGIONS,
)
from roughness_correction.retrieve_pipeline import retrieve_month


def main() -> None:
    p = argparse.ArgumentParser(
        description="Retrieve soil moisture from roughness-corrected CYGNSS "
                    "months (invert sr_rough through mironov + fresnel).",
    )
    p.add_argument("--region", required=True,
                   help=f"Region key (choices: {list(REGIONS.keys())})")
    p.add_argument("--year", type=int, default=None,
                   help="Single year to retrieve (shortcut)")
    p.add_argument("--year-start", type=int, default=None)
    p.add_argument("--year-end",   type=int, default=None)
    p.add_argument("--month-start", type=int, default=1)
    p.add_argument("--month-end",   type=int, default=12)
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing month files")
    args = p.parse_args()

    if args.region not in REGIONS:
        p.error(f"unknown region {args.region!r}. Available: {list(REGIONS.keys())}")

    if args.year is not None:
        ys, ye = args.year, args.year
    elif args.year_start is not None and args.year_end is not None:
        ys, ye = args.year_start, args.year_end
    else:
        p.error("Provide either --year or both --year-start and --year-end.")

    print("\n" + "=" * 70)
    print(f"  SM retrieval - region={args.region}  years={ys}-{ye}  "
          f"months={args.month_start}-{args.month_end}")
    print(f"  CYGNSS rough src : {os.path.relpath(os.path.join(CYGNSS_ROUGH_DIR, args.region))}")
    print(f"  SM dst           : {os.path.relpath(os.path.join(CYGNSS_SM_DIR, args.region))}")
    print(f"  Clay             : per-observation clay_fraction column in rough files")
    print(f"  Force            : {args.force}")
    print("=" * 70)

    targets = []
    for year in range(ys, ye + 1):
        m_lo = args.month_start if year == ys else 1
        m_hi = args.month_end   if year == ye else 12
        for month in range(m_lo, m_hi + 1):
            targets.append((year, month))

    written = skipped = empty = no_rough = no_clay = 0
    total_rows = 0
    total_retrieved = 0

    bar = tqdm(targets, desc=args.region, unit="mo", dynamic_ncols=True)
    for year, month in bar:
        bar.set_postfix_str(f"{year}-{month:02d}")
        stats = retrieve_month(
            region=args.region, year=year, month=month,
            rough_dir=CYGNSS_ROUGH_DIR, sm_dir=CYGNSS_SM_DIR,
            force=args.force,
        )
        tag = f"{year}-{month:02d}"
        s = stats["status"]
        if s == "written":
            written += 1
            total_rows      += stats["n_rows"]
            total_retrieved += stats["n_retrieved"]
            tqdm.write(
                f"  [WRITE] {tag}  rows={stats['n_rows']:>9,}  "
                f"retrieved={stats['n_retrieved']:>9,}  "
                f"time={stats['elapsed_s']:>6.1f}s  "
                f"-> {os.path.relpath(stats['out_path'])}"
            )
        elif s == "skipped":
            skipped += 1
            tqdm.write(f"  [SKIP ] {tag}  (exists; use --force to rebuild)")
        elif s == "no_rough":
            no_rough += 1
            tqdm.write(f"  [NORGH] {tag}  no cygnss_rough month-file")
        elif s == "no_clay":
            no_clay += 1
            tqdm.write(f"  [NOCLY] {tag}  rough file lacks clay_fraction "
                       f"(rebuild with build_rough_apply --force)")
        else:
            empty += 1
            tqdm.write(f"  [EMPTY] {tag}  empty input")

    bar.close()
    print("\n" + "-" * 70)
    print(f"  Done. written={written} skipped={skipped} empty={empty} "
          f"no_rough={no_rough} no_clay={no_clay}  total_rows={total_rows:,}  "
          f"total_retrieved={total_retrieved:,}")
    print("-" * 70 + "\n")


if __name__ == "__main__":
    main()
