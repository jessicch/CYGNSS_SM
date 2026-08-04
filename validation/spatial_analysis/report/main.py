# build report tables and figures from saved spatial results

import os
import argparse
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime

import base                       # owns the sys.path setup, import first
import context as ctx

from skill_tables import write_skill_tables
from delta_tables import write_delta_tables
from montage import plot_region_montage
from bundles import region_bundle
from vwc import plot_delta_vs_vwc
from igbp import plot_delta_vs_igbp
from afs_maps import plot_afs_map
from excluded import plot_excluded_obs


def main():
    parser = argparse.ArgumentParser(
        description="Report tables and figures from the saved spatial results"
    )
    parser.add_argument("--year",  type=int, default=2023)
    parser.add_argument("--month", type=int, default=8)
    parser.add_argument("--sigma", type=float, default=ctx.GAUSSIAN_SIGMA)
    parser.add_argument("--train-tag", default="2019-2021",
                        help="AFS training tag in the table file names")
    parser.add_argument("--only", nargs="+", default=None,
                        choices=["tables", "montage", "delta", "vwc", "igbp",
                                 "afs", "excluded"],
                        help="Run a subset of the outputs (default: everything)")
    args = parser.parse_args()

    sections = set(args.only) if args.only else \
        {"tables", "montage", "delta", "vwc", "igbp", "afs", "excluded"}
    year, month, sigma = args.year, args.month, args.sigma
    ctx.YEAR, ctx.MONTH = year, month

    print("\n" + "=" * 70)
    print("  Report outputs — spatial evaluation")
    print(f"  Scene   : {datetime(year, month, 1).strftime('%B %Y')}   "
          f"grid=EASE-2 9 km   σ_G={sigma}")
    print(f"  Regions : {', '.join(base.STUDY_REGIONS)}")
    print(f"  Output  : {os.path.relpath(base.REPORT_DIR)}")
    print("=" * 70)

    if sections & {"tables", "delta"}:
        print("\n[Skill]  Reading r / nRMSE from the npz results ...")
        skill = base.collect_skill(year, month, sigma)

    if "tables" in sections:
        print("\n[Tables]  Raw and smoothed skill per region ...")
        write_skill_tables(skill)

    if "delta" in sections:
        print("\n[Delta]  Δr per correction, raw and smoothed ...")
        write_delta_tables(skill)

    if "montage" in sections:
        print("\n[Montage]  Six-panel maps per region ...")
        for region in base.STUDY_REGIONS:
            plot_region_montage(region, year, month, sigma)

    if sections & {"vwc", "igbp"}:
        print("\n[Stratified]  Building per-region grids ...")
        bundles = {}
        for region in base.STUDY_REGIONS:
            b = region_bundle(region, year, month, sigma)
            if b is not None:
                bundles[region] = b
        print(f"  {len(bundles)}/{len(base.STUDY_REGIONS)} regions ready")

        if "vwc" in sections and bundles:
            plot_delta_vs_vwc(bundles, year, month, sigma)
        if "igbp" in sections and bundles:
            plot_delta_vs_igbp(bundles, year, month, sigma)

    if "afs" in sections:
        print("\n[AFS]  Calibration maps per region ...")
        for region in base.STUDY_REGIONS:
            plot_afs_map(region, args.train_tag)

    if "excluded" in sections:
        print("\n[Excluded]  Non-retrievable observations per region ...")
        for region in base.STUDY_REGIONS:
            plot_excluded_obs(region, year, month)

    print("\n" + "=" * 70)
    print("  COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
