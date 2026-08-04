# per-region skill tables: full-scene r and nrmse for every variant

import pandas as pd

import base
import context as ctx


def write_skill_tables(skill):
    for mode, suffix in (("raw", "raw"), ("smooth", "smoothed")):
        rows = []
        for region, per in skill.items():
            for k, v in enumerate(base.VARIANTS):
                if v not in per:
                    continue
                s = per[v]
                rows.append({
                    "region":     ctx.REGIONS[region]["label"] if k == 0 else "",
                    "variant":    base.short_variant(v),
                    "r_smap":     s[f"r_cs_{mode}"],
                    "nrmse_smap": s[f"nrmse_cs_{mode}"],
                    "r_era5":     s[f"r_ce_{mode}"],
                    "nrmse_era5": s[f"nrmse_ce_{mode}"],
                })
        base.save_table(pd.DataFrame(rows), f"skill_{suffix}")
