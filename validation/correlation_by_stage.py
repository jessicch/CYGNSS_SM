# result: correlation by processing stage — full-scene r for every region
# as one line over the four pipeline stages, spatial and temporal side by
# side, one figure per reference. computes nothing new: reads the skill
# tables written by the two report builders, so the figure always matches
# the archived numbers. spatial uses the smoothed maps and temporal the raw
# weekly series, the same convention the results chapter uses when quoting
# values.

import os
import sys

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# make the repo root importable so config resolves
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import RESULTS_DIR

SPATIAL_CSV  = os.path.join(RESULTS_DIR, "spatial", "report", "tables", "skill_smoothed.csv")
TEMPORAL_CSV = os.path.join(RESULTS_DIR, "temporal", "tables", "skill_raw.csv")

STAGES = ["SR", "SR_veg", "SR_rough", "SM"]

# one colour per region, keyed by the label the tables use
REGION_COLORS = {
    "Pakistan (Sindh)":         "goldenrod",
    "Mosambique":               "seagreen",
    "Central African Republic": "olive",
    "Guatemala & Honduras":     "purple",
    "Brasil":                   "steelblue",
    "Mississippi, US":          "tomato",
}

# the skill csvs leave the region cell blank after the first variant row
def load_skill(csv_path):
    df = pd.read_csv(csv_path)
    df["region"] = df["region"].ffill()
    return df


def main():
    spatial  = load_skill(SPATIAL_CSV)
    temporal = load_skill(TEMPORAL_CSV)

    for ref, ref_name in (("smap", "SMAP"), ("era5", "ERA5-Land")):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
        for ax, df, title in (
            (axes[0], spatial,  "Spatial — August 2023 maps, σ_G = 1.5"),
            (axes[1], temporal, "Temporal — raw weekly series, 2023–2025"),
        ):
            for region, grp in df.groupby("region", sort=False):
                r = grp.set_index("variant").loc[STAGES, f"r_{ref}"].to_numpy()
                ax.plot(range(len(STAGES)), r, marker="o", ms=4,
                        color=REGION_COLORS[region], label=region)
            ax.axhline(0, color="grey", lw=0.8)
            ax.set_xticks(range(len(STAGES)))
            ax.set_xticklabels(STAGES, fontsize=9)
            ax.set_ylim(-1, 1)
            ax.set_title(title, fontsize=10)
            ax.grid(True, alpha=0.3)
        axes[0].set_ylabel(f"Pearson r vs {ref_name}", fontsize=9)
        axes[1].legend(fontsize=8, loc="lower right")

        plt.suptitle(f"Correlation along the processing chain — vs {ref_name}", fontsize=11)
        plt.tight_layout()
        out = os.path.join(RESULTS_DIR, f"correlation_by_stage_{ref}_202308_2023_2025.png")
        plt.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Correlation-by-stage figure saved → {os.path.relpath(out)}")


if __name__ == "__main__":
    main()
