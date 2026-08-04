import os
import sys
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import pearsonr, linregress

# make validation/ importable so the shared common module resolves
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import shared

import context as ctx


# consistent date formatting on the x-axis
def _date_axis(axes):
    loc   = mdates.MonthLocator(bymonth=[1, 4, 7, 10])
    minor = mdates.MonthLocator()
    for ax in axes:
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_minor_locator(minor)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)


# Plot 1: triple-axis time series (CYGNSS left, SMAP + ERA5 right)
def plot_triple_timeseries(dates, cygnss_s, smap_s, era5_s,
                           cygnss_r, smap_r, era5_r,
                           sigma_cygnss, sigma_sm, out_path):
    v = shared.variant_label(ctx.SR_COL)
    fig, ax1 = plt.subplots(figsize=(14, 5))
    ax2 = ax1.twinx()

    # CYGNSS on left axis
    ax1.plot(dates, cygnss_r, color="steelblue", alpha=0.4, lw=0.9,
             linestyle="--", label="_nolegend_")
    ax1.plot(dates, cygnss_s, color="steelblue", lw=1.8,
             label=f"{v['short']} (σ_G={sigma_cygnss:g})")
    ax1.set_ylabel(f"{v['name']} ({v['unit']})", color="steelblue")
    ax1.tick_params(axis="y", colors="steelblue")

    # SMAP on right axis
    ax2.plot(dates, smap_r, color="tomato", alpha=0.4, lw=0.9,
             linestyle="--", label="_nolegend_")
    ax2.plot(dates, smap_s, color="tomato", lw=1.8,
             label=f"SMAP SM (σ_G={sigma_sm:g})")

    # ERA5 also on right axis
    ax2.plot(dates, era5_r, color="seagreen", alpha=0.4, lw=0.9,
             linestyle="--", label="_nolegend_")
    ax2.plot(dates, era5_s, color="seagreen", lw=1.8,
             label=f"ERA5 SM (σ_G={sigma_sm:g})")
    ax2.set_ylabel("Soil Moisture (m³ m⁻³)", color="dimgray")

    # merge legends from both axes
    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2,
               loc="upper left", fontsize=8, framealpha=0.8)

    _date_axis([ax1])
    ax1.set_xlim(dates[0], dates[-1])
    region_label = ctx.REGIONS[ctx.REGION]['label']
    ax1.set_title(
        f"{ctx.WINDOW_DAYS}-Day Averaged {v['short']}, SMAP SM, and ERA5 SM — {region_label}\n"
        f"{ctx.WINDOW_DAYS}-day windows, σ_G(CYGNSS)={sigma_cygnss:g}  "
        f"σ_G(SM)={sigma_sm:g}",
        fontsize=11
    )
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot 1 saved → {os.path.relpath(out_path)}")


# Plot 2: scatter — CYGNSS vs SMAP and CYGNSS vs ERA5
def plot_scatter(cygnss, smap, era5, out_path, start_date=None, end_date=None):
    v = shared.variant_label(ctx.SR_COL)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    for ax, (y, ylabel, color, pair_label) in zip(
        axes,
        [
            (smap, "SMAP SM (m³ m⁻³)", "tomato",   f"{v['short']} – SMAP"),
            (era5, "ERA5 SM (m³ m⁻³)", "seagreen", f"{v['short']} – ERA5"),
        ]
    ):
        mask = np.isfinite(cygnss) & np.isfinite(y)
        if mask.sum() < 2:   # Pearson needs at least two pairs
            ax.set_title(f"{pair_label} — insufficient data")
            continue

        x_m, y_m = cygnss[mask], y[mask]
        r, p      = pearsonr(x_m, y_m)
        slope, intercept, *_ = linregress(x_m, y_m)

        # density colouring via 2-D histogram
        density = shared.scatter_density(x_m, y_m)

        sc = ax.scatter(x_m, y_m, c=density, cmap="plasma",
                        s=12, alpha=0.7, linewidths=0)
        plt.colorbar(sc, ax=ax, label="Density")

        x_fit = np.linspace(x_m.min(), x_m.max(), 200)
        ax.plot(x_fit, slope * x_fit + intercept,
                color=color, lw=1.5, label=f"fit  (r={r:.3f})")

        p_str = f"p<0.001" if p < 0.001 else f"p={p:.3f}"
        ax.set_title(f"{pair_label}\nr = {r:.3f}, {p_str}", fontsize=10)
        ax.set_xlabel(f"{v['short']} ({v['unit']})")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    region_label = ctx.REGIONS[ctx.REGION]['label']
    start_str = start_date.strftime('%b %Y') if start_date else ctx.START_DATE.strftime('%b %Y')
    end_str = end_date.strftime('%b %Y') if end_date else ctx.END_DATE.strftime('%b %Y')
    plt.suptitle(f"{v['short']} vs Soil Moisture — {region_label} ({start_str} – {end_str})",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot 2 saved → {os.path.relpath(out_path)}")
