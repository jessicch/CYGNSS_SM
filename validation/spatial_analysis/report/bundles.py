# shared by vwc.py and igbp.py
import numpy as np
import pandas as pd

import base
import context as ctx
from common import shared
from grids import EASE, make_ease2_grid, bin_points_to_ease2

from config.settings import CYGNSS_SM_DIR
from processing.cygnss_sm import load_sm_cache_month


# cell-mean vwc and the dominant igbp class on the region grid
def veg_grids(region, year, month):
    df = load_sm_cache_month(CYGNSS_SM_DIR, region, year, month)
    if df.empty:
        return None, None
    lat_c, lon_c, row_south, col_west = make_ease2_grid(
        ctx.LAT_MIN, ctx.LAT_MAX, ctx.LON_MIN, ctx.LON_MAX
    )
    nlat, nlon = len(lat_c), len(lon_c)

    vwc_grid, _ = bin_points_to_ease2(
        df["sp_lat"].values, df["sp_lon"].values, df["vwc_cell"].values,
        nlat, nlon, row_south, col_west,
    )

    # majority vote so each cell gets a single igbp class
    row, col = EASE.cell_of(df["sp_lat"].values, df["sp_lon"].values)
    i = row_south - row
    j = col - col_west
    cls = df["modis_land_cover"].values.astype(float)
    ok = (row >= 0) & (col >= 0) & \
         (i >= 0) & (i < nlat) & (j >= 0) & (j < nlon) & np.isfinite(cls)

    igbp_grid = np.full((nlat, nlon), np.nan)
    if ok.any():
        cells = pd.DataFrame({"i": i[ok], "j": j[ok], "cls": cls[ok].astype(int)})
        dom = cells.groupby(["i", "j"])["cls"].agg(lambda s: s.mode().iat[0])
        idx = np.array(list(dom.index))
        igbp_grid[idx[:, 0], idx[:, 1]] = dom.values
    return vwc_grid, igbp_grid


# smoothed grids for every variant plus the veg grids, everything on one grid
def region_bundle(region, year, month, sigma):
    base.set_region(region)
    res = {v: base.load_results(region, v, year, month, sigma)
           for v in base.VARIANTS}
    if any(r is None for r in res.values()):
        print(f"  {region}: incomplete npz set — skipped in the stratified analysis")
        return None
    vwc_grid, igbp_grid = veg_grids(region, year, month)
    if vwc_grid is None:
        print(f"  {region}: no sm month-file — skipped in the stratified analysis")
        return None
    smap = res["sr"]["smap_smooth"]
    if vwc_grid.shape != smap.shape:
        print(f"  {region}: veg grid {vwc_grid.shape} != npz grid {smap.shape} — skipped")
        return None
    b = {v: res[v]["cygnss_smooth"] for v in base.VARIANTS}
    b.update(smap=smap, era5=res["sr"]["era5_smooth"], vwc=vwc_grid, igbp=igbp_grid)
    return b


# delta r against the sr baseline inside one set of cells, None when too thin
def delta_r_in_mask(b, mask, ref):
    base_r, _, n = shared.corr_nrmse(b["sr"][mask], b[ref][mask])
    if n < base.MIN_CELLS_PER_BIN:
        return None
    out = {}
    for v in base.CORR_VARIANTS:
        r, _, n_v = shared.corr_nrmse(b[v][mask], b[ref][mask])
        out[v] = r - base_r if n_v >= base.MIN_CELLS_PER_BIN else np.nan
    return out
