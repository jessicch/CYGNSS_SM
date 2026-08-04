import os
import sys
import numpy as np
from scipy.ndimage import gaussian_filter

# make repo root and validation/ importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from roughness_correction.ease2_grid import Ease2Grid
from common import shared

# one shared instance, building it is slow
EASE = Ease2Grid()


# ease-2 9 km cell centres covering the bbox, rows ordered south to north
def make_ease2_grid(lat_min, lat_max, lon_min, lon_max):
    row_n, col_w = EASE.cell_of(lat_max, lon_min)   # north-west corner
    row_s, col_e = EASE.cell_of(lat_min, lon_max)   # south-east corner
    rows = np.arange(int(row_s), int(row_n) - 1, -1)   # south→north
    cols = np.arange(int(col_w), int(col_e) + 1)       # west→east
    lat_centres, _ = EASE.cell_center(rows, np.full_like(rows, cols[0]))
    _, lon_centres = EASE.cell_center(np.full_like(cols, rows[0]), cols)
    return lat_centres, lon_centres, int(row_s), int(col_w)


# mean of point values per ease-2 cell
def bin_points_to_ease2(lats, lons, values, nlat, nlon, row_south, col_west):
    row, col = EASE.cell_of(lats, lons)
    i = row_south - row      # row_south is the bottom row → array index 0
    j = col - col_west

    valid = (row >= 0) & (col >= 0) & \
            (i >= 0) & (i < nlat) & (j >= 0) & (j < nlon) & \
            np.isfinite(values)

    sums   = np.zeros((nlat, nlon), dtype=float)
    counts = np.zeros((nlat, nlon), dtype=int)
    np.add.at(sums,   (i[valid], j[valid]), values[valid])
    np.add.at(counts, (i[valid], j[valid]), 1)

    grid = np.full((nlat, nlon), np.nan)
    has_obs = counts > 0
    grid[has_obs] = sums[has_obs] / counts[has_obs]
    return grid, counts


# true where the cell centre is on land, used to clip interpolated fields
def land_mask(lat_centres, lon_centres):
    return shared.natural_earth_land_mask(lat_centres, lon_centres)


# fills grid gaps with linear interpolation
def interpolate_grid_2d(grid, lat_centres, lon_centres):
    lons_2d, lats_2d = np.meshgrid(lon_centres, lat_centres)

    known = np.isfinite(grid)
    if known.sum() < 4:
        return grid.copy()

    return shared.interp_points_to_grid(
        lats_2d[known], lons_2d[known], grid[known], lat_centres, lon_centres
    )


# 2d gaussian smoothing, nan gaps stay nan
def smooth_2d(grid, sigma):
    if sigma <= 0:
        return grid.copy()

    finite = np.isfinite(grid)
    if finite.sum() < 4:
        return grid.copy()

    filled = grid.copy()
    filled[~finite] = np.nanmean(grid)

    smoothed = gaussian_filter(filled, sigma=sigma)
    smoothed[~finite] = np.nan
    return smoothed
