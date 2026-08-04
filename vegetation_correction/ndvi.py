from __future__ import annotations
import os
import re
import glob
from datetime import datetime, timedelta
from dataclasses import dataclass

import numpy as np

from roughness_correction.ease2_grid import Ease2Grid


# MOD13A2 filename (DDD = from 1 - 366).
_NDVI_RE = re.compile(r"_(?P<yyyy>\d{4})(?P<ddd>\d{3})\.tif$", re.I)

# Find center date of the 16 day composite for the interpolation later
def _composite_date(fpath: str) -> datetime | None:
    m = _NDVI_RE.search(os.path.basename(fpath))
    if not m:
        return None
    start = datetime(int(m.group("yyyy")), 1, 1) + timedelta(days=int(m.group("ddd")) - 1)
    # Center the composite at its midpoint (start + 8 days).
    return start + timedelta(days=8)

# List of all ndvi files w center dates, sorted chronologically
def list_ndvi_files(region_dir: str) -> list[tuple[datetime, str]]:
    pairs = []
    for tiffile in glob.glob(os.path.join(region_dir, "mod13a2_*_*.tif")):
        center_date = _composite_date(tiffile)
        if center_date is not None:
            pairs.append((center_date, tiffile))
    pairs.sort(key=lambda p: p[0])
    return pairs

# Loads a tif file as numpy array
def _load_ndvi_array(fpath: str) -> np.ndarray:
    import rasterio
    with rasterio.open(fpath) as ds:
        array = ds.read(1).astype(np.float32)
        nd  = ds.nodata
    if nd is not None:
        array = np.where(array == nd, np.nan, array)
    return np.clip(array, 0.0, 1.0) # naturally between 0,1 from modis but clip just incase


@dataclass
class CellNDVI:
    dates:  list[datetime]                      
    ndvi_timeseries: dict[int, np.ndarray]      

    # mean NDVI at a specific day in cell, linear interpolation between composites
    def interpolate_day(self, cell_id: int, day: datetime) -> float:
        series = self.ndvi_timeseries.get(int(cell_id))
        if series is None or series.size == 0:
            return float("nan")
        return _interpolate_one(self.dates, series, day)

    # Annual max/min of the cell-mean NDVI 
    def annual_maxmin(self, cell_id: int, year: int) -> tuple[float, float]:
        series = self.ndvi_timeseries.get(int(cell_id))
        if series is None:
            return float("nan"), float("nan")
        mask = np.array([d.year == year for d in self.dates])
        if not mask.any():
            return float("nan"), float("nan")
        vals = series[mask]
        vals = vals[np.isfinite(vals)]
        if vals.size:
            return float(vals.max()), float(vals.min())
        return float("nan"), float("nan")


# Linear interpolation for a date using two nearest composites
def _interpolate_one(dates: list[datetime], series: np.ndarray, target: datetime) -> float:
    finite = np.isfinite(series)
    if not finite.any():
        return float("nan")
    t  = np.array([(d - dates[0]).total_seconds() for d in dates], dtype=np.float64)
    tx = (target - dates[0]).total_seconds()
    tF = t[finite]
    ndvi_series = series[finite].astype(np.float64)   # NDVI series for one cell across all dates
    if tx <= tF[0]:
        return float(ndvi_series[0])
    if tx >= tF[-1]:
        return float(ndvi_series[-1])
    j = np.searchsorted(tF, tx)
    t1, t2 = tF[j - 1], tF[j]
    n1, n2 = ndvi_series[j - 1], ndvi_series[j]
    if t2 == t1:
        return float(n1)
    return float(n1 + (tx - t1) / (t2 - t1) * (n2 - n1))


# groups ndvi pixels by their ease-2 cell
def _ndvi_pixel_cell_groups(transform, shape, grid: Ease2Grid,
                            cells_w_cygnssobs: set[int] | None):
    height, width = shape
    # pixel-centre lon/lat: lon = c + (col+0.5)*a, lat = f + (row+0.5)*e
    cols = np.arange(width)  + 0.5
    rows = np.arange(height) + 0.5
    lon  = transform.c + cols * transform.a
    lat  = transform.f + rows * transform.e
    lat2d = np.broadcast_to(lat[:, None], (height, width))
    lon2d = np.broadcast_to(lon[None, :], (height, width))

    ids  = grid.cell_id(lat2d.ravel(), lon2d.ravel())
    flat = np.arange(ids.size, dtype=np.int64)
    keep = ids >= 0
    ids, flat = ids[keep], flat[keep]

    order = np.argsort(ids, kind="stable")
    ids, flat = ids[order], flat[order]

    groups: list[tuple[int, np.ndarray]] = []
    if ids.size == 0:
        return groups
    boundaries = np.flatnonzero(np.diff(ids)) + 1
    starts = np.concatenate(([0], boundaries))
    ends   = np.concatenate((boundaries, [ids.size]))
    for s, e in zip(starts, ends):
        cid = int(ids[s])
        if cells_w_cygnssobs is not None and cid not in cells_w_cygnssobs:
            continue
        groups.append((cid, flat[s:e]))
    return groups


# cell-mean ndvi time series for one year, one value per cell per composite
def build_cell_ndvi(
    region_dir: str,
    year:       int,
    grid:       Ease2Grid,
    cells_w_cygnssobs: set[int] | None = None,
) -> CellNDVI:

    all_pairs = list_ndvi_files(region_dir)
    if not all_pairs:
        return CellNDVI(dates=[], ndvi_timeseries={})

    pairs = [(d, f) for d, f in all_pairs if d.year == year]
    if not pairs:
        return CellNDVI(dates=[], ndvi_timeseries={})

    dates = [d for d, _ in pairs]
    n_t   = len(pairs)

    # NDVI geotransform (assume all composites share the same grid)
    import rasterio
    with rasterio.open(pairs[0][1]) as ds:
        transform = ds.transform
        shape     = (ds.height, ds.width)

    groups = _ndvi_pixel_cell_groups(transform, shape, grid, cells_w_cygnssobs)
    if not groups:
        return CellNDVI(dates=dates, ndvi_timeseries={})

    series: dict[int, np.ndarray] = {
        cid: np.full(n_t, np.nan, dtype=np.float32) for cid, _ in groups
    }

    for date_i, (_, fpath) in enumerate(pairs):
        ndvi = _load_ndvi_array(fpath)
        if (ndvi.shape[0], ndvi.shape[1]) != shape:
            continue  # skip composites whose grid doesn't match the reference one
        flat_vals = ndvi.ravel()
        for cid, idx in groups:
            vals = flat_vals[idx]
            finite = np.isfinite(vals)
            if finite.any():
                series[cid][date_i] = float(vals[finite].mean())

    return CellNDVI(dates=dates, ndvi_timeseries=series)
