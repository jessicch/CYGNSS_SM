from __future__ import annotations
import numpy as np

# NSIDC EASE-2 9km grid constants
N_COLS      = 3856
N_ROWS      = 1624
CELL_SIZE_M = 9008.055210146
X_MIN       = -17367530.44
Y_MAX       =   7314540.83
PROJ4       = "+proj=cea +lat_ts=30 +lon_0=0 +datum=WGS84 +units=m +no_defs" #epsg: 6933


class Ease2Grid:
    N_COLS = N_COLS
    N_ROWS = N_ROWS
    CELL_SIZE_M = CELL_SIZE_M
    X_MIN = X_MIN
    Y_MAX = Y_MAX

    def __init__(self):
        from pyproj import Transformer
        self._forward = Transformer.from_crs("EPSG:4326", PROJ4, always_xy=True)
        self._inverse = Transformer.from_crs(PROJ4, "EPSG:4326", always_xy=True)

    # converts lat,lon to row, col
    def cell_of(self, lat, lon):
        lat = np.asarray(lat, dtype=np.float64)
        lon = np.asarray(lon, dtype=np.float64)
        x, y = self._forward.transform(lon, lat) # has to be lon lat not lat long
        x = np.asarray(x, dtype=np.float64) #just incase
        y = np.asarray(y, dtype=np.float64)# just incase

        col = np.floor((x - X_MIN) / CELL_SIZE_M).astype(np.int32) #how many 9km cells from left, rounded down
        row = np.floor((Y_MAX - y) / CELL_SIZE_M).astype(np.int32) #how many 9km cells down, round down

        # Check that its not nan, not outside max nr cols/rows. -1 if outside.
        limit = (
            ~np.isfinite(x) | ~np.isfinite(y) |
            (col < 0) | (col >= N_COLS) |
            (row < 0) | (row >= N_ROWS)
        )
        col = np.where(limit, -1, col).astype(np.int32) 
        row = np.where(limit, -1, row).astype(np.int32)
        return row, col

    # Converts lat, lon to cell_id
    def cell_id(self, lat, lon) -> np.ndarray:
        row, col = self.cell_of(lat, lon)
        # cell_id = row*nr_columns + col
        cell_id = row.astype(np.int64) * np.int64(N_COLS) + col.astype(np.int64)
        cell_id = np.where((row < 0) | (col < 0), np.int64(-1), cell_id) #carry on marking invalid
        return cell_id

    # Converts row, col to lat,lon
    def cell_center(self, row, col):
        col = np.asarray(col, dtype=np.int64)
        x = X_MIN + (col + 0.5) * CELL_SIZE_M 
        y = Y_MAX - (row + 0.5) * CELL_SIZE_M #since it starts in to left corner
        lon, lat = self._inverse.transform(x, y)
        return np.asarray(lat, dtype=np.float64), np.asarray(lon, dtype=np.float64)

    # Converts cell_id to lat, lon
    def cell_center_from_id(self, cell_id):
        cell_id = np.asarray(cell_id, dtype=np.int64)
        row = cell_id // N_COLS
        col = cell_id %  N_COLS
        return self.cell_center(row, col)
