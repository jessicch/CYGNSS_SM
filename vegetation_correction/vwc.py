from __future__ import annotations
import numpy as np

from .constants import (
    HUNT_A, HUNT_B,
    STEM_FACTOR,
    CROPLAND_CLASSES, NON_VEGETATED_CLASSES,
)


def vwc_hunt(ndvi: float,ndvi_min:float,  ndvi_max: float, igbp_class: int) -> float:
    if not np.isfinite(ndvi):
        return float("nan")
    if int(igbp_class) in NON_VEGETATED_CLASSES:
        return 0.0
    if int(igbp_class) in CROPLAND_CLASSES or not np.isfinite(ndvi_max):
        ndvi_max = ndvi
    
    ndvi = float(ndvi)
    ndvi_max = float(ndvi_max)
    ndvi_min = float(ndvi_min)

    # equation 1, p.3 ancillary 2013. R.hunt 1996
    soil_term = HUNT_A*ndvi**2 + HUNT_B*ndvi
    stem = STEM_FACTOR.get(int(igbp_class), 0.0)
    stem_term = stem*(ndvi_max - ndvi_min) / (1.0 - ndvi_min)
    vwc = soil_term + stem_term
    return max(vwc, 0.0)
