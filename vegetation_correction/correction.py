from __future__ import annotations 
import numpy as np

from .constants import (
    GAMMA_DB,
    VWC_THRESH,
)

# Volume-scattering term for one observation
def gamma_db(vwc: float) -> float:
    if np.isfinite(vwc) and float(vwc) > VWC_THRESH:
        return GAMMA_DB
    return 0.0

# tau = M_cell,day / cos(theta_inc).
def tau_from_M(M_cell_day: float | np.ndarray,
               sp_inc_angle_deg: float | np.ndarray) -> np.ndarray:

    inc_rad = np.deg2rad(np.asarray(sp_inc_angle_deg, dtype=np.float64))
    cos_theta = np.cos(inc_rad)

    # to avoid 0 division
    cos_theta = np.where(cos_theta < 1e-3, np.nan, cos_theta)
    return np.asarray(M_cell_day, dtype=np.float64) / cos_theta

# Finalizes the vegetation correction
def correct_sr_db(sr_db: float | np.ndarray,
                  tau: float | np.ndarray,
                  gamma_db: float | np.ndarray) -> np.ndarray:

    #gamma_db to gamma_linear 

    sr_db = np.asarray(sr_db, dtype=np.float64)
    tau_array = np.asarray(tau, dtype=np.float64)
    gamma_db = np.asarray(gamma_db, dtype=np.float64)

    sr_linear = np.power(10.0, sr_db / 10.0)
    gamma_linear = np.where(gamma_db == 0.0, 0.0, np.power(10.0, gamma_db / 10.0))

    sr_unscattered = sr_linear - gamma_linear
    # If gamma >= SR (impossible but just in case
    sr_unscattered = np.where(sr_unscattered <= 0.0, np.nan, sr_unscattered)

    sr_veg_linear = sr_unscattered*np.exp(2.0*tau_array)
    return 10.0*np.log10(sr_veg_linear)
 