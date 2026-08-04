from __future__ import annotations
import numpy as np
from scipy.optimize import brentq

from .mironov import mironov_epsilon
from .fresnel import rl_reflectivity


# soil moisture search bounds for the inversion (m3/m3)
SM_RETRIEVAL_MIN = 0.0
SM_RETRIEVAL_MAX = 0.60
SM_RETRIEVAL_TOL = 1.0e-4   # brentq absolute tolerance on soil moisture


# forward helper. soil moisture to reflectivity
def _rl_reflectivity_of_sm(sm: float, clay_fraction: float, theta_deg: float) -> float:
    eps_m = mironov_epsilon(sm, clay_fraction)
    return float(rl_reflectivity(eps_m, theta_deg))


# inverts reflectivity (db) to soil moisture for one observation.
# nan when outside the achievable range, unless clamp=True.
def soil_moisture_from_reflectivity(reflectivity_db: float,
                                    clay_fraction:   float,
                                    theta_deg:       float,
                                    sm_min: float = SM_RETRIEVAL_MIN,
                                    sm_max: float = SM_RETRIEVAL_MAX,
                                    tol:    float = SM_RETRIEVAL_TOL,
                                    clamp:  bool  = False) -> float:

    if not (np.isfinite(reflectivity_db) and np.isfinite(clay_fraction)
            and np.isfinite(theta_deg)):
        return np.nan

    # db to linear power
    target = 10.0**(reflectivity_db/10.0)

    # residual whose root is the retrieved soil moisture
    def residual(sm: float) -> float:
        return _rl_reflectivity_of_sm(sm, clay_fraction, theta_deg) - target

    r_lo = residual(sm_min)
    r_hi = residual(sm_max)
    if not (np.isfinite(r_lo) and np.isfinite(r_hi)):
        return np.nan

    # target below the dry-soil floor -> drier than the model can represent
    if r_lo > 0.0:
        return sm_min if clamp else np.nan
    # target above the saturated ceiling -> wetter than the model can represent
    if r_hi < 0.0:
        return sm_max if clamp else np.nan

    return float(brentq(residual, sm_min, sm_max, xtol=tol))


# vectorised inversion, loops per observation since mironov is scalar-only
def retrieve_soil_moisture(reflectivity_db,
                           clay_fraction,
                           theta_deg,
                           sm_min: float = SM_RETRIEVAL_MIN,
                           sm_max: float = SM_RETRIEVAL_MAX,
                           tol:    float = SM_RETRIEVAL_TOL,
                           clamp:  bool  = False) -> np.ndarray:

    refl  = np.asarray(reflectivity_db, dtype=np.float64)
    clay  = np.broadcast_to(np.asarray(clay_fraction, dtype=np.float64), refl.shape)
    theta = np.broadcast_to(np.asarray(theta_deg,     dtype=np.float64), refl.shape)

    out = np.full(refl.shape, np.nan, dtype=np.float64)
    flat_out, flat_refl = out.ravel(), refl.ravel()
    flat_clay, flat_theta = clay.ravel(), theta.ravel()
    for i in range(flat_refl.size):
        flat_out[i] = soil_moisture_from_reflectivity(
            flat_refl[i], flat_clay[i], flat_theta[i],
            sm_min=sm_min, sm_max=sm_max, tol=tol, clamp=clamp,
        )
    return out
