from __future__ import annotations
import numpy as np

from .constants import (
    L_BAND_FREQ_HZ, EPS_0, EPS_INF,
    ND_A0, ND_A1, ND_A2,
    KD_A0, KD_A1,
    MVT_A0, MVT_A1,
    EPS0B_A0, EPS0B_A1, EPS0B_A2,
    TAUB_A0, TAUB_A1,
    SIGB_A0, SIGB_A1,
    EPS0U, TAUU,
    SIGU_A0, SIGU_A1,
)



# Debye equations and calculations
def debye_eps(eps_0: float | np.ndarray,
              tau:   float | np.ndarray,
              sigma: float | np.ndarray,
              frequency:  float = L_BAND_FREQ_HZ,
              eps_inf: float = EPS_INF) -> tuple:

    tau_factor = 2.0*np.pi*frequency*np.asarray(tau, dtype=np.float64)
    denominator = 1.0 + tau_factor**2
    numerator = np.asarray(eps_0, dtype=np.float64) - eps_inf
    eps_first = eps_inf + numerator/denominator
    eps_second = numerator/denominator*tau_factor + np.asarray(sigma, dtype=np.float64) / (2.0*np.pi*EPS_0*frequency)
    return eps_first, eps_second


# Converts eps'and eps'' to n, k
def eps_to_n_k(eps_first: float | np.ndarray,
               eps_second: float | np.ndarray) -> tuple:
    e1 = np.asarray(eps_first, dtype=np.float64)
    e11 = np.asarray(eps_second, dtype=np.float64)
    inner_root = np.sqrt(e1**2 + e11**2)
    n = np.sqrt((inner_root + e1)/2)
    k = np.sqrt((inner_root - e1)/2)
    return n, k


# Full Mironov for each cell soilmoisture ad clay fraction. SM is soilmoisture
def mironov_epsilon(sm: float, clay_fraction: float,
                    frequency: float = L_BAND_FREQ_HZ) -> complex:

    c = float(clay_fraction)

    # Dry soil equations
    n_d = ND_A0 + ND_A1*c + ND_A2*c**2
    k_d = KD_A0 + KD_A1*c
    m_vt = MVT_A0 + MVT_A1*c

    # Bound soil eqs
    eps_0b = EPS0B_A0 + EPS0B_A1*c + EPS0B_A2*c**2
    tau_b = TAUB_A0  + TAUB_A1*c
    sigma_b = SIGB_A0  + SIGB_A1*c

    # Check wether soil moisture is above max bound water fraction (m_vt)
    sm = float(sm)
    if sm <= m_vt:
        m_b = sm
        m_u = 0.0
    else:
        m_b = m_vt
        m_u = sm - m_vt

   # Bound water 
    eps_b_1, eps_b_11 = debye_eps(eps_0b, tau_b, sigma_b, frequency)
    n_b, k_b = eps_to_n_k(eps_b_1, eps_b_11)

    # free water, depends on whether waterlimit met
    if m_u > 0.0:
        sigma_u = SIGU_A0 + SIGU_A1 * c
        eps_u_1, eps_u_11 = debye_eps(EPS0U, TAUU, sigma_u, frequency)
        n_u, k_u = eps_to_n_k(eps_u_1, eps_u_11)
    else:
        # When m_v <= m_vt, free-water terms are 0.
        n_u = 0.0
        k_u = 0.0

    # GRMDM 
    n_m = n_d + (n_b - 1.0)*m_b + (n_u - 1.0)*m_u
    k_m = k_d + k_b*m_b + k_u*m_u

    # Back to complex dielectric
    eps_1 = n_m**2 - k_m**2
    eps_11 = 2.0*n_m*k_m

    return complex(eps_1, -eps_11)
