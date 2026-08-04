from __future__ import annotations
import numpy as np

# Calculates fresnel R_V and R_H
def fresnel(eps_m: complex | np.ndarray, theta_deg: float | np.ndarray):
    theta = np.deg2rad(np.asarray(theta_deg, dtype=np.float64))
    cos_theta = np.cos(theta)
    sin2_theta = np.sin(theta)**2

    epsilon_m = np.asarray(eps_m)
    sqrt_term = np.sqrt(epsilon_m - sin2_theta)

    R_V = (epsilon_m*cos_theta - sqrt_term) / (epsilon_m*cos_theta + sqrt_term)
    R_H = (cos_theta - sqrt_term)/(cos_theta + sqrt_term)
    return R_V, R_H

# Calculates R_RL reflectivity. righthanded transit - Left handed receive
def rl_reflectivity(epsilon_m: complex | np.ndarray, theta_deg: float | np.ndarray):
    R_V, R_H = fresnel(epsilon_m, theta_deg)
    R_RL = (R_V - R_H)/2 # right hand circular polarization
    return np.abs(R_RL)**2
