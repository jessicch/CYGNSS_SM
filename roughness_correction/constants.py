# L-BAND FREQUENCY (used in Debye spectra)
L_BAND_FREQ_HZ = 1.575e9 # GPS L1
EPS_0          = 8.854e-12 # F/m, free-space permittivity
EPS_INF        = 4.9 # high-frequency dielectric limit (BSW and FSW)

### Equations for dry soil (d), bound soil water (b) and free soil water(f)

# Dry
# Dry soil refractive index nd = 1.634 - 0.539*c + 0.2748*c**2
ND_A0 =  1.634
ND_A1 = -0.539
ND_A2 =  0.2748

# Dry soil normalized attenuation k_d = 0.03952 - 0.04038*c
KD_A0 =  0.03952
KD_A1 = -0.04038

# Maximum bound water fraction m_vt = 0.02863 + 0.30673*c
MVT_A0 = 0.02863
MVT_A1 = 0.30673

#Bound (BSW)
# BSW Low frequency dielectric limit eps_0b = 79.8 - 85.4*c + 32.7*c**2
EPS0B_A0 =  79.8
EPS0B_A1 = -85.4
EPS0B_A2 =  32.7

# BSW Relaxation time  tau_b = 1.062e-11 + 3.450e-12 * c
TAUB_A0 = 1.062e-11
TAUB_A1 = 3.450e-12

# BSW conductivity sigma_b = 0.3112 + 0.467*c
SIGB_A0 = 0.3112
SIGB_A1 = 0.467

# Free (FSW)
# FSW low-frequency dielectric limit 
EPS0U   = 100.0 #constant

# FSW relaxation time 
TAUU    = 8.5e-12 #constant 

# FSW conductivity sigma_u = 0.3631 + 1.217*c
SIGU_A0 = 0.3631
SIGU_A1 = 1.217


# Roughness training parameters, only use angles between 35 and 45, same as yueh.
INC_ANGLE_TRAIN_MIN = 35.0
INC_ANGLE_TRAIN_MAX = 45.0
