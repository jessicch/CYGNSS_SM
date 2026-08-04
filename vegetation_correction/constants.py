# IGBP class definitions (MODIS MCD12Q1 classes 1-17)
IGBP_NAMES = {
    1:  "Evergreen Needleleaf Forest", #not included in yueh so set to 0, excplicitly excluded in the paper 
    2:  "Evergreen Broadleaf Forest",
    3:  "Deciduous Needleleaf Forest", 
    4:  "Deciduous Broadleaf Forest",
    5:  "Mixed Forest",
    6:  "Closed Shrublands",
    7:  "Open Shrublands",
    8:  "Woody Savannas",
    9:  "Savannas",
    10: "Grasslands",
    11: "Permanent Wetlands", # since not included in yueh, we set to 0 in b table
    12: "Croplands",
    13: "Urban and Built-up",  # we dont actually correct for these, so values just set to 0
    14: "Cropland/Natural Vegetation Mosaic",
    15: "Snow and Ice",  # not corrected either, 0
    16: "Barren or Sparsely Vegetated",
    17: "Water Bodies",  # these are actually excluded in the qualityfilter sp_land_valid, but included to be safe. 
}

# Cropland classes, says in page 3 of smap ancillay report, section 2 that croplands and grasslands use current NDVI instead of max. Used later.
CROPLAND_CLASSES = frozenset({10, 12, 14})

# Classes that contribute zero vegetation (skipped in M and in NDVI lookups).
NON_VEGETATED_CLASSES = frozenset({11, 13, 15, 17})

# STEM FACTORS From the Ancillary smap report
STEM_FACTOR = {
    1: 15.96,
    2: 19.15,
    3: 7.98,
    4: 12.77,
    5: 12.77,
    6: 3.00,
    7: 1.50,
    8: 4.00,
    9: 3.00,
    10: 1.50,
    11: 4.00,
    12: 3.50,
    13: 6.49,
    14: 3.25,
    15: 0.00,
    16: 0.00,
    17: 0.00,
}

# Yueh 2022 Table I,  per class b coefficients
# Classes not in the table get 0.0 (do not contribute to M_cell)

YUEH_B = {
    1: 0.000,   # needle 
    2: 0.045,
    3: 0.000,   # needle
    4: 0.071,
    5: 0.054,
    6: 0.109,
    7: 0.166,
    8: 0.037,
    9: 0.125,
    10: 0.088,
    11: 0.000,  # not in Yueh table
    12: 0.055,
    13: 0.000,  # urban 
    14: 0.050,
    15: 0.000,  # ice
    16: 0.025,
    17: 0.000,  # watwe
}



# NDVI to VWC parameters (Hunt formula) 
HUNT_A = 1.9134         # NDVI**2 coefficient
HUNT_B = -0.3215         # NDVI    coefficient

# Forest volume scattering, triggered when forest and vwc > 5. yueh
GAMMA_DB = -30.5

# Threshold for activating the gamma correction
VWC_THRESH = 5.0  


