import os
import sys
import glob
import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr
from scipy.interpolate import griddata

# make the repo root importable so config/processing resolve
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.settings import (
    CYGNSS_QC_DIR, CYGNSS_VEG_DIR, CYGNSS_ROUGH_DIR, CYGNSS_SM_DIR, ERA5_DIR,
)
from processing.build_qc_cache import load_qc_cache_month
from processing.cygnss_veg   import load_veg_cache_month
from processing.cygnss_rough import load_rough_cache_month
from processing.cygnss_sm    import load_sm_cache_month


# cache directory matching the chosen SR column
def select_cache_dir(sr_col):
    if sr_col == "sm":
        return CYGNSS_SM_DIR
    if sr_col == "sr_rough":
        return CYGNSS_ROUGH_DIR
    if sr_col == "sr_veg":
        return CYGNSS_VEG_DIR
    return CYGNSS_QC_DIR


# load cygnss from cache
def load_cygnss_cache_month(sr_col, cache_dir, region, year, month):
    if sr_col == "sm":
        return load_sm_cache_month(cache_dir, region, year, month)
    if sr_col == "sr_rough":
        return load_rough_cache_month(cache_dir, region, year, month)
    if sr_col == "sr_veg":
        return load_veg_cache_month(cache_dir, region, year, month)
    return load_qc_cache_month(cache_dir, region, year, month)


# get value column for variant
def value_column(sr_col):
    return "sm_retrieved" if sr_col == "sm" else sr_col


# labels per pipeline variant, used in figures
VARIANT_LABELS = {
    "sr": {
        "name":  "CYGNSS Surface Reflectivity",
        "short": "CYGNSS SR",
        "cbar":  "SR (dB)",
        "unit":  "dB",
    },
    "sr_veg": {
        "name":  "CYGNSS Vegetation-Corrected Reflectivity",
        "short": "CYGNSS SR_veg",
        "cbar":  "SR_veg (dB)",
        "unit":  "dB",
    },
    "sr_rough": {
        "name":  "CYGNSS Roughness-Corrected Reflectivity",
        "short": "CYGNSS SR_rough",
        "cbar":  "SR_rough (dB)",
        "unit":  "dB",
    },
    "sm": {
        "name":  "CYGNSS Retrieved Soil Moisture",
        "short": "CYGNSS SM",
        "cbar":  "SM (m³ m⁻³)",
        "unit":  "m³ m⁻³",
    },
}


def variant_label(sr_col):
    return VARIANT_LABELS.get(sr_col, VARIANT_LABELS["sr"])


# one soil_moisture column pooling AM and PM, same pooling as the training
def pool_soil_moisture(df):
    sm = df["soil_moisture"] if "soil_moisture" in df.columns else None
    if "soil_moisture_pm" in df.columns:
        pm = df["soil_moisture_pm"]
        sm = pm if sm is None else sm.fillna(pm)
    return sm


# land masks 

# load era5 land-sea mask
def load_era5_lsm_native(region):
    cands = [p for p in glob.glob(os.path.join(ERA5_DIR, region, "*lsm*.nc"))
             if not p.endswith(".tmp.nc")]
    if not cands:
        return None
    ds   = xr.open_dataset(cands[0])
    name = "lsm" if "lsm" in ds else list(ds.data_vars)[0]
    lsm  = ds[name]
    for d in ("valid_time", "time"):
        if d in lsm.dims:
            lsm = lsm.isel({d: 0})
    lat_var = next((v for v in ("latitude", "lat") if v in ds.coords), None)
    lon_var = next((v for v in ("longitude", "lon") if v in ds.coords), None)
    # older lsm downloads came in 0-360 longitudes, wrap to -180..180
    lons = ds[lon_var].values
    lons = np.where(lons > 180, lons - 360, lons)
    out = (ds[lat_var].values, lons, lsm.values)
    ds.close()
    return out


# land mask from natural earth polygons, true = land
def natural_earth_land_mask(lat_centres, lon_centres):
    import shapely
    from shapely.ops import unary_union
    from cartopy.io import shapereader

    shp  = shapereader.natural_earth(resolution="10m", category="physical", name="land")
    bbox = shapely.box(
        float(np.min(lon_centres)), float(np.min(lat_centres)),
        float(np.max(lon_centres)), float(np.max(lat_centres)),
    )
    geoms = [g.intersection(bbox) for g in shapereader.Reader(shp).geometries()
             if g.intersects(bbox)]
    if not geoms:
        return np.zeros((len(lat_centres), len(lon_centres)), dtype=bool)

    land = unary_union(geoms)
    shapely.prepare(land)

    lons_2d, lats_2d = np.meshgrid(lon_centres, lat_centres)
    mask = shapely.contains_xy(land, lons_2d.ravel(), lats_2d.ravel())
    return mask.reshape(lons_2d.shape)


# interpolate points to grid
def interp_points_to_grid(src_lats, src_lons, values, lat_centres, lon_centres,
                          method="linear"):
    lons_2d, lats_2d = np.meshgrid(lon_centres, lat_centres)
    points = np.column_stack([np.ravel(src_lons), np.ravel(src_lats)])
    xi     = np.column_stack([lons_2d.ravel(), lats_2d.ravel()])
    interp = griddata(points, np.ravel(values), xi, method=method)
    return interp.reshape(len(lat_centres), len(lon_centres))


# era5 land mask resampled to grid
def era5_land_mask_on_grid(region, lat_centres, lon_centres, threshold=0.5):
    native = load_era5_lsm_native(region)
    if native is None:
        return None
    lat, lon, arr = native
    src_lon, src_lat = np.meshgrid(lon, lat)
    resampled = interp_points_to_grid(
        src_lat, src_lon, arr, lat_centres, lon_centres, method="nearest"
    )
    return resampled >= threshold


# normalise
def minmax(x):
    mn, mx = np.nanmin(x), np.nanmax(x)
    return (x - mn) / (mx - mn) if mx > mn else np.zeros_like(x)


# correlation and normalized rmse
def corr_nrmse(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:   # Pearson needs at least two pairs
        return np.nan, np.nan, 0
    xm, ym = x[m], y[m]
    r, _ = pearsonr(xm, ym)
    nrmse = float(np.sqrt(np.mean((minmax(xm) - minmax(ym)) ** 2)))
    return float(r), nrmse, int(m.sum())


# point density for scatter colouring
def scatter_density(xm, ym, bins=40, sigma=1.5):
    h, xe, ye = np.histogram2d(xm, ym, bins=bins)
    h = gaussian_filter(h, sigma=sigma)
    xi = np.clip(np.searchsorted(xe[1:], xm), 0, h.shape[0] - 1)
    yi = np.clip(np.searchsorted(ye[1:], ym), 0, h.shape[1] - 1)
    return h[xi, yi]
