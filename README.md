# CYGNSS soil moisture pipeline

Master's thesis code. Rebuilds the CYGNSS SM retrieval over 6 regions (Pakistan, Brasil, Mississippi, Guatemala, CAR, Mozambique) and compares vegetation + roughness corrections against SMAP and ERA5-Land.

**Pipeline:** fetch → QC → vegetation correction → roughness correction (train + apply) → SM retrieval → validation

## Prerequisites

- **NASA Earthdata credentials**: `_netrc` (Windows) or `.netrc` in home folder
- **Copernicus CDS API key**: `~/.cdsapirc`
- **Google Earth Engine**: project ID + `acquisition/gdrive_credentials.json` for MODIS

Edit regions and dates in [config/settings.py](config/settings.py).

## Data Fetching

**CYGNSS:**
```
python acquisition/fetch_cygnss.py --region <REGION> --year <YEAR> --workers 2
```

**SMAP:**
```
python acquisition/fetch_smap.py --region <REGION> --year <YEAR> --workers 4
```

**MODIS NDVI:**
```
python acquisition/fetch_modis_ndvi.py --region <REGION> --year <YEAR> --project <GEE_PROJECT>
```

**ERA5:**
```
python acquisition/fetch_era5.py --region <REGION> --year <YEAR>
```

*Example:*
```
python acquisition/fetch_cygnss.py --region pakistan --year 2023
```

## Quality Control

```
python processing/build_qc_cache.py --region <REGION> --year <YEAR>
```

*Example:*
```
python processing/build_qc_cache.py --region pakistan --year 2023
```

## Vegetation Correction

```
python processing/build_veg_cache.py --region <REGION> --year <YEAR>
```

*Example:*
```
python processing/build_veg_cache.py --region pakistan --year 2023
```

## Roughness Correction

**Train (2019–2021 data):**
```
python processing/build_rough_train.py --region <REGION> --year-start 2019 --year-end 2021
```

**Apply:**
```
python processing/build_rough_apply.py --region <REGION> --train-tag 2019-2021 --year-start 2023 --year-end 2025
```

*Example:*
```
python processing/build_rough_train.py --region pakistan --year-start 2019 --year-end 2021
python processing/build_rough_apply.py --region pakistan --train-tag 2019-2021 --year-start 2023 --year-end 2025
```

## Soil Moisture Retrieval

```
python processing/build_sm_retrieve.py --region <REGION> --year-start 2023 --year-end 2025
```

*Example:*
```
python processing/build_sm_retrieve.py --region pakistan --year-start 2023 --year-end 2025
```

## Validation

**Spatial (one month):**
```
python validation/spatial_analysis/main.py --region <REGION> --year <YEAR> --month <MONTH> --sr-col <sr|sr_veg|sr_rough|sm>
```

**Temporal (multi-year):**
```
python validation/temporal_analysis/main.py --region <REGION> --year-start 2023 --year-end 2025 --sr-col <sr|sr_veg|sr_rough|sm>
```

*Example:*
```
python validation/spatial_analysis/main.py --region pakistan --year 2023 --month 8 --sr-col sm
python validation/temporal_analysis/main.py --region pakistan --year-start 2023 --year-end 2025 --sr-col sm
```
