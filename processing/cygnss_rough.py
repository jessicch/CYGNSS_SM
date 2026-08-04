# reads the rough cache, imported by validation

import os

import pandas as pd

# reads one month-file, empty dataframe if missing
def load_rough_cache_month(rough_dir: str, region: str,
                           year: int, month: int) -> pd.DataFrame:
    import geopandas as gpd
    fpath = os.path.join(
        rough_dir, region, str(year),
        f"cygnss_rough_{region}_{year}{month:02d}.parquet",
    )
    if not os.path.exists(fpath):
        return pd.DataFrame()
    return gpd.read_parquet(fpath)
