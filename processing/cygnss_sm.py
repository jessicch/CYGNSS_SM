# Load SM cache (imported by validation)

import os

import pandas as pd

# Load one month-file from SM cache
def load_sm_cache_month(sm_dir: str, region: str,
                        year: int, month: int) -> pd.DataFrame:
    import geopandas as gpd
    fpath = os.path.join(
        sm_dir, region, str(year),
        f"cygnss_sm_{region}_{year}{month:02d}.parquet",
    )
    if not os.path.exists(fpath):
        return pd.DataFrame()
    return gpd.read_parquet(fpath)
