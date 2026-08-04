# downloads cygnss l1 via cmr search + opendap, only the variables we need.
# clips to the region bbox, saves as netcdf. skips files already on disk.

import os
import sys
import re
import argparse
import tempfile
import warnings #debug
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import netCDF4 as nc4
import requests
import time
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import CYGNSS_DIR, REGIONS, START_DATE, END_DATE
from acquisition._earthdata import setup_netrc_env, make_session
from acquisition._dates import build_date_list, to_iso


# CMR and OPeNDAP endpoints
CMR_SEARCH    = "https://cmr.earthdata.nasa.gov/search/granules.json"
COLLECTION_ID = "C2832195379-POCLOUD"
OPENDAP_BASE  = "https://opendap.earthdata.nasa.gov/collections/C2832195379-POCLOUD/granules"

# variables we extract from each cygnss obs
DDM_VARIABLES = [
    "ddm_snr",
    "sp_lat",
    "sp_lon",
    "sp_rx_gain",
    "tx_to_sp_range",
    "rx_to_sp_range",
    "gps_tx_power_db_w",
    "gps_ant_gain_db_i",
    "quality_flags",
    "prn_code",
    "sp_inc_angle",
    "sp_land_valid",
    "modis_land_cover",
]

# Variables with shape (sample,) — no DDM dimension
SAMPLE_VARIABLES = [
    "ddm_timestamp_utc",
]



# query cmr for granules
def get_granules(start_date: str, end_date: str) -> list:
    params = {
        "collection_concept_id": COLLECTION_ID, 
        "temporal": f"{start_date}T00:00:00Z,{end_date}T23:59:59Z",
        "page_size": 2000,
    }
    try:
        r = requests.get(CMR_SEARCH, params=params, timeout=60)
        r.raise_for_status()
    except Exception as e:
        print(f"    CMR error: {e}")
        return []

    results = []
    for entry in r.json()["feed"]["entry"]:
        title  = entry["title"]
        sat_id = title.split(".")[0]
        results.append((sat_id, title))
    return results



# output path for nc files
def output_path(region_name: str, sat: str, date_str: str) -> str:
    folder = os.path.join(CYGNSS_DIR, region_name, date_str[:4], date_str[4:6])
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f"{sat}_{date_str}.nc")



# get sample count from dmr
def get_sample_count(opendap_url: str, session: requests.Session) -> int:
    r = session.get(opendap_url + ".dmr", timeout=60)
    r.raise_for_status()
    dims = dict(re.findall(r'<Dimension\s+name="([^"]+)"\s+size="(\d+)"', r.text))
    return int(dims.get("sample", 0))


# build constraint expression for all variables
def build_ce(n_samples: int) -> str:
    parts  = [f"/{v}[0:{n_samples-1}][0:3]" for v in DDM_VARIABLES]
    parts += [f"/{v}[0:{n_samples-1}]"      for v in SAMPLE_VARIABLES]
    return ";".join(parts)


# save dataframe to netcdf4 file
def _save_netcdf(df: pd.DataFrame, path: str):
    ds = nc4.Dataset(path, "w", format="NETCDF4")
    ds.createDimension("sample", len(df))
    for col in df.columns:
        arr = df[col].values
        dtype = arr.dtype
        # pick the right netCDF4 dtype
        if np.issubdtype(dtype, np.integer):
            nc_dtype = "i4" if dtype.itemsize <= 4 else "i8"
        elif np.issubdtype(dtype, np.floating):
            nc_dtype = "f4" if dtype.itemsize <= 4 else "f8"
        else:
            nc_dtype = "f8"
        v = ds.createVariable(col, nc_dtype, ("sample",), zlib=True, complevel=1)
        v[:] = arr
    ds.close()


# download granule, clip to region, save as netcdf
def process_granule(sat: str, date_str: str, granule_name: str,
                    region_name: str, bounds: dict,
                    session: requests.Session) -> int:
    from pydap.client import UNPACKDAP4DATA

    out = output_path(region_name, sat, date_str)
    if os.path.exists(out):
        return -1

    opendap_url = f"{OPENDAP_BASE}/{granule_name}"
    label       = f"{sat} {date_str}"

    try:
        t_start = time.perf_counter()

        # get sample count from DMR
        n_samples = get_sample_count(opendap_url, session)
        if n_samples == 0:
            tqdm.write(f"  SKIP {label}: could not read sample count from DMR")
            return 0
        t_dmr = time.perf_counter() - t_start

        # fetch all variables across all 4 DDM channels
        ce      = build_ce(n_samples)
        dap_url = opendap_url + ".dap?dap4.ce=" + requests.utils.quote(ce, safe="")

        t0 = time.perf_counter()
        r  = session.get(dap_url, timeout=180, stream=True)
        r.raise_for_status()

        tmp = tempfile.NamedTemporaryFile(suffix=".dap4", delete=False)
        downloaded = 0
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            tmp.write(chunk)
            downloaded += len(chunk)
        tmp.flush()
        tmp.close()
        t_dap = time.perf_counter() - t0

        # parse the binary response
        t0 = time.perf_counter()
        with open(tmp.name, "rb") as f:
            dataset = UNPACKDAP4DATA(f).dataset
        os.unlink(tmp.name)
        t_parse = time.perf_counter() - t0

        # parse variables from dap4 response
        raw = {}
        for v in DDM_VARIABLES + SAMPLE_VARIABLES:
            if v not in dataset:
                continue
            arr = np.array(dataset[v].data)
            if v == "sp_lon":
                arr = np.where(arr > 180, arr - 360, arr)
            raw[v] = arr

        sp_lat_all = raw.get("sp_lat")
        sp_lon_all = raw.get("sp_lon")
        if sp_lat_all is None or sp_lon_all is None:
            return 0

        # iterate over each DDM channel, apply bbox filter, tag with channel index
        n_ddms = sp_lat_all.shape[1] if sp_lat_all.ndim == 2 else 1
        dfs = []
        for ddm_idx in range(n_ddms):
            cols = {}
            for v in DDM_VARIABLES:
                if v not in raw:
                    continue
                arr = raw[v]
                # DDM vars are (N, 4). slice the current channel
                cols[v] = arr[:, ddm_idx] if arr.ndim == 2 else arr
            for v in SAMPLE_VARIABLES:
                if v not in raw:
                    continue
                # sample vars are (N,). same timestamp shared 4 all DDMs
                cols[v] = raw[v]

            sp_lat = cols.get("sp_lat")
            sp_lon = cols.get("sp_lon")
            if sp_lat is None or sp_lon is None:
                continue

            mask = (
                (sp_lat >= bounds["lat_min"]) & (sp_lat <= bounds["lat_max"]) &
                (sp_lon >= bounds["lon_min"]) & (sp_lon <= bounds["lon_max"])
            )
            if not np.any(mask):
                continue

            df_ddm = pd.DataFrame({v: cols[v][mask] for v in cols})
            df_ddm["ddm_channel"] = ddm_idx
            dfs.append(df_ddm)

        if not dfs:
            return 0

        df = pd.concat(dfs, ignore_index=True)
        if df.empty:
            return 0

        # save
        t0 = time.perf_counter()
        _save_netcdf(df, out)
        t_save = time.perf_counter() - t0

        n_match = len(df)
        mb = downloaded / 1_048_576
        t_total = time.perf_counter() - t_start
        tqdm.write(
            f"  {label}  |  {mb:.1f} MB  "
            f"dmr={t_dmr:.1f}s  dap={t_dap:.1f}s  "
            f"parse={t_parse:.1f}s  save={t_save:.1f}s  "
            f"total={t_total:.1f}s  samples={n_match}"
        )
        return n_match

    except Exception as e:
        tqdm.write(f"  ERROR {label}: {type(e).__name__}: {e}")
        try:
            if "tmp" in dir() and os.path.exists(tmp.name):
                os.unlink(tmp.name)
        except Exception:
            pass
        if os.path.exists(out):
            try: os.remove(out)
            except: pass
        return 0


# download all granules for region
def download_region(region_name: str, bounds: dict,
                    dates: list, session: requests.Session,
                    max_workers: int = 1):
    print(f"\n{'='*60}")
    print(f"  {bounds['label']}")
    print(f"  {dates[0]} -> {dates[-1]}  ({len(dates)} days)")
    print(f"  Workers: {max_workers}")
    print(f"{'='*60}")

    # group by month so we can batch CMR calls
    months: dict = {}
    for d in dates:
        months.setdefault(d[:6], []).append(d)

    # build task list from CMR results
    print("  Querying CMR...")
    tasks = []
    for ym, month_dates in tqdm(months.items(), desc="  CMR", unit="month"):
        for sat_id, granule_name in get_granules(to_iso(month_dates[0]),
                                                 to_iso(month_dates[-1])):
            try:
                date_str = granule_name.split(".")[2][1:9]
            except Exception:
                continue
            if date_str in month_dates:
                out = output_path(region_name, sat_id, date_str)
                if not os.path.exists(out):
                    tasks.append((sat_id, date_str, granule_name))

    print(f"  {len(tasks)} granules to download\n")
    if not tasks:
        print("  Nothing to do.")
        return

    saved = empty = error = 0

    def _run(args):
        sat, date_str, granule_name = args
        n = process_granule(sat, date_str, granule_name, region_name, bounds, session)
        return sat, date_str, n

    if max_workers == 1:
        for sat, date_str, granule_name in tqdm(tasks, desc=f"  {region_name}", unit="file"):
            _, _, n = _run((sat, date_str, granule_name))
            if n > 0:
                saved += 1
            elif n == 0:
                empty += 1
            else:
                error += 1
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading
        _lock = threading.Lock()

        with tqdm(total=len(tasks), desc=f"  {region_name}", unit="file") as pbar:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = {ex.submit(_run, t): t for t in tasks}
                for future in as_completed(futures):
                    sat, date_str, n = future.result()
                    with _lock:
                        if n > 0:
                            saved += 1
                        elif n == 0:
                            empty += 1
                        else:
                            error += 1
                        pbar.update(1)
                        pbar.set_postfix(
                            saved=saved, empty=empty, err=error, refresh=False
                        )

    print(f"\n  Saved: {saved}  Empty: {empty}  Error: {error}")


def main():
    # run before pydap imports to find _netrc file on Windows
    netrc_path = setup_netrc_env()
    if netrc_path:
        print(f"  Using netrc: {netrc_path}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--year",    type=int, default=None,
                        help="Download only this year (2022/2023/2024)")
    parser.add_argument("--region",  default=None,
                        help="Download only this region (e.g. pakistan)")
    parser.add_argument("--workers", type=int, default=2,
                        help="Parallel downloads per terminal (default 2)")
    parser.add_argument("--start",   default=None,
                        help="Resume from this date YYYYMMDD (e.g. 20220815)")
    args = parser.parse_args()

    # set date range
    if args.start:
        start = args.start
    elif args.year:
        start = f"{args.year}0101"
    else:
        start = START_DATE

    if args.year:
        end = f"{args.year}1231"
    else:
        end = END_DATE

    print("\n" + "="*60)
    print("  CYGNSS L1 Data Acquisition  [OPeNDAP DAP4 mode]")
    print(f"  Period  : {start} -> {end}")
    print(f"  Workers : {args.workers} per terminal")
    if args.region:
        print(f"  Region  : {args.region}")
    print("="*60)

    session = make_session()
    dates   = build_date_list(start, end)

    for region_name, bounds in REGIONS.items():
        if args.region and region_name != args.region:
            print(f"\n  Skipping {bounds['label']}")
            continue
        download_region(region_name, bounds, dates, session,
                        max_workers=args.workers)

    print("\n  Done.\n")


if __name__ == "__main__":
    main()
