# downloads smap 9 km soil moisture per region

import argparse
import os
import sys
import warnings
import threading
import tempfile
warnings.filterwarnings("ignore")

import numpy as np
import h5py
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import SMAP_DIR, REGIONS, START_DATE, END_DATE
from acquisition._earthdata import setup_netrc_env, make_session
from acquisition._dates import build_date_list, to_iso


# settings
PRODUCT_9KM = "SPL3SMP_E"

CMR_URL    = "https://cmr.earthdata.nasa.gov/search/granules.json"
CHUNK_SIZE = 65536
TIMEOUT    = 180
MAX_WORKERS = 4

AM_GROUP = "Soil_Moisture_Retrieval_Data_AM"
PM_GROUP = "Soil_Moisture_Retrieval_Data_PM"

# variables to download (pm ones are renamed later to match am)
SMAP_VARIABLES_AM = [
    "soil_moisture",
    "latitude",
    "longitude",
    "retrieval_qual_flag",
    "clay_fraction",
]
SMAP_VARIABLES_PM = [
    "soil_moisture_pm",
    "latitude_pm",
    "longitude_pm",
    "retrieval_qual_flag_pm",
    "clay_fraction",
]

SMAP_FILL = -9999.0


# ask cmr for granule urls
def _cmr_search(short_name: str, start_iso: str, end_iso: str) -> list:
    try:
        r = requests.get(
            CMR_URL,
            params={
                "short_name": short_name,
                "temporal"  : f"{start_iso}T00:00:00Z,{end_iso}T23:59:59Z",
                "page_size" : 2000,
            },
            timeout=60,
        )
        r.raise_for_status()
    except Exception as exc:
        print(f"    CMR search failed: {exc}")
        return []

    results = []
    for entry in r.json()["feed"]["entry"]:
        title = entry.get("title", "")
        # pull the yyyymmdd date out of the filename
        date_str = None
        for part in title.split("_"):
            if len(part) == 8 and part.isdigit():
                date_str = part
                break
        if date_str is None:
            continue
        data_url = None
        dap_url  = None
        for link in entry.get("links", []):
            rel  = link.get("rel", "")
            href = link.get("href", "")
            if data_url is None and "data#" in rel and href.endswith(".h5"):
                data_url = href
            if dap_url is None and "service#" in rel and "opendap" in href:
                dap_url = href
        if data_url:
            results.append((date_str, data_url, dap_url))
    return results


# these stay integers
INTEGER_VARS = {
    "surface_flag", "retrieval_qual_flag", "retrieval_qual_flag_pm",
    "landcover_class", "coherency_state",
}


# read variables from hdf5 group
def _read_group(h5file: h5py.File, group_name: str, variables: list) -> pd.DataFrame:
    if group_name not in h5file:
        return pd.DataFrame()
    grp  = h5file[group_name]
    cols = {}
    for var in variables:
        if var not in grp:
            continue
        raw = grp[var][:]
        if var in INTEGER_VARS:
            cols[var] = raw.ravel().astype(np.int32)
        else:
            arr = raw.astype(np.float32)
            arr[arr <= SMAP_FILL + 1] = np.nan   # fill -> nan
            cols[var] = arr.ravel()
    return pd.DataFrame(cols) if cols else pd.DataFrame()


# read am and pm groups and stack into table
def _open_h5_regional(path: str) -> pd.DataFrame:
    try:
        with h5py.File(path, "r") as f:
            df_AM = _read_group(f, AM_GROUP, SMAP_VARIABLES_AM)
            df_PM = _read_group(f, PM_GROUP, SMAP_VARIABLES_PM)
    except Exception as exc:
        print(f"    HDF5 read error: {exc}")
        return pd.DataFrame()

    if not df_AM.empty:
        df_AM["pass"] = "AM"
    if not df_PM.empty:
        df_PM = df_PM.rename(columns={
            "latitude_pm":            "latitude",
            "longitude_pm":           "longitude",
            "soil_moisture_pm":       "soil_moisture",
            "retrieval_qual_flag_pm": "retrieval_qual_flag",
        })
        df_PM["pass"] = "PM"

    df = pd.concat([df_AM, df_PM], ignore_index=True)
    return df.reset_index(drop=True)


# clip to region bbox
def _clip(df: pd.DataFrame, bounds: dict) -> pd.DataFrame:
    if df.empty or "latitude" not in df.columns:
        return df
    mask = (
        (df["latitude"]  >= bounds["lat_min"]) &
        (df["latitude"]  <= bounds["lat_max"]) &
        (df["longitude"] >= bounds["lon_min"]) &
        (df["longitude"] <= bounds["lon_max"])
    )
    return df[mask].reset_index(drop=True)


def _save(df: pd.DataFrame, path: str) -> None:
    df.to_hdf(path, key="smap", mode="w", complevel=4, complib="zlib")


# output path for saved file
def _regional_path(region_name: str, date_str: str) -> str:
    folder = os.path.join(SMAP_DIR, region_name, date_str[:4], date_str[4:6])
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f"smap_{region_name}_{date_str}.h5")


# subsetting from opendap
OPENDAP_WINDOW_MARGIN = 1

_GRID_PROFILE_PATH = os.path.join(SMAP_DIR, "ease2_9km_profile.npz")


# fetch dap4 response and unpack to dataset
def _fetch_dap(dap_url: str, session: requests.Session):
    from pydap.client import UNPACKDAP4DATA

    r = session.get(dap_url, stream=True, timeout=TIMEOUT)
    r.raise_for_status()
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".dap4")
    os.close(tmp_fd)
    try:
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                f.write(chunk)
        with open(tmp_path, "rb") as f:
            return UNPACKDAP4DATA(f).dataset
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# cache 9km ease2 grid lat/lon
def _grid_profile(opendap_url: str, session: requests.Session):
    if os.path.exists(_GRID_PROFILE_PATH):
        npz = np.load(_GRID_PROFILE_PATH)
        return npz["row_lat"], npz["col_lon"]

    print("  Building EASE-2 grid profile (one-time lat/lon fetch)...")
    ce = f"/{AM_GROUP}/latitude;/{AM_GROUP}/longitude"
    ds = _fetch_dap(opendap_url + ".dap?dap4.ce="
                    + requests.utils.quote(ce, safe=""), session)
    grp = ds[AM_GROUP]
    lat = np.asarray(grp["latitude"].data[:], dtype=np.float64)
    lon = np.asarray(grp["longitude"].data[:], dtype=np.float64)
    lat[lat <= SMAP_FILL + 1] = np.nan
    lon[lon <= SMAP_FILL + 1] = np.nan
    row_lat = np.nanmean(lat, axis=1)   # constant along each row
    col_lon = np.nanmean(lon, axis=0)   # constant along each column
    os.makedirs(SMAP_DIR, exist_ok=True)
    np.savez(_GRID_PROFILE_PATH, row_lat=row_lat, col_lon=col_lon)
    return row_lat, col_lon


# find ease2 window covering bbox
def _window_for(bounds: dict, row_lat: np.ndarray, col_lon: np.ndarray) -> tuple:
    rows = np.where((row_lat >= bounds["lat_min"]) &
                    (row_lat <= bounds["lat_max"]))[0]
    cols = np.where((col_lon >= bounds["lon_min"]) &
                    (col_lon <= bounds["lon_max"]))[0]
    if len(rows) == 0 or len(cols) == 0:
        raise ValueError(f"bbox '{bounds['label']}' maps to an empty EASE-2 window")
    m = OPENDAP_WINDOW_MARGIN
    return (max(int(rows[0]) - m, 0), min(int(rows[-1]) + m, len(row_lat) - 1),
            max(int(cols[0]) - m, 0), min(int(cols[-1]) + m, len(col_lon) - 1))


# check which variables exist in product
def _dmr_variable_lists(opendap_url: str, session: requests.Session):
    import xml.etree.ElementTree as ET

    r = session.get(opendap_url + ".dmr", timeout=TIMEOUT)
    r.raise_for_status()
    root = ET.fromstring(r.content)

    def group_vars(name: str) -> set:
        for grp in root.iter("{http://xml.opendap.org/ns/DAP/4.0#}Group"):
            if grp.get("name") == name:
                return {c.get("name") for c in grp
                        if c.get("name") and not c.tag.endswith("Attribute")}
        return set()

    vars_am = [v for v in SMAP_VARIABLES_AM if v in group_vars(AM_GROUP)]
    vars_pm = [v for v in SMAP_VARIABLES_PM if v in group_vars(PM_GROUP)]
    return vars_am, vars_pm


# read variables from pydap response
def _read_group_dap(ds, group_name: str, variables: list) -> pd.DataFrame:
    if group_name not in ds:
        return pd.DataFrame()
    grp = ds[group_name]
    have = set(grp.keys())
    cols = {}
    for var in variables:
        if var not in have:
            continue
        raw = np.asarray(grp[var].data[:])
        if var in INTEGER_VARS:
            cols[var] = raw.ravel().astype(np.int32)
        else:
            arr = raw.astype(np.float32)
            arr[arr <= SMAP_FILL + 1] = np.nan
            cols[var] = arr.ravel()
    return pd.DataFrame(cols) if cols else pd.DataFrame()


# download granule from opendap, clip to region window
def _download_and_process_opendap(
    date_str: str, opendap_url: str,
    region_name: str, bounds: dict,
    session: requests.Session,
    window: tuple, vars_am: list, vars_pm: list,
) -> int:
    out_path = _regional_path(region_name, date_str)
    if os.path.exists(out_path):
        return -1

    r0, r1, c0, c1 = window
    sl = f"[{r0}:{r1}][{c0}:{c1}]"
    parts  = [f"/{AM_GROUP}/{v}{sl}" for v in vars_am]
    parts += [f"/{PM_GROUP}/{v}{sl}" for v in vars_pm]
    dap_url = (opendap_url + ".dap?dap4.ce="
               + requests.utils.quote(";".join(parts), safe=""))

    ds = _fetch_dap(dap_url, session)

    df_AM = _read_group_dap(ds, AM_GROUP, vars_am)
    df_PM = _read_group_dap(ds, PM_GROUP, vars_pm)

    if not df_AM.empty:
        df_AM["pass"] = "AM"
    if not df_PM.empty:
        df_PM = df_PM.rename(columns={
            "latitude_pm":            "latitude",
            "longitude_pm":           "longitude",
            "soil_moisture_pm":       "soil_moisture",
            "retrieval_qual_flag_pm": "retrieval_qual_flag",
        })
        df_PM["pass"] = "PM"

    df = pd.concat([df_AM, df_PM], ignore_index=True).reset_index(drop=True)
    df = _clip(df, bounds)

    if df.empty:
        return 0

    _save(df, out_path)
    return len(df)


# download granule from http, clip and save
def _download_and_process(
    date_str: str, url: str,
    region_name: str, bounds: dict,
    session: requests.Session,
) -> int:
    out_path = _regional_path(region_name, date_str)

    if os.path.exists(out_path):
        return -1

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".h5")
    os.close(tmp_fd)

    try:
        r = session.get(url, stream=True, timeout=TIMEOUT)
        if r.status_code == 401:
            r = session.get(r.url, stream=True, timeout=TIMEOUT)
        r.raise_for_status()

        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                f.write(chunk)

        df = _open_h5_regional(tmp_path)
        df = _clip(df, bounds)

        if df.empty:
            return 0

        _save(df, out_path)
        return len(df)

    except Exception as exc:
        print(f"    Error ({date_str}): {exc}")
        return -1
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# download spl3smp_e for region
def download_regional(region_name: str, bounds: dict, dates: list,
                      session: requests.Session, max_workers: int = 4,
                      method: str = "opendap"):
    print(f"\n{'='*60}")
    print(f"  {bounds['label']}")
    print(f"  {dates[0]} -> {dates[-1]}  ({len(dates)} days)")
    print(f"  Workers: {max_workers}   Method: {method}")
    print(f"{'='*60}")

    # group by month so we can batch CMR calls (like fetch_cygnss)
    months: dict = {}
    for d in dates:
        months.setdefault(d[:6], []).append(d)

    # build task list from CMR results
    print("  Querying CMR...")
    tasks = []
    for ym, month_dates in tqdm(months.items(), desc="  CMR", unit="month"):
        for date_str, url, dap_url in _cmr_search(PRODUCT_9KM, to_iso(month_dates[0]),
                                                  to_iso(month_dates[-1])):
            try:
                if date_str in month_dates:
                    out = _regional_path(region_name, date_str)
                    if not os.path.exists(out):
                        tasks.append((date_str, url, dap_url))
            except Exception:
                continue

    print(f"  {len(tasks)} granules to download\n")
    if not tasks:
        print("  Nothing to do.")
        return

    # one-time OPeNDAP prep: grid window + variable availability
    window = vars_am = vars_pm = None
    if method == "opendap":
        ref = next((t[2] for t in tasks if t[2]), None)
        if ref is None:
            print("  No OPeNDAP links in CMR results — using full downloads.")
            method = "full"
        else:
            try:
                row_lat, col_lon = _grid_profile(ref, session)
                window = _window_for(bounds, row_lat, col_lon)
                vars_am, vars_pm = _dmr_variable_lists(ref, session)
                r0, r1, c0, c1 = window
                print(f"  OPeNDAP window: rows {r0}:{r1}, cols {c0}:{c1}  "
                      f"({(r1-r0+1)}x{(c1-c0+1)} of 1624x3856 cells)\n")
            except Exception as exc:
                print(f"  OPeNDAP prep failed ({exc}) — using full downloads.")
                method = "full"

    saved = empty = error = 0

    def _run(args):
        date_str, url, dap_url = args
        if method == "opendap" and dap_url:
            for attempt in (1, 2):   # transient Hyrax 5xx happen; retry once
                try:
                    n = _download_and_process_opendap(
                        date_str, dap_url, region_name, bounds,
                        session, window, vars_am, vars_pm)
                    return date_str, n
                except Exception as exc:
                    if attempt == 2:
                        tqdm.write(f"    OPeNDAP failed ({date_str}): {exc} "
                                   f"— falling back to full download")
        n = _download_and_process(date_str, url, region_name, bounds, session)
        return date_str, n

    if max_workers == 1:
        for task in tqdm(tasks, desc=f"  {region_name}", unit="file"):
            _, n = _run(task)
            if n > 0:
                saved += 1
            elif n == 0:
                empty += 1
            else:
                error += 1
    else:
        _lock = threading.Lock()

        with tqdm(total=len(tasks), desc=f"  {region_name}", unit="file") as pbar:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = {ex.submit(_run, t): t for t in tasks}
                for future in as_completed(futures):
                    date_str, n = future.result()
                    with _lock:
                        if n > 0:
                            saved += 1
                        elif n == 0:
                            empty += 1
                        else:
                            error += 1
                        pbar.update(1)
                        pbar.set_postfix(saved=saved, empty=empty, err=error, refresh=False)

    print(f"\n  Saved: {saved}  Empty: {empty}  Error: {error}")


# download for all regions
def fetch_regional(region_filter: str = None, max_workers: int = 4,
                   year: int = None, start_date: str = None,
                   method: str = "opendap"):
    start = f"{year}0101" if year else START_DATE
    end   = f"{year}1231" if year else END_DATE
    if start_date:
        start = start_date

    print(f"\n{'='*60}")
    print("  SMAP L3 9 km — Regional download  [CMR + OPeNDAP subset / direct HTTP]")
    print(f"  Product : {PRODUCT_9KM}")
    print(f"  Period  : {start} → {end}")
    print(f"  Workers : {max_workers}")
    print(f"  Method  : {method}")
    if region_filter:
        print(f"  Region  : {region_filter}")
    print(f"{'='*60}")

    session = make_session()
    dates   = build_date_list(start, end)

    for region_name, bounds in REGIONS.items():
        if region_filter and region_name != region_filter:
            print(f"\n  Skipping {bounds['label']}")
            continue
        download_regional(region_name, bounds, dates, session,
                         max_workers=max_workers, method=method)


# print summary of saved file
def inspect(path: str) -> None:
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return
    df = pd.read_hdf(path, key="smap")
    print(f"\nFile : {path}")
    print(f"Rows : {len(df):,}")
    print(f"Cols : {list(df.columns)}")
    print(df.describe().round(4))


def main():
    # setup netrc for authenticated downloads
    netrc_path = setup_netrc_env()
    if netrc_path:
        print(f"  Using netrc: {netrc_path}")

    parser = argparse.ArgumentParser(
        description="Download SMAP L3 data (CMR + direct HTTP, fast).",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--region", metavar="NAME", default=None,
        help="Only download this region (e.g. 'pakistan').",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help=f"Parallel download workers (default 4).",
    )
    parser.add_argument(
        "--method", choices=["opendap", "full"], default="opendap",
        help="Regional download method: 'opendap' fetches only the region's\n"
             "EASE-2 window server-side (default, ~40x faster); 'full'\n"
             "streams whole granules and clips locally (legacy).",
    )
    parser.add_argument(
        "--year", type=int, default=None,
        help="Download only this calendar year (e.g. 2022, 2023, 2024).",
    )
    parser.add_argument(
        "--start", metavar="YYYYMMDD", default=None,
        help="Override the start date (e.g. 20230601). Overrides --year start.",
    )
    parser.add_argument(
        "--inspect", metavar="FILE", default=None,
    )
    args = parser.parse_args()

    if args.inspect:
        inspect(args.inspect)
        return

    fetch_regional(region_filter=args.region, max_workers=args.workers,
                   year=args.year, start_date=args.start,
                   method=args.method)

    print("\n  Done.\n")


if __name__ == "__main__":
    main()