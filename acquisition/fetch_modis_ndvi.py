# export modis ndvi via gee, download from drive
import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import MODISKM_DIR, REGIONS, START_DATE, END_DATE

try:
    from config.credentials import GEE_PROJECT
except (ImportError, AttributeError):
    GEE_PROJECT = os.environ.get("GEE_PROJECT", None)


GEE_DRIVE_FOLDER = "thesis_data"

# NDVI (16-day, 1 km native)
NDVI_COLLECTION = "MODIS/061/MOD13A2"
NDVI_SCALE_M    = 1000
NDVI_SCALE      = 0.0001    # raw DN to NDVI 

CRS_OUT = "EPSG:4326"


# Path helpers
def _date_to_iso(date_str: str) -> str:
    return datetime.strptime(date_str, "%Y%m%d").strftime("%Y-%m-%d")


def _out_dir(region_name: str, sub: str | None = None) -> str:
    d = os.path.join(MODISKM_DIR, region_name)
    if sub:
        d = os.path.join(d, sub)
    os.makedirs(d, exist_ok=True)
    return d


# GeoTIFF export

# Filename for one composite
def _tif_prefix_ndvi(region_name: str, year: int, doy: int) -> str:
    return f"mod13a2_{region_name}_{year}{doy:03d}"


# escape special chars for gee
def _gee_description(stem: str) -> str:
    return stem.replace(".", "_").replace("/", "_")


# GEE helpers
def _bbox(bounds: dict):
    import ee
    return ee.Geometry.BBox(
        bounds["lon_min"], bounds["lat_min"],
        bounds["lon_max"], bounds["lat_max"],
    )


# loads mod13a2, masks and scales ndvi.

def _load_ndvi_collection(ee, aoi, start_iso: str, end_iso: str):
    raw = (
        ee.ImageCollection(NDVI_COLLECTION)
        .filterDate(start_iso, end_iso)
        .filterBounds(aoi)
        .select(["NDVI", "SummaryQA"])
    )

    def prep(img):
        proj = img.select("NDVI").projection()
        mask = img.select("SummaryQA").lte(1)   # 0=good, 1=marginal
        ndvi = (img.select("NDVI")
                   .multiply(NDVI_SCALE)
                   .updateMask(mask)
                   .setDefaultProjection(proj))
        return (ndvi
                .copyProperties(img, ["system:time_start"])
                .set("system:time_start", img.get("system:time_start")))

    return raw.map(prep)


# log task ids
def _record_task(region_name: str, task_id: str, description: str) -> None:
    task_file = os.path.join(_out_dir(region_name), "task_id.txt")
    mode_flag = "a" if os.path.exists(task_file) else "w"
    with open(task_file, mode_flag) as f:
        f.write(f"{task_id}\t{description}\n")


# gee timestamps
def _ts_ms_to_year_doy(ts_ms: int) -> tuple[int, int]:
    dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
    return dt.year, dt.timetuple().tm_yday


# per-composite NDVI geotiff
def export_gridded(region_name: str, bounds: dict,
                   start_iso: str, end_iso: str, project: str):
    import ee
    ee.Initialize(project=project)
    
    print(f"\n  [NDVI GeoTIFF @ ~1 km]  {region_name}  "
          f"{start_iso} → {end_iso}")

    aoi   = _bbox(bounds)
    clean = _load_ndvi_collection(ee, aoi, start_iso, end_iso)

    count = clean.size().getInfo()
    print(f"  Composites found: {count}")
    if count == 0:
        print("  No images — skipping."); return

    # Aggregate all timestamps
    timestamps = clean.aggregate_array("system:time_start").getInfo()
    imgs       = clean.toList(count)

    # starts the tasks
    submitted = 0
    for i, ts in enumerate(timestamps):
        year, doy = _ts_ms_to_year_doy(int(ts))
        stem      = _tif_prefix_ndvi(region_name, year, doy)
        desc      = _gee_description(f"MODIS_NDVI_{stem}")

        img = ee.Image(imgs.get(i)).toFloat()
        task = ee.batch.Export.image.toDrive(
            image          = img,
            description    = desc,
            folder         = GEE_DRIVE_FOLDER,
            fileNamePrefix = stem,
            region         = aoi,
            scale          = NDVI_SCALE_M,
            crs            = CRS_OUT,
            maxPixels      = 1e9,
            fileFormat     = "GeoTIFF",
        )
        task.start()
        _record_task(region_name, task.id, desc)
        submitted += 1

    print(f"  Submitted {submitted} NDVI tasks  "
          f"(monitor at https://code.earthengine.google.com/tasks)")


# Google Drive download

# get authenticated drive service
def _drive_service():
    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
        from google.auth.transport.requests import Request
        from googleapiclient.discovery import build
        import pickle
    except ImportError:
        raise ImportError(
            "Drive download requires:\n"
            "    pip install google-auth google-auth-oauthlib google-api-python-client"
        )

    SCOPES     = ["https://www.googleapis.com/auth/drive.readonly"]
    token_path = os.path.join(os.path.dirname(__file__), ".gdrive_token.pickle")

    creds = None
    if os.path.exists(token_path):
        with open(token_path, "rb") as f:
            creds = pickle.load(f)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception:
                os.remove(token_path)
                creds = None
        if not creds:
            flow = InstalledAppFlow.from_client_secrets_file(
                os.path.join(os.path.dirname(__file__), "gdrive_credentials.json"),
                SCOPES,
            )
            creds = flow.run_local_server(port=0, open_browser=True, timeout_seconds=600)
        with open(token_path, "wb") as f:
            pickle.dump(creds, f)

    return build("drive", "v3", credentials=creds)


def _download_drive_file(service, file_id: str, out_path: str) -> None:
    from googleapiclient.http import MediaIoBaseDownload
    import io
    request = service.files().get_media(fileId=file_id)
    buf     = io.BytesIO()
    dl      = MediaIoBaseDownload(buf, request)
    done    = False
    while not done:
        _, done = dl.next_chunk()
    with open(out_path, "wb") as f:
        f.write(buf.getvalue())


# download ndvi geotiffs from drive
def download_from_drive(region_name: str) -> None:
    try:
        service = _drive_service()
    except ImportError as e:
        print(f"  {e}")
        return

    prefix = f"mod13a2_{region_name}_"

    # check all matching tif files
    out_folder = _out_dir(region_name, sub="ndvi")
    page_token = None
    matched, downloaded, skipped = 0, 0, 0
    while True:
        results = service.files().list(
            q=f"name contains '{prefix}' and name contains '.tif' and trashed=false",
            spaces="drive",
            fields="nextPageToken, files(id, name)",
            pageSize=1000,
            pageToken=page_token,
        ).execute()
        for f in results.get("files", []):
            matched += 1
            out_path = os.path.join(out_folder, f["name"])
            if os.path.exists(out_path):
                skipped += 1
                continue
            _download_drive_file(service, f["id"], out_path)
            downloaded += 1
            if downloaded % 10 == 0:
                print(f"  ...{downloaded} files downloaded")
        page_token = results.get("nextPageToken")
        if not page_token:
            break

    print(f"  Drive matches: {matched}   downloaded: {downloaded}   "
          f"already-on-disk: {skipped}")
    if matched == 0:
        print(f"  No '{prefix}*.tif' files found in Drive — exports still running?")


def main():
    parser = argparse.ArgumentParser(
        description="Export MODIS NDVI GeoTIFFs (~1 km, per 16-day composite) "
                    "via Google Earth Engine.",
    )
    parser.add_argument(
        "--region", default=None,
        help="Region key from config.settings.REGIONS. Default: all.",
    )
    parser.add_argument("--year",  type=int, default=None,
                        help="Single calendar year shortcut (e.g. 2023).")
    parser.add_argument("--start", default=None,
                        help="Start date YYYYMMDD (overrides --year).")
    parser.add_argument("--end",   default=None,
                        help="End date YYYYMMDD (overrides --year).")
    parser.add_argument("--project", default=GEE_PROJECT,
                        help="GEE project ID. Also settable via credentials.py.")
    parser.add_argument("--download-only", action="store_true",
                        help="Skip GEE export — download finished files from Drive.")
    args = parser.parse_args()

    # Resolve date range
    if args.start:
        start_iso = _date_to_iso(args.start)
    elif args.year:
        start_iso = f"{args.year}-01-01"
    else:
        start_iso = _date_to_iso(START_DATE)

    if args.end:
        end_iso = _date_to_iso(args.end)
    elif args.year:
        end_iso = f"{args.year}-12-31"
    else:
        end_iso = _date_to_iso(END_DATE)

    # validate gee project
    if not args.project and not args.download_only:
        print(
            "ERROR: No GEE project ID.\n"
            "  Use --project YOUR_ID or set GEE_PROJECT in config/credentials.py."
        )
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  MODIS NDVI export")
    print(f"  Period  : {start_iso} → {end_iso}")
    if args.region:
        print(f"  Region  : {args.region}")
    print("=" * 60)

    for region_name, bounds in REGIONS.items():
        if args.region and region_name != args.region:
            continue
        print(f"\n  [{bounds['label']}]")

        if args.download_only:
            download_from_drive(region_name)
        else:
            export_gridded(region_name, bounds, start_iso, end_iso, args.project)

    print("\n  Done.\n")


if __name__ == "__main__":
    main()
