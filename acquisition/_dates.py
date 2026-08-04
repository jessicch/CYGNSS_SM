from datetime import datetime, timedelta


# YYYYMMDD -> YYYY-MM-DD
def to_iso(date_str: str) -> str:
    return datetime.strptime(date_str, "%Y%m%d").strftime("%Y-%m-%d")


# YYYYMMDD list from start to end
def build_date_list(start: str, end: str) -> list:
    cur, end_dt = datetime.strptime(start, "%Y%m%d"), datetime.strptime(end, "%Y%m%d")
    dates = []
    while cur <= end_dt:
        dates.append(cur.strftime("%Y%m%d"))
        cur += timedelta(days=1)
    return dates
