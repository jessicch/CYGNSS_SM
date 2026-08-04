# earthdata authentication

import os
import requests

try:
    from config.credentials import EARTHDATA_USERNAME, EARTHDATA_PASSWORD
except ImportError:
    EARTHDATA_USERNAME = None
    EARTHDATA_PASSWORD = None


# set netrc env var
def setup_netrc_env():
    home = os.path.expanduser("~")
    for fname in ("_netrc", ".netrc"):
        path = os.path.join(home, fname)
        if os.path.exists(path):
            os.environ["NETRC"] = path
            return path
    return None


# create authenticated session
def make_session() -> requests.Session:
    session = requests.Session()

    if EARTHDATA_USERNAME and EARTHDATA_PASSWORD:
        session.auth = (EARTHDATA_USERNAME, EARTHDATA_PASSWORD)
        return session

    import netrc as _netrc_mod
    home = os.path.expanduser("~")
    for fname in ("_netrc", ".netrc"):
        path = os.path.join(home, fname)
        if os.path.exists(path):
            try:
                nrc = _netrc_mod.netrc(path)
                auth = nrc.authenticators("urs.earthdata.nasa.gov")
                if auth:
                    session.auth = (auth[0], auth[2])
                    return session
            except Exception:
                pass

    print("  WARNING: No Earthdata credentials found. Downloads may fail.")
    return session
