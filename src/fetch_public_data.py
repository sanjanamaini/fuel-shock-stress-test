"""Public data pipeline for the fuel shock stress test.

Pulls oil and diesel price series from the US EIA's public history files
(no API key needed) and saves tidy CSVs to data/public/.
Note: FRED is unreachable from this network, so EIA is the source of truth
here; it is the primary source for these series anyway.

Week 2 extensions: Eurostat (Germany diesel), PPAC (India), ANP (Brazil),
BLS OES (US driver wages).

Run:  python src/fetch_public_data.py
"""
import os
import urllib.request

import pandas as pd

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "public")

# EIA legacy history workbooks: sheet "Data 1", data starts after 2 header rows.
SOURCES = {
    "brent_usd_bbl_daily": {
        "url": "https://www.eia.gov/dnav/pet/hist_xls/RBRTEd.xls",
        "desc": "Brent spot FOB, USD per barrel, daily",
    },
    "wti_usd_bbl_daily": {
        "url": "https://www.eia.gov/dnav/pet/hist_xls/RWTCd.xls",
        "desc": "WTI Cushing spot FOB, USD per barrel, daily",
    },
    "us_diesel_usd_gal_weekly": {
        "url": "https://www.eia.gov/dnav/pet/hist_xls/EMD_EPD2D_PTE_NUS_DPGw.xls",
        "desc": "US No 2 diesel retail, USD per gallon, weekly",
    },
}


def fetch_eia_xls(url: str) -> pd.DataFrame:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=90) as resp:
        raw = resp.read()
    tmp = os.path.join(OUT_DIR, "_tmp.xls")
    with open(tmp, "wb") as f:
        f.write(raw)
    df = pd.read_excel(tmp, sheet_name="Data 1", skiprows=2)
    os.remove(tmp)
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for name, meta in SOURCES.items():
        df = fetch_eia_xls(meta["url"])
        path = os.path.join(OUT_DIR, f"{name}.csv")
        df.to_csv(path, index=False)
        print(f"{name}: {len(df)} rows, {df['date'].min().date()} to "
              f"{df['date'].max().date()}  ({meta['desc']})")


if __name__ == "__main__":
    main()
