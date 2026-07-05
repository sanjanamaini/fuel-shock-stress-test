"""Public company screen: a curated list of real, named, publicly listed trucking
and logistics companies across the five focus countries, with financials pulled
from Yahoo Finance (free, no API key). This replaces the CapitalIQ screen that
was originally planned -- Sanjana does not currently have CapitalIQ access, so
this project is built to be fully reproducible on public data alone. CapitalIQ
can be added later as an optional upgrade for broader/cleaner coverage; nothing
here depends on it.

Coverage is real but uneven by design: Germany's trucking sector is mostly
private/family-owned (Rhenus, Dachser), so Deutsche Post DHL is the one clean
public name there. That is an honest limitation of public markets, not a
shortcut taken by this pipeline.

Run:  python src/fetch_company_screen.py
"""
import os

import pandas as pd
import yfinance as yf

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "public")

TICKERS = {
    # ticker: country
    "JBHT": "US", "KNX": "US", "ODFL": "US", "WERN": "US", "SAIA": "US",
    "CHRW": "US", "SNDR": "US", "ARCB": "US", "MRTN": "US",
    "VRLLOG.NS": "India", "TCI.NS": "India", "MAHLOG.NS": "India",
    "ALLCARGO.NS": "India", "TCIEXP.NS": "India",
    "600233.SS": "China", "002352.SZ": "China", "601598.SS": "China", "ZTO": "China",
    "DHL.DE": "Germany",
    "JSLG3.SA": "Brazil", "LOGN3.SA": "Brazil", "RAPT4.SA": "Brazil",
}

FIELDS = [
    "longName", "marketCap", "totalRevenue", "operatingMargins", "ebitdaMargins",
    "grossMargins", "totalCash", "totalDebt", "fullTimeEmployees", "currency",
]


def fetch_one(ticker: str, country: str) -> dict:
    info = yf.Ticker(ticker).info
    row = {"ticker": ticker, "country": country}
    for f in FIELDS:
        row[f] = info.get(f)
    return row


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for ticker, country in TICKERS.items():
        try:
            row = fetch_one(ticker, country)
            rows.append(row)
            print(f"{ticker:14} {country:8} {str(row.get('longName'))[:35]}")
        except Exception as e:
            print(f"{ticker:14} {country:8} FAILED: {e}")

    df = pd.DataFrame(rows)
    # Revenue is in local currency for non-US/USD-reporting tickers (INR, CNY, EUR, BRL);
    # keep as reported for now and note it -- FX normalization is a documented next step.
    path = os.path.join(OUT_DIR, "company_screen.csv")
    df.to_csv(path, index=False)
    print(f"\nwrote {len(df)} companies to {path}")
    print(df.groupby("country").size())


if __name__ == "__main__":
    main()
