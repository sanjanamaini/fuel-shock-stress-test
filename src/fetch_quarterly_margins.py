"""Quarterly operating margins for the 22-company screen, from Yahoo Finance (free, no key).

Used to check the stress test against what actually happened in the March to June 2026 oil
shock: each company's reported operating margin before the shock (quarter ending December 2025)
against the shock quarter (ending June 2026). Writes data/public/quarterly_margins.csv with
revenue, operating income and their ratio per company and quarter; quarters a company has not
reported, or that Yahoo does not carry, are simply absent.

Run:  python src/fetch_quarterly_margins.py
"""
import os
import time

import pandas as pd
import yfinance as yf

from fetch_company_screen import TICKERS

OUT = os.path.join(os.path.dirname(__file__), "..", "data", "public", "quarterly_margins.csv")


def one(ticker: str) -> pd.DataFrame:
    q = yf.Ticker(ticker).quarterly_income_stmt
    if q is None or q.empty:
        return pd.DataFrame()
    rows = []
    for col in q.columns:
        rev = q.loc["Total Revenue", col] if "Total Revenue" in q.index else None
        op = q.loc["Operating Income", col] if "Operating Income" in q.index else None
        rows.append({"ticker": ticker, "quarter_end": pd.Timestamp(col).date(), "revenue": rev, "operating_income": op})
    return pd.DataFrame(rows)


def main():
    frames = []
    for ticker, country in TICKERS.items():
        try:
            f = one(ticker)
            f["country"] = country
            frames.append(f)
            print(f"{ticker:14} {len(f)} quarters, latest {f['quarter_end'].max() if len(f) else None}")
        except Exception as e:  # network or missing data: record and move on
            print(f"{ticker:14} FAILED: {e}")
        time.sleep(1)
    out = pd.concat(frames, ignore_index=True)
    out["operating_margin"] = out["operating_income"] / out["revenue"]
    out.to_csv(OUT, index=False)
    print("wrote", OUT, len(out), "rows")


if __name__ == "__main__":
    main()
