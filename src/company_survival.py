"""Applies the OU Monte Carlo shock engine to the real public company screen
(data/public/company_screen.csv) instead of one representative operator per
country. Each company's own trailing operating margin is used as its baseline
(a real, observed number), while fuel share of revenue and crude-to-pump beta
still come from the country-level assumptions in excel/build_model.py, since
per-company fuel cost breakdowns are not disclosed. That is an honest, stated
limitation: this is company-level margin heterogeneity applied on top of a
country-level shock sensitivity model, not a fully bottom-up per-company cost
model.

Run:  python src/company_survival.py
"""
import os

import numpy as np
import pandas as pd

from monte_carlo import COUNTRIES, HORIZON_MONTHS, N_PATHS, SEED, calibrate_ou, simulate_ou_paths

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "public")


def margin_path_for_company(price_path: np.ndarray, baseline_price: float,
                             company_margin: float, country_key: str) -> np.ndarray:
    c = COUNTRIES[country_key]
    pct_move = (price_path - baseline_price) / baseline_price
    months = np.arange(price_path.shape[-1])
    ramp = np.clip((months - 2) / max(1, (6 - 2)), 0, 1) * 0.50  # same lag/pass-through as monte_carlo.py
    hit = c["fuel_share_rev"] * pct_move * c["beta"] * (1 - ramp)
    return company_margin - hit


def main():
    brent = pd.read_csv(os.path.join(DATA_DIR, "brent_usd_bbl_daily.csv"), parse_dates=["date"])
    brent = brent.set_index("date")["value"]
    params = calibrate_ou(brent)
    paths = simulate_ou_paths(params, HORIZON_MONTHS, N_PATHS, SEED)
    baseline_price = params["last_price"]

    companies = pd.read_csv(os.path.join(DATA_DIR, "company_screen.csv"))

    rows = []
    for _, row in companies.iterrows():
        country = row["country"]
        margin0 = row["operatingMargins"]
        if pd.isna(margin0) or country not in COUNTRIES:
            continue
        m = margin_path_for_company(paths, baseline_price, margin0, country)
        rows.append({
            "ticker": row["ticker"], "company": row["longName"], "country": country,
            "revenue_local": row["totalRevenue"], "market_cap_local": row["marketCap"],
            "baseline_operating_margin": margin0,
            "p_breach_ever_12mo": float((m < 0).any(axis=1).mean()),
            "median_min_margin": float(np.median(m.min(axis=1))),
        })

    result = pd.DataFrame(rows).sort_values("p_breach_ever_12mo", ascending=False)
    out_path = os.path.join(DATA_DIR, "..", "..", "company_survival_results.csv")
    result.to_csv(out_path, index=False)

    print(result[["ticker", "country", "baseline_operating_margin", "p_breach_ever_12mo"]]
          .to_string(index=False))
    print(f"\n{(result['p_breach_ever_12mo'] > 0.05).sum()} of {len(result)} companies "
          f"have >5% chance of breaching zero margin within 12 months")
    print(f"wrote {out_path}")
    return result


if __name__ == "__main__":
    main()
