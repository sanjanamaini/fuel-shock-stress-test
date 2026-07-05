"""Mean-reverting (Ornstein-Uhlenbeck) Monte Carlo on Brent crude, run through the
same shock-to-margin logic as the Excel ShockEngine sheet, to answer: what is the
probability that a representative operator in each country breaches zero EBIT
margin over a 12-month horizon.

OU calibration (see data/public/ou_calibration.json, built by calibrate this
module's calibrate_ou()) uses an AR(1) fit on log Brent since 2010: oil mean-reverts,
so a plain GBM would overstate long-horizon drift and understate how often prices
snap back.

Run:  python src/monte_carlo.py
"""
import json
import os

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "public")

# Country baseline economics, mirrored from excel/build_model.py Inputs sheet (v0,
# except US crude-to-pump beta which is now empirically calibrated -- see README).
COUNTRIES = {
    #            beta   fuelShareRev  baselineMargin
    "US":      dict(beta=0.36, fuel_share_rev=0.242, margin0=0.090),
    "India":   dict(beta=0.50, fuel_share_rev=0.466, margin0=0.086),
    "China":   dict(beta=0.60, fuel_share_rev=0.389, margin0=0.070),
    "Germany": dict(beta=0.45, fuel_share_rev=0.324, margin0=0.076),
    "Brazil":  dict(beta=0.65, fuel_share_rev=0.463, margin0=0.060),
}

PASS_THROUGH = 0.50   # steady-state fuel surcharge recovery
LAG_MONTHS = 2         # months before surcharges kick in
HORIZON_MONTHS = 12
N_PATHS = 10_000
SEED = 42


def calibrate_ou(brent: pd.Series, start="2010-01-01") -> dict:
    recent = brent[brent.index >= start]
    x = np.log(recent).values
    xt, xt1 = x[1:], x[:-1]
    phi, c = np.polyfit(xt1, xt, 1)
    resid = xt - (c + phi * xt1)
    sigma_daily = resid.std()
    dt = 1 / 252
    theta = -np.log(phi) / dt
    mu = c / (1 - phi)
    sigma = sigma_daily * np.sqrt(2 * theta / (1 - phi ** 2))
    return {
        "theta": theta, "mu_log": mu, "mu_level": float(np.exp(mu)),
        "sigma_annual": sigma, "last_price": float(recent.iloc[-1]),
        "last_date": str(recent.index[-1].date()),
    }


def simulate_ou_paths(params: dict, months: int, n_paths: int, seed: int) -> np.ndarray:
    """Returns array shape (n_paths, months+1) of simulated crude prices, monthly steps."""
    rng = np.random.default_rng(seed)
    dt = 1 / 12
    theta, mu, sigma = params["theta"], params["mu_log"], params["sigma_annual"]
    x0 = np.log(params["last_price"])
    paths = np.zeros((n_paths, months + 1))
    paths[:, 0] = x0
    for t in range(1, months + 1):
        x_prev = paths[:, t - 1]
        drift = x_prev + theta * (mu - x_prev) * dt
        shock = sigma * np.sqrt(dt) * rng.standard_normal(n_paths)
        paths[:, t] = drift + shock
    return np.exp(paths)


def margin_path(price_path: np.ndarray, baseline_price: float, country: dict) -> np.ndarray:
    """Shock = % move in crude vs path start. Effective pass-through ramps in after
    LAG_MONTHS and reaches PASS_THROUGH by month 6 (linear ramp), matching the
    Excel engine's (duration-lag)/duration logic, evaluated path-dependently here."""
    pct_move = (price_path - baseline_price) / baseline_price  # can be negative (relief) too
    months = np.arange(price_path.shape[-1])
    ramp = np.clip((months - LAG_MONTHS) / max(1, (6 - LAG_MONTHS)), 0, 1) * PASS_THROUGH
    hit = country["fuel_share_rev"] * pct_move * country["beta"] * (1 - ramp)
    return country["margin0"] - hit


def main():
    brent = pd.read_csv(os.path.join(DATA_DIR, "brent_usd_bbl_daily.csv"), parse_dates=["date"])
    brent = brent.set_index("date")["value"]
    params = calibrate_ou(brent)
    with open(os.path.join(DATA_DIR, "ou_calibration.json"), "w") as f:
        json.dump(params, f, indent=2)

    price_paths = simulate_ou_paths(params, HORIZON_MONTHS, N_PATHS, SEED)
    baseline = params["last_price"]

    results = []
    for name, c in COUNTRIES.items():
        m = margin_path(price_paths, baseline, c)  # (n_paths, months+1)
        ever_negative = (m < 0).any(axis=1)
        end_negative = m[:, -1] < 0
        min_margin = m.min(axis=1)
        results.append({
            "country": name,
            "baseline_margin": c["margin0"],
            "p_breach_ever_12mo": float(ever_negative.mean()),
            "p_breach_at_12mo": float(end_negative.mean()),
            "median_min_margin": float(np.median(min_margin)),
            "p5_min_margin": float(np.percentile(min_margin, 5)),
        })

    df = pd.DataFrame(results).sort_values("p_breach_ever_12mo", ascending=False)
    df.to_csv(os.path.join(DATA_DIR, "..", "..", "montecarlo_results.csv"), index=False)
    print(f"OU calibration: half-life {np.log(2)/params['theta']:.2f} yrs, "
          f"long-run mean ${params['mu_level']:.1f}/bbl, start ${baseline:.1f}/bbl\n")
    print(df.to_string(index=False))
    return df, price_paths, params


if __name__ == "__main__":
    main()
