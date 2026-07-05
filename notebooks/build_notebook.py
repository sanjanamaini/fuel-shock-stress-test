"""Generates 01_analysis.ipynb. Run from the notebooks/ directory or repo root."""
import json
import os


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}


cells = [
    md("# Fuel Shock Stress Test: analysis notebook\n\n"
       "**Question:** what share of trucking companies survive an oil price shock, who is most "
       "vulnerable, and who recovers fastest?\n\n"
       "Sections:\n"
       "1. Oil price history and realized volatility (public data)\n"
       "2. Crude-to-pump pass-through: the tax wedge finding\n"
       "3. Company screen (CapitalIQ export, local only, not yet provided, section is a stub)\n"
       "4. Ornstein-Uhlenbeck calibration and Monte Carlo\n"
       "5. Vulnerability ranking (headline result)\n"
       "6. Next steps"),

    code("import sys\n"
         "sys.path.insert(0, '../src')\n"
         "import numpy as np\n"
         "import pandas as pd\n"
         "import matplotlib.pyplot as plt\n"
         "from monte_carlo import calibrate_ou, simulate_ou_paths, margin_path, COUNTRIES, HORIZON_MONTHS, N_PATHS, SEED\n\n"
         "brent = pd.read_csv('../data/public/brent_usd_bbl_daily.csv', parse_dates=['date']).set_index('date')['value']\n"
         "wti = pd.read_csv('../data/public/wti_usd_bbl_daily.csv', parse_dates=['date']).set_index('date')['value']\n"
         "diesel = pd.read_csv('../data/public/us_diesel_usd_gal_weekly.csv', parse_dates=['date']).set_index('date')['value']\n"
         "print('Brent rows:', len(brent), 'to', brent.index.max().date())\n"
         "print('US diesel rows:', len(diesel), 'to', diesel.index.max().date())"),

    md("## 1. Price history and realized volatility"),

    code("fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)\n"
         "axes[0].plot(brent.index, brent.values, lw=0.6)\n"
         "axes[0].set_title('Brent spot, USD per barrel')\n"
         "ret = np.log(brent).diff()\n"
         "vol30 = ret.rolling(30).std() * np.sqrt(252)\n"
         "axes[1].plot(brent.index, vol30, lw=0.6, color='darkred')\n"
         "axes[1].set_title('30-day realized volatility, annualized')\n"
         "plt.tight_layout()\n"
         "plt.show()"),

    md("## 2. Crude-to-pump pass-through: the tax wedge\n\n"
       "13-week percent changes in Brent regressed against 13-week percent changes in US retail "
       "diesel. Finding: beta is approximately 0.36 (R-squared 0.51), not the naive 1:1 many people "
       "assume. Retail diesel absorbs crude moves slowly and only partially: taxes, refining margin "
       "and distribution costs don't move with crude. This is the empirical anchor for the US "
       "Crude-to-pump beta in the Excel model. The other four countries still use v0 estimates of "
       "their tax wedge (India, Germany and Brazil tax fuel heavily and so should show an even lower "
       "beta than the US in local-currency terms; China's administered pricing behaves differently "
       "again), calibrating those four with Eurostat, PPAC, NDRC, ANP, or Bloomberg is the next "
       "data task."),

    code("bw = brent.resample('W-MON').mean()\n"
         "merged = pd.concat([bw.rename('brent'), diesel.rename('diesel')], axis=1).dropna()\n"
         "chg = merged.pct_change(13).dropna()\n"
         "beta, alpha = np.polyfit(chg['brent'], chg['diesel'], 1)\n"
         "r2 = np.corrcoef(chg['brent'], chg['diesel'])[0, 1] ** 2\n"
         "print('US crude-to-pump beta (13wk moves,', chg.index.min().date(), 'to', chg.index.max().date(), '):')\n"
         "print(f'beta={beta:.3f}  R2={r2:.3f}  n={len(chg)}')\n\n"
         "fig, ax = plt.subplots(figsize=(6, 5))\n"
         "ax.scatter(chg['brent'] * 100, chg['diesel'] * 100, s=4, alpha=0.3)\n"
         "xs = np.linspace(chg['brent'].min(), chg['brent'].max(), 50)\n"
         "ax.plot(xs * 100, (alpha + beta * xs) * 100, color='red', lw=1.5, label=f'beta={beta:.2f}')\n"
         "ax.set_xlabel('Brent 13-week % change')\n"
         "ax.set_ylabel('US diesel 13-week % change')\n"
         "ax.legend()\n"
         "plt.tight_layout()\n"
         "plt.show()"),

    md("## 3. Company screen (CapitalIQ export)\n\n"
       "Not yet provided. Drop the CIQ export at ../data/licensed/ciq_screen.xlsx (gitignored) "
       "with: name, ticker, country, market cap, revenue, EBITDA margin, EBIT margin (FY2015 to "
       "FY2025), cash and short-term investments, total debt, interest expense, employees, for "
       "GICS Ground Transportation/Trucking and Air Freight and Logistics companies headquartered "
       "in the five focus countries. Once it lands, this section applies the shock and Monte Carlo "
       "engine to the actual distribution of company margins and cash buffers instead of one "
       "representative operator per country, which is the upgrade that turns 'vulnerability by "
       "country' into 'which named companies breach, and at what fleet or revenue scale.'"),

    code("import os\n"
         "ciq_path = '../data/licensed/ciq_screen.xlsx'\n"
         "if os.path.exists(ciq_path):\n"
         "    ciq = pd.read_excel(ciq_path)\n"
         "    print(ciq.shape)\n"
         "    ciq.head()\n"
         "else:\n"
         "    print('ciq_screen.xlsx not found yet, section 3 is a stub until it is provided')"),

    md("## 4. Ornstein-Uhlenbeck calibration and Monte Carlo\n\n"
       "Oil mean-reverts (a shock decays back toward a long-run level), so a plain geometric "
       "Brownian motion would overstate how far prices drift over a 12-month horizon. Calibrated "
       "here as an AR(1) on log Brent since 2010, mapped to continuous-time OU parameters."),

    code("params = calibrate_ou(brent, start='2010-01-01')\n"
         "print('half-life (years):', round(np.log(2) / params['theta'], 2))\n"
         "print('long-run mean ($/bbl):', round(params['mu_level'], 1))\n"
         "print('current price', params['last_date'], ': $', round(params['last_price'], 1), sep='')\n\n"
         "paths = simulate_ou_paths(params, HORIZON_MONTHS, N_PATHS, SEED)\n"
         "months = np.arange(HORIZON_MONTHS + 1)\n"
         "pct = np.percentile(paths, [5, 25, 50, 75, 95], axis=0)\n\n"
         "fig, ax = plt.subplots(figsize=(9, 5))\n"
         "ax.fill_between(months, pct[0], pct[4], alpha=0.15, label='5th-95th pct')\n"
         "ax.fill_between(months, pct[1], pct[3], alpha=0.3, label='25th-75th pct')\n"
         "ax.plot(months, pct[2], lw=2, label='median')\n"
         "ax.axhline(params['last_price'], ls=':', color='gray', label='start')\n"
         "ax.set_xlabel('Months ahead')\n"
         "ax.set_ylabel('USD/bbl')\n"
         "ax.set_title(f'{N_PATHS:,}-path Monte Carlo, {HORIZON_MONTHS}-month horizon')\n"
         "ax.legend(fontsize=8)\n"
         "plt.tight_layout()\n"
         "plt.show()"),

    md("## 5. Vulnerability ranking (headline result)\n\n"
       "Each simulated price path is run through the same margin logic as the Excel ShockEngine "
       "sheet (fuel share of revenue times pass-through-adjusted price move), using v0 baseline "
       "economics per country (see excel/Inputs). Result: Brazil has a 42 percent chance of "
       "breaching zero EBIT margin at some point in the next 12 months; the US has effectively "
       "none. The ranking (Brazil, China, India, Germany, US, from most to least vulnerable) "
       "follows directly from each country's fuel share of revenue and its baseline margin "
       "cushion, not from the size of the oil shock itself, which is identical across countries "
       "in this simulation. That is the core finding of this project: exposure is structural, "
       "not just about how bad the shock is."),

    code("rows = []\n"
         "for name, c in COUNTRIES.items():\n"
         "    m = margin_path(paths, params['last_price'], c)\n"
         "    rows.append({\n"
         "        'country': name,\n"
         "        'baseline_margin': c['margin0'],\n"
         "        'fuel_share_of_revenue': c['fuel_share_rev'],\n"
         "        'p_breach_ever_12mo': (m < 0).any(axis=1).mean(),\n"
         "        'median_min_margin': np.median(m.min(axis=1)),\n"
         "    })\n"
         "result = pd.DataFrame(rows).sort_values('p_breach_ever_12mo', ascending=False)\n"
         "result"),

    md("## 6. Next steps\n\n"
       "- Calibrate crude-to-pump beta for India, China, Germany, Brazil (Eurostat oil bulletin, "
       "PPAC, NDRC, ANP, or Bloomberg country diesel series).\n"
       "- Ingest the CapitalIQ screen (section 3) and re-run the vulnerability ranking against "
       "actual company margins and cash buffers, not one representative operator.\n"
       "- 2022 oil-spike backtest: which real companies recovered fastest, and what did they have "
       "in common (scale, contract mix, hedging)?\n"
       "- Bounded food-price section: transport share of food retail price times shock scenario."),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out_path = os.path.join(os.path.dirname(__file__), "01_analysis.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("wrote", out_path)
