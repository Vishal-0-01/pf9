# Flexi-Cap Portfolio Optimizer: multi-stage framework (v5.1, Vercel-native)

Dissertation title: "Design and Development of a Multi-Stage Quantitative Portfolio Optimization Framework for
Mutual Fund Selection and Dynamic Multi-Asset Allocation Using Monte Carlo Simulation".
Research prototype by Vishal Singh. Not investment advice.

One Vercel project: static page in `public/`, Flask API as a Vercel Python Function (`app.py` -> `backend/`). No Render service, no external backend URL, no environment variables. See `DEPLOYMENT.md`.

Investable asset classes: equity (flexi-cap fund sleeve), gold and debt. No cash asset class. The universe is exactly the 12 funds in `backend/universe.py`.

## Layout
```
app.py            Vercel entrypoint (exports the Flask app)
vercel.json       function duration and bundle excludes
requirements.txt  flask, numpy, pandas, scipy (pinned)
.python-version   3.13
public/           static frontend (relative /api calls)
backend/          Python modules (optimizer, scenario, montecarlo, hypotheses, ...)
data/             raw/ (as supplied), cleaned/, provenance.json, data_quality_report.json, DATA_QUALITY_REPORT.md
tools/            developer tools (ingest_raw.py, fetch_gsec_nse.py, generate_results.py, verify_bundle.py); not part of the deployed bundle
tests/  docs/  results/
```

## Stages
| Stage | Content | Code |
|---|---|---|
| 1 Fund evaluation | 12 Direct-Growth Flexi Cap schemes; CAGR, volatility, Sharpe, Sortino, max drawdown, beta, alpha on own window and on the common window | `universe.py`, `data_fetcher.py`, `fund_metrics.py` |
| 2 Constrained optimisation | One joint SLSQP over the fund weights and the debt/gold split of the remainder; fund weights 3-25%, sum 1, hard 10% ceiling on the complete portfolio; no minimum for debt or gold; infeasibility reported (HTTP 422); independent verification in the API | `optimizer.py`, `assets.py`, `verify.py` |
| 3 Valuation-based equity weight | Nifty 50 P/E and P/B z-score sets equity: E = clip(0.70 - 0.10 z, 0.50, 0.90), i.e. 70% at fair value, 50% rich, 90% cheap; if the 10% cap cannot be met equity is lowered (not below 50%) and flagged | `optimizer.py`, `assets.py` |
| Historical scenario | Actual daily Nifty TRI, G-Sec index and gold on common dates | `scenario.py`, `perf.py`, `datasets.py` |
| Monte Carlo | Seeded, 10,000 monthly multivariate-normal paths, horizons 1/3/5/10 years | `montecarlo.py` |
| Hypotheses H1-H3 | Fixed verdict rules, block bootstrap, hold-out | `hypotheses.py` |

## What is actual and what is not
| Output | Nature |
|---|---|
| Fund metrics, covariance | Actual NAVs from the bundled, checksummed `data/cleaned/fund_nav.csv`. No live provider exists in the code |
| Historical scenario, H1, H3 | Actual daily data, 2018 to 2024 (limited by the supplied G-Sec series 2016-04-05 to 2024-12-31 and the Nifty TRI end). H1 directionally supported only (not significant, not robust), H3 supported; the 10% cap could not be met in 2021 to 2023 at the 50% equity floor (flagged); see `docs/HYPOTHESES.md` |
| H2 | Evaluated on the real fund data (NOT SUPPORTED, see `docs/HYPOTHESES.md`) |
| Debt and gold inputs to optimiser and Monte Carlo | Historical (G-Sec series as supplied; its TRI status is unconfirmed). Assumed constants are a flagged fallback only if the dataset is removed |
| Equity weight | Specified rule (70% neutral, 50-90%), not estimated |
| Fund returns used by the optimiser | Shrunk historical CAGRs from a 3.75 year bull market (not forecasts) |
| Monte Carlo | Simulation, never historical performance; conditional on a short bull-market window |

## Endpoints (all under /api)
`health` `funds` `fund-metrics` `data-quality` `data-report` `datasets` `diagnostics` `optimize?pe=&pb=` `frontier` `allocation-history` `scenario` `hypotheses` `montecarlo[?paths=&seed=&pe=&pb=]` `meta`

## Run locally
```bash
pip install -r requirements.txt
python app.py            # http://localhost:5000  (page and API on the same origin)
```
## Tests
```bash
pip install -r requirements-dev.txt
pytest tests -q
python -m venv /tmp/v && /tmp/v/bin/pip install -r requirements.txt && python tools/verify_bundle.py --python /tmp/v/bin/python
CHARTJS_PATH=/path/to/chart.umd.js python tests/e2e_frontend.py      # headless-browser check
```
Tests use synthetic fixtures to test logic, never to produce results. Documents: `DEPLOYMENT.md`, `docs/`.
