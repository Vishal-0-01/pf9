# Monte Carlo simulation

A genuine seeded simulation of the recommended allocation. It is NOT historical performance and NOT a forecast.

* Assets: equity sleeve, debt, gold (no cash). Monthly steps, multivariate normal returns via a Cholesky factor of the monthly covariance, weights rebalanced annually to the recommended allocation.
* The allocation simulated is what the application recommends for the latest valuation reading: equity from the valuation rule, fund weights and the debt/gold split from the optimiser. The fixed 70/20/10 mix is simulated only as a comparison.
* Seed 20260825 (horizon h uses seed + h), 10,000 paths by default (API `paths`, `seed`; capped at 50,000). Same inputs and seed give identical output.
* Horizons 1, 3, 5, 10 years: terminal-wealth percentiles, mean, standard deviation, probability of loss, probability of falling short of the risk-free rate, maximum-drawdown distribution, histogram.

## Historical inputs versus assumptions
| Input | Basis |
|---|---|
| Equity covariance (fund NAVs, Ledoit-Wolf) | Historical (921 daily returns, 2023-02-23 to 2026-11-25) |
| Equity mean (base mode) | The optimiser's shrunk historical fund CAGRs (shrinkage 25% toward an assumed 12% long-run prior is a modelling choice) + sigma^2/2 to convert to an arithmetic mean: 19.7% a year. It comes from a short bull market and is NOT a forward-looking expectation |
| Equity mean (`historical_unshrunk` mode) | Raw historical fund CAGRs + sigma^2/2 (higher still) |
| Debt: mean, volatility, correlations | Historical: actual G-Sec index returns (2016-04-05 to 2024-12-31): mean 7.19%, volatility 2.27% |
| Gold: mean, volatility, correlations | Historical (gold dataset): mean 11.62%, volatility 13.84% |
| Covariance of funds with debt and gold | Historical, estimated only where all series overlap: 2023-02-23 to 2024-12-31 (429 daily observations), because the G-Sec series ends 2024-12-31 |
| Normality, annual rebalancing, no costs | Modelling assumptions |

`parameter_source` is `historical_daily_data` when every input comes from the supplied data (as now) and `partly_assumed` when a debt/gold dataset is missing and flagged constants stand in. "Historical" here does not mean "representative": the equity mean in particular reflects 2023 to 2026.

## Current real-data run (base mode)
Allocation: equity 72.55% (rule target 72.55%), debt 0.00%, gold 27.45% (valuation P/E 21.08, P/B 3.18 on 2026-11-26).

| Horizon | P5 | P50 | P95 | P(loss) | P(max drawdown > 10%) |
|---|---|---|---|---|---|
| 1 year | 100.3 | 118.5 | 139.7 | 4.6% | 4.1% |
| 3 years | 124.3 | 165.9 | 219.7 | 0.2% | 14.6% |
| 5 years | 160.1 | 232.7 | 334.2 | 0.0% | 24.6% |
| 10 years | 324.7 | 542.4 | 904.9 | 0.0% | 42.3% |

(terminal wealth from 100; 10,000 paths.) The same allocation with raw historical fund returns:

| Horizon | P5 | P50 | P95 | P(loss) | P(max drawdown > 10%) |
|---|---|---|---|---|---|
| 1 year | 102.0 | 120.5 | 142.2 | 3.2% | 3.2% |
| 3 years | 130.5 | 174.4 | 231.0 | 0.1% | 11.1% |
| 5 years | 174.0 | 252.8 | 363.8 | 0.0% | 19.3% |
| 10 years | 382.9 | 640.9 | 1069.4 | 0.0% | 33.4% |

**How to read this.** The outcomes are strong mainly because the equity mean is a 3.75-year bull-market estimate (about 20% a year); lower the equity mean and they fall sharply. Returns are normal, so tail risk is likely understated.

## Tests
Reproducibility, seed sensitivity, single-asset mean and variance against closed form, an independent step-by-step recursion reproducing the paths to 1e-12, simulated correlations against input, hand-built paths for the summary statistics, and a check that the allocation simulated equals the optimiser's.
