# Hypotheses and results

Verdicts are computed by `hypotheses.py` from the data; the rules were fixed before results. Nothing is declared supported without a calculation.

## Status
| Hypothesis | Test | Result |
|---|---|---|
| H1: dynamic allocation Sharpe > fixed 70/20/10 Sharpe | Historical scenario 2017-12-29 to 2024-12-31, paired moving-block bootstrap (block 21, 2,000 resamples, seed 20260825) | **DIRECTIONALLY SUPPORTED (not statistically significant)**: Sharpe 0.540 vs 0.515 (difference +0.025, 95% interval [-0.20, 0.23]) |
| H2: optimised sleeve volatility < equal-weight sleeve volatility, comparable return | Ex-ante, realised in-sample, hold-out on the 12 funds | **SUPPORTED in-sample, not confirmed out of sample** (below) |
| H3: dynamic max drawdown < Nifty 50 TRI max drawdown | Same scenario, bootstrap interval on the difference | **SUPPORTED**: -21.9% vs -38.3%; 95% bootstrap interval of the difference [0.054, 0.264] excludes zero |

## Evaluation window and data (actual, overlapping)
Equity = Nifty 50 TRI (ends 2024-12-31), debt = the Investing.com "Nifty GS Composite" series supplied by the author (2016-04-05 to 2024-12-31, 2,158 observations, one Saturday row excluded), gold = INR spot. Common period 2016-04-05 to 2024-12-31; each decision needs 252 returns of history, so the first decision is 31 Dec 2017 and the evaluation runs 2018-01-02 to 2024-12-31 (1626 daily returns, 7 calendar years). Nothing was filled or extended. **Caveats:** the debt series is not confirmed as the NSE Total Return Index (see DATA_PROVENANCE.md) and contains a few isolated one-day dips that reverse (for example -3.3% on 2022-09-12) that may be bad prints; they were kept as supplied.

## H1 and H3: scenario results
Strategy: equity from the valuation rule (70% at fair value, 50% to 90%), debt/gold split optimised, 10% cap.

| Portfolio | CAGR | Volatility | Sharpe (rf 6.5%) | Max drawdown |
|---|---|---|---|---|
| Dynamic (valuation rule + optimised rest) | 12.1% | 10.3% | 0.540 | -21.9% |
| Fixed 70/20/10 (comparator only) | 12.6% | 11.9% | 0.515 | -26.0% |
| Valuation-blind (equity fixed at 70%, rest optimised) | 11.6% | 11.2% | 0.453 | -25.8% |
| Nifty 50 TRI | 13.6% | 17.5% | 0.405 | -38.3% |

Yearly decisions (inputs available on 31 Dec of the previous year):

| Year | P/E / P/B | z | Rule target equity | Equity | Debt | Gold | Ex-ante vol | Cap status |
|---|---|---|---|---|---|---|---|---|
| 2018 | 26.92 / 3.55 | +1.10 | 59.0% | 59.0% | 41.0% | 0.0% | 6.68% | target met |
| 2019 | 26.17 / 3.38 | +0.77 | 62.3% | 62.3% | 37.7% | 0.0% | 7.50% | target met |
| 2020 | 28.30 / 3.75 | +1.14 | 58.6% | 58.6% | 0.0% | 41.4% | 7.86% | target met |
| 2021 | 38.45 / 3.96 | +2.25 | 50.0% | 50.0% | 41.6% | 8.4% | 10.71% | cap infeasible min variance at 50pct equity |
| 2022 | 24.11 / 4.37 | +0.86 | 61.4% | 50.0% | 45.5% | 4.5% | 10.94% | cap infeasible min variance at 50pct equity |
| 2023 | 21.79 / 4.25 | +0.54 | 64.6% | 50.0% | 43.6% | 6.4% | 11.58% | cap infeasible min variance at 50pct equity |
| 2024 | 23.17 / 3.81 | +0.39 | 66.1% | 66.1% | 15.3% | 18.5% | 9.99% | target met |

**The 10% cap could not be met in 2021, 2022 and 2023.** In those years even 50% equity (the rule's floor) gives an ex-ante volatility of 10.7% to 11.6%, because the trailing three-year window includes the 2020 crash. The scenario then uses the minimum-volatility split at 50% equity and flags the year as a cap breach; it does not hide it. The realised volatility of the dynamic portfolio over the window is 10.31%, slightly above 10%. The live recommendation meets the cap.

Observations. (1) H1: the dynamic portfolio's Sharpe is slightly higher than the fixed mix (+0.025) but the bootstrap interval is wide and includes zero, so this is directional only; it is also higher than the valuation-blind variant (+0.087). (2) H3: the drawdown reduction is large and the interval excludes zero, but it comes from one seven-year window with one crash, and the lower equity weights (50% to 66%) plus gold and debt explain much of it. (3) Sensitivities, reported whatever they show:

| Variant | Sharpe dynamic | Sharpe fixed | Difference | Max DD dynamic | Max DD Nifty TRI |
|---|---|---|---|---|---|
| primary (slope 0.10, expanding z, lag 0) | 0.540 | 0.515 | +0.025 | -21.9% | -38.3% |
| synopsis constants for z (P/E 22 +/- 5, P/B 3.2 +/- 0.6) | 0.555 | 0.515 | +0.041 | -22.0% | -38.3% |
| valuation reading lagged one month | 0.508 | 0.515 | -0.007 | -22.0% | -38.3% |
| slope 0.05 | 0.482 | 0.515 | -0.032 | -23.7% | -38.3% |
| slope 0.15 | 0.584 | 0.515 | +0.069 | -20.1% | -38.3% |

The sign of the H1 difference depends on the specification (from -0.032 to +0.069), so H1 is not robust. H3 holds in every variant.

## H2 in detail (real fund data, common window 2023-02-23 to 2026-11-25)
Expected returns are the shrunk historical fund returns. Comparable return: absolute difference of at most 1.0 percentage point (assumption).

| Measure | Optimised | Equal weight |
|---|---|---|
| Ex-ante volatility (in-sample) | 12.74% | 13.06% |
| Ex-ante expected return | 18.85% | 18.24% |
| Realised volatility (in-sample, 921 obs) | 12.66% | 12.98% |
| Hold-out volatility (fit 2023-02-23 to 2025-01-10, test 2025-01-13 to 2026-11-25, 461 obs) | 13.87% | 13.73% |

The verdict rests on the ex-ante in-sample comparison (SUPPORTED, by 0.3 pp), and the realised in-sample figure agrees. **In the hold-out the optimised sleeve is NOT lower** (13.87% vs 13.73%), so the result does not hold out of sample. The optimiser maximises return, not minimises volatility, and with an average pairwise correlation near 0.90 there is little diversification to exploit. Supplementary (not the formal test): a minimum-variance sleeve at the same expected return has volatility 11.70% in-sample and 11.93% in the hold-out (equal weight 13.06% and 13.73%), a larger and out-of-sample-robust reduction that the return-maximising optimiser does not select.

## Reporting rules
* SUPPORTED needs the stated test to pass; a positive difference with a bootstrap interval containing zero is "directionally supported (not statistically significant)".
* Sensitivities are computed by the same pipeline and reported whatever they show.
* On synthetic test fixtures the pipeline returns both positive and negative verdicts; fixtures are never shown as results.
