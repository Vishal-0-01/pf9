# Methodology (v5.1)

Investable asset classes: equity (a sleeve of 12 flexi-cap mutual funds), gold and debt. Cash is not an asset class.

## What is specified and what is optimised
* **Specified by the author (not estimated):** the equity weight follows the Nifty valuation: 70% at fair valuation, within 50% (rich) to 90% (cheap):  E_target = clip(0.70 - 0.10 z, 0.50, 0.90), where z is the mean of the P/E and P/B z-scores. The hard 10% portfolio volatility ceiling and the 3% to 25% single-fund limits are also fixed design inputs.
* **Optimised:** the fund weights inside the equity sleeve and the split of the remaining 1 - E between debt and gold. There is **no minimum or maximum for debt or gold**: either can be 0%.

## Stage 1: fund evaluation
For each of the 12 funds: CAGR, volatility, Sharpe (risk-free 6.5%), Sortino, downside deviation, maximum drawdown (every observation), best and worst day, beta and alpha against the Nifty 50 TRI (overlap only), and a model estimate. Two windows: the fund's own window and the common window of all 12 funds (2023-02-23 to 2026-11-25, 921 returns). Only the common window feeds covariance and optimisation. Returns are formed between consecutive observed NAV dates; gaps over 7 calendar days and moves above 15% are excluded and counted. Nothing is filled.

## Inputs: observed, estimated, assumed
| Input | Basis | Nature |
|---|---|---|
| Fund covariance | Common-window daily returns, Ledoit-Wolf shrinkage, PSD check | Estimated from history |
| Covariances of funds with debt and gold, debt and gold variances | Actual daily returns of the G-Sec series and the gold series; estimated where all series overlap (2023-02-23 to 2024-12-31, 429 observations) | Estimated from history |
| Expected return of debt and gold | Annualised arithmetic mean of each series' full history (debt 2016 to 2024, gold 2010 to 2026) | Historical average, not a forecast |
| Expected return of each fund | Historical CAGR (capped at 30% for estimation) shrunk 25% (more for shorter histories) toward an ASSUMED 12% long-run prior | Estimated from a short bull market, partly assumed |
| z-scores | P/E and P/B of the Nifty 50 against the expanding history of the monthly readings (synopsis constants P/E 22 +/- 5, P/B 3.2 +/- 0.6 if fewer than 24 readings) | Estimated from history |
| Equity rule (70%, slope 0.10, 50% to 90%), 10% cap, 6.5% risk-free, 3%/25% fund limits | Author's specification | Assumptions |
If the G-Sec or gold dataset were absent the model falls back to flagged assumed constants (debt 7.5%, volatility 4%; gold 10%, 15%; assumed correlations) and says so in `defensive_parameter_source`.

## Optimisation
Given E_target, ONE joint SLSQP problem chooses the 12 fund weights s and the debt weight D and gold weight G with D + G = 1 - E:

    maximise    E s'mu_f + D mu_debt + G mu_gold - lambda_div HHI(s) - turnover penalty
    subject to  sqrt(p' S p) <= 10%   p = [E s, D, G], S = full covariance of (funds, debt, gold)   (hard)
                sum(s) = 1,  3% <= s_i <= 25%,  D, G >= 0,  E = E_target

HHI and turnover act on the sleeve weights only (soft regularisers). Resolution:
1. `target_met`: the optimiser meets the cap at E_target.
2. `equity_reduced_to_meet_vol_cap`: the cap cannot be met at E_target, so equity is lowered to the largest feasible weight (bisection on the minimum achievable volatility), never below 50%; flagged in the output and the UI.
3. If even 50% equity cannot meet 10%, `OptimizationInfeasible` is raised (HTTP 422 with the lowest achievable volatility). A portfolio that breaks the cap is never returned by the live optimiser.
Every result is re-checked independently from the rounded numbers (`verify.py`, full (n+2)x(n+2) covariance). E, D, G are reported to 4 decimals and sum to exactly 100%. The efficient-frontier chart shows, for equity weights from 50% to 90%, the best return with the cap not applied, so the cost of the cap is visible; points above the cap are flagged.
The volatility cap has priority over the equity rule. On the current data the cap is binding.

## Historical scenario (actual daily data)
Equity = Nifty 50 TRI, debt = the supplied G-Sec series, gold = INR spot, on common dates, no filling. For year Y the decision is taken on 31 Dec of Y-1 with information available then: z from the last monthly valuation reading against the expanding history; the debt/gold split optimised from expanding historical mean returns and the trailing 3-year 3x3 covariance (at least 252 returns), subject to the 10% cap, with the same cap handling as above. If even 50% equity cannot meet the cap in a historical year the scenario cannot "report infeasibility and stop" without losing the whole evaluation, so that year uses the minimum-volatility split at 50% equity and is flagged `cap_infeasible...` with its ex-ante volatility; `ceiling_breach_years` lists such years. Weights drift during the year and reset at the next decision.
Comparators on identical dates: the fixed 70/20/10 mix (the synopsis 70/18/10/2 with 2% cash moved to debt), for comparison only; a valuation-blind variant (equity fixed at 70%, same optimisation); and the Nifty 50 TRI. Sensitivities: synopsis z constants, a one-month valuation lag, slope 0.05 and 0.15. Conventions: wealth from 100, years = days/365.25, Sharpe = (CAGR - 6.5%)/volatility, drawdown at every observation, no transaction costs or taxes.
Window: 2018-01-02 to 2024-12-31 (G-Sec starts 2016-04-05, 252-observation requirement, Nifty TRI ends 2024-12-31). Results: `HYPOTHESES.md`.

## Observations, estimates and assumptions
* Observations: the cleaned datasets in `data/cleaned/`.
* Estimates: covariances, shrunk fund returns (the 12% prior is an assumption), historical debt and gold means, z-score statistics.
* Assumptions: the equity rule, the 10% cap, 6.5% risk-free, fund limits, shrinkage and its 12% prior, normality in the Monte Carlo.
The portfolio return and Sharpe shown by the optimiser use historical (shrunk) fund returns from a bull market; they are not forecasts.
