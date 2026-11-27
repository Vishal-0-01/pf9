"""
montecarlo.py - Monte Carlo simulation of forward portfolio wealth (a SIMULATION, not historical performance).

Model (every assumption is also returned by the API):
  assets            equity sleeve (optimised fund weights), debt, gold  (cash is not an asset class)
  step              one month; returns are simple returns drawn jointly from a multivariate normal
                    N(mu_a / 12, Sigma_a / 12)  (i.i.d. across months; no fat tails, no volatility clustering)
  equity sleeve     variance w' Sigma_funds w from the fund covariance (Ledoit-Wolf shrunk, from actual NAVs);
                    mean (mode "base") = the optimiser's shrunk historical sleeve CAGR + sigma^2/2 (arithmetic).
                    Mode "historical_unshrunk" uses the raw historical fund CAGR (+ sigma^2/2) instead. Both come from
                    a short (about 3.8 year) bull-market window.
  covariance        3x3 covariance of (sleeve, debt, gold) = assets.asset_cov3: fund covariance from actual NAVs
                    (Ledoit-Wolf shrunk), debt/gold variances and covariances with the funds from actual daily returns
                    on common dates. An asset without a dataset uses ASSUMED constants and is labelled so
                    (parameter_source = "partly_assumed_constants").
  defensive means   annualised arithmetic mean of the full dataset history of debt and gold (historical averages,
                    not forecasts).
  allocation        the recommended weights [E, D, G]: equity from the valuation rule (50% to 90%, asserted), debt and
                    gold from the optimiser (>= 0, no floor); sum 1 (asserted); each sleeve fund weight in [3%, 25%] (asserted);
                    rebalanced to target weights every 12 months.
  outputs           per horizon: terminal-wealth distribution (start 100), percentiles, probabilities of loss and of
                    falling short of the risk-free rate, maximum-drawdown distribution (monthly path), histogram.
  reproducibility   numpy default_rng(seed); n_paths; all returned.
"""
import numpy as np
import pandas as pd

import assets as AS
import optimizer as OPT
import perf

ASSET_NAMES = ["equity", "debt", "gold"]

DEFAULT_SEED = 20260825
DEFAULT_PATHS = 10000
DEFAULT_HORIZONS = (1, 3, 5, 10)
MAX_PATHS = 50000
PCTS = (5, 25, 50, 75, 95)
ALLOC_SUM_TOL = 2e-4      # the optimiser reports asset weights rounded to 4 decimals


def build_parameters(fund_w, cov_funds, ann_returns, raw_cagrs, mode="base"):
    """
    mu (3), S (3x3) for [equity sleeve, debt, gold].
    S = asset_cov3(fund_w, cov_funds): the SAME extended covariance the optimiser uses (fund covariance from actual NAVs,
    debt/gold blocks from actual daily returns where the datasets exist, assumed constants otherwise).
    Means: sleeve = shrunk (base) or raw (historical_unshrunk) historical fund CAGR + sigma^2/2. Debt and gold = annualised arithmetic mean of the full
    dataset history (or the assumed constant if the dataset is missing; flagged in parameter_source).
    """
    w = np.asarray(fund_w, float)
    cross = AS.cross_of(cov_funds)
    S = AS.asset_cov3(w, cov_funds)
    s_sleeve = float(np.sqrt(S[0, 0]))
    if mode == "historical_unshrunk":
        sleeve_in, eq_basis = float(w @ np.asarray(raw_cagrs, float)), "raw historical fund CAGR + sigma^2/2"
    else:
        sleeve_in = float(w @ np.asarray(ann_returns, float))      # optimiser's shrunk historical CAGR
        eq_basis = "shrunk historical fund CAGR + sigma^2/2 (shrinkage toward an assumed 12% prior is a modelling choice)"
    mu_sleeve = sleeve_in + 0.5 * s_sleeve ** 2
    cagr_sleeve = sleeve_in
    mu = np.array([mu_sleeve, cross["mu"][0], cross["mu"][1]])
    S, rep = OPT.repair_covariance(S)
    src = dict(cross["source"])
    allhist = all(v == "historical_daily_data" for v in src.values()) 
    sd = np.sqrt(np.diag(S))
    return mu, S, {"parameter_source": "historical_daily_data" if allhist else "partly_assumed",
                   "equity_mean_basis": eq_basis,
                   "parameter_components": {"equity_covariance": "historical (Ledoit-Wolf, fund NAVs)",
                                            "equity_mean": eq_basis,
                                            "debt": src.get("debt"), "gold": src.get("gold")},
                   "defensive_parameter_source": src, "mean_mode": mode, "sleeve_vol": s_sleeve,
                   "sleeve_cagr_input": cagr_sleeve,
                   "mu_annual_arithmetic": dict(zip(ASSET_NAMES, map(float, mu))),
                   "vol_annual": dict(zip(ASSET_NAMES, map(float, sd))),
                   "correlation": [[round(float(x), 4) for x in r] for r in S / np.outer(sd, sd)],
                   "psd_repair": rep, "historical_window": {"window": cross.get("window"), "n_obs": cross.get("n_obs")},
                   "note": "Means of debt and gold are full-history historical averages, not forecasts."}


def validate_allocation(weights3, fund_w=None):
    p = np.asarray(weights3, float)
    ok = (len(p) == 3 and abs(p.sum() - 1) < ALLOC_SUM_TOL and (p >= -1e-12).all()
          and 0.5 - 1e-6 <= p[0] <= 0.9 + 1e-6)          # equity within the valuation rule's 50-90% band; no floor for debt/gold
    if fund_w is not None:
        f = np.asarray(fund_w, float)
        ok = ok and abs(f.sum() - 1) < 1e-5 and f.min() >= OPT.MIN_FUND_W - 1e-6 and f.max() <= OPT.MAX_FUND_W + 1e-6
    if not ok:
        raise ValueError(f"allocation violates framework constraints: {p.tolist()}")


def simulate_paths(weights3, mu, S, n_paths, years, seed, rebalance_months=12):
    """Return monthly wealth array (n_paths, 12*years + 1), start 100, and count of floored returns."""
    rng = np.random.default_rng(seed)
    m = 12 * years
    L = np.linalg.cholesky(S / 12.0)
    z = rng.standard_normal((n_paths, m, 3))
    r = mu / 12.0 + z @ L.T                      # (n, m, 3) simple monthly returns
    floored = int((r <= -1.0).sum())
    r = np.maximum(r, -0.999999)
    p = np.asarray(weights3, float)
    vals = np.tile(100.0 * p, (n_paths, 1))
    W = np.empty((n_paths, m + 1)); W[:, 0] = 100.0
    for t in range(m):
        vals = vals * (1.0 + r[:, t, :])
        tot = vals.sum(axis=1)
        W[:, t + 1] = tot
        if (t + 1) % rebalance_months == 0:
            vals = tot[:, None] * p[None, :]
    return W, floored


def summarize_paths(W, years, rf=perf.RF_DEFAULT, bins=40):
    term = W[:, -1]
    peak = np.maximum.accumulate(W, axis=1)
    mdd = (W / peak - 1.0).min(axis=1)
    cagr = (term / 100.0) ** (1.0 / years) - 1.0
    pct = lambda x: {f"p{q}": float(np.percentile(x, q)) for q in PCTS}
    hist, edges = np.histogram(term, bins=bins)
    n = len(term)
    return {"years": years, "n_paths": int(n),
            "terminal_wealth": {**pct(term), "mean": float(term.mean()), "std": float(term.std(ddof=1)),
                                "std_error_of_mean": float(term.std(ddof=1) / np.sqrt(n))},
            "cagr": pct(cagr),
            "prob_loss": float((term < 100.0).mean()),
            "prob_below_risk_free": float((term < 100.0 * (1 + rf) ** years).mean()),
            "max_drawdown": {**pct(mdd), "mean": float(mdd.mean()),
                             "prob_exceeds_10pct": float((mdd <= -0.10).mean()),
                             "prob_exceeds_20pct": float((mdd <= -0.20).mean()),
                             "prob_exceeds_30pct": float((mdd <= -0.30).mean())},
            "histogram": {"counts": hist.tolist(), "edges": [round(float(e), 3) for e in edges],
                          "density": (hist / (n * np.diff(edges))).tolist()}}


def run_monte_carlo(fund_w, weights3, cov_funds, ann_returns, raw_cagrs, n_paths=DEFAULT_PATHS,
                    horizons=DEFAULT_HORIZONS, seed=DEFAULT_SEED, mode="base", label="proposed"):
    n_paths = int(min(max(n_paths, 1000), MAX_PATHS))
    validate_allocation(weights3, fund_w)
    weights3 = np.asarray(weights3, float) / np.sum(weights3)      # remove rounding residue; sums to exactly 1
    mu, S, info = build_parameters(fund_w, cov_funds, ann_returns, raw_cagrs, mode)
    res, floored_total = {}, 0
    for h in horizons:
        W, fl = simulate_paths(weights3, mu, S, n_paths, h, seed + h)
        floored_total += fl
        res[str(h)] = summarize_paths(W, h)
    return {"type": "monte_carlo_simulation", "NOT_HISTORICAL_PERFORMANCE": True, "label": label,
            "n_paths": n_paths, "seed": seed, "seed_note": "horizon h uses seed + h", "horizons_years": list(horizons),
            "monthly_steps": True, "rebalance": "to target weights every 12 months", "start_value": 100.0,
            "allocation": dict(zip(ASSET_NAMES, map(float, weights3))),
            "parameters": info, "returns_floored_at_minus_100pct": floored_total, "results": res,
            "assumptions": __doc__.strip()}
