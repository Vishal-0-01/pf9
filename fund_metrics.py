"""
fund_metrics.py - fund-level statistics computed from historical NAV data.

Everything here is computed from the fund's own observed NAVs (no filling,
no adjustment).  These are the RAW historical statistics, reported next to the
shrunk / regularised values that the optimiser uses, so the two can be
reconciled.  Volatility is never floored, capped or otherwise adjusted.

Definitions (trading year = 252 days; RF is the model's constant risk-free rate)
  cagr_calendar   (NAV_end / NAV_start) ** (365.25 / calendar_days) - 1
  cagr_log_daily  exp(252 * mean(ln(1 + r))) - 1  over valid daily returns
                  (the estimator the optimiser starts from)
  vol_sample      std(r, ddof=1) * sqrt(252) over valid daily returns
  sharpe_raw      (cagr_log_daily - RF) / vol_sample
  sortino_std     (cagr_log_daily - RF) / DD, DD = sqrt(mean(min(r - RF/252, 0)^2) * 252)
                  with the mean taken over ALL valid returns (standard definition)
  max_drawdown    min over t of NAV_t / max(NAV_s, s <= t) - 1, on observed NAVs
  beta, alpha     daily regression against the market-proxy fund on commonly
                  observed dates; alpha is Jensen's alpha, annualised (x252)
"""

import numpy as np
import pandas as pd

from data_fetcher import MAX_GAP_CALENDAR_DAYS

TRADING_DAYS = 252
LOW_EQUITY_VOL_FLAG = 0.08     # below this an equity fund's vol is flagged (not altered)
HIGH_CAGR_FLAG = 0.30


def _log_cagr(r):
    r = np.asarray(r, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) < 2:
        return float("nan")
    return float(np.expm1(np.log1p(r).mean() * TRADING_DAYS))


def max_drawdown(nav):
    s = nav.dropna()
    if len(s) < 2:
        return float("nan")
    return float((s / s.cummax() - 1.0).min())


def beta_alpha(nav_fund, nav_proxy, rf):
    """
    Beta and Jensen's alpha of a fund against the proxy, from returns over
    consecutive commonly observed dates only.  Returns dict (values None if
    there are fewer than 60 overlapping returns).
    """
    common = nav_fund.dropna().index.intersection(nav_proxy.dropna().index)
    if len(common) < 61:
        return {"beta": None, "alpha_annual": None, "r_squared": None, "n_overlap": 0}
    f = nav_fund.loc[common]
    p = nav_proxy.loc[common]
    gap = common.to_series().diff().dt.days
    rf_d = f.pct_change(fill_method=None)
    rp_d = p.pct_change(fill_method=None)
    ok = (gap <= MAX_GAP_CALENDAR_DAYS) & rf_d.notna() & rp_d.notna() & (rf_d.abs() < 0.15) & (rp_d.abs() < 0.15)
    x, y = rp_d[ok].values, rf_d[ok].values
    n = len(x)
    if n < 60 or np.var(x, ddof=1) == 0:
        return {"beta": None, "alpha_annual": None, "r_squared": None, "n_overlap": int(n)}
    cov = np.cov(y, x, ddof=1)
    beta = float(cov[0, 1] / cov[1, 1])
    d = rf / TRADING_DAYS
    alpha_daily = float(np.mean(y - d) - beta * np.mean(x - d))
    r2 = float(np.corrcoef(x, y)[0, 1] ** 2)
    return {"beta": beta, "alpha_annual": alpha_daily * TRADING_DAYS, "r_squared": r2, "n_overlap": int(n)}


def compute_fund_metrics(label, nav_est, returns_est, rf, proxy_nav=None):
    """
    nav_est     : the fund's NAV Series over the estimation window (own dates)
    returns_est : the fund's valid daily returns over the same window (NaN = excluded)
    """
    r = returns_est.dropna().values
    nav = nav_est.dropna()
    days = (nav.index[-1] - nav.index[0]).days
    vol = float(np.std(r, ddof=1) * np.sqrt(TRADING_DAYS)) if len(r) > 1 else float("nan")
    cagr_log = _log_cagr(r)
    cagr_cal = float((nav.iloc[-1] / nav.iloc[0]) ** (365.25 / days) - 1.0) if days > 0 else float("nan")
    d = rf / TRADING_DAYS
    dd = float(np.sqrt(np.mean(np.minimum(r - d, 0.0) ** 2) * TRADING_DAYS)) if len(r) else float("nan")
    out = {
        "fund": label,
        "n_returns": int(len(r)),
        "window_start": str(nav.index[0].date()),
        "window_end": str(nav.index[-1].date()),
        "cagr_calendar": cagr_cal,
        "cagr_log_daily": cagr_log,
        "vol_sample": vol,
        "sharpe_raw": (cagr_log - rf) / vol if vol and vol > 0 else None,
        "sortino_std": (cagr_log - rf) / dd if dd and dd > 0 else None,
        "downside_deviation": dd,
        "max_drawdown": max_drawdown(nav),
        "min_daily_return": float(r.min()) if len(r) else None,
        "max_daily_return": float(r.max()) if len(r) else None,
        "beta": None, "alpha_annual": None, "r_squared": None, "n_overlap": 0,
        "flags": [],
    }
    if proxy_nav is not None:
        out.update(beta_alpha(nav, proxy_nav, rf))
    if vol == vol and vol < LOW_EQUITY_VOL_FLAG:
        out["flags"].append(f"sample volatility {vol*100:.1f}% is implausibly low for a Flexi Cap equity fund "
                            "(check scheme identity and NAV series); value NOT adjusted")
    if cagr_log == cagr_log and cagr_log > HIGH_CAGR_FLAG:
        out["flags"].append(f"historical CAGR {cagr_log*100:.1f}% exceeds {HIGH_CAGR_FLAG*100:.0f}%")
    if abs(cagr_log - cagr_cal) > 0.01 and cagr_log == cagr_log and cagr_cal == cagr_cal:
        out["flags"].append("log-daily CAGR and calendar CAGR differ by more than 1 percentage point "
                            "(possible missing returns); values reported as computed")
    return out
