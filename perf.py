"""
perf.py - performance statistics from a series of periodic returns.

One convention for every portfolio and benchmark (so comparisons are like-for-like):
  observations      returns between consecutive observed dates (gaps are real multi-day returns, never filled)
  wealth            starts at 100 on the base date; W_t = 100 * prod(1 + r)
  years             calendar days between first and last wealth point / 365.25
  periods_per_year  n_returns / years  (empirical; avoids assuming 252)
  CAGR              (W_end / 100) ** (1 / years) - 1
  volatility        std(r, ddof=1) * sqrt(periods_per_year)
  Sharpe            (CAGR - RF) / volatility          (same definition as the fund-level Sharpe)
  Sortino           (CAGR - RF) / DD, DD = sqrt(mean(min(r - RF/ppy, 0)^2) * ppy)
  drawdown_t        W_t / max(W_s, s <= t) - 1, at EVERY observation (base point included)
  max drawdown      min drawdown_t
"""
import numpy as np
import pandas as pd

RF_DEFAULT = 0.065


def wealth_index(returns, base_date, start=100.0):
    """Wealth Series (index = [base_date] + returns.index) from a returns Series."""
    r = pd.Series(returns).astype(float)
    w = start * np.cumprod(1.0 + r.values)
    return pd.Series(np.concatenate([[start], w]), index=pd.DatetimeIndex([base_date]).append(r.index))


def drawdown_series(wealth):
    return wealth / wealth.cummax() - 1.0


def years_between(a, b):
    return (pd.Timestamp(b) - pd.Timestamp(a)).days / 365.25


def cagr_from_returns(r, years):
    return float(np.expm1(np.log1p(np.asarray(r, dtype=float)).sum() / years))


def sharpe_from_returns(r, years, rf=RF_DEFAULT):
    r = np.asarray(r, dtype=float)
    ppy = len(r) / years
    vol = float(np.std(r, ddof=1) * np.sqrt(ppy))
    return (cagr_from_returns(r, years) - rf) / vol if vol > 0 else float("nan")


def max_drawdown_detail(wealth):
    dd = drawdown_series(wealth)
    trough = dd.idxmin()
    peak = wealth.loc[:trough].idxmax()
    rec = wealth.loc[trough:]
    recovered = rec[rec >= wealth.loc[peak]]
    return {"max_drawdown": float(dd.min()), "peak_date": str(peak.date()), "trough_date": str(trough.date()),
            "recovery_date": str(recovered.index[0].date()) if len(recovered) else None}


def calendar_year_returns(wealth):
    """Return per calendar year from year-end wealth points (last observation on/before 31 Dec)."""
    out = {}
    ye = wealth.groupby(wealth.index.year).last()
    first_year = wealth.index[0].year
    prev = float(wealth.iloc[0])
    for y, v in ye.items():
        if y == first_year and wealth.index[0].month == 12 and wealth.index[0].day >= 24:
            prev = float(v); continue              # base point is the previous year-end
        out[int(y)] = float(v) / prev - 1.0
        prev = float(v)
    return out


def summarize(returns, base_date, rf=RF_DEFAULT):
    r = pd.Series(returns).astype(float)
    w = wealth_index(r, base_date)
    years = years_between(w.index[0], w.index[-1])
    ppy = len(r) / years
    vol = float(np.std(r.values, ddof=1) * np.sqrt(ppy))
    cagr = cagr_from_returns(r.values, years)
    d = rf / ppy
    dd = float(np.sqrt(np.mean(np.minimum(r.values - d, 0.0) ** 2) * ppy))
    out = {"n_returns": int(len(r)), "start": str(w.index[0].date()), "end": str(w.index[-1].date()),
           "years": years, "periods_per_year": ppy, "cagr": cagr, "volatility": vol,
           "sharpe": (cagr - rf) / vol if vol > 0 else None, "sortino": (cagr - rf) / dd if dd > 0 else None,
           "total_return": float(w.iloc[-1] / 100.0 - 1.0), "rf": rf}
    out.update(max_drawdown_detail(w))
    out["calendar_year_returns"] = calendar_year_returns(w)
    return out, w
