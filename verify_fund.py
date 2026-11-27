#!/usr/bin/env python3
"""
Independent check of ONE scheme against raw provider JSON, without using the
app's own pipeline code for the statistics (plain pandas/numpy).

Usage (needs network access to api.mfapi.in):
    python tools/verify_fund.py 148990 [--label ICICI]
    python tools/verify_fund.py --file raw.json --label ICICI        # offline, from a saved response

It prints the scheme name the provider returns, the identity check result,
the observation count, date range, missing-day count, and CAGR / annualised
volatility / max drawdown computed from the raw NAVs, then the same numbers
from the app's pipeline so the two can be compared.
"""
import argparse, json, os, sys, urllib.request
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend"))

def load(code, path):
    if path:
        return json.load(open(path))
    with urllib.request.urlopen(f"https://api.mfapi.in/mf/{code}", timeout=30) as r:
        return json.load(r)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("code", nargs="?"); ap.add_argument("--label"); ap.add_argument("--file")
    a = ap.parse_args()
    raw = load(a.code, a.file)
    meta = raw.get("meta", {})
    name = meta.get("scheme_name")
    print("provider scheme name :", name)
    print("provider category    :", meta.get("scheme_category"))
    df = pd.DataFrame(raw["data"]); df["date"] = pd.to_datetime(df["date"], dayfirst=True)
    nav = pd.to_numeric(df["nav"]).set_axis(df["date"]).sort_index()
    nav = nav[~nav.index.duplicated(keep="last")]
    r = nav.pct_change(fill_method=None).dropna()           # independent computation
    gap = nav.index.to_series().diff().dt.days.loc[r.index]
    r_ok = r[(gap <= 7) & (r.abs() <= 0.15)]
    n_bd = int(np.busday_count(nav.index[0].date(), nav.index[-1].date())) + 1
    print(f"observations         : {len(nav)}  ({nav.index[0].date()} -> {nav.index[-1].date()}); "
          f"business days in span {n_bd}, apparently missing {max(0, n_bd - len(nav))}")
    print(f"valid daily returns  : {len(r_ok)} (excluded: {len(r) - len(r_ok)})")
    last5 = r_ok.iloc[-1260:]
    print("--- last <=1260 valid returns (independent) ---")
    print(f"annualised vol       : {last5.std(ddof=1) * np.sqrt(252) * 100:.2f}%")
    print(f"CAGR (log-daily)     : {np.expm1(np.log1p(last5).mean() * 252) * 100:.2f}%")
    w = nav.iloc[-1261:]
    print(f"max drawdown (NAV)   : {(w / w.cummax() - 1).min() * 100:.2f}%")
    print(f"min / max daily ret  : {last5.min() * 100:.2f}% / {last5.max() * 100:.2f}%")
    if a.label:
        from universe import validate_scheme_identity
        ok, reasons = validate_scheme_identity(a.label, a.code or meta.get("scheme_code"), {**meta})
        print("identity check       :", "PASS" if ok else "FAIL " + "; ".join(reasons))
    from data_fetcher import returns_from_nav
    from fund_metrics import compute_fund_metrics
    rr, _ = returns_from_nav(nav)
    sub = nav.iloc[-1261:]
    m = compute_fund_metrics(a.label or "x", sub, rr.loc[sub.index[1]:], 0.065)
    print("--- app pipeline ---")
    print(f"vol {m['vol_sample']*100:.2f}%  CAGR {m['cagr_log_daily']*100:.2f}%  maxDD {m['max_drawdown']*100:.2f}%  flags {m['flags']}")

if __name__ == "__main__":
    main()
