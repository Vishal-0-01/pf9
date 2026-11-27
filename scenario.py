"""
scenario.py - historical evaluation on ACTUAL daily data (equity = Nifty 50 TRI, debt = Nifty Composite G-Sec Index,
gold = INR gold). It is a historical evaluation, not a forecast.

Asset classes: equity, debt, gold. Cash is not an asset class.
The equity leg is the Nifty 50 Total Returns Index (the funds do not all have a daily NAV history back to the start
of the window); Stages 1-2 (fund evaluation, sleeve optimisation) are evaluated on the fund NAVs elsewhere.

Evaluation period
  Determined by the data, not chosen: all three series are put on their COMMON observation dates (no filling).
  A decision for year Y is taken on 31 December of Y-1 and needs at least MIN_TRAIL_OBS (252) common daily
  returns before that date to estimate the 3x3 covariance.  Because the G-Sec series starts later than the
  other two (see data/provenance.json), the first evaluable year is the first one with enough common history.
  Every comparator is evaluated over exactly the same dates.

Allocation for year Y (only information available at the decision date d = 31 Dec Y-1 is used)
  Equity weight: E_target = clip(0.70 - 0.10 z, 0.50, 0.90), z = mean of the P/E and P/B z-scores of the last monthly
  reading on or before d, against the expanding history of the readings up to d (70% at fair valuation, 50% when rich,
  90% when cheap). This is the author's specification.
  Debt and gold: the remaining 1 - E is split by optimisation (no minimum for either): maximise the Sharpe ratio
  (default; or the expected return in the alternative objective) subject to the 3-asset portfolio volatility <= 10%
  (hard), using the trailing 3-year daily covariance of equity, debt and gold and the expanding historical mean
  returns of all three assets up to d (the equity mean enters only the Sharpe numerator).
  If E_target cannot meet the 10% ceiling with any split, equity is lowered to the largest feasible weight (not below
  50%) and the year is flagged "equity_reduced_to_meet_vol_cap". If even 50% equity cannot meet it, the year uses the
  minimum-volatility split at 50% equity and is flagged "cap_infeasible..." with its ex-ante volatility (> 10%): the
  breach is explicit in the output (ceiling_breach_years), not silent.
  Weights are held from the first observation of year Y, drift with returns, reset at the next decision.

Comparators over the same dates:
  static_70_20_10     the synopsis fixed mix 70/18/10/2 with its 2% cash added to debt (cash removed), reset yearly.
                      Fixed weights, for COMPARISON ONLY; it does not influence the recommended allocation.
  neutral_optimised   valuation-blind: equity fixed at 70% every year (cap rule as above), debt/gold split optimised.
  nifty_50_tri        100% equity, buy and hold.
No transaction costs or taxes are modelled.
"""
import numpy as np
import pandas as pd

import assets as AS
import optimizer as OPT
import perf
from datasets import SCENARIO_START, SCENARIO_END

ASSETS = ["equity", "debt", "gold"]
STATIC_WEIGHTS = np.array([0.70, 0.20, 0.10])
TRAIL_YEARS = 3
MIN_TRAIL_OBS = 252
MAX_VAL_STALENESS_DAYS = 45          # a monthly reading may be at most this old at the decision date


class ScenarioError(Exception):
    pass


def common_returns(levels):
    """levels: dict name -> Series of levels. Returns (levels_df on common dates, returns_df between those dates)."""
    df = pd.concat({k: pd.Series(v).dropna() for k, v in levels.items()}, axis=1, join="inner").sort_index()
    return df, df.pct_change(fill_method=None).iloc[1:]


def _cov_ann(r):
    n_years = perf.years_between(r.index[0], r.index[-1])
    return r.cov().values * (len(r) / n_years)


def trailing_cov(rets, d):
    tr = rets.loc[d - pd.DateOffset(years=TRAIL_YEARS) + pd.Timedelta(days=1):d]
    if len(tr) < MIN_TRAIL_OBS:
        return None, len(tr)
    return _cov_ann(tr[ASSETS]), len(tr)


def allocate_equity_rule(E_target, S3, mu_dg, vol_cap=OPT.VOL_CAP, objective=OPT.OBJECTIVE_DEFAULT, mu_eq=None, rf=perf.RF_DEFAULT):
    """
    Given the equity weight target, choose the debt/gold split of the remainder (no minimum for either) that maximises
    expected return subject to the 3-asset volatility cap. If no split meets the cap at E_target, equity is lowered
    (0.1 pp steps, not below EQUITY_MIN). Deterministic grid search. If even 50% equity cannot meet the cap, the
    minimum-volatility split at 50% is returned and flagged cap_infeasible (breach reported in the output).
    """
    S3 = np.asarray(S3, float); mu_dg = np.asarray(mu_dg, float)
    if objective == "sharpe" and mu_eq is None:
        raise ValueError("the Sharpe objective needs mu_eq (historical equity mean)")
    limit = vol_cap - OPT.VOL_SOLVER_MARGIN
    E = float(E_target)
    while E >= OPT.EQUITY_MIN - 1e-12:
        d = np.linspace(0.0, 1.0 - E, 2001)
        P = np.column_stack([np.full_like(d, E), d, (1.0 - E) - d])
        vol = np.sqrt(np.einsum("ij,jk,ik->i", P, S3, P))
        ok = vol <= limit
        if ok.any():
            ret = P[:, 1] * mu_dg[0] + P[:, 2] * mu_dg[1]
            if objective == "sharpe":                     # maximise (E mu_eq + D mu_d + G mu_g - rf) / vol
                crit = (E * float(mu_eq) + ret - rf) / np.maximum(vol, 1e-9)
            else:
                crit = ret
            i = int(np.argmax(np.where(ok, crit, -np.inf)))
            w = [float(P[i, 0]), float(P[i, 1]), float(P[i, 2])]
            w[2] = 1.0 - w[0] - w[1]
            return {"weights": w, "vol": float(vol[i]), "equity_target": float(E_target),
                    "constraint_status": "target_met" if abs(E - E_target) < 1e-9 else "equity_reduced_to_meet_vol_cap",
                    "ceiling_binding": bool(vol[i] >= vol_cap - 5 * OPT.VOL_SOLVER_MARGIN)}
        E = round(E - 0.001, 6)
    # No split at >= 50% equity meets the cap. The historical evaluation must still run, so this year uses the
    # minimum-volatility split at 50% equity and is FLAGGED (cap_infeasible); the breach is reported, never hidden.
    d = np.linspace(0.0, 1.0 - OPT.EQUITY_MIN, 2001)
    P = np.column_stack([np.full_like(d, OPT.EQUITY_MIN), d, (1.0 - OPT.EQUITY_MIN) - d])
    vol = np.sqrt(np.einsum("ij,jk,ik->i", P, S3, P)); i = int(np.argmin(vol))
    w = [float(P[i, 0]), float(P[i, 1]), float(P[i, 2])]; w[2] = 1.0 - w[0] - w[1]
    return {"weights": w, "vol": float(vol[i]), "equity_target": float(E_target),
            "constraint_status": "cap_infeasible_min_variance_at_50pct_equity", "ceiling_binding": True,
            "cap_breached": True}


def decide_allocation(Y, valuation, rets, vol_cap=OPT.VOL_CAP, lag_months=0, slope=OPT.EQUITY_SLOPE,
                      neutral=False, z_method="expanding", objective=OPT.OBJECTIVE_DEFAULT):
    d = pd.Timestamp(f"{Y-1}-12-31")
    cutoff = d - pd.DateOffset(months=lag_months)
    v = valuation.loc[:cutoff]
    if v.empty or (cutoff - v.index[-1]).days > MAX_VAL_STALENESS_DAYS:
        raise ScenarioError(f"no valuation reading within {MAX_VAL_STALENESS_DAYS} days before {cutoff.date()}")
    vdate, pe, pb = v.index[-1], float(v["pe"].iloc[-1]), float(v["pb"].iloc[-1])
    stats = AS.valuation_stats(valuation, d, method=z_method, lag_months=lag_months)
    z_pe, z_pb, z = OPT.valuation_z(pe, pb, stats)
    E_target = OPT.EQUITY_NEUTRAL if neutral else OPT.equity_from_z(z, slope)
    S3, n_tr = trailing_cov(rets, d)
    rec = {"year": Y, "decision_date": str(d.date()), "valuation_date": str(vdate.date()), "pe": pe, "pb": pb,
           "z_pe": float(z_pe), "z_pb": float(z_pb), "z": float(z), "equity_target": float(E_target),
           "trailing_obs": int(n_tr)}
    if S3 is None:
        rec["status"] = "no_covariance"
        return rec
    hist = rets.loc[:d]
    n_years = perf.years_between(hist.index[0], hist.index[-1])
    mu_hist = hist[ASSETS].mean().values * (len(hist) / n_years)        # annualised arithmetic means up to d
    a = allocate_equity_rule(E_target, S3, mu_hist[1:], vol_cap, objective, mu_eq=mu_hist[0])
    w = a["weights"]
    rec.update(status="ok", equity_final=w[0], debt_final=w[1], gold_final=w[2], weights_final=w,
               constraint_status=a["constraint_status"], mu_debt_hist=float(mu_hist[1]), mu_gold_hist=float(mu_hist[2]),
               exante_vol_final=a["vol"], ceiling_binding=a["ceiling_binding"], n_history_obs=int(len(hist)),
               z_basis=stats.get("method"), objective=objective, mu_equity_hist=float(mu_hist[0]))
    assert abs(sum(w) - 1) < 1e-9 and min(w) >= 0 and OPT.EQUITY_MIN - 1e-9 <= w[0] <= OPT.EQUITY_MAX + 1e-9
    return rec


def simulate(rets_period, weights_by_year, base_date):
    """Daily portfolio simulation with drifting weights, reset at the first observation of each year."""
    R = rets_period[ASSETS].values
    years = rets_period.index.year.values
    out = np.empty(len(R)); vals = None; cur_year = None
    for i in range(len(R)):
        if years[i] != cur_year:
            cur_year = years[i]
            tot = 1.0 if vals is None else vals.sum()
            vals = tot * np.asarray(weights_by_year[cur_year], dtype=float)
        before = vals.sum()
        vals = vals * (1.0 + R[i])
        out[i] = vals.sum() / before - 1.0
    return pd.Series(out, index=rets_period.index)


def run_scenario(datasets, rf=perf.RF_DEFAULT, start=SCENARIO_START, end=SCENARIO_END, z_method="expanding",
                 lag_months=0, slope=OPT.EQUITY_SLOPE, full=True, objective=OPT.OBJECTIVE_DEFAULT):
    missing = [k for k, v in datasets.items() if v["status"] != "ok"]
    if missing:
        return {"status": "datasets_unavailable", "evaluated": False,
                "message": "Historical evaluation NOT run: required dataset(s) missing or invalid: " + ", ".join(missing) + ".",
                "datasets": {k: {"status": v["status"], "problems": v["problems"]} for k, v in datasets.items()}}
    levels = {"equity": datasets["nifty_tri"]["df"]["value"], "debt": datasets["debt"]["df"]["value"],
              "gold": datasets["gold"]["df"]["value"]}
    lv, rets = common_returns(levels)
    s_year, e = pd.Timestamp(start).year, pd.Timestamp(end)
    valuation = datasets["nifty_valuation"]["df"]
    first = None
    for Y in range(s_year, e.year + 1):
        d = pd.Timestamp(f"{Y-1}-12-31")
        if len(rets.loc[d - pd.DateOffset(years=TRAIL_YEARS) + pd.Timedelta(days=1):d]) >= MIN_TRAIL_OBS:
            first = Y
            break
    if first is None:
        raise ScenarioError("no year has enough common history for a covariance estimate")
    d0 = pd.Timestamp(f"{first-1}-12-31")
    base_date = lv.index[lv.index <= d0][-1]
    rp = rets.loc[base_date + pd.Timedelta(days=1):e]
    if len(rp) < 500:
        raise ScenarioError(f"only {len(rp)} common daily returns in the evaluation window")
    gaps = lv.index.to_series().diff().dt.days.loc[rp.index]
    years_in = sorted(set(rp.index.year))
    dec, dec_neutral, wy_c, wy_n = [], [], {}, {}
    for Y in years_in:
        r = decide_allocation(Y, valuation, rets, z_method=z_method, lag_months=lag_months, slope=slope, objective=objective)
        if r["status"] != "ok":
            raise ScenarioError(f"year {Y}: {r['status']}")
        rn = decide_allocation(Y, valuation, rets, z_method=z_method, lag_months=lag_months, slope=slope, neutral=True, objective=objective)
        dec.append(r); dec_neutral.append(rn)
        wy_c[Y], wy_n[Y] = r["weights_final"], rn["weights_final"]
    series = {
        "dynamic_constrained": simulate(rp, wy_c, base_date),
        "neutral_optimised": simulate(rp, wy_n, base_date),
        "static_70_20_10": simulate(rp, {y: STATIC_WEIGHTS for y in years_in}, base_date),
        "nifty_50_tri": rp["equity"].copy(),
    }
    metrics, wealth = {}, {}
    for k, r in series.items():
        metrics[k], wealth[k] = perf.summarize(r, base_date, rf)
    out = {
        "status": "ok", "evaluated": True, "backtest_type": "historical_scenario_actual_daily_data",
        "parameters": {"z_method": z_method, "lag_months": lag_months, "slope": slope, "objective": objective, "rf": rf, "vol_cap": OPT.VOL_CAP,
                       "trail_years": TRAIL_YEARS, "min_trail_obs": MIN_TRAIL_OBS},
        "period": {"start": str(rp.index[0].date()), "end": str(rp.index[-1].date()), "base_date": str(base_date.date()),
                   "first_decision_year": first, "n_daily_returns": int(len(rp)), "max_gap_days": int(gaps.max()),
                   "gaps_over_7_days": int((gaps > 7).sum()),
                   "common_data_start": str(lv.index[0].date()), "common_data_end": str(lv.index[-1].date()),
                   "series_first_dates": {k: str(v.index[0].date()) for k, v in levels.items()},
                   "note": "The evaluation window is the common period of the three series, shortened further because "
                           "each decision needs >= 252 common daily returns of history for the covariance."},
        "decisions": dec, "decisions_neutral_optimised": dec_neutral, "metrics": metrics,
        "ceiling_binding_years": [r["year"] for r in dec if r["ceiling_binding"]],
        "ceiling_breach_years": [r["year"] for r in dec if r["constraint_status"].startswith("cap_infeasible")],
        "equity_reduced_years": [r["year"] for r in dec if r["constraint_status"] == "equity_reduced_to_meet_vol_cap"],
        "_returns": {k: r for k, r in series.items()}, "_base_date": base_date,
    }
    if full:
        asset_stats = {a: perf.summarize(rp[a], base_date, rf)[0] for a in ASSETS}
        dd = {k: perf.drawdown_series(w) for k, w in wealth.items()}
        tri_full = perf.summarize(lv["equity"].pct_change(fill_method=None).dropna().loc[:e], lv.index[0], rf)[0]
        out.update({
            "convention": perf.__doc__.strip(), "method": __doc__.strip(), "asset_class_stats": asset_stats,
            "context_nifty_tri_common_period": tri_full,
            "series": {"dates": [str(d.date()) for d in wealth["nifty_50_tri"].index],
                       "wealth": {k: [round(float(x), 4) for x in w.values] for k, w in wealth.items()},
                       "drawdown": {k: [round(float(x), 6) for x in d.values] for k, d in dd.items()}},
            "limitations": [
                "Equity leg = Nifty 50 TRI, not the optimised fund sleeve (funds lack ten-year daily histories).",
                "Debt = Nifty Composite G-Sec Index and gold = INR spot parity (no duties); see data provenance.",
                "The equity rule (70% neutral, 0.10 slope, 50-90% band) is the author's specification, not estimated. "
                "The debt/gold split is optimised from historical expanding means and a trailing covariance.",
                "No transaction costs or taxes. One historical path; differences are not statistically conclusive "
                "without the bootstrap interval reported with each hypothesis."]})
    return out


SENSITIVITY_VARIANTS = [
    {"name": "primary (slope 0.10, expanding z, lag 0)", "z_method": "expanding", "lag_months": 0, "slope": 0.10},
    {"name": "synopsis constants for z (P/E 22 +/- 5, P/B 3.2 +/- 0.6)", "z_method": "synopsis_constants", "lag_months": 0, "slope": 0.10},
    {"name": "valuation reading lagged one month", "z_method": "expanding", "lag_months": 1, "slope": 0.10},
    {"name": "slope 0.05", "z_method": "expanding", "lag_months": 0, "slope": 0.05},
    {"name": "slope 0.15", "z_method": "expanding", "lag_months": 0, "slope": 0.15},
]


def run_sensitivities(datasets, rf=perf.RF_DEFAULT):
    rows = []
    for v in SENSITIVITY_VARIANTS:
        try:
            r = run_scenario(datasets, rf, z_method=v["z_method"], lag_months=v["lag_months"], slope=v["slope"], full=False,
                             objective=v.get("objective", OPT.OBJECTIVE_DEFAULT))
        except Exception as exc:
            rows.append({"variant": v["name"], "error": f"{type(exc).__name__}: {exc}"}); continue
        if not r.get("evaluated"):
            rows.append({"variant": v["name"], "error": r.get("message")}); continue
        m = r["metrics"]
        rows.append({"variant": v["name"], "sharpe_dynamic": m["dynamic_constrained"]["sharpe"],
                     "sharpe_static": m["static_70_20_10"]["sharpe"],
                     "sharpe_difference": m["dynamic_constrained"]["sharpe"] - m["static_70_20_10"]["sharpe"],
                     "sharpe_neutral_optimised": m["neutral_optimised"]["sharpe"],
                     "mdd_dynamic": m["dynamic_constrained"]["max_drawdown"], "mdd_tri": m["nifty_50_tri"]["max_drawdown"],
                     "mdd_static": m["static_70_20_10"]["max_drawdown"],
                     "cagr_dynamic": m["dynamic_constrained"]["cagr"],
                     "equity_final_range": [min(d["equity_final"] for d in r["decisions"]),
                                            max(d["equity_final"] for d in r["decisions"])]})
    return rows
