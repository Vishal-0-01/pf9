"""
hypotheses.py - evaluation of the three approved hypotheses. Verdict rules are declared here, before any result.

H1 Sharpe(dynamic, optimiser-determined weights) > Sharpe(static 70/20/10, fixed weights used only as a comparator)             [historical scenario, daily data; the synopsis static
   70/18/10/2 mix with its 2% cash weight moved to debt because cash is not an asset class]
   point estimate of the difference; paired moving-block bootstrap (block 21 obs, B=BOOT_B, seed BOOT_SEED) gives a
   95% percentile CI. Verdict: SUPPORTED if diff > 0 and CI lower bound > 0; DIRECTIONALLY SUPPORTED (not
   statistically significant) if diff > 0 and CI includes 0; NOT SUPPORTED if diff <= 0.
H2 Optimised, vol-constrained equity sleeve has lower volatility than the equal-weight sleeve over the same funds,
   for a comparable expected return [fund NAVs]. Comparable = optimised expected return >= equal-weight expected
   return - COMPARABLE_RETURN_TOL (1.0 pp). Verdict: SUPPORTED if vol_opt < vol_eq and return comparable;
   LOWER VOLATILITY BUT RETURN NOT COMPARABLE if vol_opt < vol_eq and return lower by more than the tolerance;
   NOT SUPPORTED if vol_opt >= vol_eq. Evidence: (a) ex-ante (model covariance, in-sample), (b) realised on the
   common daily window (in-sample), (c) single-split hold-out: weights fitted on the first half of the common window,
   volatility measured on the second half.
H3 Max drawdown of the dynamic portfolio is lower (smaller in magnitude) than that of the Nifty 50 TRI [historical
   scenario, daily data]. Verdict: SUPPORTED if |MDD_dynamic| < |MDD_TRI|, else NOT SUPPORTED. A bootstrap interval for
   the drawdown difference is also reported (same resampling scheme).
Nothing is "assumed to hold": if the data needed are unavailable the verdict is NOT EVALUATED.
"""
import numpy as np
import pandas as pd

import optimizer as OPT
import perf

BOOT_B = 2000
BOOT_SEED = 20260825
BLOCK = 21
COMPARABLE_RETURN_TOL = 0.01



def _block_indices(n, rng):
    nb = int(np.ceil(n / BLOCK))
    starts = rng.integers(0, n - BLOCK + 1, size=nb)
    return (starts[:, None] + np.arange(BLOCK)[None, :]).ravel()[:n]


def _mdd_from_returns(r):
    w = np.cumprod(1.0 + r)
    w = np.concatenate([[1.0], w])
    return float((w / np.maximum.accumulate(w) - 1.0).min())


def bootstrap_pair(r_a, r_b, years, stat, B=BOOT_B, seed=BOOT_SEED):
    """Paired moving-block bootstrap of stat(a) - stat(b). Returns array of differences."""
    a, b = np.asarray(r_a, float), np.asarray(r_b, float)
    n = len(a)
    rng = np.random.default_rng(seed)
    out = np.empty(B)
    for i in range(B):
        ix = _block_indices(n, rng)
        out[i] = stat(a[ix], years) - stat(b[ix], years)
    return out


def _sharpe_stat(r, years):
    return perf.sharpe_from_returns(r, years, perf.RF_DEFAULT)


def _mdd_stat(r, years):
    return _mdd_from_returns(r)


def evaluate_h1_h3(scenario):
    base = {"evaluated": False, "verdict": "NOT EVALUATED"}
    if not scenario.get("evaluated"):
        why = scenario.get("message", "historical scenario not available")
        return ({**base, "reason": why}, {**base, "reason": why})
    R = scenario["_returns"]
    dyn, sta, tri = R["dynamic_constrained"], R["static_70_20_10"], R["nifty_50_tri"]
    m = scenario["metrics"]
    years = m["dynamic_constrained"]["years"]
    # H1
    d1 = m["dynamic_constrained"]["sharpe"] - m["static_70_20_10"]["sharpe"]
    bs = bootstrap_pair(dyn.values, sta.values, years, _sharpe_stat)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    verdict = ("SUPPORTED" if d1 > 0 and lo > 0 else "DIRECTIONALLY SUPPORTED (not statistically significant)"
               if d1 > 0 else "NOT SUPPORTED")
    h1 = {"evaluated": True, "hypothesis": "Dynamic allocation Sharpe > static 70/20/10 Sharpe",
          "sharpe_dynamic": m["dynamic_constrained"]["sharpe"], "sharpe_static": m["static_70_20_10"]["sharpe"],
          "difference": d1, "ci95": [float(lo), float(hi)], "bootstrap": {"B": BOOT_B, "block": BLOCK, "seed": BOOT_SEED},
          "sharpe_neutral_optimised": m["neutral_optimised"]["sharpe"],
          "sharpe_nifty_tri": m["nifty_50_tri"]["sharpe"],
          "difference_vs_neutral_optimised": m["dynamic_constrained"]["sharpe"] - m["neutral_optimised"]["sharpe"],
          "sensitivity": scenario.get("sensitivity"),
          "verdict": verdict, "period": f"{m['dynamic_constrained']['start']} to {m['dynamic_constrained']['end']}"}
    # H3
    mdd_d, mdd_t = m["dynamic_constrained"]["max_drawdown"], m["nifty_50_tri"]["max_drawdown"]
    bs3 = bootstrap_pair(dyn.values, tri.values, years, _mdd_stat)       # difference of (negative) MDDs
    lo3, hi3 = np.percentile(bs3, [2.5, 97.5])
    h3 = {"evaluated": True, "hypothesis": "Dynamic portfolio max drawdown < Nifty 50 TRI max drawdown",
          "max_drawdown_dynamic": mdd_d, "max_drawdown_nifty_tri": mdd_t,
          "difference_in_magnitude": abs(mdd_t) - abs(mdd_d), "mdd_difference_ci95_dynamic_minus_tri": [float(lo3), float(hi3)],
          "dynamic_peak_trough": [m["dynamic_constrained"]["peak_date"], m["dynamic_constrained"]["trough_date"]],
          "tri_peak_trough": [m["nifty_50_tri"]["peak_date"], m["nifty_50_tri"]["trough_date"]],
          "verdict": "SUPPORTED" if abs(mdd_d) < abs(mdd_t) else "NOT SUPPORTED",
          "period": h1["period"]}
    return h1, h3


def _sleeve_vol_realised(returns_df, w):
    ok = returns_df.dropna()
    r = ok.values @ w
    return float(np.std(r, ddof=1) * np.sqrt(len(r) / perf.years_between(ok.index[0], ok.index[-1]))), len(r)


def matched_return_min_vol(mu, cov, target):
    """Supplementary: minimum-variance sleeve with expected return >= target (SLSQP, 3-25% bounds). None if infeasible."""
    from scipy.optimize import minimize
    n = len(mu)
    best = None
    for w0 in (np.ones(n) / n, OPT.project_weights(np.asarray(mu) / np.sum(mu))):
        r = minimize(lambda w: float(w @ cov @ w), w0, method="SLSQP", bounds=[(OPT.MIN_FUND_W, OPT.MAX_FUND_W)] * n,
                     constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0},
                                  {"type": "ineq", "fun": lambda w: float(w @ mu) - target}],
                     options={"ftol": 1e-14, "maxiter": 2000})
        if r.success and abs(r.x.sum() - 1) < 1e-8 and float(r.x @ mu) >= target - 1e-9:
            if best is None or r.fun < best.fun:
                best = r
    return None if best is None else OPT.project_weights(best.x)


def evaluate_h2(state, pe=None, pb=None, objective=OPT.OBJECTIVE_DEFAULT):
    """Uses the live optimiser state (actual NAV based). pe/pb default to the latest valuation reading in the state
    (else the dataset-mean P/E and P/B, flagged). Expected returns are the optimiser's shrunk historical returns."""
    if not state.get("ready") or state.get("returns_df") is None:
        return {"evaluated": False, "verdict": "NOT EVALUATED", "reason": "live fund NAV data unavailable"}
    funds, ann, cov, vols, shp = state["funds"], state["ann_returns"], state["cov_mat"], state["vols"], state["fund_sharpes"]
    semi = state["extra_stats"]["semi_vols"]
    vs = state.get("val_stats")
    lv = state.get("latest_valuation") or {}
    neutral_pe = lv.get("pe") or (vs["pe"][0] if vs else OPT.NIFTY_MEAN_PE)
    neutral_pb = lv.get("pb") or (vs["pb"][0] if vs else OPT.NIFTY_MEAN_PB)
    pe_u, pb_u = (pe if pe is not None else neutral_pe), (pb if pb is not None else neutral_pb)
    try:
        res = OPT.optimize_for_pe_pb(pe_u, pb_u, ann, cov, vols, shp, funds, semi_vols=semi, val_stats=vs, objective=objective)
    except OPT.OptimizationInfeasible as exc:
        return {"evaluated": False, "verdict": "NOT EVALUATED", "reason": f"optimisation infeasible: {exc}"}
    w_o = np.array(res["fund_weights"]); n = len(funds); w_e = np.ones(n) / n
    vol_o, vol_e = float(np.sqrt(w_o @ cov @ w_o)), float(np.sqrt(w_e @ cov @ w_e))
    ret_o, ret_e = float(w_o @ ann), float(w_e @ ann)
    comparable = abs(ret_o - ret_e) <= COMPARABLE_RETURN_TOL
    def verdict(vo, ve, ro, re_):
        if vo < ve:
            return "SUPPORTED" if abs(ro - re_) <= COMPARABLE_RETURN_TOL else "LOWER VOLATILITY BUT RETURN NOT COMPARABLE"
        return "NOT SUPPORTED"
    out = {"evaluated": True, "hypothesis": "Optimised equity sleeve volatility < equal-weight sleeve volatility, comparable expected return",
           "valuation_inputs": {"pe": pe_u, "pb": pb_u, "source": "supplied" if pe is not None else ("latest valuation reading" if lv.get("pe") else "dataset mean (no latest reading)")},
           "defensive_parameter_source": res.get("defensive_parameter_source"),
           "optimised_allocation": {k: res[k] for k in ("equity", "debt", "gold", "port_vol", "status")},
           "comparable_return_tolerance": COMPARABLE_RETURN_TOL,
           "exante_in_sample": {"vol_optimised": vol_o, "vol_equal_weight": vol_e, "return_optimised": ret_o,
                                "return_equal_weight": ret_e, "return_comparable": bool(comparable),
                                "verdict": verdict(vol_o, vol_e, ret_o, ret_e)},
           "optimised_weights": {f: float(x) for f, x in zip(funds, w_o)}}
    rdf = state["returns_df"][funds]
    try:
        vo_r, n_obs = _sleeve_vol_realised(rdf, w_o)
        ve_r, _ = _sleeve_vol_realised(rdf, w_e)
        out["realised_in_sample"] = {"vol_optimised": vo_r, "vol_equal_weight": ve_r, "n_obs": n_obs,
                                     "window": [str(rdf.dropna().index[0].date()), str(rdf.dropna().index[-1].date())],
                                     "lower": bool(vo_r < ve_r)}
    except Exception as exc:
        out["realised_in_sample"] = {"error": str(exc)}
    # hold-out
    ok = rdf.dropna()
    if len(ok) >= 2 * OPT.MIN_OVERLAP_OBS * 4:
        half = len(ok) // 2
        try:
            a, c, v, s, ex = OPT.build_cov_from_returns(ok.iloc[:half])
            if state.get("fund_levels") is not None:
                import assets as AS
                c = AS.CovModel(c, AS.build_cross(c, state["fund_levels"][funds], state.get("def_levels") or {},
                                                  ok.index[0] - pd.Timedelta(days=5), ok.index[half - 1]))
            r2 = OPT.optimize_for_pe_pb(pe_u, pb_u, a, c, v, s, funds, semi_vols=ex["semi_vols"], val_stats=vs, objective=objective)
            wh = np.array(r2["fund_weights"]); test = ok.iloc[half:]
            vh, nt = _sleeve_vol_realised(test, wh); ve2, _ = _sleeve_vol_realised(test, w_e)
            out["holdout"] = {"fit_window": [str(ok.index[0].date()), str(ok.index[half - 1].date())],
                              "test_window": [str(test.index[0].date()), str(test.index[-1].date())],
                              "n_test_obs": nt, "vol_optimised_test": vh, "vol_equal_weight_test": ve2,
                              "lower": bool(vh < ve2),
                              "expected_return_optimised_fit": float(wh @ a), "expected_return_equal_fit": float(np.ones(n) / n @ a)}
        except Exception as exc:
            out["holdout"] = {"evaluated": False, "reason": f"{type(exc).__name__}: {exc}"}
    else:
        out["holdout"] = {"evaluated": False, "reason": f"common daily window has only {len(ok)} observations"}
    # supplementary: minimum-variance sleeve at the SAME expected return as the equal-weight sleeve (matched return).
    try:
        wm = matched_return_min_vol(np.asarray(ann, float), np.asarray(cov, float), ret_e)
        if wm is not None:
            rm = {"expected_return": float(wm @ ann), "vol_matched_min_variance": float(np.sqrt(wm @ cov @ wm)),
                  "vol_equal_weight": vol_e, "lower": bool(np.sqrt(wm @ cov @ wm) < vol_e), "scope": "in-sample, ex-ante"}
            if len(ok) >= 2 * OPT.MIN_OVERLAP_OBS * 4:
                a2, c2, _v, _s, _ex = OPT.build_cov_from_returns(ok.iloc[:half])
                wh2 = matched_return_min_vol(np.asarray(a2, float), np.asarray(c2, float), float(np.ones(n) / n @ a2))
                if wh2 is not None:
                    test = ok.iloc[half:]
                    vh2, _ = _sleeve_vol_realised(test, wh2); ve3, _ = _sleeve_vol_realised(test, w_e)
                    rm["holdout"] = {"vol_matched_min_variance_test": vh2, "vol_equal_weight_test": ve3, "lower": bool(vh2 < ve3)}
            out["supplementary_matched_return_min_variance"] = rm
    except Exception as exc:
        out["supplementary_matched_return_min_variance"] = {"error": f"{type(exc).__name__}: {exc}"}
    out["verdict"] = out["exante_in_sample"]["verdict"]
    out["note"] = ("Verdict rests on the in-sample ex-ante comparison; realised in-sample and hold-out evidence are "
                   "reported alongside and may disagree. The optimiser maximises return, not minimises volatility.")
    return out
