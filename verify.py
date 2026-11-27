"""
verify.py - independent re-computation of a portfolio's constraints and volatility.

Uses a different formulation from optimizer.portfolio_vol_from_weights: it assembles the FULL (n+2) x (n+2) covariance
of (funds, debt, gold) from the cross blocks and evaluates p' S p with p = [E * w_1 ... E * w_n, D, G], instead of the
3 x 3 reduction (sleeve, debt, gold) used by the optimiser. It also re-checks every weight constraint from the
returned (rounded) numbers. No minimum or maximum asset-class weight exists, so none is checked.
"""
import numpy as np

import assets as AS
import optimizer as OPT


def independent_check(result, cov_funds, vol_cap=OPT.VOL_CAP, tol=1e-9):
    w = np.asarray(result["fund_weights"], float)
    E, D, G = (float(result[k]) for k in ("equity", "debt", "gold"))
    n = len(w)
    S = AS.full_covariance(cov_funds, AS.cross_of(cov_funds))
    p = np.concatenate([E * w, [D, G]])
    vol = float(np.sqrt(p @ S @ p))
    checks = {k: (v if k == "n_funds" else bool(v)) for k, v in {
        "fund_weights_sum_to_one": abs(w.sum() - 1) <= 1e-6,
        "fund_weights_within_3_25": bool(w.min() >= OPT.MIN_FUND_W - 1e-6 and w.max() <= OPT.MAX_FUND_W + 1e-6),
        "asset_weights_sum_to_one": abs(E + D + G - 1) <= 2e-4,
        "asset_weights_non_negative": min(E, D, G) >= -1e-12,
        "volatility_within_cap": vol <= vol_cap + tol,
        "n_funds": n,
    }.items()}
    return {"independent_vol": vol, "reported_vol": float(result.get("port_vol", float("nan"))),
            "vol_difference": abs(vol - float(result.get("port_vol", float("nan")))), "checks": checks,
            "all_ok": all(v for k, v in checks.items() if k != "n_funds")}
