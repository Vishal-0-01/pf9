"""V5.1: equity follows the valuation rule (70% fair, 50-90%); fund weights and the debt/gold split are optimised."""
import numpy as np
import pytest

import optimizer as OPT
from assets import CovModel, assumed_cross


def _setup(n=12, seed=1, corr=0.7, vol_lo=.12, vol_hi=.2):
    rng = np.random.default_rng(seed)
    vols = rng.uniform(vol_lo, vol_hi, n)
    C = corr * np.ones((n, n)) + (1 - corr) * np.eye(n)
    cov = CovModel(np.outer(vols, vols) * C); cov.cross = assumed_cross(cov)
    raw = rng.uniform(.10, .25, n)
    return raw, cov, vols


def _opt(pe, pb, raw, cov, vols, **kw):
    return OPT.optimize_for_pe_pb(pe, pb, raw, cov, vols, raw / vols, list(range(len(raw))), **kw)


STATS = {"pe": (22.0, 5.0), "pb": (3.2, 0.6)}


def test_fair_valuation_gives_70_percent_equity():
    raw, cov, vols = _setup(corr=0.4, vol_lo=.10, vol_hi=.13)
    r = _opt(22.0, 3.2, raw, cov, vols, val_stats=STATS)
    assert r["z"] == pytest.approx(0, abs=1e-9) and r["equity_target"] == 0.70
    assert r["equity"] == pytest.approx(0.70, abs=1e-4) and r["constraint_status"] == "target_met"


def test_rich_and_cheap_hit_the_50_and_90_limits():
    raw, cov, vols = _setup(corr=0.2, vol_lo=.06, vol_hi=.09)         # low risk so the cap never interferes
    rich = _opt(45, 7.0, raw, cov, vols, val_stats=STATS); cheap = _opt(9, 1.2, raw, cov, vols, val_stats=STATS)
    assert rich["equity"] == pytest.approx(0.50, abs=1e-4) and cheap["equity"] == pytest.approx(0.90, abs=1e-4)
    assert cheap["constraint_status"] == "target_met"


def test_equity_monotone_in_valuation_and_always_in_band():
    raw, cov, vols = _setup(corr=0.3, vol_lo=.07, vol_hi=.1)
    eq = [_opt(pe, pb, raw, cov, vols, val_stats=STATS)["equity"] for pe, pb in [(12, 1.8), (18, 2.6), (22, 3.2), (28, 4.0), (40, 6.0)]]
    assert eq == sorted(eq, reverse=True) and all(0.5 - 1e-6 <= e <= 0.9 + 1e-6 for e in eq)


def test_weights_sum_to_one_and_vol_cap_enforced():
    raw, cov, vols = _setup()
    for pe, pb in [(15, 2), (22, 3.2), (30, 4.5), (40, 6)]:
        r = _opt(pe, pb, raw, cov, vols, val_stats=STATS)
        assert r["equity"] + r["debt"] + r["gold"] == pytest.approx(1, abs=1e-9)
        assert r["port_vol"] <= OPT.VOL_CAP + 1e-9 and r["constraint_ok"] and r["equity_range_ok"]
        assert OPT.check_portfolio(r["fund_weights"], r["equity"], r["debt"], r["gold"], cov)["ok"]


def test_cap_lowers_equity_below_target_and_flags_it():
    raw, cov, vols = _setup()                         # equity sleeve vol about 14%: 90% equity target cannot meet the cap
    r = _opt(12, 1.8, raw, cov, vols, val_stats=STATS)
    assert r["equity_target"] > r["equity"] >= 0.5 - 1e-9 and r["constraint_status"] == "equity_reduced_to_meet_vol_cap"
    assert r["port_vol"] <= OPT.VOL_CAP + 1e-9


def test_debt_and_gold_have_no_floor():
    """Gold far better than debt, both low risk: the optimiser puts the whole remainder in gold, debt 0 (no 3% floor)."""
    raw, cov, vols = _setup(corr=0.3, vol_lo=.06, vol_hi=.09)
    cr = {**cov.cross, "mu": np.array([0.04, 0.15])}
    c2 = CovModel(np.asarray(cov)); c2.cross = cr
    r = _opt(22.0, 3.2, raw, c2, vols, val_stats=STATS, objective="return")
    assert r["objective"] == "return"
    assert r["debt"] < 1e-3 and "debt" in r["debt_gold_at_zero"] and r["gold"] == pytest.approx(0.30, abs=2e-3)


def test_sharpe_objective_not_worse_than_return_objective_on_sharpe():
    raw, cov, vols = _setup(corr=0.3, vol_lo=.06, vol_hi=.09)
    rs = _opt(22.0, 3.2, raw, cov, vols, val_stats=STATS, objective="sharpe")
    rr = _opt(22.0, 3.2, raw, cov, vols, val_stats=STATS, objective="return")
    assert rs["objective"] == "sharpe" and rs["equity"] == pytest.approx(rr["equity"], abs=1e-9)
    assert rs["port_vol"] <= OPT.VOL_CAP + 1e-9 and rr["port_vol"] <= OPT.VOL_CAP + 1e-9
    sh = lambda r: (r["port_ret"] - OPT.RF) / r["port_vol"]
    assert sh(rs) >= sh(rr) - 1e-3


def test_infeasible_is_reported_not_returned():
    raw, cov, vols = _setup()
    big = CovModel(np.asarray(cov) * 25)
    big.cross = {**assumed_cross(cov), "cov_dd": np.eye(2) * 0.25, "cov_fd": np.full((12, 2), 0.1)}
    with pytest.raises(OPT.OptimizationInfeasible) as e:
        _opt(22, 3.2, raw, big, vols * 5, val_stats=STATS)
    assert e.value.details["min_achievable_vol_at_50pct_equity"] > OPT.VOL_CAP
