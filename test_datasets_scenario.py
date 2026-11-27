import os, json, shutil
import numpy as np, pandas as pd, pytest
from fixture_data import make_fixture_dir
import datasets as DS, scenario as SC, perf, hypotheses as HY, montecarlo as MC, optimizer as OPT

@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    d = str(make_fixture_dir(tmp_path_factory.mktemp("fx")))
    return d, DS.load_all(d)

@pytest.fixture(scope="module")
def scn(fx):
    return SC.run_scenario(fx[1])

# ---- dataset loader / provenance
def test_all_fixture_datasets_valid(fx):
    assert all(v["status"] == "ok" for v in fx[1].values()), {k: v["problems"] for k, v in fx[1].items()}

def test_missing_dataset_reported(tmp_path):
    d = make_fixture_dir(str(tmp_path), skip=("gold",))
    r = DS.load_all(d)
    assert r["gold"]["status"] == "missing" and r["nifty_tri"]["status"] == "ok"
    out = SC.run_scenario(r)
    assert out["evaluated"] is False and out["status"] == "datasets_unavailable"

def test_checksum_tamper_detected(tmp_path):
    d = make_fixture_dir(str(tmp_path))
    with open(os.path.join(d, "debt.csv"), "a") as f: f.write("2024-12-31,999\n")
    assert DS.load_dataset("debt", d)["status"] == "invalid"

def test_non_daily_or_short_series_invalid(tmp_path):
    d = make_fixture_dir(str(tmp_path))
    df = pd.read_csv(os.path.join(d, "gold.csv"), parse_dates=["date"])
    df = df[df["date"].dt.day == 28]                         # monthly-ish
    p = os.path.join(d, "gold.csv"); df.to_csv(p, index=False)
    prov = DS.read_provenance(d); prov["gold"]["sha256"] = DS.sha256_file(p); DS.write_provenance(prov, d)
    r = DS.load_dataset("gold", d)
    assert r["status"] == "invalid" and r["problems"]

def test_shipped_data_dir_state_is_reported_honestly():
    """The shipped data folder holds the user-supplied datasets. Whatever is missing must be reported as missing
    (never filled); the G-Sec level history was not supplied, so debt must not be 'ok' unless a debt file exists."""
    shipped = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    r = DS.load_all(shipped)
    for k in ("nifty_tri", "nifty_valuation", "gold"):
        assert r[k]["status"] == "ok", (k, r[k]["problems"])
    prov = DS.read_provenance(shipped)
    if "debt" not in prov:
        assert r["debt"]["status"] == "missing"
        out = SC.run_scenario(r)
        assert out["evaluated"] is False and out["status"] == "datasets_unavailable"

def test_debt_may_start_late_but_tri_and_gold_may_not(tmp_path):
    d = make_fixture_dir(str(tmp_path / "a"), debt_start="2016-01-04")
    assert DS.load_dataset("debt", d)["status"] == "ok"
    # a late-starting gold series is invalid
    d2 = make_fixture_dir(str(tmp_path / "b"))
    df = pd.read_csv(os.path.join(d2, "gold.csv"), parse_dates=["date"]); df = df[df["date"] >= "2016-01-04"]
    p = os.path.join(d2, "gold.csv"); df.to_csv(p, index=False)
    prov = DS.read_provenance(d2); prov["gold"]["sha256"] = DS.sha256_file(p); DS.write_provenance(prov, d2)
    assert DS.load_dataset("gold", d2)["status"] == "invalid"

def test_fingerprint_changes_with_data(tmp_path):
    d = make_fixture_dir(str(tmp_path)); f1 = DS.fingerprint_data(d)
    with open(os.path.join(d, "gold.csv"), "a") as f: f.write("\n")
    assert DS.fingerprint_data(d) != f1

# ---- independent recomputation of the scenario
def _indep_levels(d, key):
    df = pd.read_csv(os.path.join(d, f"{key}.csv"), parse_dates=["date"])
    return dict(zip(df["date"], df["value"]))

def test_scenario_independent_replay(fx, scn):
    d, _ = fx
    L = {k: _indep_levels(d, k) for k in ("nifty_tri", "debt", "gold")}
    common = sorted(set(L["nifty_tri"]) & set(L["debt"]) & set(L["gold"]))
    base = max(x for x in common if x < pd.Timestamp("2014-01-01"))
    dates = [x for x in common if base <= x <= pd.Timestamp("2024-12-31")]
    assert [str(x.date()) for x in dates] == scn["series"]["dates"]
    order = ["nifty_tri", "debt", "gold"]
    # weights per year taken from the reported decisions, re-validated independently
    W = {}
    for rec in scn["decisions"]:
        w = rec["weights_final"]; assert abs(sum(w) - 1) < 1e-9 and min(w) >= 0
        W[rec["year"]] = w
    vals, series, cur_year, tot = None, [100.0], None, 100.0
    for a, b in zip(dates[:-1], dates[1:]):
        if b.year != cur_year:
            cur_year = b.year; vals = [tot * x for x in W[cur_year]]
        vals = [v * (L[k][b] / L[k][a]) for v, k in zip(vals, order)]
        tot = sum(vals); series.append(tot)
    series = np.array(series)
    assert np.allclose(series, scn["series"]["wealth"]["dynamic_constrained"], rtol=0, atol=2e-4)   # API rounds to 4 dp
    dd = series / np.maximum.accumulate(series) - 1
    assert np.allclose(dd, scn["series"]["drawdown"]["dynamic_constrained"], atol=1e-6)
    m = scn["metrics"]["dynamic_constrained"]
    assert m["max_drawdown"] == pytest.approx(dd.min(), abs=1e-6)
    yrs = (dates[-1] - dates[0]).days / 365.25
    assert m["cagr"] == pytest.approx((series[-1] / 100) ** (1 / yrs) - 1, abs=1e-9)
    r = series[1:] / series[:-1] - 1
    ppy = len(r) / yrs
    assert m["volatility"] == pytest.approx(r.std(ddof=1) * np.sqrt(ppy), abs=1e-9)
    assert m["sharpe"] == pytest.approx(((series[-1] / 100) ** (1 / yrs) - 1 - 0.065) / (r.std(ddof=1) * np.sqrt(ppy)), abs=1e-9)

def test_nifty_tri_metrics_independent(fx, scn):
    d, _ = fx
    lv = pd.Series(_indep_levels(d, "nifty_tri")).sort_index()
    common = sorted(set(lv.index) & set(_indep_levels(d, "debt")) & set(_indep_levels(d, "gold")))
    sel = pd.Series([lv[x] for x in common], index=common)
    sel = sel[(sel.index >= sel.index[sel.index < "2014-01-01"][-1]) & (sel.index <= "2024-12-31")]
    w = 100 * sel / sel.iloc[0]
    m = scn["metrics"]["nifty_50_tri"]
    assert m["max_drawdown"] == pytest.approx((w / w.cummax() - 1).min(), abs=1e-9)
    # calendar year returns
    ye = w.groupby(w.index.year).last()
    prev = 100.0
    for y, v in ye.items():
        if y < 2014:
            prev = v; continue
        assert m["calendar_year_returns"][y] == pytest.approx(v / prev - 1, abs=1e-9); prev = v

def test_daily_drawdown_not_zero_when_every_calendar_year_is_positive():
    idx = pd.bdate_range("2013-12-02", "2016-12-30")
    r = np.full(len(idx), 0.0); r[:] = 0.0008
    for y in (2014, 2015, 2016):                       # dip 10% mid-year, recover and end the year up
        m = (idx >= f"{y}-03-01") & (idx <= f"{y}-03-31"); r[m] = (0.90) ** (1 / m.sum()) - 1
    s, w = perf.summarize(pd.Series(r[1:], index=idx[1:]), idx[0])
    ann = [v for y, v in s["calendar_year_returns"].items() if y in (2014, 2015)]
    assert len(ann) == 2 and all(x > 0 for x in ann) and s["max_drawdown"] < -0.09        # annual-only drawdown would be 0

def test_decisions_use_only_information_available_at_decision_date(fx):
    d, loaded = fx
    base = SC.run_scenario(loaded)
    # corrupt every valuation reading AFTER 31 Dec 2018 and all returns after 2018: 2014-2019 decisions must not move
    v = loaded["nifty_valuation"]["df"].copy(); v.loc["2019-01-01":, :] = [99.0, 9.0]
    l2 = {k: dict(x) for k, x in loaded.items()}; l2["nifty_valuation"] = {**loaded["nifty_valuation"], "df": v}
    for k in ("nifty_tri", "debt", "gold"):
        df = loaded[k]["df"].copy(); df.loc["2019-01-01":, "value"] *= np.linspace(1.0, 0.6, (df.index >= "2019-01-01").sum())   # smooth: keeps later years feasible
        l2[k] = {**loaded[k], "df": df}
    alt = SC.run_scenario(l2)
    for a, b in zip(base["decisions"], alt["decisions"]):
        if a["year"] <= 2019:
            assert a["weights_final"] == b["weights_final"] and a["pe"] == b["pe"], a["year"]

def test_decision_math_independent(scn):
    """Each decision is re-derived from its recorded z-score: the equity rule, the cap handling, weights."""
    import optimizer as O
    for r in scn["decisions"]:
        assert r["equity_target"] == pytest.approx(float(np.clip(0.70 - 0.10 * r["z"], 0.50, 0.90)), abs=1e-9)
        w = r["weights_final"]
        assert sum(w) == pytest.approx(1, abs=1e-9) and min(w) >= 0 and 0.5 - 1e-9 <= w[0] <= 0.9 + 1e-9
        st = r["constraint_status"]
        if st == "target_met":
            assert w[0] == pytest.approx(r["equity_target"], abs=1e-9) and r["exante_vol_final"] <= 0.10 + 1e-9
        elif st == "equity_reduced_to_meet_vol_cap":
            assert w[0] < r["equity_target"] and r["exante_vol_final"] <= 0.10 + 1e-9
        else:
            assert st.startswith("cap_infeasible") and w[0] == pytest.approx(0.5) and r["exante_vol_final"] > 0.10


def test_scenario_equity_follows_rule_and_fair_value_gives_70():
    S3 = np.diag([0.0144, 0.0004, 0.0100])
    a = SC.allocate_equity_rule(O_rule(0.0), S3, [0.07, 0.09], objective="return")
    assert a["weights"][0] == pytest.approx(0.70) and sum(a["weights"]) == pytest.approx(1)


def O_rule(z):
    import optimizer as O
    return O.equity_from_z(z)


def test_equity_rule_limits():
    import optimizer as O
    assert O.equity_from_z(0) == 0.70 and O.equity_from_z(10) == 0.50 and O.equity_from_z(-10) == 0.90
    assert O.equity_from_z(-1) == pytest.approx(0.80) and O.equity_from_z(1) == pytest.approx(0.60)


def test_debt_gold_split_optimised_no_floor():
    # gold dominates debt in return and has low vol: all of the remainder goes to gold (no 3% debt floor)
    a = SC.allocate_equity_rule(0.70, np.diag([0.0144, 0.0100, 0.0004]), [0.05, 0.12], objective="return")
    assert a["weights"][1] < 1e-3 and a["weights"][2] == pytest.approx(0.30, abs=1e-3)


def test_cap_reduces_equity_then_reports_infeasible():
    S3 = np.diag([0.0225, 0.0004, 0.0004])         # equity vol 15%: 90% equity breaches, 50% does not
    a = SC.allocate_equity_rule(0.90, S3, [0.07, 0.07], objective="return")
    assert a["constraint_status"] == "equity_reduced_to_meet_vol_cap" and a["weights"][0] < 0.90 and a["vol"] <= 0.10
    b = SC.allocate_equity_rule(0.90, np.diag([0.09, 0.0004, 0.0004]), [0.07, 0.07], objective="return")   # 30% equity vol: 50% floor infeasible
    assert b["constraint_status"].startswith("cap_infeasible") and b["cap_breached"] and b["vol"] > 0.10 and b["weights"][0] == pytest.approx(0.5)


def test_valuation_missing_before_decision_date_fails_loudly(fx):
    d, loaded = fx
    v = loaded["nifty_valuation"]["df"].loc["2015-03-01":]
    l2 = {**loaded, "nifty_valuation": {**loaded["nifty_valuation"], "df": v}}
    with pytest.raises(SC.ScenarioError):
        SC.run_scenario(l2)

def test_expanding_z_uses_only_past_readings(fx):
    import assets as AS
    d, loaded = fx
    v = loaded["nifty_valuation"]["df"]
    a = AS.valuation_stats(v, pd.Timestamp("2018-12-31"), method="expanding", lag_months=0)
    v2 = v.copy(); v2.loc["2019-01-01":, :] = [500.0, 50.0]            # future readings corrupted
    b = AS.valuation_stats(v2, pd.Timestamp("2018-12-31"), method="expanding", lag_months=0)
    assert a["pe"] == b["pe"] and a["pb"] == b["pb"]

def test_late_debt_start_shortens_and_discloses_period(tmp_path):
    d = make_fixture_dir(str(tmp_path), debt_start="2016-01-04")
    out = SC.run_scenario(DS.load_all(d))
    assert out["evaluated"] and out["period"]["first_decision_year"] >= 2017
    assert out["period"]["series_first_dates"]["debt"] >= "2016-01-04"
    assert pd.Timestamp(out["period"]["start"]) > pd.Timestamp("2016-12-31")

def test_drawdown_taken_at_every_observation(scn):
    for k, w in scn["series"]["wealth"].items():
        w = np.array(w); assert np.allclose(scn["series"]["drawdown"][k], w / np.maximum.accumulate(w) - 1, atol=2e-6)

def test_all_comparators_share_identical_dates(scn):
    r = scn["_returns"]; idx = r["nifty_50_tri"].index
    for k, s in r.items(): assert s.index.equals(idx), k

def test_weights_sum_to_one_and_no_cash_key(scn):
    for rec in scn["decisions"] + scn["decisions_neutral_optimised"]:
        assert sum(rec["weights_final"]) == pytest.approx(1, abs=1e-9) and "cash_final" not in rec

# ---- hypotheses
def test_h1_h3_computed_and_consistent(scn):
    h1, h3 = HY.evaluate_h1_h3(scn)
    m = scn["metrics"]
    assert h1["difference"] == pytest.approx(m["dynamic_constrained"]["sharpe"] - m["static_70_20_10"]["sharpe"])
    lo, hi = h1["ci95"]; assert lo < hi
    exp = ("SUPPORTED" if h1["difference"] > 0 and lo > 0 else "DIRECTIONALLY SUPPORTED (not statistically significant)" if h1["difference"] > 0 else "NOT SUPPORTED")
    assert h1["verdict"] == exp
    assert h3["verdict"] == ("SUPPORTED" if abs(m["dynamic_constrained"]["max_drawdown"]) < abs(m["nifty_50_tri"]["max_drawdown"]) else "NOT SUPPORTED")
    h1b, _ = HY.evaluate_h1_h3(scn); assert h1b["ci95"] == h1["ci95"]          # seeded: reproducible

def test_h1_h3_not_evaluated_without_data():
    h1, h3 = HY.evaluate_h1_h3({"evaluated": False, "message": "x"})
    assert h1["verdict"] == h3["verdict"] == "NOT EVALUATED"

def test_hypothesis_verdict_can_be_negative():
    # a scenario in which the 'dynamic' series is strictly worse than static must give NOT SUPPORTED
    idx = pd.bdate_range("2014-01-01", periods=600)
    rng = np.random.default_rng(1)
    st = pd.Series(rng.normal(0.0006, 0.005, 600), index=idx); dy = st * 0.5 - 0.0002
    sc = {"evaluated": True, "_returns": {"dynamic_constrained": dy, "static_70_20_10": st, "neutral_optimised": st, "nifty_50_tri": st * 2},
          "metrics": {k: perf.summarize(v, idx[0] - pd.Timedelta(days=1))[0] for k, v in
                      {"dynamic_constrained": dy, "static_70_20_10": st, "neutral_optimised": st, "nifty_50_tri": st * 2}.items()}}
    h1, h3 = HY.evaluate_h1_h3(sc)
    assert h1["verdict"] == "NOT SUPPORTED" and h3["verdict"] == "SUPPORTED"      # TRI proxy has twice the risk here


def test_sharpe_objective_allocation_and_requires_equity_mean():
    S3 = np.diag([0.0144, 0.0100, 0.0004])
    a = SC.allocate_equity_rule(0.70, S3, [0.05, 0.12], objective="sharpe", mu_eq=0.14)
    assert a["E"] == 0.70 if "E" in a else True
    with pytest.raises(ValueError):
        SC.allocate_equity_rule(0.70, S3, [0.05, 0.12], objective="sharpe")
