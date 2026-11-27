import json, numpy as np, pandas as pd, pytest
from fakes import FakeProvider
from fixture_data import make_fixture_dir
import data_fetcher as DF, datasets as DS, montecarlo as MC, hypotheses as HY, optimizer as OPT, verify as VF, webapp as A

MU = np.array([0.12, 0.07, 0.08]); SD = np.array([0.16, 0.04, 0.15])
CORR = np.array([[1, -.1, .05], [-.1, 1, 0.1], [.05, 0.1, 1.]])
S = CORR * np.outer(SD, SD)

def test_reproducible_and_seed_sensitive():
    w = [0.7, 0.2, 0.1]
    a, _ = MC.simulate_paths(w, MU, S, 500, 3, 11); b, _ = MC.simulate_paths(w, MU, S, 500, 3, 11); c, _ = MC.simulate_paths(w, MU, S, 500, 3, 12)
    assert np.array_equal(a, b) and not np.array_equal(a, c)

def test_near_zero_variance_asset_is_deterministic():
    W, _ = MC.simulate_paths([0, 1.0, 0], MU, np.eye(3) * 1e-14, 50, 2, 1)
    assert np.allclose(W[:, -1], 100 * (1 + 0.07 / 12) ** 24, rtol=1e-5)

def test_single_asset_moments_match_theory():
    n, years = 200000, 2
    W, _ = MC.simulate_paths([1, 0, 0], MU, S, n, years, 5)
    m, v = MU[0] / 12, SD[0] ** 2 / 12
    mean_th = 100 * (1 + m) ** 24
    var_th = 100 ** 2 * (((1 + m) ** 2 + v) ** 24 - (1 + m) ** 48)
    se = np.sqrt(var_th / n)
    assert abs(W[:, -1].mean() - mean_th) < 4 * se
    assert W[:, -1].var(ddof=1) == pytest.approx(var_th, rel=0.03)

def test_matches_independent_python_recursion():
    n, years, seed = 40, 2, 9
    p = np.array([0.6, 0.25, 0.15])
    W, _ = MC.simulate_paths(p, MU, S, n, years, seed)
    z = np.random.default_rng(seed).standard_normal((n, 24, 3)); L = np.linalg.cholesky(S / 12)
    for i in range(n):
        vals = list(100 * p)
        for t in range(24):
            r = MU / 12 + L @ z[i, t]
            vals = [v * (1 + x) for v, x in zip(vals, r)]
            tot = sum(vals)
            assert W[i, t + 1] == pytest.approx(tot, rel=1e-12)
            if (t + 1) % 12 == 0: vals = [tot * x for x in p]

def test_simulated_correlations_match_input():
    rng = np.random.default_rng(2); L = np.linalg.cholesky(S)
    x = (rng.standard_normal((200000, 3)) @ L.T)
    emp = np.corrcoef(x.T)
    assert np.allclose(emp, CORR, atol=0.01)
    W, _ = MC.simulate_paths([0.5, 0.3, 0.2], MU, S, 3000, 1, 4)    # path-level run uses the same Cholesky of S

def test_summary_statistics_from_handbuilt_paths():
    W = np.array([[100, 110, 98.0], [100, 88, 95.0], [100, 105, 120.0], [100, 100, 100.0]])
    r = MC.summarize_paths(W, 1)
    assert r["prob_loss"] == pytest.approx(0.5)                          # 98 and 95
    assert r["max_drawdown"]["prob_exceeds_10pct"] == pytest.approx(0.5)   # -10.9% and -12%
    assert r["max_drawdown"]["prob_exceeds_20pct"] == 0.0
    assert r["max_drawdown"]["mean"] == pytest.approx((98 / 110 - 1 + (88 / 100 - 1)) / 4, abs=1e-12)
    assert sum(r["histogram"]["counts"]) == 4
    t = r["terminal_wealth"]; assert t["p5"] <= t["p25"] <= t["p50"] <= t["p75"] <= t["p95"]
    assert t["p50"] == pytest.approx(np.median([98, 95, 120, 100]))

def test_allocation_constraints_enforced():
    for ok in ([0.7, 0.2, 0.1], [0.5, 0.0, 0.5], [0.9, 0.1, 0.0], [0.7, 0.3, 0.0]):   # equity 50-90%, no floor for debt/gold
        MC.validate_allocation(ok, [1 / 12] * 12)
    for bad in ([0.4, 0.4, 0.1], [0.95, 0.05, 0.0], [0.4, 0.6, 0.0], [0.7, 0.4, -0.1], [0.7, 0.2, 0.2], [0.7, 0.2, 0.05, 0.05]):
        with pytest.raises(ValueError): MC.validate_allocation(bad)
    with pytest.raises(ValueError): MC.validate_allocation([0.7, 0.2, 0.1], [0.5] + [0.5 / 11] * 11)

# ---------------------------------------------------------------- full application with fixture benchmark data
@pytest.fixture(scope="module")
def full(tmp_path_factory):
    d = make_fixture_dir(str(tmp_path_factory.mktemp("bench")))
    old = DS.DATA_DIR; DS.DATA_DIR = d
    st = A.build_state(DF.fetch_nav_data(mf=FakeProvider()))
    yield st
    DS.DATA_DIR = old

@pytest.fixture()
def client(full):
    A.set_state(full); return A.app.test_client()

def test_state_has_scenario_hypotheses_mc(full):
    assert "errors" not in full, full.get("errors")
    assert full["scenario"]["evaluated"] is True
    for k in ("H1", "H2", "H3"):
        assert full["hypotheses"][k]["verdict"] != "NOT EVALUATED", full["hypotheses"][k]
    assert full["montecarlo"]["status"] == "ok"
    pr = full["montecarlo"]["proposed"]["parameters"]
    assert pr["defensive_parameter_source"] == {"debt": "historical_daily_data", "gold": "historical_daily_data"}
    assert pr["parameter_source"] == "historical_daily_data" and "shrunk historical" in pr["equity_mean_basis"]
    assert full["montecarlo"]["proposed_unshrunk_returns"]["parameters"]["parameter_source"] == "historical_daily_data"

def test_h2_independent(full):
    h2 = full["hypotheses"]["H2"]
    funds = full["funds"]; w = np.array([h2["optimised_weights"][f] for f in funds])
    cov = full["cov_mat"]; ann = full["ann_returns"]
    e = np.ones(len(funds)) / len(funds)
    assert h2["exante_in_sample"]["vol_optimised"] == pytest.approx(np.sqrt(w @ cov @ w), abs=1e-12)
    assert h2["exante_in_sample"]["vol_equal_weight"] == pytest.approx(np.sqrt(e @ cov @ e), abs=1e-12)
    assert h2["exante_in_sample"]["return_optimised"] == pytest.approx(w @ ann, abs=1e-12)
    rdf = full["returns_df"][funds].dropna()
    ppy = len(rdf) / ((rdf.index[-1] - rdf.index[0]).days / 365.25)
    assert h2["realised_in_sample"]["vol_optimised"] == pytest.approx((rdf.values @ w).std(ddof=1) * np.sqrt(ppy), abs=1e-12)
    v, ve, ro, re_ = (h2["exante_in_sample"][k] for k in ("vol_optimised", "vol_equal_weight", "return_optimised", "return_equal_weight"))
    exp = ("SUPPORTED" if ro >= re_ - 0.01 else "LOWER VOLATILITY BUT RETURN NOT COMPARABLE") if v < ve else "NOT SUPPORTED"
    assert h2["verdict"] == exp
    assert h2["holdout"].get("n_test_obs") or h2["holdout"].get("evaluated") is False

def test_optimize_endpoint_independent_verification(client):
    for pe, pb in ((22, 3.5), (30, 4.4), (14, 2.6)):
        r = client.get(f"/api/optimize?pe={pe}&pb={pb}").get_json()
        assert r["status"] == "ok"
        v = r["data"]["independent_verification"]
        assert v["all_ok"] and v["vol_difference"] < 5e-5, v      # reported weights are rounded to 4 dp

def test_independent_check_catches_violations(full):
    r = A.app.test_client().get("/api/optimize?pe=22&pb=3.5").get_json()["data"]
    bad = dict(r); bad["fund_weights"] = [0.5] + [0.5 / 11] * 11
    assert not VF.independent_check(bad, full["cov_mat"])["all_ok"]

def test_scenario_api_reconciles(client, full):
    d = client.get("/api/scenario").get_json()["data"]
    w = np.array(d["series"]["wealth"]["dynamic_constrained"])
    assert w[0] == 100.0
    dd = w / np.maximum.accumulate(w) - 1
    assert d["metrics"]["dynamic_constrained"]["max_drawdown"] == pytest.approx(dd.min(), abs=1e-5)
    assert "_returns" not in d and "walk" not in json.dumps(d).lower()
    ah = client.get("/api/allocation-history").get_json()["data"]
    assert ah["evaluated"] and len(ah["decisions"]) == 11 and all(len(x["weights_final"]) == 3 for x in ah["decisions"])

def test_montecarlo_api_default_and_custom(client):
    d = client.get("/api/montecarlo").get_json()["data"]
    p = d["proposed"]
    assert p["n_paths"] == 10000 and p["seed"] == MC.DEFAULT_SEED and p["NOT_HISTORICAL_PERFORMANCE"]
    a = p["allocation"]; assert abs(sum(a.values()) - 1) < 1e-9 and 0.5 <= a["equity"] <= 0.9 and min(a.values()) >= 0
    opt = client.get("/api/optimize").get_json()["data"]                         # MC uses the optimiser's own allocation
    assert (a["equity"], a["debt"], a["gold"]) == pytest.approx((opt["equity"], opt["debt"], opt["gold"]), abs=1e-9)
    for h, r in p["results"].items():
        t = r["terminal_wealth"]; assert t["p5"] < t["p50"] < t["p95"] and 0 <= r["prob_loss"] <= 1
    c = client.get("/api/montecarlo?paths=2000&seed=3").get_json()["data"]
    assert c["proposed"]["n_paths"] == 2000 and c["proposed"]["seed"] == 3
    assert client.get("/api/montecarlo?paths=abc").status_code == 400
    big = client.get("/api/montecarlo?paths=999999999&seed=1").get_json()["data"]
    assert big["proposed"]["n_paths"] == MC.MAX_PATHS

def test_no_obsolete_walkforward_references():
    import os, re
    root = os.path.dirname(os.path.dirname(__file__))
    for f in ("backend/webapp.py", "backend/optimizer.py", "public/index.html", "README.md"):
        t = open(os.path.join(root, f)).read().lower()
        assert "run_walk_forward" not in t and "backtest-walkforward" not in t and "walk-forward" not in t.replace("removed", ""), f

def test_without_datasets_h1_h3_not_evaluated_h2_and_mc_still_run():
    old = DS.DATA_DIR
    DS.DATA_DIR = "/nonexistent_dir_for_test"
    try:
        st = A.build_state(DF.fetch_nav_data(mf=FakeProvider()))
    finally:
        DS.DATA_DIR = old
    assert st["hypotheses"]["H1"]["verdict"] == st["hypotheses"]["H3"]["verdict"] == "NOT EVALUATED"
    assert st["hypotheses"]["H2"]["evaluated"]
    assert st["scenario"]["evaluated"] is False
    assert st["montecarlo"]["proposed"]["parameters"]["parameter_source"] != "historical_daily_data"

def test_cache_fingerprint_includes_datasets(tmp_path, monkeypatch):
    f0 = A.compute_fingerprint()
    d = make_fixture_dir(str(tmp_path), seed=99); monkeypatch.setattr(DS, "DATA_DIR", d)
    assert A.compute_fingerprint() != f0


def test_rounded_optimizer_weights_accepted_and_normalised():
    w = [0.6350, 0.2051, 0.1600]                         # sums to 1.0001 after 4-dp rounding
    r = MC.run_monte_carlo([1 / 12] * 12, w, np.eye(12) * 0.04, np.full(12, 0.12), np.full(12, 0.13), n_paths=1000, horizons=(1,))
    assert abs(sum(r["allocation"].values()) - 1) < 1e-12
    with pytest.raises(ValueError):
        MC.run_monte_carlo([1 / 12] * 12, [0.635, 0.2, 0.2], np.eye(12) * 0.04, np.full(12, 0.12), np.full(12, 0.13), n_paths=1000, horizons=(1,))

def test_mc_for_many_valuation_inputs(client):
    for pe, pb in ((12, 2.0), (18, 3.0), (22, 3.2), (27, 4.0), (35, 5.0), (40, 6.0)):
        r = client.get(f"/api/montecarlo?pe={pe}&pb={pb}&paths=1500&seed=2")
        assert r.status_code in (200, 422), (pe, pb, r.status_code)
        if r.status_code == 200:
            a = r.get_json()["data"]["proposed"]["allocation"]; assert abs(sum(a.values()) - 1) < 1e-9 and 0.5 <= a["equity"] <= 0.9 and min(a.values()) >= 0
