import json, os, time, copy
import numpy as np, pandas as pd, pytest
from fakes import FakeProvider, make_nav, raw_from, NAMES
import data_fetcher as DF
import optimizer as OPT
import universe as U
import fund_metrics as FM
import webapp as A

TOL = 1e-6

@pytest.fixture(scope="module")
def ds():
    return DF.fetch_nav_data(mf=FakeProvider())

@pytest.fixture(scope="module")
def state(ds):
    return A.build_state(ds)

# ---------------------------------------------------------------- 1. NAV preprocessing
def test_no_cross_fund_ffill_and_no_artificial_zeros():
    idx = pd.bdate_range("2023-01-02", periods=300)
    nav = pd.Series(100 * np.cumprod(1 + np.random.default_rng(0).normal(0.0005, 0.01, 300)), index=idx)
    nav_gap = nav.drop(nav.index[100:104])                  # 4 missing business days
    r, st = DF.returns_from_nav(nav_gap)
    assert (r == 0).sum() == 0
    # return across the gap spans two observed NAVs exactly
    d0, d1 = nav.index[99], nav.index[104]
    assert r.loc[d1] == pytest.approx(nav[d1] / nav[d0] - 1)
    r_ref = nav_gap.pct_change(fill_method=None).iloc[1:]
    assert np.allclose(r.dropna(), r_ref[r.notna()])

def test_missing_days_in_one_fund_do_not_create_zero_returns_in_df():
    prov = FakeProvider(drop={"Tata": list(range(500, 520))})
    d = DF.fetch_nav_data(mf=prov)
    assert d["returns_df"]["Tata"].dropna().eq(0).sum() == 0
    q = d["quality"]["Tata"]
    assert q["missing_business_days"] >= 20

def test_gap_return_excluded():
    idx = pd.to_datetime(["2023-01-02", "2023-01-03", "2023-02-01", "2023-02-02"])
    r, st = DF.returns_from_nav(pd.Series([100, 101, 105, 106.0], index=idx))
    assert np.isnan(r.loc["2023-02-01"]) and st["n_gap_excluded"] == 1

def test_outlier_return_excluded():
    idx = pd.bdate_range("2023-01-02", periods=4)
    r, st = DF.returns_from_nav(pd.Series([100, 101, 150, 151.0], index=idx))
    assert st["n_outlier_excluded"] == 1

def test_covariance_matches_pairwise_numpy(ds):
    rd = ds["returns_df"]
    ann, cov, vols, shp, extra = OPT.build_cov_from_returns(rd)
    assert np.isfinite(cov).all() and np.all(np.linalg.eigvalsh(cov) > 0)
    # unshrunk pairwise covariance check on one pair using only overlapping days
    a, b = rd["Parag Parikh"], rd["HDFC"]
    m = a.notna() & b.notna()
    assert m.sum() > 60
    raw = np.cov(a[m], b[m], ddof=1)[0, 1] * 252
    assert np.isfinite(raw)

def test_insufficient_overlap_raises():
    rd = pd.DataFrame(np.random.default_rng(0).normal(0, .01, (400, 3)), columns=list("ABC"),
                      index=pd.bdate_range("2022-01-03", periods=400))
    rd.loc[rd.index[:380], "C"] = np.nan
    with pytest.raises(OPT.InsufficientDataError):
        OPT.build_cov_from_returns(rd)

# ---------------------------------------------------------------- 2. Universe
def test_universe_consistency():
    assert len(U.SCHEME_SPECS) == 12
    assert list(U.SELECTED_SCHEMES) == U.FUNDS_DEFAULT
    assert len(set(U.SELECTED_SCHEMES.values())) == 12
    assert OPT.FUNDS_DEFAULT == U.FUNDS_DEFAULT if hasattr(OPT, "FUNDS_DEFAULT") else True

def test_frontend_labels_match_universe():
    html = open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "public", "index.html")).read()
    # the frontend must take fund labels from the API (single source of truth), never a hard-coded list
    assert "/api/funds" in html
    for stale in ("Mirae", "Canara", "SBI Flexi", "UTI Flexi", "Nippon", "Axis Flexi", "DSP Flexi"):
        assert stale not in html, f"stale hard-coded fund label {stale}"
    assert html.count("EQ_COLORS") >= 1 and "'#" in html
    import re
    m = re.search(r"EQ_COLORS\s*=\s*\[(.*?)\]", html, re.S)
    assert len(re.findall(r"#[0-9a-fA-F]{6}", m.group(1))) >= len(U.FUNDS_DEFAULT)

def test_all_12_validate(ds):
    assert ds["dataset_complete"]
    for f, v in ds["validation"].items():
        assert v["ok"], (f, v)

@pytest.mark.parametrize("label,bad", [
    ("Invesco", "Invesco India Large Cap Fund - Direct Plan - Growth"),
    ("WhiteOak", "WhiteOak Capital Large Cap Fund - Direct Plan - Growth"),
    ("HDFC", "HDFC Flexi Cap Fund - Regular Plan - Growth"),
    ("Tata", "Tata Flexi Cap Fund - Direct Plan - IDCW Payout"),
    ("Quant", "Quantum Flexi Cap Fund - Direct Plan - Growth"),
    ("HSBC", "UTI Ultra Short Term Fund - Direct Plan - Growth"),
    ("ITI", "ITI Large Cap Fund - Direct Plan - Growth"),
])
def test_wrong_identity_rejected(label, bad):
    prov = FakeProvider(names={label: bad})
    with pytest.raises(DF.DataUnavailableError) as e:
        DF.fetch_nav_data(mf=prov)
    assert label in e.value.report["rejected"]

def test_wrong_identity_nonstrict_excluded_and_flagged():
    prov = FakeProvider(names={"Tata": "Tata Equity P/E Fund - Direct Plan - Growth"})
    d = DF.fetch_nav_data(mf=prov, strict=False)
    assert "Tata" not in d["returns_df"].columns and d["dataset_complete"] is False
    assert d["validation"]["Tata"]["status"] == "rejected_identity"

def test_missing_name_rejected():
    prov = FakeProvider()
    code = U.SCHEME_SPECS["Tata"]["code"]
    prov.raw[code] = dict(prov.raw[code]); prov.raw[code]["scheme_name"] = ""
    with pytest.raises(DF.DataUnavailableError):
        DF.fetch_nav_data(mf=prov)

def test_code_mismatch_rejected():
    ok, reasons = U.validate_scheme_identity("Tata", "999999", {"scheme_name": NAMES["Tata"]})
    assert not ok

def test_category_not_flexi_rejected():
    ok, _ = U.validate_scheme_identity("Tata", U.SCHEME_SPECS["Tata"]["code"],
        {"scheme_name": NAMES["Tata"], "scheme_category": "Equity Scheme - Large Cap Fund"})
    assert not ok

# ---------------------------------------------------------------- failure modes
def test_failed_api_strict():
    code = U.SCHEME_SPECS["Tata"]["code"]
    with pytest.raises(DF.DataUnavailableError) as e:
        DF.fetch_nav_data(mf=FakeProvider(fail=[code]))
    assert "Tata" in e.value.report["failed"]

def test_failed_api_nonstrict_flagged():
    code = U.SCHEME_SPECS["Tata"]["code"]
    d = DF.fetch_nav_data(mf=FakeProvider(fail=[code]), strict=False)
    assert "Tata" not in d["returns_df"].columns and not d["dataset_complete"]

def test_none_and_garbage_responses():
    prov = FakeProvider()
    prov.raw[U.SCHEME_SPECS["HSBC"]["code"]] = None
    prov.raw[U.SCHEME_SPECS["Edelweiss"]["code"]] = {"scheme_name": NAMES["Edelweiss"], "data": []}
    with pytest.raises(DF.DataUnavailableError) as e:
        DF.fetch_nav_data(mf=prov)
    assert set(e.value.report["failed"]) >= {"HSBC", "Edelweiss"}

def test_all_providers_fail():
    prov = FakeProvider(fail=[s["code"] for s in U.SCHEME_SPECS.values()])
    with pytest.raises(DF.DataUnavailableError):
        DF.fetch_nav_data(mf=prov, strict=False)

def test_insufficient_history_excluded():
    prov = FakeProvider()
    code = U.SCHEME_SPECS["WhiteOak"]["code"]
    prov.raw[code] = raw_from(make_nav(120, seed=5), NAMES["WhiteOak"], code)
    d = DF.fetch_nav_data(mf=prov, strict=False)
    assert "WhiteOak" in [e["label"] for e in d["excluded_short"]] and "WhiteOak" not in d["returns_df"].columns
    assert d["dataset_complete"] is False

def test_load_dataset_has_no_live_or_synthetic_path(monkeypatch, tmp_path):
    import datasets as DS
    monkeypatch.setattr(DS, "DATA_DIR", str(tmp_path))                 # no bundled data here
    with pytest.raises(DF.DataUnavailableError):
        DF.load_dataset()
    with pytest.raises(DF.DataUnavailableError):                        # no provider supplied: nothing is fetched
        DF.fetch_nav_data()
    assert not hasattr(DF, "synthetic_dataset") and not hasattr(DF, "_get_provider")

# ---------------------------------------------------------------- 3. Fund metrics
def test_metrics_reconcile_with_independent_numpy(ds, state):
    for m in state["fund_metrics"]:
        f = m["fund"]
        nav = ds["nav_full"][f].dropna()
        nav = nav.iloc[-(DF.MAX_LOOKBACK_DAYS + 1):]
        r = nav.pct_change(fill_method=None).iloc[1:]
        gap = nav.index.to_series().diff().dt.days.iloc[1:]
        r = r[(gap <= DF.MAX_GAP_CALENDAR_DAYS) & (r.abs() <= DF.MAX_ABS_DAILY_RETURN)]
        vol = r.std(ddof=1) * np.sqrt(252)
        cagr = np.expm1(np.log1p(r).mean() * 252)
        mdd = (nav / nav.cummax() - 1).min()
        assert m["vol_sample"] == pytest.approx(vol, abs=1e-9)
        assert m["cagr_log_daily"] == pytest.approx(cagr, abs=1e-9)
        assert m["max_drawdown"] == pytest.approx(mdd, abs=1e-12)
        assert m["n_returns"] == len(r)

def test_beta_alpha_match_ols(ds):
    f = "Tata"
    nav = ds["nav_full"][f].dropna()
    p = make_nav(len(nav) + 40, seed=99, start=str(nav.index[0].date()))   # synthetic proxy, test only
    out = FM.beta_alpha(nav, p, OPT.RF)
    c = nav.index.intersection(p.index)
    y = nav[c].pct_change(fill_method=None).dropna(); x = p[c].pct_change(fill_method=None).dropna()
    b = np.polyfit(x, y, 1)[0]
    assert out["beta"] == pytest.approx(b, rel=1e-6)

def test_low_vol_flagged_not_adjusted():
    idx = pd.bdate_range("2022-01-03", periods=600)
    nav = pd.Series(100 * np.cumprod(1 + np.random.default_rng(3).normal(0.0004, 0.0017, 600)), index=idx)
    r, _ = DF.returns_from_nav(nav)
    m = FM.compute_fund_metrics("X", nav, r, OPT.RF)
    assert m["vol_sample"] == pytest.approx(r.std(ddof=1) * np.sqrt(252))
    assert any("implausibly low" in f for f in m["flags"])

def test_model_vol_matches_sample_vol_no_floor(state):
    # model vol is derived from raw covariance; ensure it is within shrinkage distance of raw
    for m in state["fund_metrics"]:
        assert abs(m["vol_model"] - m["vol_sample"]) < 0.03

# ---------------------------------------------------------------- 4. Optimiser feasibility
VAL_GRID = [(pe, pb) for pe in (12, 16, 19, 22, 25, 28, 34, 40) for pb in (2.0, 2.8, 3.5, 4.2, 5.5)]

def _run(state, pe, pb):
    return OPT.optimize_for_pe_pb(pe, pb, state["ann_returns"], state["cov_mat"], state["vols"],
                                  state["fund_sharpes"], state["funds"], semi_vols=state["extra_stats"]["semi_vols"],
                                  val_stats=state.get("val_stats"))

def test_optimizer_feasible_across_valuations(state):
    statuses = set()
    for pe, pb in VAL_GRID:
        try:
            r = _run(state, pe, pb)
        except OPT.OptimizationInfeasible:
            continue
        w = np.array(r["fund_weights"])
        assert abs(w.sum() - 1) < 1e-5
        assert w.min() >= OPT.MIN_FUND_W - 1e-6 and w.max() <= OPT.MAX_FUND_W + 1e-6
        assert 0.5 - 1e-6 <= r["equity"] <= 0.9 + 1e-6 and min(r["debt"], r["gold"]) >= 0
        assert r["equity"] + r["debt"] + r["gold"] == pytest.approx(1, abs=2e-4)
        # independent vol recomputation from the returned (rounded) numbers
        E = r["equity"]; D, G = r["debt"], r["gold"]
        v = OPT.portfolio_vol_from_weights(E, w, D, G, state["cov_mat"])[0]
        assert v <= OPT.VOL_CAP + 1e-9, (pe, pb, v)
        assert r["constraint_ok"] is True
        assert "cash" not in r
        statuses.add(r["status"])
    assert statuses  # at least something solved

def _unmeetable_cov(state):
    """Every asset (funds, debt, gold) has volatility far above 10%, so no allocation can meet the cap."""
    from assets import CovModel
    cr = dict(state["cov_mat"].cross)
    cr["cov_dd"] = np.diag([0.25, 0.25]); cr["cov_fd"] = np.asarray(cr["cov_fd"]) * 5
    cov = CovModel(np.asarray(state["cov_mat"]) * 9); cov.cross = cr
    return cov


def test_infeasible_reported_not_returned(state):
    cov = _unmeetable_cov(state)
    with pytest.raises(OPT.OptimizationInfeasible) as e:
        OPT.optimize_for_pe_pb(22, 3.5, state["ann_returns"], cov, state["vols"], state["fund_sharpes"],
                               state["funds"], semi_vols=state["extra_stats"]["semi_vols"], val_stats=state.get("val_stats"))
    assert e.value.details["min_achievable_vol_at_50pct_equity"] > OPT.VOL_CAP

def test_project_weights_bounds():
    rng = np.random.default_rng(0)
    for _ in range(200):
        w = OPT.project_weights(rng.normal(0.07, 0.3, 12))
        assert abs(w.sum() - 1) < 1e-9 and w.min() >= OPT.MIN_FUND_W - 1e-9 and w.max() <= OPT.MAX_FUND_W + 1e-9






# ---------------------------------------------------------------- 6. Cache and API
def test_fingerprint_changes_with_params(monkeypatch):
    A._FP.clear(); f1 = A.compute_fingerprint()
    monkeypatch.setattr(OPT, "VOL_CAP", 0.11); A._FP.clear()
    assert A.compute_fingerprint() != f1
    monkeypatch.undo(); A._FP.clear()
    monkeypatch.setitem(U.SCHEME_SPECS["Tata"], "code", "000000"); A._FP.clear()
    assert A.compute_fingerprint() != f1
    monkeypatch.undo(); A._FP.clear()

def test_no_disk_cache_is_written(state, tmp_path):
    import glob
    before = set(glob.glob(os.path.join(A.ROOT_DIR, "**", "*.pkl"), recursive=True))
    A.set_state(state)
    assert not hasattr(A, "_save_cache") and not hasattr(A, "_load_cache") and not hasattr(A, "CACHE_FILE")
    assert set(glob.glob(os.path.join(A.ROOT_DIR, "**", "*.pkl"), recursive=True)) == before

@pytest.fixture()
def client(state):
    A.set_state(state)
    return A.app.test_client()

def test_api_endpoints_live(client, state):
    for path in ("/health", "/api/funds", "/api/fund-metrics", "/api/data-quality", "/api/diagnostics",
                 "/api/frontier", "/api/allocation-history", "/api/scenario", "/api/hypotheses", "/api/montecarlo", "/api/datasets", "/api/meta",
                 "/api/health", "/api/data-report"):
        r = client.get(path)
        assert r.status_code == 200, (path, r.data[:200])
    j = client.get("/api/funds").get_json()
    assert len(j["data"]["funds"]) == 12 and j["meta"]["source"] in A.REAL_SOURCES and not j["meta"]["is_synthetic"]
    # displayed numbers reconcile with fund-metrics
    fm = {m["fund"]: m for m in client.get("/api/fund-metrics").get_json()["data"]["funds"]}
    for row in j["data"]["funds"]:
        assert row["raw_vol"] == pytest.approx(fm[row["name"]]["vol_sample"])
        assert row["max_drawdown"] == pytest.approx(fm[row["name"]]["max_drawdown"])
    dq = client.get("/api/data-quality").get_json()["data"]
    assert dq["dataset_complete"] and dq["snapshot"]["data_end"]

def test_api_optimize_ok_and_422(client, state):
    r = client.get("/api/optimize?pe=22&pb=3.5").get_json()
    assert r["status"] == "ok" and r["data"]["constraint_ok"] and r["data"]["port_vol"] <= OPT.VOL_CAP + 1e-9
    bad = dict(state, cov_mat=_unmeetable_cov(state))
    A.set_state(bad)
    resp = A.app.test_client().get("/api/optimize?pe=22&pb=3.5")
    assert resp.status_code == 422 and resp.get_json()["constraint_ok"] is False

def test_api_503_when_unavailable():
    A.set_state(A._unavailable_state("simulated outage"))
    c = A.app.test_client()
    assert c.get("/api/funds").status_code == 503
    assert c.get("/api/optimize").status_code == 503
    h = c.get("/health").get_json()["data"]
    assert h["ready"] is False and h["source"] == "unavailable"
    assert c.get("/api/data-quality").status_code == 200
