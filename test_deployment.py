"""Deployment-oriented tests: exact universe, bundled data only, every route, frontend contract, Vercel config,
lazy in-memory state, behaviour with and without the G-Sec dataset. Synthetic debt data appear ONLY in the tests
that check that sections populate when a complete dataset exists; they are never shipped or reported."""
import importlib.util, json, os, re, shutil, subprocess, sys, threading
import numpy as np, pandas as pd, pytest
import data_fetcher as DF, datasets as DS, universe as U, webapp as A

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROUTES = ["/api/health", "/api/funds", "/api/fund-metrics", "/api/data-quality", "/api/data-report", "/api/datasets",
          "/api/diagnostics", "/api/optimize?pe=22&pb=3.5", "/api/frontier", "/api/allocation-history", "/api/scenario",
          "/api/hypotheses", "/api/montecarlo", "/api/meta"]
EXPECTED_12 = ["Parag Parikh", "Quant", "Motilal Oswal", "HDFC", "Aditya Birla", "ITI", "Tata", "Invesco",
               "Bank of India", "HSBC", "Edelweiss", "WhiteOak"]


@pytest.fixture(scope="module")
def real_state():
    DS.DATA_DIR = os.path.join(ROOT, "data")
    return A.build_state(DF.load_dataset())


@pytest.fixture()
def client(real_state):
    A.set_state(real_state)
    return A.app.test_client()


def test_universe_is_exactly_the_12_funds():
    assert list(U.SCHEME_SPECS) == EXPECTED_12 and len(U.SELECTED_SCHEMES) == 12
    assert len(set(U.SELECTED_SCHEMES.values())) == 12


def test_api_returns_exactly_12_funds_from_bundled_data(client):
    j = client.get("/api/funds").get_json()
    assert [f["name"] for f in j["data"]["funds"]] == EXPECTED_12
    assert j["meta"]["source"] == "bundled_dataset" and not j["meta"]["is_synthetic"]
    assert not {"ICICI", "Kotak"} & {f["name"] for f in j["data"]["funds"]}
    assert all(f["scheme_code"] for f in j["data"]["funds"])


@pytest.mark.parametrize("route", ROUTES)
def test_every_route_returns_200_json(client, route):
    r = client.get(route)
    assert r.status_code == 200, (route, r.data[:300])
    j = r.get_json()
    assert j["status"] == "ok" and "data" in j


def test_formerly_404_routes_exist(client):
    for route in ("/api/montecarlo", "/api/datasets", "/api/data-report", "/api/health"):
        assert client.get(route).status_code == 200


def test_unknown_api_route_is_json_404(client):
    r = client.get("/api/nope")
    assert r.status_code == 404 and r.get_json()["status"] == "error"


@pytest.fixture(scope="module")
def no_debt_state(tmp_path_factory):
    """The bundled data with the G-Sec dataset removed (simulates a deployment before G-Sec was added)."""
    d = str(tmp_path_factory.mktemp("nodebt") / "data")
    shutil.copytree(os.path.join(ROOT, "data"), d)
    os.remove(os.path.join(d, "cleaned", "debt.csv"))
    prov = DS.read_provenance(d); prov.pop("debt"); DS.write_provenance(prov, d)
    old = DS.DATA_DIR; DS.DATA_DIR = d
    try:
        yield A.build_state(DF.load_dataset())
    finally:
        DS.DATA_DIR = old


def test_without_debt_dataset_sections_report_honestly(no_debt_state):
    A.set_state(no_debt_state); client = A.app.test_client()
    d = client.get("/api/datasets").get_json()["data"]
    debt = d["datasets"]["debt"] if "datasets" in d else d["debt"]
    assert debt["status"] == "missing"
    sc = client.get("/api/scenario").get_json()["data"]
    assert sc["evaluated"] is False and "debt" in sc["message"]
    h = client.get("/api/hypotheses").get_json()["data"]
    assert h["H1"]["verdict"] == h["H3"]["verdict"] == "NOT EVALUATED"
    assert client.get("/api/allocation-history").get_json()["data"]["evaluated"] is False
    mc = client.get("/api/montecarlo").get_json()["data"]
    assert mc["status"] == "ok" and mc["proposed"]["parameters"]["parameter_source"] == "partly_assumed"
    assert mc["proposed"]["parameters"]["defensive_parameter_source"]["debt"] == "assumed_constants"


def test_real_bundled_gsec_activates_scenario_hypotheses_and_mc(client):
    """With the supplied G-Sec series the sections populate from ACTUAL data (no synthetic input)."""
    p = DS.read_provenance(os.path.join(ROOT, "data"))["debt"]
    assert p["start"] == "2016-04-05" and p["end"] == "2024-12-31" and p["n_obs"] == 2158 and len(p["sha256"]) == 64
    sc = client.get("/api/scenario").get_json()["data"]
    assert sc["evaluated"] and sc["period"]["series_first_dates"]["debt"] == "2016-04-05"
    assert sc["period"]["first_decision_year"] >= 2017 and sc["period"]["end"] <= "2024-12-31"
    h = client.get("/api/hypotheses").get_json()["data"]
    assert h["H1"]["evaluated"] and h["H3"]["evaluated"] and h["H2"]["evaluated"]
    ah = client.get("/api/allocation-history").get_json()["data"]
    assert ah["evaluated"] and all(abs(sum(x["weights_final"]) - 1) < 1e-9 and 0.5 - 1e-9 <= x["weights_final"][0] <= 0.9 + 1e-9
                                   and (x["exante_vol_final"] <= 0.10 + 1e-9 or x["constraint_status"].startswith("cap_infeasible"))
                                   for x in ah["decisions"])
    mc = client.get("/api/montecarlo").get_json()["data"]["proposed"]["parameters"]
    assert mc["defensive_parameter_source"] == {"debt": "historical_daily_data", "gold": "historical_daily_data"}


def test_saturday_observation_excluded_from_debt_series():
    import pandas as pd
    s = pd.read_csv(os.path.join(ROOT, "data", "cleaned", "debt.csv")); s.columns = ["date", "value"]; s["date"] = pd.to_datetime(s["date"])
    assert (s["date"].dt.dayofweek < 5).all() and pd.Timestamp("2024-05-18") not in set(s["date"])
    q = json.load(open(os.path.join(ROOT, "data", "data_quality_report.json")))["datasets"]["debt"]
    assert [x["date"] for x in q["weekend_rows_excluded"]] == ["2024-05-18"]


def test_no_live_nav_fetching_possible(monkeypatch):
    import socket
    calls = []
    def blocked(*a, **k):
        calls.append(a); raise OSError("network blocked in test")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)
    DS.DATA_DIR = os.path.join(ROOT, "data")
    st = A.build_state(DF.load_dataset())
    assert st["ready"] and len(st["funds"]) == 12 and calls == []


def test_production_code_and_requirements_have_no_live_provider():
    for fn in os.listdir(os.path.join(ROOT, "backend")):
        if fn.endswith(".py"):
            src = open(os.path.join(ROOT, "backend", fn)).read()
            assert not re.search(r"^\s*(import|from)\s+(mftool|requests|urllib\.request|flask_cors)", src, re.M), fn
    req = open(os.path.join(ROOT, "requirements.txt")).read().lower()
    assert "mftool" not in req and "flask-cors" not in req and "requests" not in req
    out = subprocess.run([sys.executable, "-c",
        "import sys; sys.path.insert(0, 'backend'); import webapp as A, datasets as DS, data_fetcher as DF;"
        "A.build_state(DF.load_dataset());"
        "print([m for m in ('mftool','requests','flask_cors') if m in sys.modules])"],
        cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-500:]
    assert out.stdout.strip().splitlines()[-1] == "[]"


def test_frontend_uses_relative_api_paths_and_every_path_has_a_route():
    html = open(os.path.join(ROOT, "public", "index.html")).read()
    assert re.search(r'const API_BASE\s*=\s*""', html)
    assert "onrender.com" not in html and "http://" not in re.sub(r"http://www\.w3\.org[^\"']*", "", html).replace("http://localhost", "")
    used = set(re.findall(r"/api/[a-z\-]+", html))
    rules = {r.rule for r in A.app.url_map.iter_rules()}
    assert used and used <= rules, used - rules


def test_vercel_project_files():
    cfg = json.load(open(os.path.join(ROOT, "vercel.json")))
    # Root cause of the "unused_function" build error: without a detected framework Vercel only maps `functions`
    # patterns to files in api/. The framework is therefore pinned here, independent of the dashboard preset.
    assert cfg.get("framework") == "flask"
    assert not os.path.isdir(os.path.join(ROOT, "api")) and "builds" not in cfg and "routes" not in cfg
    assert "app.py" in cfg["functions"] and cfg["functions"]["app.py"]["maxDuration"] >= 60
    ex = cfg["functions"]["app.py"]["excludeFiles"]
    assert "data" not in ex.replace("tools", "") and "backend" not in ex and "public" not in ex   # data is NOT excluded
    assert open(os.path.join(ROOT, ".python-version")).read().strip() in ("3.12", "3.13", "3.14")
    spec = importlib.util.spec_from_file_location("vercel_entry", os.path.join(ROOT, "app.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    from flask import Flask
    assert isinstance(m.app, Flask)
    assert os.path.exists(os.path.join(ROOT, "public", "index.html"))
    prov = DS.read_provenance(os.path.join(ROOT, "data"))
    for k, p in prov.items():
        assert DS.sha256_file(os.path.join(ROOT, "data", p["file"])) == p["sha256"], k


def test_state_is_built_lazily_once_in_memory_and_concurrently_safe(monkeypatch):
    monkeypatch.delenv("PORTFOLIO_SKIP_INIT", raising=False)
    DS.DATA_DIR = os.path.join(ROOT, "data")
    A.set_state({}); A._LAST_ATTEMPT = 0.0
    n = {"builds": 0}
    real = A.build_state
    def counting(ds):
        n["builds"] += 1; return real(ds)
    monkeypatch.setattr(A, "build_state", counting)
    codes = []
    def hit():
        codes.append(A.app.test_client().get("/api/funds").status_code)
    ts = [threading.Thread(target=hit) for _ in range(5)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert codes == [200] * 5 and n["builds"] == 1
    A.app.test_client().get("/api/scenario"); assert n["builds"] == 1            # warm instance reuses memory


def test_unavailable_when_bundled_data_missing(monkeypatch, tmp_path):
    monkeypatch.delenv("PORTFOLIO_SKIP_INIT", raising=False)
    monkeypatch.setattr(DS, "DATA_DIR", str(tmp_path))
    A.set_state({}); A._LAST_ATTEMPT = 0.0
    c = A.app.test_client()
    assert c.get("/api/funds").status_code == 503
    j = c.get("/api/health").get_json()["data"]
    assert j["ready"] is False and j["source"] == "unavailable"


def test_success_responses_are_cacheable_but_health_and_errors_are_not(client):
    assert "s-maxage" in client.get("/api/funds").headers["Cache-Control"]
    assert "s-maxage" not in client.get("/api/health").headers.get("Cache-Control", "")
    assert "s-maxage" not in client.get("/api/nope").headers.get("Cache-Control", "")


# ---- sections populate when a complete debt dataset exists (synthetic debt used ONLY here, for logic testing)
@pytest.fixture(scope="module")
def with_debt(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("withdebt") / "data")
    shutil.copytree(os.path.join(ROOT, "data"), d)
    g = pd.read_csv(os.path.join(d, "cleaned", "gold.csv"), parse_dates=["date"])["date"]
    g = g[g >= "2015-10-13"]
    lv = 100 * np.cumprod(1 + np.random.default_rng(1).normal(0.0002, 0.0015, len(g)))
    p = os.path.join(d, "cleaned", "debt.csv")
    pd.DataFrame({"date": g, "value": lv}).to_csv(p, index=False)
    prov = DS.read_provenance(d)
    prov["debt"] = {"file": "cleaned/debt.csv", "source": "TEST FIXTURE (synthetic)", "retrieved_on": "2026-11-27",
                    "sha256": DS.sha256_file(p)}
    DS.write_provenance(prov, d)
    old = DS.DATA_DIR; DS.DATA_DIR = d
    st = A.build_state(DF.load_dataset())
    yield d, st
    DS.DATA_DIR = old


def test_scenario_allocation_history_h1_h3_mc_populate_with_complete_data(with_debt):
    d, st = with_debt
    A.set_state(st); c = A.app.test_client()
    sc = c.get("/api/scenario").get_json()["data"]
    assert sc["evaluated"] and sc["period"]["start"] >= "2016-12-31"           # G-Sec starts later: period is shorter
    assert sc["period"]["series_first_dates"]["debt"] >= "2015-10-13"
    ah = c.get("/api/allocation-history").get_json()["data"]
    assert ah["evaluated"] and len(ah["decisions"]) >= 5 and all(len(x["weights_final"]) == 3 for x in ah["decisions"])
    h = c.get("/api/hypotheses").get_json()["data"]
    assert h["H1"]["verdict"] != "NOT EVALUATED" and h["H3"]["verdict"] != "NOT EVALUATED"
    mc = c.get("/api/montecarlo").get_json()["data"]
    assert mc["proposed"]["parameters"]["defensive_parameter_source"]["debt"] == "historical_daily_data"
    assert "errors" not in st, st.get("errors")


def test_invalid_debt_dataset_is_reported_not_used(with_debt, tmp_path):
    d, _ = with_debt
    bad = str(tmp_path / "data"); shutil.copytree(d, bad)
    with open(os.path.join(bad, "cleaned", "debt.csv"), "a") as f:
        f.write("2024-12-31,1\n")                                               # checksum no longer matches
    old = DS.DATA_DIR; DS.DATA_DIR = bad
    try:
        st = A.build_state(DF.load_dataset())
    finally:
        DS.DATA_DIR = old
    assert st["datasets"]["debt"]["status"] == "invalid"
    assert st["scenario"]["evaluated"] is False and "debt" in st["scenario"]["message"]
    assert st["hypotheses"]["H1"]["verdict"] == "NOT EVALUATED"
