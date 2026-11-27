import os, subprocess, sys, numpy as np, pandas as pd
import datasets as DS
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _run(args, data_dir):
    env = dict(os.environ, PORTFOLIO_DATA_DIR=str(data_dir))
    return subprocess.run([sys.executable, os.path.join(ROOT, "tools", "build_dataset.py"), *args], capture_output=True, text=True, env=env)

def test_import_csv_roundtrip_and_status(tmp_path):
    idx = pd.bdate_range("2010-06-01", "2025-01-31"); rng = np.random.default_rng(1)
    raw = pd.DataFrame({"Date": idx.strftime("%d %b %Y"), "Index Name": "TEST", "Close": (1000 * np.cumprod(1 + rng.normal(0, .01, len(idx)))).round(2)})
    f = tmp_path / "raw_tri.csv"; raw.to_csv(f, index=False)
    out = _run(["import-csv", "--key", "nifty_tri", "--file", str(f), "--source", "TEST FIXTURE", "--retrieved-on", "2026-11-27", "--date-format", "%d %b %Y"], tmp_path / "d")
    assert out.returncode == 0, out.stderr
    r = DS.load_dataset("nifty_tri", str(tmp_path / "d"))
    assert r["status"] == "ok" and r["provenance"]["n_obs"] == len(idx) and len(r["provenance"]["sha256"]) == 64
    assert np.allclose(r["df"]["value"].values, raw["Close"].values)          # values copied, not altered
    st = _run(["status"], tmp_path / "d"); assert "nifty_tri" in st.stdout and "ok" in st.stdout

def test_import_rejects_duplicates_and_bad_key(tmp_path):
    raw = pd.DataFrame({"Date": ["2014-01-01", "2014-01-01"], "Close": [1, 2]}); f = tmp_path / "x.csv"; raw.to_csv(f, index=False)
    assert _run(["import-csv", "--key", "nifty_tri", "--file", str(f), "--source", "s", "--retrieved-on", "2026-01-01"], tmp_path / "d").returncode != 0
    assert _run(["import-csv", "--key", "bogus", "--file", str(f), "--source", "s", "--retrieved-on", "2026-01-01"], tmp_path / "d").returncode != 0

def test_valuation_import_columns(tmp_path):
    me = pd.date_range("2010-06-30", "2025-01-31", freq="ME")
    raw = pd.DataFrame({"Date": me.strftime("%d-%m-%Y"), "P/E": np.linspace(18, 24, len(me)), "P/B": np.linspace(3, 3.6, len(me)), "Div Yield": 1.2})
    f = tmp_path / "v.csv"; raw.to_csv(f, index=False)
    out = _run(["import-csv", "--key", "nifty_valuation", "--file", str(f), "--source", "TEST", "--retrieved-on", "2026-11-27"], tmp_path / "d")
    assert out.returncode == 0, out.stderr
    assert DS.load_dataset("nifty_valuation", str(tmp_path / "d"))["status"] == "ok"


def test_ingest_autodetects_gsec_level_file_and_registers_debt(tmp_path):
    import shutil, json
    d = tmp_path / "data"; shutil.copytree(os.path.join(ROOT, "data"), d)
    idx = pd.bdate_range("2015-10-13", "2026-11-26"); rng = np.random.default_rng(2)
    pd.DataFrame({"Date": idx.strftime("%d-%b-%Y"), "Close": (1000 * np.cumprod(1 + rng.normal(0.0002, 0.0015, len(idx)))).round(4)}
                 ).to_csv(d / "raw" / "NIFTY-COMPOSITE-G-SEC-INDEX-levels.csv", index=False)       # TEST FIXTURE
    env = dict(os.environ, PORTFOLIO_DATA_DIR=str(d))
    out = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "ingest_raw.py"), "--raw", str(d / "raw"),
                          "--retrieved-on", "2026-11-27"], capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr[-600:]
    prov = DS.read_provenance(str(d))
    assert "debt" in prov and DS.load_dataset("debt", str(d))["status"] == "ok"


def test_ingest_ignores_the_about_sheet_and_reports_debt_not_supplied(tmp_path):
    import shutil, json
    d = tmp_path / "data"; shutil.copytree(os.path.join(ROOT, "data"), d)
    env = dict(os.environ, PORTFOLIO_DATA_DIR=str(d))
    out = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "ingest_raw.py"), "--raw", str(d / "raw"),
                          "--retrieved-on", "2026-11-27"], capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr[-600:]
    assert "debt" not in DS.read_provenance(str(d))
    assert json.load(open(d / "data_quality_report.json"))["datasets"]["debt"]["status"] == "NOT SUPPLIED"
