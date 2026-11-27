"""
datasets.py - project-stored historical datasets with provenance and checksums.

Nothing in this module downloads or fabricates data. Datasets are CSV files in data/ (columns: date,value)
registered in data/provenance.json by tools/build_dataset.py, which records the source, retrieval date,
date range, number of observations and a SHA-256 checksum. On load, the checksum is re-computed; any mismatch
makes the dataset INVALID and it is not used.

Required datasets (keys):
  nifty_tri        Nifty 50 Total Returns Index, daily (official NSE Indices data, NOT the price index)
  nifty_valuation  Nifty 50 trailing P/E and P/B (columns date,pe,pb)
  gold             INR gold price or gold ETF NAV, daily
  debt             Nifty Composite G-Sec Index (daily levels)
Cash is NOT an asset class in this framework (removed by design); the investable classes are equity, gold, debt.
"""
import hashlib
import json
import os

import numpy as np
import pandas as pd

# The datasets ship inside the repository: <repo>/data (backend/ is one level below the repository root).
# PORTFOLIO_DATA_DIR is an optional override used only by the tests and the import tools.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("PORTFOLIO_DATA_DIR") or os.path.join(ROOT_DIR, "data")
PROVENANCE_FILE = "provenance.json"
REQUIRED = ("nifty_tri", "nifty_valuation", "gold", "debt")
DESCRIPTIONS = {
    "nifty_tri": "Nifty 50 Total Returns Index (daily) - NSE Indices",
    "nifty_valuation": "Nifty 50 trailing P/E and P/B - NSE Indices",
    "gold": "Gold in INR per troy oz (USD spot x INR/USD), daily",
    "debt": "Nifty Composite G-Sec Index, daily",
}
SCENARIO_START, SCENARIO_END = "2014-01-01", "2024-12-31"
MAX_GAP_DAYS = 10            # larger gaps inside the scenario window invalidate a daily series


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_provenance(data_dir=None):
    p = os.path.join(data_dir or DATA_DIR, PROVENANCE_FILE)
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        return json.load(f)


def write_provenance(prov, data_dir=None):
    with open(os.path.join(data_dir or DATA_DIR, PROVENANCE_FILE), "w") as f:
        json.dump(prov, f, indent=2, sort_keys=True)


def _read_csv(path, kind):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "date" not in df.columns:
        raise ValueError("missing 'date' column")
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df = df.set_index("date").sort_index()
    if df.index.has_duplicates:
        raise ValueError("duplicate dates")
    cols = ["pe", "pb"] if kind == "nifty_valuation" else ["value"]
    for c in cols:
        if c not in df.columns:
            raise ValueError(f"missing '{c}' column")
    df = df[cols].apply(pd.to_numeric, errors="raise")
    if df.isna().any().any() or (df <= 0).any().any() or not np.isfinite(df.values).all():
        raise ValueError("non-numeric, non-positive or non-finite values")
    return df


def validate_series(df, kind, start=SCENARIO_START, end=SCENARIO_END):
    """Coverage and frequency checks. Returns list of problems (empty = ok).
    The debt (G-Sec) series is allowed to start after the scenario start: the evaluation period is then the
    common period and is disclosed (scenario.py reports the actual period)."""
    problems = []
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    idx = df.index
    if idx[-1] < e - pd.Timedelta(days=10):
        problems.append(f"ends {idx[-1].date()}, before {e.date()}")
    if kind == "nifty_valuation":
        if idx[0] > s - pd.Timedelta(days=15):
            problems.append(f"starts {idx[0].date()}; needs a reading on/before {(s - pd.Timedelta(days=15)).date()}")
        return problems
    if kind != "debt" and idx[0] > s + pd.Timedelta(days=7):
        problems.append(f"starts {idx[0].date()}; must start within a week of {s.date()}")
    win = idx[(idx >= s - pd.Timedelta(days=10)) & (idx <= e)]
    if len(win) < 200:
        problems.append(f"only {len(win)} observations inside the scenario window")
    else:
        gaps = pd.Series(win).diff().dt.days.dropna()
        if gaps.max() > MAX_GAP_DAYS:
            problems.append(f"gap of {int(gaps.max())} days inside the scenario window (max {MAX_GAP_DAYS}); "
                            "series is not daily")
        med = gaps.median()
        if med > 3:
            problems.append(f"median spacing {med:.0f} days: series is not daily")
    return problems


def load_dataset(key, data_dir=None):
    """Return dict(status, df, provenance, problems). status: ok | missing | invalid."""
    d = data_dir or DATA_DIR
    prov = read_provenance(d).get(key)
    res = {"key": key, "description": DESCRIPTIONS[key], "status": "missing", "df": None, "provenance": prov,
           "problems": []}
    if not prov:
        res["problems"].append("no entry in data/provenance.json (import it with tools/ingest_raw.py; for debt supply --debt-file with the Nifty Composite G-Sec Index levels)")
        return res
    path = os.path.join(d, prov.get("file", ""))
    if not os.path.exists(path):
        res.update(status="invalid", problems=[f"file {prov.get('file')} not found"]); return res
    if sha256_file(path) != prov.get("sha256"):
        res.update(status="invalid", problems=["checksum mismatch: the file changed after import"]); return res
    for fld in ("source", "retrieved_on", "sha256"):
        if not prov.get(fld):
            res.update(status="invalid", problems=[f"provenance field '{fld}' missing"]); return res
    try:
        df = _read_csv(path, key)
    except Exception as exc:
        res.update(status="invalid", problems=[f"unreadable: {exc}"]); return res
    probs = validate_series(df, key)
    res.update(df=df, problems=probs, status="ok" if not probs else "invalid")
    if probs:
        res["df"] = None
    res["observed"] = {"start": str(df.index[0].date()), "end": str(df.index[-1].date()), "n_obs": int(len(df))}
    return res


_LOAD_ALL_MEMO = {}


def load_all(data_dir=None):
    """Load and validate every required dataset. The result is memoised in process memory (serverless-safe: no
    disk cache), keyed by the data directory and the provenance file's size and modification time, so any change to
    the registered data triggers a fresh checksum validation. Treat the returned frames as read-only."""
    d = data_dir or DATA_DIR
    try:
        st = os.stat(os.path.join(d, PROVENANCE_FILE))
        key = (os.path.abspath(d), st.st_size, st.st_mtime_ns)
    except OSError:
        key = None
    if key is not None and key in _LOAD_ALL_MEMO:
        return _LOAD_ALL_MEMO[key]
    out = {k: load_dataset(k, d) for k in REQUIRED}
    if key is not None:
        _LOAD_ALL_MEMO.clear()
        _LOAD_ALL_MEMO[key] = out
    return out


def dataset_report(loaded):
    """JSON-safe summary for the API (no DataFrames)."""
    out = {}
    for k, v in loaded.items():
        out[k] = {"description": v["description"], "status": v["status"], "problems": v["problems"],
                  "provenance": v["provenance"], "observed": v.get("observed")}
    return out


def fingerprint_data(data_dir=None):
    """Hash of the provenance file and every referenced data file (feeds the cache fingerprint)."""
    d = data_dir or DATA_DIR
    h = hashlib.sha256()
    prov = read_provenance(d)
    h.update(json.dumps(prov, sort_keys=True).encode())
    for k in sorted(prov):
        p = os.path.join(d, prov[k].get("file", ""))
        if os.path.exists(p):
            h.update(sha256_file(p).encode())
    return h.hexdigest()
