"""TEST FIXTURES ONLY. Synthetic benchmark datasets for exercising the code. Never shipped as data."""
import os, datetime as dt
import numpy as np, pandas as pd
import datasets as DS

def _levels(idx, mu, sig, rng, start=1000.0, crash=None):
    r = rng.normal(mu / 252, sig / np.sqrt(252), len(idx))
    if crash is not None:
        a, b, size = crash
        mask = (idx >= a) & (idx <= b)
        r[mask] = (1 + size) ** (1 / mask.sum()) - 1
    return pd.Series(start * np.cumprod(1 + r), index=idx)

def make_fixture_dir(path, seed=7, drop_frac=0.02, skip=(), debt_start=None):
    rng = np.random.default_rng(seed)
    os.makedirs(path, exist_ok=True)
    full = pd.bdate_range("2010-06-01", "2025-01-31")
    series = {
        "nifty_tri": _levels(full, 0.13, 0.17, rng, crash=(pd.Timestamp("2020-02-20"), pd.Timestamp("2020-03-25"), -0.35)),
        "gold": _levels(full, 0.08, 0.14, rng, 3000),
        "debt": _levels(full, 0.07, 0.03, rng, 20),
    }
    prov = {}
    for k, s in series.items():
        if k in skip: continue
        keep = rng.random(len(s)) > drop_frac           # each series misses different random days
        keep[0] = keep[-1] = True
        if k == "debt" and debt_start is not None:
            keep &= (s.index >= pd.Timestamp(debt_start))
            keep[np.argmax(keep) if keep.any() else 0] = True
        s = s[keep]
        df = pd.DataFrame({"value": s.values}, index=s.index); df.index.name = "date"
        p = os.path.join(path, f"{k}.csv"); df.to_csv(p, date_format="%Y-%m-%d")
        prov[k] = {"file": f"{k}.csv", "source": "TEST FIXTURE (synthetic)", "retrieved_on": "2026-01-01", "sha256": DS.sha256_file(p)}
    if "nifty_valuation" not in skip:
        me = pd.date_range("2001-06-30", "2025-01-31", freq="ME")
        v = pd.DataFrame({"pe": rng.uniform(15, 36, len(me)), "pb": rng.uniform(2.5, 4.6, len(me))}, index=me)
        v.index.name = "date"
        p = os.path.join(path, "nifty_valuation.csv"); v.to_csv(p, date_format="%Y-%m-%d")
        prov["nifty_valuation"] = {"file": "nifty_valuation.csv", "source": "TEST FIXTURE (synthetic)", "retrieved_on": "2026-01-01", "sha256": DS.sha256_file(p)}
    DS.write_provenance(prov, path)
    return path
