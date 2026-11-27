#!/usr/bin/env python3
"""
Import a historical dataset into data/ with provenance and a SHA-256 checksum. Run this on a machine that
has the raw files / network access. The tool never invents values: it only reformats what you give it.

  python tools/build_dataset.py status
  python tools/build_dataset.py search --q "gold etf"                      # mfapi.in scheme search (network)
  python tools/build_dataset.py import-csv --key nifty_tri --file NIFTY50_TRI.csv \
        --source "NSE Indices, Nifty 50 Total Returns Index, historical data download" \
        --retrieved-on 2026-11-27 [--date-col Date] [--value-col Close] [--date-format "%d-%b-%Y"]
  python tools/build_dataset.py import-csv --key nifty_valuation --file PE_PB.csv --source "..." --retrieved-on ... \
        [--pe-col "P/E"] [--pb-col "P/B"]
  python tools/build_dataset.py import-mfapi --key gold --code <AMFI code> --expect "Gold" --source-note "why this proxy"

Keys: nifty_tri, nifty_valuation, gold, debt, cash  (see datasets.py for what each must be).
The Nifty 50 *Total Returns* Index must be used (not the price index); you are responsible for choosing the right
file, the tool records what you declare in --source.
"""
import argparse, datetime as dt, json, os, sys, urllib.parse, urllib.request
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))
import numpy as np, pandas as pd
import datasets as DS


def _std_frame(df, key, a):
    cols = {c.strip().lower(): c for c in df.columns}
    dcol = (a.date_col or next((c for c in df.columns if "date" in c.lower()), None))
    if dcol is None:
        sys.exit("cannot find the date column; pass --date-col")
    d = pd.to_datetime(df[dcol], format=a.date_format, dayfirst=not a.date_format, errors="raise")
    out = pd.DataFrame(index=d)
    if key == "nifty_valuation":
        pc = a.pe_col or next((c for c in df.columns if c.strip().lower() in ("p/e", "pe")), None)
        bc = a.pb_col or next((c for c in df.columns if c.strip().lower() in ("p/b", "pb")), None)
        if pc is None or bc is None:
            sys.exit("cannot find P/E and P/B columns; pass --pe-col and --pb-col")
        out["pe"] = pd.to_numeric(df[pc].astype(str).str.replace(",", ""), errors="raise").values
        out["pb"] = pd.to_numeric(df[bc].astype(str).str.replace(",", ""), errors="raise").values
    else:
        vc = a.value_col or next((c for c in df.columns if c.strip().lower() in ("close", "value", "nav", "total returns index")), None)
        if vc is None:
            sys.exit("cannot find the value column; pass --value-col")
        out["value"] = pd.to_numeric(df[vc].astype(str).str.replace(",", ""), errors="raise").values
    out = out.sort_index()
    out.index.name = "date"
    return out


def _register(key, out, source, retrieved_on, extra):
    os.makedirs(DS.DATA_DIR, exist_ok=True)
    fname = f"{key}.csv"
    path = os.path.join(DS.DATA_DIR, fname)
    out.to_csv(path, date_format="%Y-%m-%d")
    gaps = pd.Series(out.index).diff().dt.days.dropna()
    prov = DS.read_provenance()
    prov[key] = {"file": fname, "source": source, "retrieved_on": retrieved_on, "sha256": DS.sha256_file(path),
                 "start": str(out.index[0].date()), "end": str(out.index[-1].date()), "n_obs": int(len(out)),
                 "median_spacing_days": float(gaps.median()) if len(gaps) else None,
                 "max_gap_days": int(gaps.max()) if len(gaps) else None, "imported_utc": dt.datetime.utcnow().isoformat() + "Z",
                 **extra}
    DS.write_provenance(prov)
    print(json.dumps(prov[key], indent=2))
    r = DS.load_dataset(key)
    print("VALIDATION:", r["status"], r["problems"])


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    s = sub.add_parser("search"); s.add_argument("--q", required=True)
    c = sub.add_parser("import-csv")
    for n in ("key", "file", "source", "retrieved_on"): c.add_argument("--" + n.replace("_", "-"), required=True, dest=n)
    for n in ("date_col", "value_col", "date_format", "pe_col", "pb_col"): c.add_argument("--" + n.replace("_", "-"), dest=n)
    m = sub.add_parser("import-mfapi")
    m.add_argument("--key", required=True); m.add_argument("--code", required=True)
    m.add_argument("--expect", required=True, help="text that must appear in the provider's scheme name")
    m.add_argument("--source-note", required=True)
    a = ap.parse_args()
    if a.cmd == "status":
        for k, v in DS.load_all().items():
            print(f"{k:16s} {v['status']:8s} {v['problems']} {v.get('observed') or ''}")
        return
    if a.cmd == "search":
        with urllib.request.urlopen("https://api.mfapi.in/mf/search?q=" + urllib.parse.quote(a.q), timeout=30) as r:
            for row in json.load(r)[:50]:
                print(row)
        return
    if a.cmd == "import-csv":
        if a.key not in DS.REQUIRED: sys.exit(f"key must be one of {DS.REQUIRED}")
        raw = pd.read_csv(a.file)
        out = _std_frame(raw, a.key, a)
        if out.index.has_duplicates:
            sys.exit("duplicate dates in the raw file; clean them explicitly, the tool will not pick one")
        _register(a.key, out, a.source, a.retrieved_on, {"raw_file": os.path.basename(a.file), "raw_sha256": DS.sha256_file(a.file)})
        return
    if a.cmd == "import-mfapi":
        with urllib.request.urlopen(f"https://api.mfapi.in/mf/{a.code}", timeout=60) as r:
            j = json.load(r)
        name = (j.get("meta") or {}).get("scheme_name", "")
        print("provider scheme name:", name)
        if a.expect.lower() not in name.lower():
            sys.exit(f"REJECTED: '{a.expect}' not in provider scheme name")
        df = pd.DataFrame(j["data"]); df["date"] = pd.to_datetime(df["date"], dayfirst=True)
        out = pd.DataFrame({"value": pd.to_numeric(df["nav"]).values}, index=df["date"]).sort_index()
        out = out[out["value"] > 0]; out.index.name = "date"
        out = out[~out.index.duplicated(keep="last")]
        _register(a.key, out, f"mfapi.in (AMFI NAV) scheme {a.code}: {name}. {a.source_note}", dt.date.today().isoformat(),
                  {"scheme_code": a.code, "scheme_name": name})


if __name__ == "__main__":
    main()
