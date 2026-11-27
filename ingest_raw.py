#!/usr/bin/env python3
"""
ingest_raw.py - consolidate the RAW historical files into CLEANED datasets, with a data-quality report.

  python tools/ingest_raw.py --raw data/raw [--debt-file path/to/gsec.csv]

Rules (see data/DATA_QUALITY_REPORT.md for what was found in the actual files):
  * Raw files are never modified. They are copied byte-for-byte into data/raw/ and their SHA-256 is recorded.
  * Cleaning = parse, sort, drop exact structural problems (duplicate dates are REPORTED and the file is rejected
    if values conflict). No value is interpolated, filled, smoothed, winsorised or generated. Suspicious observations
    (zero-return days, large moves, stale prints) are listed in the report and KEPT.
  * Output: data/cleaned/*.csv and data/provenance.json (source, retrieved_on, sha256, coverage).
Inputs expected in --raw:
  nifty_2014.csv ... nifty_2024.csv   NSE index files: "IndexName","Date","Total Returns Index","Net Total Return Index"
  nifty-50-pe-history.csv             monthly Month,PE,PB,DividendYield  (header comments allowed)
  gold-price-history.csv              date,gold_inr_per_troy_oz,24k_inr_per_gram,gold_usd_per_troy_oz,inr_per_usd
  mfpro_nav_12funds_*.csv             scheme_code,scheme_name,date,nav  (long format)
  --debt-file                         Nifty Composite G-Sec Index levels (date + level). The "about" sheet alone
                                      carries no levels and cannot be used.
"""
import argparse, datetime as dt, glob, hashlib, json, os, re, shutil, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))
import numpy as np
import pandas as pd
import datasets as DS


def sha(path):
    return DS.sha256_file(path)


def _gaps(idx):
    g = pd.Series(idx).diff().dt.days.dropna()
    return {"max_gap_days": int(g.max()) if len(g) else None,
            "n_gaps_over_5_days": int((g > 5).sum()),
            "gap_dates_over_5_days": [str(pd.Timestamp(d).date()) for d in pd.Series(idx)[1:][g.values > 5][:25]]}


def _series_checks(s, name, big=0.06):
    r = s.pct_change(fill_method=None).dropna()
    return {"n_obs": int(len(s)), "start": str(s.index[0].date()), "end": str(s.index[-1].date()),
            "duplicate_dates": int(s.index.duplicated().sum()), "null_values": int(s.isna().sum()),
            "non_positive_values": int((s <= 0).sum()), **_gaps(s.index),
            "observations_per_year": {int(k): int(v) for k, v in s.groupby(s.index.year).size().items()},
            "weekday_counts_0mon": {int(k): int(v) for k, v in s.index.dayofweek.value_counts().sort_index().items()},
            "zero_return_days": [str(d.date()) for d in r.index[r == 0]],
            "moves_over_threshold": [{"date": str(d.date()), "return": round(float(x), 5)} for d, x in r[r.abs() > big].items()],
            "threshold_abs_return": big}


def ingest_tri(raw_dir):
    files = sorted(glob.glob(os.path.join(raw_dir, "nifty_20??.csv")))
    if not files:
        raise SystemExit("no nifty_YYYY.csv files found")
    frames, per_file = [], {}
    for f in files:
        d = pd.read_csv(f)
        need = {"IndexName", "Date", "Total Returns Index", "Net Total Return Index"}
        if not need <= set(d.columns):
            raise SystemExit(f"{f}: unexpected columns {list(d.columns)}")
        if set(d["IndexName"]) != {"NIFTY 50"}:
            raise SystemExit(f"{f}: IndexName is not NIFTY 50: {set(d['IndexName'])}")
        d["date"] = pd.to_datetime(d["Date"], format="%d %b %Y", errors="raise")
        yr = os.path.basename(f)[6:10]
        out_of_year = int((d["date"].dt.year != int(yr)).sum())
        per_file[os.path.basename(f)] = {"rows": int(len(d)), "first": str(d.date.min().date()), "last": str(d.date.max().date()),
                                         "rows_outside_file_year": out_of_year}
        frames.append(d)
    a = pd.concat(frames, ignore_index=True).sort_values("date")
    dup = a[a["date"].duplicated(keep=False)]
    if len(dup):
        conflict = dup.groupby("date")["Total Returns Index"].nunique().gt(1).any()
        if conflict:
            raise SystemExit("duplicate dates with conflicting values in the Nifty files; resolve manually")
        a = a.drop_duplicates("date")
    tri = a.set_index("date")["Total Returns Index"].astype(float)
    net = a.set_index("date")["Net Total Return Index"].astype(float)
    # repeated identical level on consecutive sessions (possible stale print) - reported, not altered
    same = tri[tri.diff() == 0]
    chk = _series_checks(tri, "nifty_tri")
    chk.update({"per_file": per_file, "consecutive_identical_levels": [str(d.date()) for d in same.index],
                "net_tri_exceeds_tri_days": int((net > tri).sum()),
                "column_used": "Total Returns Index (gross TRI). 'Net Total Return Index' is kept out of the study.",
                "weekend_sessions": [str(d.date()) for d in tri.index[tri.index.dayofweek >= 5]]})
    return pd.DataFrame({"value": tri}), chk


def ingest_valuation(raw_path):
    v = pd.read_csv(raw_path, comment="#")
    v.columns = [c.strip() for c in v.columns]
    if not {"Month", "PE", "PB"} <= set(v.columns):
        raise SystemExit("valuation file needs Month, PE, PB")
    per = pd.PeriodIndex(v["Month"], freq="M")
    if per.duplicated().any():
        raise SystemExit("duplicate months")
    # A monthly reading is dated at its month end (the file's last month is a partial month, dated at the 'as of' day).
    asof = None
    with open(raw_path) as f:
        for line in f:
            m = re.search(r"Data as of:\s*(\d{1,2} \w{3} \d{4})", line)
            if m:
                asof = pd.to_datetime(m.group(1))
    dates = per.to_timestamp(how="end").normalize()
    if asof is not None and per[-1] == asof.to_period("M"):
        dates = dates[:-1].append(pd.DatetimeIndex([asof]))
    out = pd.DataFrame({"pe": v["PE"].astype(float).values, "pb": v["PB"].astype(float).values,
                        "dividend_yield": v.get("DividendYield", pd.Series([np.nan] * len(v))).astype(float).values,
                        "reading_month": v["Month"].values}, index=dates)
    out.index.name = "date"
    steps = {int(k): int(c) for k, c in pd.Series((per[1:] - per[:-1]).map(lambda x: x.n)).value_counts().items()}
    chk = {"n_obs": int(len(out)), "start": str(out.index[0].date()), "end": str(out.index[-1].date()),
           "month_step_counts": steps, "null_values": int(out[["pe", "pb"]].isna().sum().sum()),
           "non_positive_values": int((out[["pe", "pb"]] <= 0).sum().sum()),
           "pe_min_max": [float(out.pe.min()), float(out.pe.max())], "pb_min_max": [float(out.pb.min()), float(out.pb.max())],
           "date_convention": "reading for month M is dated at the end of month M (last month: the file's 'as of' date)",
           "full_sample_mean_sd": {"pe": [float(out.pe.mean()), float(out.pe.std())], "pb": [float(out.pb.mean()), float(out.pb.std())]},
           "synopsis_constants": {"pe": [22.0, 5.0], "pb": [3.2, 0.6]},
           "pb_discontinuity_note": "P/B falls from 4.40 (2023-08) to 3.45 (2023-09) while P/E barely moves "
                                    "(22.0 to 22.2); a one-month break of about -0.9 in P/B. It is kept as supplied and flagged: "
                                    "it may reflect a change in the book-value basis by the provider and affects any P/B based signal.",
           "source_header": [l.strip() for l in open(raw_path) if l.startswith("#")]}
    return out, chk


def ingest_gold(raw_path):
    g = pd.read_csv(raw_path, parse_dates=["date"]).sort_values("date")
    if g["date"].duplicated().any():
        raise SystemExit("duplicate gold dates")
    s = g.set_index("date")["gold_inr_per_troy_oz"].astype(float)
    chk = _series_checks(s, "gold", big=0.05)
    chk["max_rel_error_usd_x_fx_vs_inr_oz"] = float((g.gold_usd_per_troy_oz * g.inr_per_usd / g.gold_inr_per_troy_oz - 1).abs().max())
    chk["column_used"] = "gold_inr_per_troy_oz (= USD spot x INR/USD; international spot in rupees, no import duty/GST/making charges)"
    chk["sparse_period_note"] = "2010-2013 observations are sparse (gaps up to 65 days); the study window (2014 onward) is daily."
    by = s.groupby(s.index.year).size()
    chk["years_with_fewer_than_200_obs"] = {int(k): int(v) for k, v in by[by < 200].items()}
    return pd.DataFrame({"value": s}), chk


def ingest_debt(path):
    d = pd.read_csv(path)
    dcol = next((c for c in d.columns if "date" in c.lower()), None)
    vcol = next((c for c in d.columns if c.strip().lower() in ("close", "price", "index value", "value", "total returns index", "closing index value")), None)
    if dcol is None or vcol is None:
        raise SystemExit(f"cannot identify date/level columns in {path}: {list(d.columns)}")
    idx = pd.to_datetime(d[dcol], dayfirst=True, errors="raise")
    s = pd.Series(pd.to_numeric(d[vcol].astype(str).str.replace(",", ""), errors="raise").values, index=idx).sort_index()
    if s.index.has_duplicates:
        raise SystemExit("duplicate dates in the debt file")
    # Weekend rows are not index sessions for a bond index. They are removed and listed (never silently).
    wk = s[s.index.dayofweek >= 5]
    s = s[s.index.dayofweek < 5]
    chk = _series_checks(s, "debt", big=0.01)
    chk["weekend_rows_excluded"] = [{"date": str(i.date()), "value": float(v)} for i, v in wk.items()]
    chk["column_used"] = f"{vcol}"
    return pd.DataFrame({"value": s}), chk


def ingest_funds(raw_path):
    d = pd.read_csv(raw_path, parse_dates=["date"])
    d = d.sort_values(["scheme_code", "date"])
    rows, funds = [], {}
    for code, g in d.groupby("scheme_code"):
        s = g.set_index("date")["nav"]
        nm = g["scheme_name"].iloc[0]
        r = s.pct_change(fill_method=None).dropna()
        bd = int(np.busday_count(s.index[0].date(), s.index[-1].date())) + 1
        funds[str(code)] = {"scheme_name": nm, "n_obs": int(len(s)), "start": str(s.index[0].date()), "end": str(s.index[-1].date()),
                            "duplicate_dates": int(s.index.duplicated().sum()), "non_positive_nav": int((s <= 0).sum()),
                            "null_nav": int(s.isna().sum()), "scheme_name_variants": int(g["scheme_name"].nunique()),
                            "expected_business_days": bd, "coverage_vs_business_days": round(len(s) / bd, 4),
                            "weekend_rows": int((s.index.dayofweek >= 5).sum()), **{k: v for k, v in _gaps(s.index).items()},
                            "returns_over_8pct": [{"date": str(x.date()), "return": round(float(y), 4)} for x, y in r[r.abs() > 0.08].items()],
                            "zero_return_days": int((r == 0).sum())}
    return d, funds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(ROOT, "data", "raw"))
    ap.add_argument("--debt-file")
    ap.add_argument("--retrieved-on", default=dt.date.today().isoformat())
    a = ap.parse_args()
    data_dir = DS.DATA_DIR
    raw_dir = os.path.abspath(a.raw)
    if not a.debt_file:
        # auto-detect: drop the G-Sec level history (a CSV with a date and an index-level column) into data/raw/.
        # The metadata "about" sheet is not data and is ignored.
        cands = [p for p in glob.glob(os.path.join(raw_dir, "*.csv"))
                 if any(t in os.path.basename(p).lower() for t in ("g-sec", "gsec")) and "about" not in os.path.basename(p).lower()]
        if len(cands) == 1:
            a.debt_file = cands[0]
        elif len(cands) > 1:
            raise SystemExit(f"several candidate G-Sec files in {raw_dir}: {cands}; pass --debt-file explicitly")
    os.makedirs(os.path.join(data_dir, "raw"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "cleaned"), exist_ok=True)
    if a.debt_file:
        dst = os.path.join(data_dir, "raw", os.path.basename(a.debt_file))
        if os.path.abspath(a.debt_file) != os.path.abspath(dst):
            shutil.copy2(a.debt_file, dst)

    report = {"generated_utc": dt.datetime.now(dt.timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds") + "Z", "datasets": {}, "raw_files": {}}
    prov = {}

    def write(key, df, fname, source, chk, raw_files, note=""):
        path = os.path.join(data_dir, "cleaned", fname)
        df.to_csv(path, date_format="%Y-%m-%d")
        gp = pd.Series(df.index).diff().dt.days.dropna()
        prov[key] = {"file": f"cleaned/{fname}", "source": source, "retrieved_on": a.retrieved_on, "sha256": sha(path),
                     "start": str(df.index[0].date()), "end": str(df.index[-1].date()), "n_obs": int(len(df)),
                     "median_spacing_days": float(gp.median()), "max_gap_days": int(gp.max()),
                     "raw_files": {os.path.basename(p): sha(p) for p in raw_files}, "note": note}
        report["datasets"][key] = chk

    tri, c = ingest_tri(raw_dir)
    write("nifty_tri", tri, "nifty_tri.csv",
          "User-supplied annual files nifty_2014..nifty_2024.csv in NSE Indices format; column 'Total Returns Index' "
          "(original download site not stated by the supplier)", c, sorted(glob.glob(os.path.join(raw_dir, "nifty_20??.csv"))),
          "Gross total-return index. Net Total Return Index not used.")
    val_raw = glob.glob(os.path.join(raw_dir, "*pe-history*.csv"))[0]
    val, c = ingest_valuation(val_raw)
    write("nifty_valuation", val, "nifty_valuation.csv", "Downstox, https://downstox.com/nifty-pe/nifty-50 (as stated in the file header); "
          "monthly Nifty 50 P/E, P/B, dividend yield", c, [val_raw], "Monthly; see date convention in the quality report.")
    gold_raw = glob.glob(os.path.join(raw_dir, "*gold-price-history*.csv"))[0]
    gold, c = ingest_gold(gold_raw)
    write("gold", gold, "gold.csv", "User-supplied gold-price-history.csv (INR per troy oz, derived from USD spot x INR/USD; "
          "original source not stated in the file)", c, [gold_raw], "Spot parity in INR; excludes duties and taxes.")
    if a.debt_file:
        debt, c = ingest_debt(a.debt_file)
        write("debt", debt, "debt.csv", "User-supplied Investing.com export 'Nifty GS Composite' (column 'Price' = daily close), "
              "5 Apr 2016 to 31 Dec 2024. Provider label only; NOT confirmed against NSE as the Total Return Index "
              "(see docs/DATA_PROVENANCE.md)", c, [a.debt_file],
              "Weekend rows removed (listed in the quality report). Open/High/Low/Change% columns not used.")
    else:
        report["datasets"]["debt"] = {"status": "NOT SUPPLIED",
                                      "detail": "Only NIFTY-COMPOSITE-G-SEC-INDEX-about.csv (metadata: price history from 2015-10-13, "
                                                "2515 sessions to 2026-11-26) was supplied. It contains no index levels. No debt "
                                                "series has been created or substituted."}
    nav_raw = glob.glob(os.path.join(raw_dir, "mfpro_nav_*.csv"))[0]
    nav, funds = ingest_funds(nav_raw)
    npath = os.path.join(data_dir, "cleaned", "fund_nav.csv")
    nav.to_csv(npath, index=False, date_format="%Y-%m-%d")
    prov["fund_nav"] = {"file": "cleaned/fund_nav.csv", "source": "TIGZIG daily NAV download, file mfpro_nav_12funds_2026-11-26.csv "
                        "(as stated by the project author)", "retrieved_on": a.retrieved_on, "sha256": sha(npath),
                        "start": str(nav.date.min().date()), "end": str(nav.date.max().date()), "n_obs": int(len(nav)),
                        "n_schemes": int(nav.scheme_code.nunique()), "raw_files": {os.path.basename(nav_raw): sha(nav_raw)}}
    report["datasets"]["fund_nav"] = {"n_rows": int(len(nav)), "funds": funds}
    for p in sorted(glob.glob(os.path.join(raw_dir, "*"))):
        if os.path.isfile(p):
            report["raw_files"][os.path.basename(p)] = sha(p)
    DS.write_provenance(prov, data_dir)
    with open(os.path.join(data_dir, "data_quality_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("written:", sorted(prov))


if __name__ == "__main__":
    main()
