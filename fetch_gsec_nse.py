"""
fetch_gsec_nse.py - download NIFTY COMPOSITE G-SEC INDEX Total Return Index levels from NSE Indices and validate them.

RUN THIS ON YOUR OWN COMPUTER (it needs internet access to www.niftyindices.com). The build environment used to
prepare this project could not reach that host (HTTP 403 from the sandbox proxy), so this tool is UNVERIFIED against
the live endpoint: its parsing is tested only against a mock response in the shape niftyindices.com is known to use.
If the live response differs, the tool stops with a clear message; it never guesses and never fills data.

    pip install requests pandas
    python tools/fetch_gsec_nse.py --start 2015-10-13 --end 2026-11-26
    python tools/ingest_raw.py            # picks up the CSV written to data/raw/ (name contains "g-sec")

What it does
  1. POSTs to https://www.niftyindices.com/Backpage.aspx/getTotalReturnIndexString in yearly chunks.
  2. Saves every original response verbatim:  data/raw/NIFTY-COMPOSITE-G-SEC-INDEX_TRI_original_responses.json
  3. Checks identity (index name in every record, when the response carries one) and that the series is the TOTAL
     RETURN index (the endpoint is the Total Return Index endpoint; the field must be a total-returns level).
  4. Checks date coverage, duplicates, non-numeric / non-positive values, large gaps, extreme daily moves.
     Nothing is interpolated, filled or substituted. Duplicate dates with different values abort the run.
  5. Writes the cleaned CSV  data/raw/NIFTY-COMPOSITE-G-SEC-INDEX-TRI-g-sec.csv  (Date,Total Returns Index) and
     data/raw/gsec_fetch_record.json with source, retrieval date, row count and SHA-256 of both files.
"""
import argparse, datetime as dt, hashlib, json, os, re, sys

import pandas as pd

URL = "https://www.niftyindices.com/Backpage.aspx/getTotalReturnIndexString"
INDEX_NAME = "NIFTY COMPOSITE G-SEC INDEX"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_JSON = "NIFTY-COMPOSITE-G-SEC-INDEX_TRI_original_responses.json"
CSV_NAME = "NIFTY-COMPOSITE-G-SEC-INDEX-TRI-g-sec.csv"
RECORD = "gsec_fetch_record.json"


class FetchError(Exception):
    pass


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(p):
    with open(p, "rb") as f:
        return sha256_bytes(f.read())


def request_body(start, end, name=INDEX_NAME):
    f = lambda d: pd.Timestamp(d).strftime("%d-%b-%Y")
    cinfo = "{'name':'%s','startDate':'%s','endDate':'%s','indexName':'%s'}" % (name, f(start), f(end), name)
    return {"cinfo": cinfo}


def _records(resp_json):
    """The endpoint wraps a JSON string in {"d": "..."}; accept a string or an already-decoded list."""
    if not isinstance(resp_json, dict) or "d" not in resp_json:
        raise FetchError("unexpected response shape: no 'd' field")
    d = resp_json["d"]
    if isinstance(d, str):
        d = json.loads(d) if d.strip() else []
    if not isinstance(d, list):
        raise FetchError("unexpected response shape: 'd' is not a list")
    return d


def _key(rec, *needles):
    for k in rec:
        kl = re.sub(r"[^a-z]", "", k.lower())
        if all(n in kl for n in needles):
            return k
    return None


def parse_response(resp_json, expect_name=INDEX_NAME):
    """-> DataFrame(date, value) from one response. Verifies the index name when the records carry one."""
    rows = _records(resp_json)
    out = []
    for rec in rows:
        kd = _key(rec, "date")
        kv = _key(rec, "total", "return") or _key(rec, "tri")
        if kd is None or kv is None:
            raise FetchError(f"cannot find date / total-returns fields in record keys {list(rec)}; "
                             "this is not the Total Return Index payload, stopping")
        kn = _key(rec, "index", "name") or _key(rec, "name")
        if kn is not None and str(rec[kn]).strip().upper() != expect_name:
            raise FetchError(f"record index name {rec[kn]!r} is not {expect_name!r}; refusing to use it")
        out.append((rec[kd], rec[kv]))
    return pd.DataFrame(out, columns=["date_raw", "value_raw"])


def clean(frames, start, end):
    """Concatenate parsed chunks and validate. Returns (series, report). No interpolation, no filling."""
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["date_raw", "value_raw"])
    rep = {"rows_downloaded": int(len(df))}
    if df.empty:
        raise FetchError("no observations returned")
    df["date"] = pd.to_datetime(df["date_raw"], errors="coerce", dayfirst=True)
    df["value"] = pd.to_numeric(df["value_raw"].astype(str).str.replace(",", "").str.strip(), errors="coerce")
    bad = df[df["date"].isna() | df["value"].isna() | (df["value"] <= 0)]
    rep["invalid_rows"] = int(len(bad))
    rep["invalid_examples"] = bad.head(5)[["date_raw", "value_raw"]].astype(str).values.tolist()
    if len(bad):
        raise FetchError(f"{len(bad)} invalid rows (unparseable date, non-numeric or non-positive level); "
                         f"examples {rep['invalid_examples']}. Not repaired.")
    dups = df[df.duplicated("date", keep=False)]
    rep["duplicate_date_rows"] = int(len(dups))
    if len(dups):
        conflict = dups.groupby("date")["value"].nunique()
        if (conflict > 1).any():
            raise FetchError(f"duplicate dates with conflicting values: {conflict[conflict > 1].index.strftime('%Y-%m-%d').tolist()[:5]}")
        df = df.drop_duplicates("date")                       # identical repeats (e.g. chunk overlap) only
    s = df.set_index("date")["value"].sort_index()
    gaps = s.index.to_series().diff().dt.days.dropna()
    ret = s.pct_change().dropna()
    rep.update({"rows_clean": int(len(s)), "first_date": str(s.index[0].date()), "last_date": str(s.index[-1].date()),
                "requested_start": str(pd.Timestamp(start).date()), "requested_end": str(pd.Timestamp(end).date()),
                "starts_within_7_days_of_request": bool((s.index[0] - pd.Timestamp(start)).days <= 7),
                "ends_within_7_days_of_request": bool((pd.Timestamp(end) - s.index[-1]).days <= 7),
                "max_gap_calendar_days": int(gaps.max()) if len(gaps) else 0,
                "gaps_over_7_days": int((gaps > 7).sum()),
                "max_abs_daily_return": float(ret.abs().max()), "days_abs_return_over_1pct": int((ret.abs() > 0.01).sum()),
                "interpolated_or_filled_values": 0})
    return s, rep


def fetch(start, end, session=None, years_per_chunk=1):
    import requests
    sess = session or requests.Session()
    sess.headers.update({"Content-Type": "application/json; charset=UTF-8", "Accept": "application/json",
                         "User-Agent": "Mozilla/5.0", "Referer": "https://www.niftyindices.com/reports/historical-data",
                         "X-Requested-With": "XMLHttpRequest"})
    raws, frames = [], []
    a = pd.Timestamp(start)
    while a <= pd.Timestamp(end):
        b = min(a + pd.DateOffset(years=years_per_chunk) - pd.Timedelta(days=1), pd.Timestamp(end))
        body = request_body(a, b)
        r = sess.post(URL, data=json.dumps(body), timeout=60)
        if r.status_code != 200:
            raise FetchError(f"HTTP {r.status_code} for {a.date()}..{b.date()}: {r.text[:200]!r}")
        try:
            j = r.json()
        except ValueError:
            raise FetchError(f"non-JSON response for {a.date()}..{b.date()}: {r.text[:200]!r}")
        raws.append({"request": body, "http_status": r.status_code, "response": j})
        frames.append(parse_response(j))
        a = b + pd.Timedelta(days=1)
    return raws, frames


def save(raws, series, rep, out_dir, retrieved_on):
    os.makedirs(out_dir, exist_ok=True)
    pj = os.path.join(out_dir, RAW_JSON)
    with open(pj, "w") as f:
        json.dump(raws, f, indent=1)
    pc = os.path.join(out_dir, CSV_NAME)
    pd.DataFrame({"Date": series.index.strftime("%d-%b-%Y"), "Total Returns Index": series.values}).to_csv(pc, index=False)
    rec = {"source": URL, "index": INDEX_NAME, "series": "Total Return Index (coupon income included by construction)",
           "retrieved_on": retrieved_on, "rows": rep["rows_clean"], "validation": rep,
           "files": {RAW_JSON: {"sha256": sha256_file(pj)}, CSV_NAME: {"sha256": sha256_file(pc)}},
           "interpolation": "none", "substitution": "none"}
    with open(os.path.join(out_dir, RECORD), "w") as f:
        json.dump(rec, f, indent=1)
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2015-10-13")
    ap.add_argument("--end", default="2026-11-26")
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "raw"))
    a = ap.parse_args(argv)
    try:
        raws, frames = fetch(a.start, a.end)
        s, rep = clean(frames, a.start, a.end)
    except FetchError as exc:
        print(f"STOPPED: {exc}\nNo file was written. Nothing was filled or substituted.", file=sys.stderr)
        return 2
    rec = save(raws, s, rep, a.out, dt.date.today().isoformat())
    print(json.dumps({k: rec[k] for k in ("source", "retrieved_on", "rows", "files")}, indent=1))
    print(json.dumps(rep, indent=1))
    print("\nNext: python tools/ingest_raw.py   (then review data/DATA_QUALITY_REPORT.md before committing)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
