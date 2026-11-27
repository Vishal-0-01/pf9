"""
data_fetcher.py - NAV retrieval, scheme-identity validation and preprocessing
==============================================================================

What this module guarantees
---------------------------
1. IDENTITY.  Every scheme is checked against the name returned by the data
   provider (universe.validate_scheme_identity).  A scheme whose identity does
   not validate is REJECTED, recorded in the report, and never used.
2. NO CROSS-FUND FILLING.  Each fund's NAV is kept on that fund's own dates.
   Nothing is forward-filled, and no fund's NAV is aligned to another fund's
   calendar before returns are computed.
3. RETURNS PER FUND.  Daily returns are computed independently for each fund
   with pct_change(fill_method=None) on that fund's own observed NAVs.  A
   return over a gap of more than MAX_GAP_CALENDAR_DAYS, or with |r| above
   MAX_ABS_DAILY_RETURN, is set to NaN (excluded), never replaced by zero.
4. COVARIANCE ON OVERLAP ONLY.  The returns table is the outer join of the
   per-fund series, with NaN where a fund has no valid return.  The optimiser
   estimates covariance from pairwise-complete observations only and refuses
   pairs with too little overlap (optimizer.build_cov_from_returns).
5. NO LIVE FETCHING AND NO SYNTHETIC DATA.  The application reads only the bundled, checksummed NAV file
   (data/cleaned/fund_nav.csv, source "bundled_dataset").  No network provider (mftool, AMFI, mfapi) is imported
   or called.  If the file is missing or fails its checksum or identity validation, DataUnavailableError is
   raised and the API reports "unavailable"; nothing is substituted.
6. TRANSPARENCY.  Per-fund observation counts, date ranges and missing-data
   statistics are returned in the "quality" report.
"""

import logging
import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from universe import (SCHEME_SPECS, SELECTED_SCHEMES, FUNDS_DEFAULT,  # noqa: F401
                      MARKET_PROXY, validate_scheme_identity)

logger = logging.getLogger(__name__)

# ── CONFIGURATION ────────────────────────────────────────────────────────────

# Minimum return observations for a fund to enter the live model (~3 years).
MIN_TRADING_DAYS: int = 756

# Estimation window cap (~5 years of trading days).  None = use all history.
MAX_LOOKBACK_DAYS = 1260

# A return spanning more than this many calendar days (a data gap beyond
# weekends and holidays) is treated as missing, not as a normal daily return.
MAX_GAP_CALENDAR_DAYS: int = 7

# Single-day returns beyond this magnitude are treated as data errors.
MAX_ABS_DAILY_RETURN: float = 0.15

# Fewest funds with which the optimiser can respect 3%-25% weight bounds.
MIN_FUNDS_REQUIRED: int = 4


class DataUnavailableError(RuntimeError):
    """Live data could not be obtained or validated.  Carries a report."""

    def __init__(self, message, report=None):
        super().__init__(message)
        self.report = report or {}


# ── PROVIDER RESPONSE PARSING ─────────────────────────────────────────────────

def extract_meta(raw):
    """Pull scheme metadata out of a provider response (mftool or mfapi shape)."""
    if not isinstance(raw, dict):
        return {}
    meta = {}
    nested = raw.get("meta") if isinstance(raw.get("meta"), dict) else {}
    for key in ("scheme_name", "scheme_category", "fund_house", "scheme_code", "scheme_type"):
        val = raw.get(key, nested.get(key))
        if val not in (None, ""):
            meta[key] = val
    return meta


def parse_nav(raw):
    """
    Convert a provider response to (nav Series on its own dates, parse_stats).
    Returns (None, stats) if no usable data.
    """
    stats = {"n_raw_rows": 0, "n_unparseable": 0, "n_nonpositive": 0,
             "n_weekend_dropped": 0, "n_duplicate_dates": 0}
    if not isinstance(raw, dict) or not raw.get("data"):
        return None, stats
    df = pd.DataFrame(raw["data"])
    stats["n_raw_rows"] = int(len(df))
    if "date" not in df.columns or "nav" not in df.columns:
        return None, stats
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
    bad = df["date"].isna() | df["nav"].isna()
    stats["n_unparseable"] = int(bad.sum())
    df = df[~bad]
    nonpos = df["nav"] <= 0
    stats["n_nonpositive"] = int(nonpos.sum())
    df = df[~nonpos]
    df = df.sort_values("date")
    wk = df["date"].dt.dayofweek >= 5
    stats["n_weekend_dropped"] = int(wk.sum())
    df = df[~wk]
    dup = df["date"].duplicated(keep="last")
    stats["n_duplicate_dates"] = int(dup.sum())
    df = df[~dup]
    if df.empty:
        return None, stats
    return df.set_index("date")["nav"].astype(float), stats


# ── RETURNS (PER FUND, NO FILLING) ────────────────────────────────────────────

def returns_from_nav(nav):
    """
    Daily returns for ONE fund from its own observed NAVs.

    pct_change(fill_method=None) is applied to the fund's own series (already
    free of NaN), so each return spans exactly two consecutive observed NAVs.
    Returns over gaps longer than MAX_GAP_CALENDAR_DAYS and implausible returns
    are set to NaN.  Returns (returns Series, stats dict).
    """
    s = nav.dropna()
    r = s.pct_change(fill_method=None)
    gap = s.index.to_series().diff().dt.days
    gap_bad = (gap > MAX_GAP_CALENDAR_DAYS) & r.notna()
    out_bad = (r.abs() > MAX_ABS_DAILY_RETURN) & ~gap_bad
    r = r.iloc[1:].copy()
    gap_bad = gap_bad.iloc[1:]
    out_bad = out_bad.iloc[1:]
    r[gap_bad | out_bad] = np.nan
    stats = {"n_gap_excluded": int(gap_bad.sum()), "n_outlier_excluded": int(out_bad.sum()),
             "max_gap_calendar_days": int(gap.max()) if len(gap.dropna()) else 0}
    return r, stats


def nav_quality(nav, parse_stats, ret_stats, returns):
    """Per-fund observation count, date range and missing-data statistics."""
    first, last = nav.index[0], nav.index[-1]
    expected_bdays = int(np.busday_count(first.date(), last.date())) + 1
    n_nav = int(len(nav))
    return {
        "n_nav_observations": n_nav,
        "first_date": str(first.date()),
        "last_date": str(last.date()),
        "expected_business_days": expected_bdays,
        "missing_business_days": max(0, expected_bdays - n_nav),
        "coverage_ratio": round(n_nav / expected_bdays, 4) if expected_bdays else None,
        "n_valid_returns": int(returns.notna().sum()),
        **ret_stats,
        **parse_stats,
    }


# ── BUNDLED NAV DATASET (the study's reproducible source) ────────────────────

class BundledProvider:
    """
    Serves NAV histories from the project's cleaned NAV file (data/cleaned/fund_nav.csv, checksummed in
    data/provenance.json) in the same shape as the live provider, so that exactly the same identity validation,
    parsing and return construction is applied. Returns None for a code that is not in the file.
    """

    def __init__(self, path, expected_sha256=None):
        import hashlib
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        self.sha256 = h.hexdigest()
        if expected_sha256 and expected_sha256 != self.sha256:
            raise DataUnavailableError("bundled NAV file checksum does not match data/provenance.json; the file "
                                       "changed after import")
        df = pd.read_csv(path, dtype={"scheme_code": str})
        self._by_code = {c: g for c, g in df.groupby("scheme_code")}

    def get_scheme_historical_nav(self, code):
        g = self._by_code.get(str(code))
        if g is None:
            return None
        g = g.sort_values("date", ascending=False)
        return {"scheme_name": g["scheme_name"].iloc[0], "scheme_code": str(code),
                "data": [{"date": pd.Timestamp(d).strftime("%d-%m-%Y"), "nav": repr(float(v))}
                         for d, v in zip(g["date"], g["nav"])]}


# ── FETCH ─────────────────────────────────────────────────────────────────────

def _fetch_raw(mf, code):
    """Call the provider; returns (raw dict or None, error string or None)."""
    try:
        raw = mf.get_scheme_historical_nav(code)
    except Exception as exc:  # provider/network failure
        return None, f"provider call failed: {type(exc).__name__}: {exc}"
    if raw is None:
        return None, "provider returned nothing"
    if not isinstance(raw, dict):
        return None, f"unexpected provider response type {type(raw).__name__}"
    return raw, None


def _spec_for(label, code):
    base = SCHEME_SPECS.get(label)
    if base is not None:
        return {**base, "code": str(code)} if str(code) != base["code"] else base
    # Unknown label (custom universe passed by a caller): require the label as AMC token.
    return {"code": str(code), "amc": [label.lower()]}


def fetch_nav_data(schemes=None, mf=None, strict=None, include_proxy=False, source_label="injected_provider"):
    """
    Fetch, validate and prepare NAV data for the universe.

    schemes : dict label -> code (default: universe.SELECTED_SCHEMES)
    mf      : provider object with get_scheme_historical_nav(code) (required; production passes the BundledProvider, tests pass a fixture provider)
    strict  : if True (default), raise DataUnavailableError
              when any scheme is rejected by identity validation or cannot be
              fetched.  If False, proceed with the schemes that validated and
              flag the dataset incomplete.

    Returns dict:
      returns_df     per-fund returns, outer join, NaN where missing (<= MAX_LOOKBACK_DAYS each)
      nav_full       per-fund NAVs on their own dates, full history, no filling
      proxy_returns  market-proxy returns Series or None
      validation     label -> {code, scheme_name, ok, reasons, status}
      quality        label -> per-fund statistics (loaded funds)
      rejected / failed / excluded_short   lists
      dataset_complete  True only if all schemes validated, loaded and long enough
      retrieved_at, source (label)
    Raises DataUnavailableError on failure.
    """
    if schemes is None:
        schemes = SELECTED_SCHEMES
    if strict is None:
        strict = True
    if mf is None:
        raise DataUnavailableError("no NAV provider supplied: the application serves only the bundled dataset "
                                   "(live NAV fetching has been removed)")

    validation, quality = {}, {}
    rejected, failed, excluded_short = [], [], []
    ret_map, nav_map = {}, {}

    def handle(label, code, spec):
        raw, err = _fetch_raw(mf, code)
        entry = {"code": str(code), "scheme_name": None, "ok": False, "reasons": [], "status": ""}
        validation[label] = entry
        if err or not raw:
            entry["status"] = "fetch_failed"
            entry["reasons"] = [err or "empty response"]
            failed.append(label)
            return None
        meta = extract_meta(raw)
        entry["scheme_name"] = meta.get("scheme_name")
        ok, reasons = validate_scheme_identity(label, code, meta, spec)
        entry["ok"], entry["reasons"] = ok, reasons
        if not ok:
            entry["status"] = "rejected_identity"
            rejected.append(label)
            logger.error("REJECTED %s (%s): %s | name=%r", label, code, "; ".join(reasons), meta.get("scheme_name"))
            return None
        nav, pstats = parse_nav(raw)
        if nav is None or len(nav) < 3:
            entry["status"] = "fetch_failed"
            entry["ok"] = False
            entry["reasons"] = ["no usable NAV rows"]
            failed.append(label)
            return None
        entry["status"] = "validated"
        return nav, pstats

    for label, code in schemes.items():
        res = handle(label, code, _spec_for(label, code))
        if res is None:
            continue
        nav, pstats = res
        nav_map[label] = nav
        full_ret, rstats = returns_from_nav(nav)
        # Estimation window: most recent MAX_LOOKBACK_DAYS observations
        nav_est = nav.iloc[-(MAX_LOOKBACK_DAYS + 1):] if MAX_LOOKBACK_DAYS else nav
        est_ret, est_stats = returns_from_nav(nav_est)
        q = nav_quality(nav, pstats, rstats, full_ret)
        q["n_valid_returns_estimation_window"] = int(est_ret.notna().sum())
        q["estimation_window_start"] = str(nav_est.index[0].date())
        quality[label] = q
        if q["n_valid_returns_estimation_window"] < MIN_TRADING_DAYS:
            excluded_short.append({"label": label, "n_valid_returns": q["n_valid_returns_estimation_window"]})
            validation[label]["status"] = "excluded_insufficient_history"
            logger.warning("Excluding %s: %d valid returns (< %d)", label,
                           q["n_valid_returns_estimation_window"], MIN_TRADING_DAYS)
            continue
        ret_map[label] = est_ret

    proxy_returns, proxy_report, proxy_nav = None, None, None
    if include_proxy:
        pres = handle(MARKET_PROXY["label"], MARKET_PROXY["code"], MARKET_PROXY)
        proxy_report = validation.pop(MARKET_PROXY["label"])
        # a failed proxy must not count against the 14-fund universe
        for lst in (rejected, failed):
            if MARKET_PROXY["label"] in lst:
                lst.remove(MARKET_PROXY["label"])
        if pres is not None:
            pr, _ = returns_from_nav(pres[0])
            proxy_returns = pr
            proxy_nav = pres[0]

    report = {
        "validation": validation, "quality": quality, "rejected": rejected,
        "failed": failed, "excluded_short": excluded_short, "market_proxy": proxy_report,
        "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    if strict and (rejected or failed):
        raise DataUnavailableError(
            "Universe not accepted (strict mode). Rejected by identity check: %s. Fetch failed: %s."
            % (rejected or "none", failed or "none"), report)
    if len(ret_map) < MIN_FUNDS_REQUIRED:
        raise DataUnavailableError(
            "Only %d fund(s) passed validation with enough history; at least %d are required."
            % (len(ret_map), MIN_FUNDS_REQUIRED), report)

    returns_df = pd.concat(ret_map.values(), axis=1, join="outer", keys=list(ret_map.keys()), sort=True).sort_index()
    returns_df.index.name = "date"
    returns_df = returns_df.dropna(how="all")
    nav_full = pd.concat(nav_map.values(), axis=1, join="outer", keys=list(nav_map.keys()), sort=True).sort_index()
    nav_full.index.name = "date"

    complete = (not rejected and not failed and not excluded_short
                and len(ret_map) == len(schemes))
    logger.info("Loaded %d/%d funds; complete=%s; window %s -> %s", len(ret_map), len(schemes),
                complete, returns_df.index[0].date(), returns_df.index[-1].date())
    return {**report, "returns_df": returns_df, "nav_full": nav_full, "proxy_returns": proxy_returns, "proxy_nav": proxy_nav,
            "dataset_complete": bool(complete), "source": source_label}


def _bundled_dataset(schemes, strict):
    """The project's checksummed NAV file, if present and registered. """
    import datasets as DS
    prov = DS.read_provenance().get("fund_nav")
    if not prov:
        return None
    path = os.path.join(DS.DATA_DIR, prov["file"])
    if not os.path.exists(path):
        return None
    ds = fetch_nav_data(schemes, mf=BundledProvider(path, prov.get("sha256")), strict=strict,
                        source_label="bundled_dataset")
    ds["dataset_provenance"] = {k: prov.get(k) for k in ("source", "retrieved_on", "sha256", "start", "end", "n_schemes")}
    tri = DS.load_dataset("nifty_tri")
    if tri["status"] == "ok":
        ds["proxy_nav"] = tri["df"]["value"]
        ds["market_proxy"] = {"ok": True, "scheme_name": "Nifty 50 Total Returns Index (dataset, ends "
                              + tri["observed"]["end"] + ")", "kind": "index_dataset"}
    return ds


def load_dataset(schemes=None, strict=None):
    """
    The ONLY production data path: the bundled, checksummed NAV dataset validated against the 12-scheme universe.
    Raises DataUnavailableError if the file is missing, altered or fails identity validation. There is no live
    provider and no synthetic fallback.
    """
    bundled = _bundled_dataset(schemes, strict)
    if bundled is None:
        raise DataUnavailableError(
            "the bundled NAV dataset (data/cleaned/fund_nav.csv registered in data/provenance.json) was not found",
            {"expected": "data/cleaned/fund_nav.csv"})
    return bundled
