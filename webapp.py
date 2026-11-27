"""
webapp.py - Flask API (runs as a Vercel Python Function via the root app.py shim).

Data integrity rules enforced here
-----------------------------------
* Bundled data only. The model is built from the checksummed datasets inside the repository (data/cleaned,
  data/provenance.json). Nothing is fetched from the network: no mftool, AMFI, mfapi or any other provider.
  If loading or scheme-identity validation fails the API answers "unavailable" (HTTP 503). There is no synthetic
  fallback.
* Serverless-safe state: the model is built lazily, in process memory, on the first API request of an instance
  (deterministic: same files and same seeds give the same numbers). Nothing is written to disk and no background
  thread is used. A warm instance reuses the state; a cold instance rebuilds it.
* A dataset that is missing or invalid (for example the G-Sec index levels) makes the dependent sections report
  "not evaluated" with the reason; nothing is generated in its place.
* /api/optimize returns HTTP 422 if no portfolio satisfies every constraint.
"""

import hashlib
import json
import logging
import os
import sys
import threading
import time

import numpy as np

from flask import Flask, jsonify, request, send_from_directory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
PUBLIC_DIR = os.path.join(ROOT_DIR, "public")
sys.path.insert(0, BASE_DIR)

import data_fetcher as DF                                  # noqa: E402
import optimizer as OPT                                    # noqa: E402
import datasets as DS                                      # noqa: E402
import scenario as SC                                      # noqa: E402
import hypotheses as HY                                    # noqa: E402
import montecarlo as MC                                    # noqa: E402
import verify as VF                                        # noqa: E402
from fund_metrics import compute_fund_metrics              # noqa: E402
import assets as AS                                        # noqa: E402
from universe import SCHEME_SPECS, MARKET_PROXY            # noqa: E402
from optimizer import (                                    # noqa: E402
    optimize_for_pe_pb, compute_frontier, build_cov_from_returns, compute_diagnostics,
    OptimizationInfeasible, VOL_CAP, RF,
    NIFTY_MEAN_PE, NIFTY_STD_PE, NIFTY_MEAN_PB, NIFTY_STD_PB,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

app = Flask(__name__, static_folder=None)      # static files are served from public/ (Vercel CDN)

# "bundled_dataset" is the only production source; "injected_provider" exists for unit tests with fixture providers.
REAL_SOURCES = ("bundled_dataset", "injected_provider")
RETRY_SECONDS = 60

_STATE = {}
_LOCK = threading.Lock()
_LAST_ATTEMPT = 0.0


# ── DEPLOYMENT FINGERPRINT ─────────────────────────────────────────────────────────

_FP = {}


def compute_fingerprint():
    """Hash of the code, scheme mapping, model parameters and datasets: identifies exactly what is deployed.
    Memoised per process (the files do not change while an instance is alive)."""
    key = DS.DATA_DIR
    if key in _FP:
        return _FP[key]
    h = hashlib.sha256()
    for fn in ("webapp.py", "assets.py", "optimizer.py", "data_fetcher.py", "universe.py", "fund_metrics.py", "perf.py",
               "datasets.py", "scenario.py", "montecarlo.py", "hypotheses.py", "verify.py"):
        with open(os.path.join(BASE_DIR, fn), "rb") as f:
            h.update(fn.encode()); h.update(f.read())
    params = {
        "schemes": {k: v for k, v in SCHEME_SPECS.items()}, "proxy": MARKET_PROXY,
        "opt": {k: getattr(OPT, k) for k in (
            "RF", "VOL_CAP", "MIN_FUND_W", "MAX_FUND_W", "NIFTY_MEAN_PE", "NIFTY_STD_PE", "NIFTY_MEAN_PB",
            "NIFTY_STD_PB", "DEBT_RET", "DEBT_VOL", "GOLD_RET", "GOLD_VOL", "EQUITY_SLOPE", "EQUITY_NEUTRAL", "OBJECTIVE_DEFAULT", "EQUITY_MIN", "EQUITY_MAX",
            "RHO_EQ_DEBT", "RHO_EQ_GOLD", "RHO_DEBT_GOLD", "NIFTY_LONGRUN_RETURN", "SHRINKAGE_ALPHA",
            "LAMBDA_DIVERSIFICATION", "LAMBDA_TURNOVER", "TC_BPS", "MAX_CAGR_THRESHOLD", "MIN_VOL_THRESHOLD",
            )},
        "data": {k: getattr(DF, k) for k in ("MIN_TRADING_DAYS", "MAX_LOOKBACK_DAYS", "MAX_GAP_CALENDAR_DAYS",
                                             "MAX_ABS_DAILY_RETURN", "MIN_FUNDS_REQUIRED")},
    }
    h.update(json.dumps(params, sort_keys=True, default=str).encode())
    h.update(DS.fingerprint_data().encode())          # imported benchmark datasets and their provenance
    _FP[key] = h.hexdigest()
    return _FP[key]


# ── STATE ─────────────────────────────────────────────────────────────────────

def _unavailable_state(message, report=None):
    return {"status": "unavailable", "ready": False, "error": message, "report": report or {},
            "funds": [], "source": "unavailable", "is_synthetic": False, "dataset_complete": False,
            "snapshot": None, "built_at": time.time()}


def build_state(dataset):
    """Compute the full model state from a dataset returned by data_fetcher."""
    import pandas as pd
    returns_all = dataset["returns_df"]
    nav_full = dataset.get("nav_full")
    source = dataset["source"]
    funds = list(returns_all.columns)

    # COMMON observation window: only days on which every fund has a valid return. The covariance, the return
    # estimates and the optimisation all use this identical window (fair comparison across funds with different
    # inception dates).  Per-fund full-history statistics are reported separately.
    common = returns_all.dropna(how="any")
    min_common = DF.MIN_TRADING_DAYS
    if len(common) < min_common:
        raise DF.DataUnavailableError(
            f"the common window of all {len(funds)} funds has only {len(common)} daily returns (< {min_common})",
            {"common_observations": int(len(common))})
    ann, cov, vols, shp, extra = build_cov_from_returns(common)

    loaded = DS.load_all()
    def_levels = {a: loaded[a]["df"]["value"] for a in AS.DEF_ORDER if loaded[a]["status"] == "ok"}
    real = source in REAL_SOURCES and nav_full is not None
    if real:
        cross = AS.build_cross(cov, nav_full[funds], def_levels,
                               common.index[0] - pd.Timedelta(days=5), common.index[-1])
    else:
        cross = AS.assumed_cross(cov)
    cov = AS.CovModel(cov, cross)
    vstat, latest_val = None, None
    if loaded["nifty_valuation"]["status"] == "ok":
        vdf = loaded["nifty_valuation"]["df"]
        vstat = AS.valuation_stats(vdf, vdf.index[-1], "expanding")
        latest_val = {"date": str(vdf.index[-1].date()), "pe": float(vdf["pe"].iloc[-1]), "pb": float(vdf["pb"].iloc[-1])}

    state = {
        "status": "ready", "ready": True, "source": source, "is_synthetic": source == "synthetic",
        "dataset_complete": bool(dataset.get("dataset_complete")),
        "funds": funds, "ann_returns": ann, "cov_mat": cov, "vols": vols, "fund_sharpes": shp,
        "extra_stats": extra, "validation": dataset.get("validation", {}), "quality": dataset.get("quality", {}),
        "market_proxy": dataset.get("market_proxy"), "excluded_short": dataset.get("excluded_short", []),
        "live_failure": dataset.get("live_failure"), "built_at": time.time(),
        "dataset_provenance": dataset.get("dataset_provenance"),
        "val_stats": vstat, "latest_valuation": latest_val, "cross_summary": {k: (v if not hasattr(v, "tolist") else v.tolist()) for k, v in cross.items()
                                              if k in ("source", "window", "n_obs", "mu", "obs_per_year")},
        "snapshot": {
            "source": source, "retrieved_at": dataset.get("retrieved_at"),
            "data_start": str(returns_all.index[0].date()), "data_end": str(returns_all.index[-1].date()),
            "common_window_start": str(common.index[0].date()), "common_window_end": str(common.index[-1].date()),
            "common_window_obs": int(len(common)),
            "n_rows": int(len(returns_all)), "n_funds": len(funds),
            "built_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "returns_df": common if real else None,
        "fund_levels": nav_full[funds] if real else None, "def_levels": def_levels if real else None,
    }

    # fund-level metrics from the funds' own NAVs: (a) own estimation window, (b) the common window
    metrics = []
    if nav_full is not None:
        proxy_nav = None
        if dataset.get("market_proxy") and dataset["market_proxy"].get("ok"):
            proxy_nav = dataset.get("proxy_nav")
        for i, f in enumerate(funds):
            nav = nav_full[f].dropna()
            nav = nav.iloc[-(DF.MAX_LOOKBACK_DAYS + 1):] if DF.MAX_LOOKBACK_DAYS else nav
            m = compute_fund_metrics(f, nav, returns_all[f].dropna().loc[nav.index[0]:], RF, proxy_nav)
            base = nav.index[nav.index < common.index[0]]
            nav_c = nav.loc[base[-1]:] if len(base) else nav
            mc = compute_fund_metrics(f, nav_c, common[f], RF, None)
            m.update({"return_model": float(ann[i]), "vol_model": float(vols[i]), "sharpe_model": float(shp[i]),
                      "sortino_model": float(extra["sortinos"][i]),
                      "raw_cagr_capped": float(extra["raw_cagrs"][i]),
                      "common": {k: mc[k] for k in ("n_returns", "window_start", "window_end", "cagr_calendar",
                                                    "cagr_log_daily", "vol_sample", "sharpe_raw", "sortino_std",
                                                    "max_drawdown")}})
            metrics.append(m)
    state["fund_metrics"] = metrics

    def safe(name, fn, default):
        try:
            return fn()
        except Exception as exc:  # recorded with the real exception text and traceback, never hidden
            import traceback
            logger.exception("%s failed", name)
            msg = f"{type(exc).__name__}: {exc}"
            state.setdefault("errors", {})[name] = {"message": msg, "traceback": traceback.format_exc()[-3000:]}
            if isinstance(default, dict):
                default = {**default, "error": msg}
            return default

    state["frontier"] = safe("frontier", lambda: compute_frontier(ann, cov), [])
    state["datasets"] = DS.dataset_report(loaded)
    scen = safe("historical_scenario", lambda: SC.run_scenario(loaded),
                {"status": "error", "evaluated": False})
    if scen.get("evaluated"):
        scen["sensitivity"] = safe("sensitivity", lambda: SC.run_sensitivities(loaded), [])
    h1, h3 = HY.evaluate_h1_h3(scen)
    h2 = safe("h2", lambda: HY.evaluate_h2(state, *_current_valuation(loaded)), {"evaluated": False, "verdict": "NOT EVALUATED"})
    state["hypotheses"] = {"H1": h1, "H2": h2, "H3": h3, "rules": HY.__doc__.strip()}
    scen_public = {k: v for k, v in scen.items() if not k.startswith("_")}
    state["scenario"] = scen_public
    state["allocation_history"] = scen_public.get("decisions") if scen_public.get("evaluated") else None
    state["montecarlo"] = safe("monte_carlo", lambda: _default_monte_carlo(state, loaded),
                               {"status": "error"})
    state["diagnostics"] = safe("diagnostics", lambda: compute_diagnostics(funds, ann, cov, vols, extra), {})
    return state


def _current_valuation(loaded):
    """Latest imported P/E and P/B if the valuation dataset is valid, else (None, None) -> neutral."""
    v = loaded["nifty_valuation"]
    if v["status"] == "ok":
        last = v["df"].iloc[-1]
        return float(last["pe"]), float(last["pb"])
    return None, None


def _optimise_state(state, pe, pb, objective=OPT.OBJECTIVE_DEFAULT):
    return optimize_for_pe_pb(pe, pb, state["ann_returns"], state["cov_mat"], state["vols"],
                              state["fund_sharpes"], state["funds"], semi_vols=state["extra_stats"]["semi_vols"],
                              val_stats=state.get("val_stats"), objective=objective)


def _allocation_for_mc(state, pe, pb):
    """The recommended allocation (equity from the valuation rule, rest optimised) for the given or latest valuation."""
    lv = state.get("latest_valuation")
    if pe is None and lv:
        pe, pb = lv["pe"], lv["pb"]
    src = "supplied / latest valuation dataset reading"
    if pe is None:                      # no valuation data at all: neutral valuation (z = 0, equity target 70%), flagged
        st = state.get("val_stats") or {"pe": OPT.SYNOPSIS_STATS["pe"], "pb": OPT.SYNOPSIS_STATS["pb"]}
        pe, pb = st["pe"][0], st["pb"][0]
        src = "no valuation dataset: neutral valuation assumed (z = 0, equity target 70%)"
    return _optimise_state(state, pe, pb), (pe, pb, src)


def _run_mc(state, loaded, pe, pb, n_paths=MC.DEFAULT_PATHS, seed=MC.DEFAULT_SEED, horizons=MC.DEFAULT_HORIZONS):
    res, (pe_u, pb_u, src) = _allocation_for_mc(state, pe, pb)
    w3 = [res["equity"], res["debt"], res["gold"]]
    raw = state["extra_stats"]["raw_cagrs"]
    common = dict(cov_funds=state["cov_mat"], ann_returns=np.array(res["fund_returns"]), raw_cagrs=raw,
                  n_paths=n_paths, horizons=horizons, seed=seed)
    return {"status": "ok", "fund_data_source": state.get("source"),
            "valuation_inputs": {"pe": pe_u, "pb": pb_u, "source": src},
            "recommended_allocation": {"equity": res["equity"], "debt": res["debt"], "gold": res["gold"],
                                       "equity_target": res["equity_target"], "constraint_status": res["constraint_status"]},
            "proposed": MC.run_monte_carlo(res["fund_weights"], w3, mode="base", label="optimiser's recommended allocation", **common),
            "proposed_unshrunk_returns": MC.run_monte_carlo(res["fund_weights"], w3, mode="historical_unshrunk",
                                                          label="recommended allocation, raw historical fund returns (short window)", **common),
            "static_70_20_10": MC.run_monte_carlo(res["fund_weights"], [0.70, 0.20, 0.10], mode="base",
                                                  label="fixed 70/20/10 (comparison only) with the same fund sleeve", **common)}


def _default_monte_carlo(state, loaded):
    pe, pb = _current_valuation(loaded)
    return _run_mc(state, loaded, pe, pb)


def set_state(state):
    global _STATE
    _STATE = state


def _jsonable_report(rep):
    return json.loads(json.dumps(rep, default=str)) if rep else {}


def _init_state(force=False):
    """Build the state from the bundled datasets, in memory only. Thread-safe: concurrent first requests wait."""
    global _LAST_ATTEMPT
    t_enter = time.time()
    with _LOCK:
        if _STATE.get("ready") and not force:
            return
        if not force and _LAST_ATTEMPT >= t_enter:        # another request finished an attempt while we waited
            return
        _LAST_ATTEMPT = time.time()
        t0 = time.time()
        try:
            st = build_state(DF.load_dataset())
            st["build_seconds"] = round(time.time() - t0, 2)
            set_state(st)
            logger.info("State ready in %.1fs | source=%s | funds=%d | complete=%s", st["build_seconds"],
                        st["source"], len(st["funds"]), st["dataset_complete"])
        except DF.DataUnavailableError as exc:
            logger.error("DATA UNAVAILABLE: %s", exc)
            set_state(_unavailable_state(str(exc), _jsonable_report(exc.report)))
        except Exception as exc:
            logger.exception("INIT FAILED")
            set_state(_unavailable_state(f"{type(exc).__name__}: {exc}"))


# ── HELPERS ───────────────────────────────────────────────────────────────────

def _meta():
    s = _STATE
    return {"source": s.get("source"), "is_synthetic": bool(s.get("is_synthetic")),
            "dataset_complete": bool(s.get("dataset_complete")), "status": s.get("status"),
            "snapshot": s.get("snapshot"),
            "warning": ("SYNTHETIC ILLUSTRATIVE DATA - NOT HISTORICAL FUND PERFORMANCE"
                        if s.get("is_synthetic") else None)}


def _ok(data):
    return jsonify({"status": "ok", "data": data, "meta": _meta()})


def _err(msg, code=400, **extra):
    return jsonify({"status": "error", "message": msg, **extra}), code


def _ready():
    """True if a usable state exists. The first request of a cold instance builds it; concurrent requests wait for
    that build instead of failing. A failed build is retried at most once a minute."""
    if _STATE.get("ready"):
        return True
    # PORTFOLIO_SKIP_INIT=1 is a unit-test switch only (tests inject their own state); it is never set in production.
    if os.environ.get("PORTFOLIO_SKIP_INIT") == "1":
        return False
    if time.time() - _LAST_ATTEMPT > RETRY_SECONDS:
        _init_state()
    else:
        with _LOCK:                    # a build may be in flight in another thread: wait for it to finish
            pass
    return bool(_STATE.get("ready"))


def _unavailable_response():
    return _err("Model data are unavailable: " + str(_STATE.get("error", "state not initialised")), 503,
                source=_STATE.get("source"), report=_STATE.get("report"))


# ── ROUTES ────────────────────────────────────────────────────────────────────

@app.route("/api/health")
@app.route("/health")
def health():
    """Readiness check: builds the model on the first call of a cold instance (add ?light=1 for liveness only)."""
    if request.args.get("light") != "1":
        _ready()
    s = _STATE
    return jsonify({"status": "ok", "data": {
        "ready": bool(s.get("ready")), "source": s.get("source"), "is_synthetic": bool(s.get("is_synthetic")),
        "dataset_complete": bool(s.get("dataset_complete")), "snapshot": s.get("snapshot"),
        "error": s.get("error"), "datasets": {k: v["status"] for k, v in (s.get("datasets") or {}).items()}, "n_funds": len(s.get("funds", [])),
        "rejected": (s.get("report") or {}).get("rejected") if not s.get("ready") else None,
        "code_fingerprint": compute_fingerprint()[:12], "build_seconds": s.get("build_seconds"),
        "deployment": {"vercel": bool(os.environ.get("VERCEL")), "region": os.environ.get("VERCEL_REGION"),
                       "python": sys.version.split()[0], "live_nav_fetching": False,
                       "data_dir_exists": os.path.isdir(DS.DATA_DIR)},
    }})


@app.route("/api/funds")
def api_funds():
    if not _ready():
        return _unavailable_response()
    fm = {m["fund"]: m for m in _STATE.get("fund_metrics", [])}
    val = _STATE.get("validation", {})
    rows = []
    for i, f in enumerate(_STATE["funds"]):
        m = fm.get(f, {})
        rows.append({
            "name": f,
            "return": round(float(_STATE["ann_returns"][i]), 4),
            "vol": round(float(_STATE["vols"][i]), 4),
            "sharpe": round(float(_STATE["fund_sharpes"][i]), 3),
            # additive: raw historical statistics for reconciliation
            "raw_cagr": m.get("cagr_log_daily"), "raw_vol": m.get("vol_sample"),
            "max_drawdown": m.get("max_drawdown"), "n_returns": m.get("n_returns"),
            "scheme_name": (val.get(f) or {}).get("scheme_name"), "scheme_code": (val.get(f) or {}).get("code"),
        })
    return _ok({"funds": rows, "source": _STATE["source"]})


@app.route("/api/fund-metrics")
def api_fund_metrics():
    if not _ready():
        return _unavailable_response()
    return _ok({"funds": _STATE.get("fund_metrics", []), "risk_free": RF,
                "market_proxy": _STATE.get("market_proxy"),
                "note": "Raw historical statistics from each fund's own NAVs; model_* fields are the shrunk "
                        "/ regularised values the optimiser uses."})


@app.route("/api/data-quality")
def api_data_quality():
    _ready()
    s = _STATE
    return jsonify({"status": "ok", "data": {
        "ready": bool(s.get("ready")), "source": s.get("source"), "is_synthetic": bool(s.get("is_synthetic")),
        "dataset_complete": bool(s.get("dataset_complete")), "snapshot": s.get("snapshot"),
        "benchmark_datasets": s.get("datasets"),
        "dataset_provenance": s.get("dataset_provenance"), "defensive_parameters": s.get("cross_summary"),
        "validation": s.get("validation") if s.get("ready") else (s.get("report") or {}).get("validation"),
        "quality": s.get("quality"), "excluded_short": s.get("excluded_short"),
        "market_proxy": s.get("market_proxy"), "error": s.get("error"), "live_failure": s.get("live_failure"),
        "component_errors": s.get("errors"),
        "expected_universe": {k: v["code"] for k, v in SCHEME_SPECS.items()},
    }})


@app.route("/api/diagnostics")
def api_diagnostics():
    if not _ready():
        return _unavailable_response()
    return _ok(_STATE.get("diagnostics", {}))


@app.route("/api/optimize")
def api_optimize():
    if not _ready():
        return _unavailable_response()
    try:
        lv = _STATE.get("latest_valuation") or {}
        pe = float(request.args.get("pe", lv.get("pe", 22)))
        pb = float(request.args.get("pb", lv.get("pb", 3.5)))
        objective = request.args.get("objective", OPT.OBJECTIVE_DEFAULT)
        if objective not in ("sharpe", "return"):
            return _err("objective must be 'sharpe' or 'return'", 400)
        result = _optimise_state(_STATE, pe, pb, objective)
    except OptimizationInfeasible as exc:
        return _err(f"Optimisation failed: {exc}", 422, details=exc.details, constraint_ok=False)
    except ValueError as exc:
        return _err(str(exc), 400)
    except Exception as exc:
        logger.exception("Optimize failed")
        return _err(str(exc), 500)

    result["fund_names"] = _STATE["funds"]
    result["independent_verification"] = VF.independent_check(result, _STATE["cov_mat"])
    n = len(result["fund_weights"])
    delta = [float(w) - 1.0 / n for w in result["fund_weights"]]
    turnover = sum(abs(d) for d in delta)
    result.update({"delta_weights": delta, "turnover": float(turnover), "cost_estimate": float(turnover * 0.002),
                   "rebalance_action": "SKIP" if turnover < 0.05 else "REBALANCE",
                   "turnover_baseline": "equal weights (not the user's actual holdings)"})
    return _ok(result)


@app.route("/api/frontier")
def api_frontier():
    if not _ready():
        return _unavailable_response()
    return _ok(_STATE.get("frontier", []))


@app.route("/api/allocation-history")
def api_allocation_history():
    if not _ready():
        return _unavailable_response()
    sc = _STATE.get("scenario", {})
    if not sc.get("evaluated"):
        return _ok({"evaluated": False, "message": sc.get("message", "historical scenario not available"),
                    "datasets": _STATE.get("datasets")})
    return _ok({"evaluated": True, "decisions": sc["decisions"]})


@app.route("/api/scenario")
def api_scenario():
    if not _ready():
        return _unavailable_response()
    return _ok(_STATE.get("scenario", {}))


@app.route("/api/hypotheses")
def api_hypotheses():
    if not _ready():
        return _unavailable_response()
    return _ok(_STATE.get("hypotheses", {}))


@app.route("/api/datasets")
def api_datasets():
    return jsonify({"status": "ok", "data": DS.dataset_report(DS.load_all())})


@app.route("/api/montecarlo")
def api_montecarlo():
    if not _ready():
        return _unavailable_response()
    q = request.args
    if not any(k in q for k in ("paths", "seed", "pe", "pb")):
        return _ok(_STATE.get("montecarlo", {}))
    try:
        loaded = DS.load_all()
        pe = float(q["pe"]) if "pe" in q else None
        pb = float(q["pb"]) if "pb" in q else None
        out = _run_mc(_STATE, loaded, pe, pb, n_paths=int(q.get("paths", MC.DEFAULT_PATHS)),
                      seed=int(q.get("seed", MC.DEFAULT_SEED)))
    except OptimizationInfeasible as exc:
        return _err(f"Optimisation failed: {exc}", 422, constraint_ok=False)
    except (ValueError, TypeError) as exc:
        return _err(str(exc), 400)
    return _ok(out)


@app.route("/api/data-report")
def api_data_report():
    """Static data-quality report written by tools/ingest_raw.py (raw vs cleaned data, provenance)."""
    p = os.path.join(DS.DATA_DIR, "data_quality_report.json")
    if not os.path.exists(p):
        return _err("data_quality_report.json not found", 404)
    with open(p) as f:
        return jsonify({"status": "ok", "data": json.load(f), "provenance": DS.read_provenance()})


@app.route("/api/meta")
def api_meta():
    _ready()
    vs = _STATE.get("val_stats") or {}
    return _ok({"vol_cap": VOL_CAP, "rf": RF, "equity_rule": "clip(0.70 - 0.10 z, 0.50, 0.90)",
                "allocation_method": "equity from the valuation rule; fund weights and the debt/gold split optimised (no minimum for debt or gold)",
                "latest_valuation": _STATE.get("latest_valuation"),
                "z_basis": {k: (list(v) if isinstance(v, tuple) else v) for k, v in vs.items()} or
                {"method": "synopsis_constants", "pe": [NIFTY_MEAN_PE, NIFTY_STD_PE], "pb": [NIFTY_MEAN_PB, NIFTY_STD_PB]},
                "asset_classes": ["equity", "debt", "gold"], "cash": "not an asset class",
                "synopsis_constants": {"pe": [NIFTY_MEAN_PE, NIFTY_STD_PE], "pb": [NIFTY_MEAN_PB, NIFTY_STD_PB]}})


@app.after_request
def _cache_headers(resp):
    """Results are deterministic for a given deployment (fixed data and seeds), so successful GET responses may be
    cached at the CDN; Vercel invalidates the cache on every new deployment. Health and errors are never cached."""
    if (request.method == "GET" and request.path.startswith("/api/") and request.path != "/api/health"
            and resp.status_code == 200):
        resp.headers["Cache-Control"] = "public, s-maxage=600, stale-while-revalidate=3600"
    else:
        resp.headers.setdefault("Cache-Control", "no-store")
    return resp


@app.errorhandler(404)
def _not_found(_e):
    if request.path.startswith("/api/") or request.path == "/health":
        return _err(f"unknown API route {request.path}", 404, routes=sorted(r.rule for r in app.url_map.iter_rules()
                                                                           if r.rule.startswith("/api/")))
    return ("Not found", 404)


@app.errorhandler(Exception)
def _unhandled(exc):
    from werkzeug.exceptions import HTTPException
    if isinstance(exc, HTTPException):
        return exc
    logger.exception("unhandled error")
    return _err(f"{type(exc).__name__}: {exc}", 500)


@app.route("/")
def index():
    return send_from_directory(PUBLIC_DIR, "index.html")


@app.route("/<path:path>")
def static_files(path):
    # On Vercel the CDN serves public/ directly; this fallback serves it when running the Flask app locally.
    if path.startswith("api/"):
        return _err(f"unknown API route /{path}", 404)
    return send_from_directory(PUBLIC_DIR, path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
