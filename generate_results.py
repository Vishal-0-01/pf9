"""Regenerate results/ from the shipped datasets (runs build_state exactly as the deployed app does).
    python tools/generate_results.py
Writes results/*.json and the fund CSV, plus optimiser sensitivity tables (growth assumption, valuation)."""
import json, os, sys, csv
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))
import numpy as np
import webapp
import optimizer as OPT

OUT = os.path.join(ROOT, "results")


def dump(name, obj):
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


def main():
    c = webapp.app.test_client()
    assert c.get("/api/health").status_code == 200
    S = webapp._STATE
    get = lambda u: c.get(u).json["data"]
    dump("hypotheses_real_data.json", get("/api/hypotheses"))
    dump("montecarlo_real_data.json", get("/api/montecarlo"))
    sc = get("/api/scenario")
    if sc.get("evaluated"):
        keep = {k: v for k, v in sc.items() if k not in ("series", "convention", "method")}
        dump("historical_scenario.json", keep)
        dump("allocation_history.json", sc["decisions"])
        dump("scenario_wealth_series.json", sc["series"])
    else:
        dump("scenario_status.json", {k: sc[k] for k in ("status", "evaluated", "message", "datasets") if k in sc})
    rec = get("/api/optimize")
    dump("recommended_allocation.json", rec)
    dump("frontier.json", get("/api/frontier"))
    lv = S["latest_valuation"]
    sens = {"latest_valuation": lv, "valuation_sensitivity": []}
    for pe, pb in ((14, 2.0), (18, 2.6), (lv["pe"], lv["pb"]), (26, 3.8), (32, 4.6), (40, 6.0)):
        try:
            r = webapp._optimise_state(S, pe, pb)
            sens["valuation_sensitivity"].append({"pe": pe, "pb": pb, "z": r["z"], "equity_target": r["equity_target"],
                                                  "equity": r["equity"], "debt": r["debt"], "gold": r["gold"],
                                                  "port_vol": r["port_vol"], "constraint_status": r["constraint_status"]})
        except Exception as exc:
            sens["valuation_sensitivity"].append({"pe": pe, "pb": pb, "infeasible": str(exc)})
    dump("allocation_sensitivity.json", sens)
    fm = get("/api/fund-metrics")
    rows = fm if isinstance(fm, list) else fm.get("funds", [])
    if rows:
        keys = sorted({k for r in rows for k, v in r.items() if not isinstance(v, (dict, list))})
        with open(os.path.join(OUT, "fund_evaluation_12_funds.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k) for k in keys})
    print("results regenerated; errors:", S.get("errors"))


if __name__ == "__main__":
    main()
