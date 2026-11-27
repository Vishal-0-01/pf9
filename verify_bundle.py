"""
Deployment emulation (what can be checked WITHOUT a Vercel account).

1. Copies the repository to a temporary directory, removing exactly what vercel.json `excludeFiles` removes, so the
   bundle contains only what a Vercel function would ship.
2. Runs the Flask app from that directory with the interpreter given by --python (a clean virtualenv in which ONLY
   requirements.txt was installed), with all network access blocked and counted.
3. Imports the Vercel entrypoint (root app.py), calls every API route and the static page, checks status codes, JSON
   format, the 12-fund universe, absence of mftool/requests, data accessibility, cold-start time and peak memory.

Usage:
  python -m venv /tmp/v && /tmp/v/bin/pip install -r requirements.txt
  python tools/verify_bundle.py --python /tmp/v/bin/python [--json out.json]

This is an emulation. It does not prove the behaviour of Vercel's build system, routing or hosting.
"""
import argparse, fnmatch, json, os, shutil, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHILD = r'''
import json, os, socket, sys, time, resource
attempts = []
def _blocked(*a, **k):
    attempts.append(str(a)[:80]); raise OSError("network blocked by verify_bundle")
socket.socket.connect = _blocked
socket.create_connection = _blocked
socket.getaddrinfo = _blocked
out = {"python": sys.version.split()[0], "cwd": os.getcwd()}
for mod in ("mftool", "requests", "flask_cors"):
    try:
        __import__(mod); out["importable_" + mod] = True
    except ImportError:
        out["importable_" + mod] = False
t = time.time()
import importlib.util
spec = importlib.util.spec_from_file_location("vercel_entry", os.path.join(os.getcwd(), "app.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
app = m.app
out["import_seconds"] = round(time.time() - t, 2)
c = app.test_client()
routes = ["/api/health", "/api/funds", "/api/fund-metrics", "/api/data-quality", "/api/data-report", "/api/datasets",
          "/api/diagnostics", "/api/optimize?pe=22&pb=3.5", "/api/frontier", "/api/allocation-history", "/api/scenario",
          "/api/hypotheses", "/api/montecarlo", "/api/montecarlo?paths=2000&seed=3", "/api/meta"]
res = {}
first = True
for r in routes:
    t = time.time(); resp = c.get(r); dt = round(time.time() - t, 2)
    try:
        j = resp.get_json()
    except Exception:
        j = None
    res[r] = {"status": resp.status_code, "seconds": dt, "json": j is not None and j.get("status") == "ok",
              "cold": first}
    first = False
    if r == "/api/funds" and j:
        out["n_funds"] = len(j["data"]["funds"]); out["fund_names"] = [f["name"] for f in j["data"]["funds"]]
        out["source"] = j["meta"]["source"]
    if r == "/api/health" and j:
        out["health"] = {k: j["data"].get(k) for k in ("ready", "source", "n_funds", "build_seconds", "datasets", "deployment")}
    if r == "/api/hypotheses" and j:
        out["hypotheses"] = {k: j["data"][k].get("verdict") for k in ("H1", "H2", "H3")}
    if r == "/api/scenario" and j:
        out["scenario_evaluated"] = j["data"].get("evaluated")
        out["scenario_message"] = j["data"].get("message")
    if r == "/api/montecarlo" and j:
        out["mc_status"] = j["data"].get("status"); out["mc_parameter_source"] = j["data"]["proposed"]["parameters"]["parameter_source"]
res["/ (static page)"] = {"status": c.get("/").status_code}
res["/api/does-not-exist"] = {"status": c.get("/api/does-not-exist").status_code,
                              "json": (c.get("/api/does-not-exist").get_json() or {}).get("status") == "error"}
out["routes"] = res
out["network_attempts"] = len(attempts)
out["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
print("JSON_RESULT=" + json.dumps(out, default=str))
'''


def excluded(rel, patterns):
    return any(fnmatch.fnmatch(rel, p) or fnmatch.fnmatch(rel, p.replace("/**", "/*")) or rel.startswith(p.replace("/**", "/"))
               for p in patterns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", required=True)
    ap.add_argument("--json")
    a = ap.parse_args()
    cfg = json.load(open(os.path.join(ROOT, "vercel.json")))
    pat = cfg["functions"]["app.py"]["excludeFiles"].strip("{}").split(",")
    bundle = tempfile.mkdtemp(prefix="vercel_bundle_")
    n = size = 0
    skip_dirs = {".git", "__pycache__", ".pytest_cache", ".venv", ".vercel"}
    for dp, dn, fn in os.walk(ROOT):
        dn[:] = [d for d in dn if d not in skip_dirs]
        for f in fn:
            full = os.path.join(dp, f); rel = os.path.relpath(full, ROOT).replace(os.sep, "/")
            if excluded(rel, pat) or f.endswith(".pyc"):
                continue
            dest = os.path.join(bundle, rel); os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy2(full, dest); n += 1; size += os.path.getsize(full)
    print(f"bundle: {n} files, {size/1e6:.1f} MB (project files only) in {bundle}")
    env = {k: v for k, v in os.environ.items() if k not in ("PORTFOLIO_SKIP_INIT", "PORTFOLIO_DATA_DIR")}
    p = subprocess.run([a.python, "-I", "-c", "import sys; sys.path.insert(0, '.'); exec(open('/dev/stdin').read())"] if False else
                       [a.python, "-c", CHILD], cwd=bundle, env=env, capture_output=True, text=True, timeout=600)
    line = next((l for l in p.stdout.splitlines() if l.startswith("JSON_RESULT=")), None)
    if not line:
        print(p.stdout[-2000:], p.stderr[-3000:]); sys.exit("child failed")
    out = json.loads(line[len("JSON_RESULT="):])
    out["bundle_project_files"] = n; out["bundle_project_mb"] = round(size / 1e6, 2)
    print(json.dumps(out, indent=1))
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)
    shutil.rmtree(bundle, ignore_errors=True)


if __name__ == "__main__":
    main()
