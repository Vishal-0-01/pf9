"""Run Vercel's own build locally (no account, no deployment) and emulate the built function.
    python tools/verify_vercel_build.py [--framework-setting null|flask|fastapi]
Needs Node/npm, internet for npm and PyPI, and Python 3.13 with the project's requirements.txt installed (for the emulation).
It copies the project to a temp dir, fakes the dashboard project settings (default: framework "Other"), runs
`vercel build`, checks vercel.json was accepted, maxDuration, excluded folders, bundled data and the static page, then
calls every GET route of the built bundle. It does NOT test a live Vercel deployment."""
import argparse, json, os, shutil, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--framework-setting", default="null", help="dashboard Framework Preset to simulate: null (Other) | flask")
    a = ap.parse_args()
    tmp = tempfile.mkdtemp(prefix="vcl_build_")
    proj = os.path.join(tmp, "p")
    shutil.copytree(ROOT, proj, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", ".vercel", ".venv"))
    os.makedirs(os.path.join(proj, ".vercel"))
    fw = "null" if a.framework_setting == "null" else json.dumps(a.framework_setting)
    open(os.path.join(proj, ".vercel", "project.json"), "w").write(
        '{"projectId":"prj_local","orgId":"team_local","settings":{"framework":%s,"rootDirectory":null}}' % fw)
    r = subprocess.run(["npx", "--yes", "vercel@latest", "build", "--yes"], cwd=proj, capture_output=True, text=True)
    print(r.stdout[-1500:], r.stderr[-1500:])
    if r.returncode != 0:
        sys.exit("vercel build FAILED")
    out = os.path.join(proj, ".vercel", "output")
    fdirs = [d for d in os.listdir(os.path.join(out, "functions")) if d.endswith(".func")]
    assert len(fdirs) == 1, fdirs
    fdir = os.path.join(out, "functions", fdirs[0])
    cfg = json.load(open(os.path.join(fdir, ".vc-config.json")))
    fmap = cfg["filePathMap"]
    assert cfg["maxDuration"] == 120, cfg["maxDuration"]
    assert not [k for k in fmap if k.startswith(("tests/", "tools/", "docs/", "results/"))], "excluded folders leaked"
    need = ["app.py", "backend/webapp.py", "data/provenance.json", "data/cleaned/fund_nav.csv", "data/cleaned/debt.csv"]
    assert all(n in fmap for n in need), [n for n in need if n not in fmap]
    assert os.path.exists(os.path.join(out, "static", "index.html"))
    bundle = os.path.join(tmp, "bundle")
    for dst, src in fmap.items():
        d = os.path.join(bundle, dst); os.makedirs(os.path.dirname(d), exist_ok=True)
        shutil.copy(src if os.path.isabs(src) else os.path.join(proj, src), d)
    code = r'''
import sys, os, time, logging; logging.disable(logging.CRITICAL)
sys.path.insert(0, os.getcwd()); import app as A
c = A.app.test_client(); bad = []
for _ in range(90):
    if c.get("/api/health").status_code == 200: break
    time.sleep(1)
rules = sorted({r.rule for r in A.app.url_map.iter_rules() if "GET" in r.methods and "<" not in r.rule})
for u in rules:
    s = c.get(u).status_code; print(u, s)
    if u.startswith("/api/") and s != 200: bad.append(u)
sys.exit(1 if bad else 0)
'''
    env = {"PATH": os.environ["PATH"], "PYTHONPATH": os.path.join(bundle, "_vendor")}
    e = subprocess.run([sys.executable, "-c", code], cwd=bundle, env=env, capture_output=True, text=True)
    print(e.stdout, e.stderr[-800:])
    print("BUILD AND BUNDLE EMULATION OK" if e.returncode == 0 else "EMULATION FAILED")
    sys.exit(e.returncode)


if __name__ == "__main__":
    main()
