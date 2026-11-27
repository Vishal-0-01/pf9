"""End-to-end: the Flask app (same origin, relative /api paths, exactly as on Vercel) + real frontend in headless Chromium.
Run: python tests/e2e_frontend.py   (needs playwright + chromium)"""
import os, sys, threading, json, time
os.environ["PORTFOLIO_SKIP_INIT"] = "1"; os.environ["DISABLE_CACHE"] = "1"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "backend"), os.path.join(ROOT, "tests")]
from werkzeug.serving import make_server
import data_fetcher as DF, webapp as A, datasets as DS
from fakes import FakeProvider
from fixture_data import make_fixture_dir
from playwright.sync_api import sync_playwright

def serve(state, port):
    A.set_state(state)
    srv = make_server("127.0.0.1", port, A.app); threading.Thread(target=srv.serve_forever, daemon=True).start(); return srv

def run(page_url_port, label):
    with sync_playwright() as p:
        b = p.chromium.launch(); pg = b.new_page(viewport={"width": 1280, "height": 1600})
        errs = []; pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.on("console", lambda m: errs.append(m.text) if m.type == "error" and "Failed to load resource" not in m.text else None)
        pg.route("**/cdn.jsdelivr.net/**", lambda r: r.abort()); pg.route("**/cdnjs.cloudflare.com/**", lambda r: r.fulfill(path=os.environ["CHARTJS_PATH"], content_type="application/javascript")); pg.route("**/fonts.*/**", lambda r: r.abort())
        pg.goto(f"http://127.0.0.1:{page_url_port}/", wait_until="load"); pg.wait_for_timeout(9000)   # optimise call takes about 2 s on real data
        out = {"label": label, "pill": pg.inner_text("#pill-source"), "banner": pg.inner_text("#data-banner"),
               "rows": pg.locator("#fund-tbl tr").count(), "vol": pg.inner_text("#m-vol") if pg.locator("#m-vol").count() else None,
               "errors": errs}
        def tab(name, label):
            pg.click(f"text={label}"); pg.wait_for_timeout(500)
        if pg.locator("#perf-seg").count():
            tab("perf", "Historical Scenario")
            out["scenario_table"] = pg.inner_text("#scn-tbl"); out["scenario_note"] = pg.inner_text("#perf-note")[:200]
            pg.screenshot(path=os.path.join(ROOT, "tests", f"e2e_{label}_scenario.png"))
            pg.click("text=Drawdown"); pg.wait_for_timeout(300)
            tab("hyp", "Hypotheses"); out["hypotheses"] = pg.inner_text("#hyp-tbl")
            pg.screenshot(path=os.path.join(ROOT, "tests", f"e2e_{label}_hyp.png"))
            tab("mc", "Monte Carlo"); out["mc_table"] = pg.inner_text("#mc-tbl")[:400]; out["mc_note"] = pg.inner_text("#mc-note")[:160]
            pg.screenshot(path=os.path.join(ROOT, "tests", f"e2e_{label}_mc.png"))
            out["fund_row0"] = pg.inner_text("#fund-tbl tr:first-child") if pg.locator("#fund-tbl tr").count() else None
        pg.screenshot(path=os.path.join(ROOT, "tests", f"e2e_{label}.png"))
        b.close(); return out

if __name__ == "__main__":
    res = []
    fxdir = make_fixture_dir("/tmp/e2e_fixture_data"); DS.DATA_DIR = fxdir
    ds = DF.fetch_nav_data(mf=FakeProvider()); st = A.build_state(ds)
    s1 = serve(st, 5871); res.append(run(5871, "live_fixture")); s1.shutdown()
    DS.DATA_DIR = os.path.join(ROOT, "data")        # the shipped REAL datasets (G-Sec levels not supplied)
    st2 = A.build_state(DF.load_dataset()); s2 = serve(st2, 5872); res.append(run(5872, "real_data")); s2.shutdown()
    s3 = serve(A._unavailable_state("simulated outage"), 5873); res.append(run(5873, "unavailable")); s3.shutdown()
    print(json.dumps(res, indent=1))
