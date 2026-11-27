import json, subprocess, sys, os
from fakes import FakeProvider
import universe as U

def test_verify_fund_tool_on_fixture(tmp_path):
    prov = FakeProvider()
    code = U.SCHEME_SPECS["Tata"]["code"]
    raw = prov.raw[code]; raw = {"meta": {k: raw[k] for k in ("scheme_name", "scheme_category", "scheme_code")}, "data": raw["data"]}
    f = tmp_path / "raw.json"; f.write_text(json.dumps(raw))
    root = os.path.dirname(os.path.dirname(__file__))
    out = subprocess.run([sys.executable, os.path.join(root, "tools", "verify_fund.py"), code, "--file", str(f), "--label", "Tata"],
                         capture_output=True, text=True, cwd=str(tmp_path))
    assert out.returncode == 0, out.stderr
    assert "PASS" in out.stdout
    import re
    vols = re.findall(r"vol\s+([\d.]+)%|annualised vol\s+:\s+([\d.]+)%", out.stdout)
    nums = [float(a or b) for a, b in vols]
    assert len(nums) == 2 and abs(nums[0] - nums[1]) < 0.01     # independent == pipeline
