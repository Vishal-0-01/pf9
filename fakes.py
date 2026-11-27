"""Fake NAV provider built from SYNTHETIC test fixtures (clearly not real fund data)."""
import numpy as np, pandas as pd
from universe import SCHEME_SPECS

NAMES = {
 "Parag Parikh": "Parag Parikh Flexi Cap Fund - Direct Plan - Growth",
 "Quant": "Quant Flexi Cap Fund - Growth Option - Direct Plan",
 "Motilal Oswal": "Motilal Oswal Flexi Cap Fund - Direct Plan - Growth",
 "HDFC": "HDFC Flexi Cap Fund - Growth Option - Direct Plan",
 "Aditya Birla": "Aditya Birla Sun Life Flexi Cap Fund - Growth - Direct Plan",
 "ITI": "ITI Flexi Cap Fund - Direct Plan - Growth Option",
 "Tata": "Tata Flexi Cap Fund - Direct Plan - Growth",
 "Invesco": "Invesco India Flexi Cap Fund - Direct Plan - Growth",
 "Bank of India": "Bank of India Flexi Cap Fund - Direct Plan - Growth",
 "HSBC": "HSBC Flexi Cap Fund - Direct Plan - Growth",
 "Edelweiss": "Edelweiss Flexi Cap Fund - Direct Plan - Growth",
 "WhiteOak": "WhiteOak Capital Flexi Cap Fund - Direct Plan - Growth",
}
PROXY_NAME = "UTI Nifty 50 Index Fund - Direct Plan - Growth Option"

def make_nav(n=1500, seed=0, mu=0.13, sig=0.15, start="2020-06-01", common=None):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    mkt = common if common is not None else rng.normal(0, 0.009, n)
    r = 0.7 * mkt + rng.normal(mu / 252, sig / np.sqrt(252) * 0.7, n)
    return pd.Series(100 * np.cumprod(1 + r), index=idx)

def raw_from(nav, name, code, category="Equity Scheme - Flexi Cap Fund"):
    return {"scheme_name": name, "scheme_category": category, "scheme_code": code, "fund_house": name.split()[0],
            "data": [{"date": d.strftime("%d-%m-%Y"), "nav": f"{v:.4f}"} for d, v in nav[::-1].items()]}

class FakeProvider:
    def __init__(self, n=1500, overrides=None, fail=(), drop=None, names=None):
        self.calls = []
        self.fail = set(map(str, fail))
        self.overrides = overrides or {}
        rng = np.random.default_rng(1)
        common = rng.normal(0, 0.009, n)
        self.raw = {}
        for i, (label, spec) in enumerate(SCHEME_SPECS.items()):
            nav = make_nav(n, seed=10 + i, common=common)
            if drop and label in drop:
                nav = nav.drop(nav.index[drop[label]])
            nm = (names or {}).get(label, NAMES[label])
            self.raw[spec["code"]] = raw_from(nav, nm, spec["code"])
        self.raw.update(self.overrides)

    def get_scheme_historical_nav(self, code):
        self.calls.append(str(code))
        if str(code) in self.fail:
            raise ConnectionError("simulated provider outage")
        return self.raw.get(str(code))
