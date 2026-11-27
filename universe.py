"""
universe.py - the fund universe and scheme-identity validation.

Single source of truth for the 12 schemes of the study universe (final dataset supplied by the author). data_fetcher.py, optimizer.py,
app.py and the tests all import from here, so the universe cannot drift.

Every scheme must be a Direct Plan, Growth-option Flexi Cap fund of the stated
asset management company. A scheme code is only accepted if the scheme name
returned by the data provider passes validate_scheme_identity(). A code whose
name cannot be read, or does not match, is REJECTED (never silently accepted).

The universe is the 12 funds present in the author's TIGZIG NAV file (mfpro_nav_12funds_2026-11-26.csv).
ICICI Prudential (148990) and Kotak (120166) were part of the earlier 14-fund list but are NOT in the supplied
dataset and are therefore not in the study universe; no substitute funds were introduced. The scheme names in the
supplied file are validated against these specifications at load time (see /api/data-quality).
"""

from collections import OrderedDict
import re

# label -> {"code": AMFI scheme code, "amc": tokens that must appear in the name}
SCHEME_SPECS = OrderedDict([
    ("Parag Parikh",  {"code": "122639", "amc": ["parag parikh"]}),
    ("Quant",         {"code": "120843", "amc": ["quant"]}),
    ("Motilal Oswal", {"code": "129046", "amc": ["motilal"]}),
    ("HDFC",          {"code": "118955", "amc": ["hdfc"]}),
    ("Aditya Birla",  {"code": "120564", "amc": ["aditya birla"]}),
    ("ITI",           {"code": "151379", "amc": ["iti"]}),
    ("Tata",          {"code": "144546", "amc": ["tata"]}),
    ("Invesco",       {"code": "149763", "amc": ["invesco"]}),
    ("Bank of India", {"code": "148404", "amc": ["bank of india"]}),
    ("HSBC",          {"code": "120046", "amc": ["hsbc"]}),
    ("Edelweiss",     {"code": "140353", "amc": ["edelweiss"]}),
    ("WhiteOak",      {"code": "150346", "amc": ["whiteoak"]}),
])

SELECTED_SCHEMES = OrderedDict((k, v["code"]) for k, v in SCHEME_SPECS.items())
FUNDS_DEFAULT = list(SCHEME_SPECS.keys())

# Beta / alpha are computed against the Nifty 50 Total Returns Index dataset (data/cleaned/nifty_tri.csv), not a fund.
MARKET_PROXY = None

_BAD_WORDS = ["idcw", "dividend", "payout", "reinvest", "bonus", "regular",
              "fund of fund", "fund of funds", "fof", "ultra short", "liquid",
              "overnight", "money market", "nasdaq", "s&p", "overseas",
              "international", "etf", "arbitrage", "debt", "gilt", "bond",
              "multi cap", "multicap", "large cap", "largecap", "mid cap",
              "midcap", "small cap", "smallcap", "focused", "elss", "tax saver",
              "value fund", "contra", "dynamic asset"]


def _norm(text):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9& ]", " ", str(text).lower())).strip()


def validate_scheme_identity(label, code, meta, spec=None):
    """
    Check that provider metadata matches the expected scheme.

    meta: dict with at least "scheme_name"; optionally "scheme_category",
          "fund_house", "scheme_code".
    Returns (ok: bool, reasons: list[str]). ok is True only if every check
    passes. Missing metadata is a rejection.
    """
    if spec is None:
        spec = SCHEME_SPECS.get(label)
    reasons = []
    if spec is None:
        return False, [f"unknown label {label!r}"]
    if str(code) != str(spec["code"]):
        reasons.append(f"code {code} differs from configured code {spec['code']}")

    name = _norm((meta or {}).get("scheme_name", ""))
    if not name:
        return False, reasons + ["provider returned no scheme name; identity cannot be confirmed"]

    pcode = (meta or {}).get("scheme_code")
    if pcode not in (None, "") and str(pcode) != str(code):
        reasons.append(f"provider scheme_code {pcode} != requested {code}")

    haystack = name + " " + _norm((meta or {}).get("fund_house", ""))
    for tok in spec.get("amc", []):
        if not re.search(r"\b" + re.escape(tok) + r"\b", haystack):
            reasons.append(f"name lacks asset manager token '{tok}'")

    if spec.get("kind") == "index":
        if "nifty 50" not in name and "nifty50" not in name:
            reasons.append("name does not contain 'Nifty 50'")
        if "index" not in name:
            reasons.append("name does not contain 'Index'")
        bad = [w for w in ("next 50", "midcap", "smallcap", "bank", "it ", "momentum", "value",
                           "equal weight", "alpha", "quality", "idcw", "dividend", "regular",
                           "etf", "fund of fund") if w in name + " "]
    else:
        if "flexi" not in name:
            reasons.append("name does not contain 'Flexi Cap'")
        bad = [w for w in _BAD_WORDS if re.search(r"\b" + re.escape(w) + r"\b", name)]
    if "direct" not in name:
        reasons.append("not a Direct Plan")
    if "growth" not in name:
        reasons.append("not the Growth option")
    if bad:
        reasons.append("name contains excluded term(s): " + ", ".join(sorted(set(bad))))

    cat = _norm((meta or {}).get("scheme_category", ""))
    if cat and spec.get("kind") != "index" and "flexi" not in cat:
        reasons.append(f"provider category '{cat}' is not Flexi Cap")

    return (len(reasons) == 0), reasons
