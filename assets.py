"""
assets.py - the three-asset-class layer (equity, debt, gold). Cash is not an asset class in this framework.

Shared by optimizer.py (live Stage 2/3), scenario.py (historical evaluation), montecarlo.py and verify.py so that
every part of the application uses the same covariance structure and the same allocation rule.

Covariance structure
  Portfolio vector p = [E * w (n funds), D, G].  Variance = p' S p, where S is the (n+2)x(n+2) covariance of
  [funds, debt, gold]:   S = [[S_ff, S_fd], [S_fd', S_dd]]   (S_ff: Ledoit-Wolf shrunk fund covariance).
  S_fd and S_dd are estimated from actual daily returns on commonly observed dates when the dataset for that asset
  is available.  For an asset with NO dataset the model falls back to ASSUMED constants (volatility and a correlation
  with every fund) and says so in `source` - those numbers are assumptions, not observations.
  The cross blocks travel with the covariance matrix as an attribute (CovModel), so every function that receives
  `cov_mat` sees the same cross-asset structure.

Allocation across the three asset classes
  The optimiser (optimizer.py) chooses E, D, G and the fund weights jointly. This module contains no allocation rule,
  no minimum or maximum weight for any asset class and no fixed percentage: only the covariance structure, the
  assumed fallback constants for an asset whose dataset is missing, and valuation context statistics.
"""
import numpy as np

DEF_ORDER = ("debt", "gold")

# ASSUMED constants - used ONLY for an asset whose dataset is unavailable (flagged in `source`)
DEBT_RET, DEBT_VOL = 0.075, 0.040
GOLD_RET, GOLD_VOL = 0.100, 0.150
RHO_EQ_DEBT, RHO_EQ_GOLD, RHO_DEBT_GOLD = -0.10, 0.05, 0.00
ASSUMED = {"debt": (DEBT_RET, DEBT_VOL, RHO_EQ_DEBT), "gold": (GOLD_RET, GOLD_VOL, RHO_EQ_GOLD)}


class CovModel(np.ndarray):
    """Fund covariance matrix that carries its cross-asset blocks as `.cross`."""

    def __new__(cls, arr, cross=None):
        obj = np.asarray(arr, dtype=float).view(cls)
        obj.cross = cross
        return obj

    def __array_finalize__(self, obj):
        self.cross = getattr(obj, "cross", None)

    def __reduce__(self):                 # keep `.cross` through pickling (state cache) and copy.deepcopy
        base = super().__reduce__()
        return (base[0], base[1], base[2] + (self.cross,))

    def __setstate__(self, state):
        self.cross = state[-1]
        super().__setstate__(state[:-1])


def assumed_cross(cov_funds, which=DEF_ORDER):
    """Cross blocks built entirely from the assumed constants (no datasets)."""
    cov = np.asarray(cov_funds, float)
    n = cov.shape[0]
    sig_f = np.sqrt(np.diag(cov))
    cov_fd = np.zeros((n, 2)); cov_dd = np.zeros((2, 2)); mu = np.zeros(2)
    for k, a in enumerate(DEF_ORDER):
        r, s, rho = ASSUMED[a]
        cov_fd[:, k] = rho * sig_f * s
        cov_dd[k, k] = s * s
        mu[k] = r
    cov_dd[0, 1] = cov_dd[1, 0] = RHO_DEBT_GOLD * DEBT_VOL * GOLD_VOL
    return {"cov_fd": cov_fd, "cov_dd": cov_dd, "mu": mu,
            "source": {"debt": "assumed_constants", "gold": "assumed_constants"}, "window": None, "n_obs": 0}


def cross_of(cov_mat):
    c = getattr(cov_mat, "cross", None)
    return c if c is not None else assumed_cross(cov_mat)


def full_covariance(cov_funds, cross):
    n = np.asarray(cov_funds).shape[0]
    S = np.zeros((n + 2, n + 2))
    S[:n, :n] = np.asarray(cov_funds, float)
    S[:n, n:] = cross["cov_fd"]; S[n:, :n] = cross["cov_fd"].T
    S[n:, n:] = cross["cov_dd"]
    return S


def asset_cov3(w, cov_mat):
    """3x3 covariance of (equity sleeve with fund weights w, debt, gold)."""
    w = np.asarray(w, float)
    c = cross_of(cov_mat)
    S = np.zeros((3, 3))
    S[0, 0] = float(w @ np.asarray(cov_mat, float) @ w)
    S[0, 1:] = w @ c["cov_fd"]; S[1:, 0] = S[0, 1:]
    S[1:, 1:] = c["cov_dd"]
    return S


def vol3(E, D, G, S3):
    p = np.array([E, D, G], float)
    return float(np.sqrt(max(p @ S3 @ p, 0.0)))


SYNOPSIS_STATS = {"pe": (22.0, 5.0), "pb": (3.2, 0.6)}
MIN_VALUATION_MONTHS = 60


def valuation_stats(val_df, as_of, method="expanding", lag_months=0):
    """
    CONTEXT ONLY (not an optimiser input): mean and standard deviation used to express the current P/E and P/B as
    z-scores relative to their own history up to a date, for display. The equity weight rule uses the combined z-score.
      expanding            all monthly readings dated on or before as_of (shifted back by lag_months); at least
                           MIN_VALUATION_MONTHS readings, else the synopsis constants are used and flagged.
      synopsis_constants   P/E 22.0 +/- 5.0, P/B 3.2 +/- 0.6 (approved synopsis; origin not documented there).
    """
    import pandas as pd
    if method == "synopsis_constants" or val_df is None:
        return {"method": "synopsis_constants", **{k: v for k, v in SYNOPSIS_STATS.items()}, "n_months": None}
    cutoff = pd.Timestamp(as_of) - pd.DateOffset(months=lag_months)
    h = val_df.loc[:cutoff]
    if len(h) < MIN_VALUATION_MONTHS:
        return {"method": "synopsis_constants (fallback: fewer than %d monthly readings)" % MIN_VALUATION_MONTHS,
                **SYNOPSIS_STATS, "n_months": int(len(h))}
    return {"method": "expanding_window", "pe": (float(h["pe"].mean()), float(h["pe"].std(ddof=1))),
            "pb": (float(h["pb"].mean()), float(h["pb"].std(ddof=1))), "n_months": int(len(h)),
            "history_start": str(h.index[0].date()), "history_end": str(h.index[-1].date())}


# ── cross-asset covariance from actual daily data ────────────────────────────

MAX_GAP_DAYS_RET = 7          # a return across a longer calendar gap is dropped (same rule as fund returns)
MAX_ABS_RET = 0.15
MIN_CROSS_OBS = 120


def _levels_returns(levels):
    """levels: DataFrame of levels on a common date index (no NaN). Returns returns across consecutive common dates."""
    rets = levels.pct_change(fill_method=None).iloc[1:]
    gaps = levels.index.to_series().diff().dt.days.iloc[1:].values
    ok = (gaps <= MAX_GAP_DAYS_RET) & (rets.abs() <= MAX_ABS_RET).all(axis=1).values
    return rets[ok]


def build_cross(cov_funds, fund_levels, def_levels, start=None, end=None):
    """
    Cross blocks S_fd (n x 2), S_dd (2 x 2) for [debt, gold] from ACTUAL daily returns.

    fund_levels : DataFrame of fund NAV levels (columns = funds, in the order of cov_funds)
    def_levels  : dict {"debt": Series, "gold": Series} of dataset levels (missing keys -> assumed constants)
    start, end  : estimation window (the common fund window).  Returns are formed across consecutive dates on
                  which the funds and the dataset(s) are ALL observed (no filling); the annualisation factor is the
                  actual number of observations per year in that sample.
    Variances/covariances of a historical asset are unshrunk sample values; the funds block is the (Ledoit-Wolf)
    cov_funds passed in. The assembled (n+2) matrix is checked for positive semi-definiteness and repaired
    (variance preserving) only if needed; the repair is reported.
    """
    import pandas as pd
    from optimizer import repair_covariance, cov_health
    cov_funds = np.asarray(cov_funds, float)
    n = cov_funds.shape[0]
    base = assumed_cross(cov_funds)
    avail = [a for a in DEF_ORDER if a in def_levels and def_levels[a] is not None and len(def_levels[a]) > MIN_CROSS_OBS]
    cross = {**base, "source": dict(base["source"]), "mu": base["mu"].copy(), "cov_fd": base["cov_fd"].copy(),
             "cov_dd": base["cov_dd"].copy(), "window": None, "n_obs": 0, "repair": None}
    if not avail:
        return cross
    fl = fund_levels
    if start is not None:
        fl = fl.loc[pd.Timestamp(start):]
    if end is not None:
        fl = fl.loc[:pd.Timestamp(end)]
    parts = [fl] + [def_levels[a].rename(a) for a in avail]
    lv = pd.concat(parts, axis=1, join="inner").dropna()
    rets = _levels_returns(lv)
    if len(rets) < MIN_CROSS_OBS:
        cross["note"] = f"only {len(rets)} joint observations; assumed constants kept for the defensive assets"
        return cross
    years = (rets.index[-1] - lv.index[0]).days / 365.25
    ppy = len(rets) / years
    S = rets.cov().values * ppy
    fcols = list(fl.columns)
    pos = {a: len(fcols) + i for i, a in enumerate(avail)}
    for a in avail:
        k = DEF_ORDER.index(a)
        cross["cov_fd"][:, k] = S[:n, pos[a]]
        cross["cov_dd"][k, k] = S[pos[a], pos[a]]
        cross["source"][a] = "historical_daily_data"
        ser = def_levels[a]
        r_full = ser.pct_change(fill_method=None).dropna()
        gaps = ser.index.to_series().diff().dt.days.iloc[1:]
        r_full = r_full[(gaps.values <= MAX_GAP_DAYS_RET)]
        yrs = (ser.index[-1] - ser.index[0]).days / 365.25
        cross["mu"][k] = float(r_full.mean() * len(r_full) / yrs)      # annualised arithmetic mean, full history
    if len(avail) == 2:
        cross["cov_dd"][0, 1] = cross["cov_dd"][1, 0] = S[pos["debt"], pos["gold"]]
    else:
        # one historical asset: covariance with the assumed one stays at the assumed correlation
        pass
    Sfull = full_covariance(cov_funds, cross)
    before = cov_health(Sfull)
    Sfull, rep = repair_covariance(Sfull)
    if rep.get("repaired"):
        cross["cov_fd"] = Sfull[:n, n:]; cross["cov_dd"] = Sfull[n:, n:]
    cross["repair"] = {"before": before, "repaired": bool(rep.get("repaired"))}
    cross["window"] = [str(rets.index[0].date()), str(rets.index[-1].date())]
    cross["n_obs"] = int(len(rets)); cross["obs_per_year"] = float(ppy)
    return cross
