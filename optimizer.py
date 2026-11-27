"""
optimizer.py — Robust Portfolio Optimization Engine
=====================================================
Production-grade Markowitz optimizer for Indian flexi-cap AMC portfolios.

Upgrades over naive implementation:
  1. Return shrinkage (James-Stein toward Nifty long-run ~12%)
  2. Ledoit-Wolf analytical covariance shrinkage
  3. Diversification penalty (L2/HHI) in objective function
  4. Turnover penalty (transaction cost model, 20bps one-way)
  5. Semi-variance / Sortino as secondary risk metric
  6. Data quality flags (CAGR cap 30%; low volatility flagged, never altered)
  7. (historical scenario, hypotheses and Monte Carlo live in scenario.py, hypotheses.py, montecarlo.py)
     Asset classes: equity (fund sleeve), debt, gold. Cash is not an asset class (see assets.py).
  8. Benchmarking vs Nifty TRI and equal-weight portfolio
  9. 3-layer vol constraint fallback chain (maintained)
 10. Geometric CAGR via log-return method (maintained)

Flask/frontend API contract: unchanged. All existing response keys preserved.
New keys added to API responses (additive, non-breaking).
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import logging

from universe import FUNDS_DEFAULT
from assets import (CovModel, assumed_cross, cross_of, asset_cov3, vol3, valuation_stats,  # noqa: F401
                    SYNOPSIS_STATS, DEBT_RET, DEBT_VOL, GOLD_RET, GOLD_VOL,
                    RHO_EQ_DEBT, RHO_EQ_GOLD, RHO_DEBT_GOLD)

logger = logging.getLogger(__name__)


# ── GLOBAL PARAMETERS ────────────────────────────────────────────────────────

RF          = 0.065   # India 91-day T-bill risk-free rate
VOL_CAP     = 0.10    # Hard portfolio volatility cap
MIN_FUND_W  = 0.03    # Minimum fund weight in equity sleeve
MAX_FUND_W  = 0.25    # Maximum fund weight in equity sleeve

# Synopsis valuation constants: used ONLY as context statistics for z-score display (not an optimiser input).
NIFTY_MEAN_PE, NIFTY_STD_PE = 22.0, 5.0
NIFTY_MEAN_PB, NIFTY_STD_PB = 3.2,  0.6

# Equity weight rule (author's specification, 27 Nov 2026): equity is 70% at fair valuation (z = 0), moves linearly
# with the combined P/E and P/B z-score, and is limited to 50% (rich) to 90% (cheap):
#       E_target = clip(0.70 - 0.10 z, 0.50, 0.90)
# Only the EQUITY weight follows this rule. The split of the remainder between debt and gold, and the fund weights
# inside the equity sleeve, are chosen by the optimiser (no minimum for debt or gold). The hard 10% portfolio
# volatility cap has priority: if E_target cannot meet it, equity is reduced below the target (never below 50%) and the
# result is flagged; if even 50% cannot meet it the optimiser reports infeasibility.
OBJECTIVE_DEFAULT = "return"   # dissertation specification: maximise expected return under the hard cap ("sharpe" is an unused option)
EQUITY_NEUTRAL = 0.70
EQUITY_SLOPE   = 0.10
EQUITY_MIN     = 0.50
EQUITY_MAX     = 0.90

# Debt / gold parameters live in assets.py: they are ASSUMED constants used only when the dataset for that
# asset is not available, and the API labels them as assumed.  Cash is not an asset class.

# ── RETURN SHRINKAGE PARAMETERS ───────────────────────────────────────────────
# Shrink raw historical CAGR toward the long-run Nifty market return.
# Rationale: individual fund CAGRs are estimated on limited samples;
# shrinkage reduces the impact of estimation error on optimized weights.
NIFTY_LONGRUN_RETURN = 0.12   # ~12% long-run geometric CAGR for Indian large/flexi-cap
SHRINKAGE_ALPHA      = 0.25   # 25% weight on market prior, 75% on historical
                               # Higher = more conservative, less sensitive to recent returns
                               # Range: 0.15 (less shrinkage) to 0.40 (more conservative)

# ── REGULARISATION PARAMETERS ─────────────────────────────────────────────────
LAMBDA_DIVERSIFICATION = 0.08   # L2 penalty coefficient on fund weight concentration
                                  # Penalises HHI: adds lambda*sum(w^2) to cost
                                  # 0 = pure Markowitz; 0.20 = strong equal-weight pull
LAMBDA_TURNOVER       = 0.30    # Turnover penalty coefficient
                                  # Multiplied by sum(|w_new - w_prev|) * TC_BPS/10000
TC_BPS                = 20      # One-way transaction cost (bps) — approx for MF STP

# ── DATA QUALITY THRESHOLDS ───────────────────────────────────────────────────
MAX_CAGR_THRESHOLD    = 0.30    # Exclude funds with CAGR > 30% (likely window bias)
MIN_VOL_THRESHOLD     = 0.05    # Exclude funds with vol < 5% (data issue / money market)



# ── RETURN ESTIMATION ─────────────────────────────────────────────────────────

def _geometric_cagr(daily_returns_series):
    """
    Geometric CAGR from daily returns series via log-return method.
    ann = exp(mean(log(1+r)) * 252) - 1
    Removes the +vol^2/2 upward bias of arithmetic annualization.
    """
    r = daily_returns_series.dropna().values
    if len(r) < 2:
        return RF
    return float(np.expm1(np.log1p(r).mean() * 252))


def shrink_returns(raw_returns, n_obs_per_fund, market_return=NIFTY_LONGRUN_RETURN,
                   alpha=SHRINKAGE_ALPHA):
    """
    James-Stein / Bayes-Stein shrinkage toward market return.

    Formula: mu_shrunk = (1 - alpha) * mu_hist + alpha * mu_market

    Rationale:
      - Raw historical CAGR is a noisy estimator on 3-5 year windows.
      - Shrinking toward a market prior reduces estimation error.
      - alpha = 0.25 means 75% historical, 25% prior. Adjustable.
      - Funds with shorter history get stronger shrinkage (larger alpha).

    Returns shrunk returns array.
    """
    n = len(raw_returns)
    shrunk = np.zeros(n)
    for i in range(n):
        # Scale shrinkage by effective sample: fewer obs → more shrinkage
        n_i = n_obs_per_fund[i]
        # Effective alpha: higher for shorter history
        alpha_eff = min(0.60, alpha * (756 / max(n_i, 1)))
        alpha_eff = max(alpha, alpha_eff)   # at least base alpha
        shrunk[i] = (1 - alpha_eff) * raw_returns[i] + alpha_eff * market_return
    return shrunk


# ── COVARIANCE HEALTH, EIGENVALUES AND PSD REPAIR ─────────────────────────────

COV_EIG_FLOOR = 1e-6      # smallest allowed eigenvalue of the CORRELATION matrix after repair


class InsufficientDataError(ValueError):
    """Raised when the returns table cannot support a covariance estimate."""


def cov_health(A):
    """Diagnostics of a square matrix: finiteness, symmetry, diagonal range, extreme eigenvalues."""
    A = np.asarray(A, dtype=float)
    h = {"shape": list(A.shape), "n_nan": int(np.isnan(A).sum()), "n_inf": int(np.isinf(A).sum())}
    fin = np.isfinite(A).all()
    h["finite"] = bool(fin)
    if fin and A.ndim == 2 and A.shape[0] == A.shape[1]:
        h["max_asymmetry"] = float(np.abs(A - A.T).max())
        d = np.diag(A)
        h["diag_min"], h["diag_max"] = float(d.min()), float(d.max())
        try:
            ev = _eigvalsh_safe(A)[0]
            h["min_eigenvalue"], h["max_eigenvalue"] = float(ev.min()), float(ev.max())
        except Exception as exc:
            h["eigen_error"] = f"{type(exc).__name__}: {exc}"
    return h


def _eigvalsh_safe(A):
    """
    Eigenvalues of a symmetric matrix. Tries numpy, then scipy LAPACK drivers
    (evr, evx, ev). Returns (eigenvalues, driver_used). Raises LinAlgError only if
    every driver fails. Non-finite input is rejected up front (LAPACK reports
    that as the unhelpful 'Eigenvalues did not converge').
    """
    A = np.asarray(A, dtype=float)
    if not np.isfinite(A).all():
        raise InsufficientDataError(f"matrix contains non-finite values: {cov_health(A)}")
    A = (A + A.T) / 2.0
    errs = []
    try:
        return np.linalg.eigvalsh(A), "numpy.eigvalsh"
    except np.linalg.LinAlgError as exc:
        errs.append(f"numpy: {exc}")
    import scipy.linalg as sla
    for drv in ("evr", "evx", "ev"):
        try:
            return sla.eigh(A, eigvals_only=True, driver=drv, check_finite=False), f"scipy.{drv}"
        except Exception as exc:
            errs.append(f"scipy.{drv}: {exc}")
    raise np.linalg.LinAlgError("all eigenvalue drivers failed: " + " | ".join(errs))


def repair_covariance(S, floor=COV_EIG_FLOOR):
    """
    Symmetrise S and make it positive definite WITHOUT changing any variance.

    Pairwise-complete covariance estimation is not guaranteed PSD. If the smallest
    eigenvalue of the correlation matrix is below `floor`, the eigenvalues are
    clipped at `floor` and the matrix is rescaled to unit diagonal (nearest-PD
    style correction); the repaired covariance keeps the ORIGINAL diagonal, so
    volatilities are identical before and after repair.
    Returns (matrix, info dict). Raises InsufficientDataError on non-finite input
    or non-positive variances.
    """
    S = np.asarray(S, dtype=float)
    if not np.isfinite(S).all():
        raise InsufficientDataError(f"covariance is not finite: {cov_health(S)}")
    asym = float(np.abs(S - S.T).max())
    S = (S + S.T) / 2.0
    var = np.diag(S).copy()
    if (var < 0).any():
        raise InsufficientDataError(f"negative variance on the diagonal: {cov_health(S)}")
    d = np.sqrt(np.maximum(var, 1e-16))
    C = S / np.outer(d, d)
    np.fill_diagonal(C, 1.0)
    info = {"asymmetry_removed": asym, "repaired": False, "method": "none"}
    try:
        ev, drv = _eigvalsh_safe(C)
        info["eig_driver"] = drv
        info["min_corr_eigenvalue_before"] = float(ev.min())
        if ev.min() >= floor:
            return S, info
        w, V = np.linalg.eigh(C) if drv == "numpy.eigvalsh" else _eigh_fallback(C)
        w = np.clip(w, floor, None)
        C2 = (V * w) @ V.T
        dd = np.sqrt(np.diag(C2)); C2 = C2 / np.outer(dd, dd)
        info.update(repaired=True, method="eigenvalue_clipping_unit_diagonal")
    except np.linalg.LinAlgError as exc:                    # every eigen driver failed: eigen-free repair
        info["eig_error"] = str(exc)
        jit, C2 = 1e-8, None
        while jit <= 10.0:
            try:
                np.linalg.cholesky(C + jit * np.eye(len(C)))
                C2 = (C + jit * np.eye(len(C))) / (1.0 + jit)
                break
            except np.linalg.LinAlgError:
                jit *= 10
        if C2 is None:
            raise InsufficientDataError(f"covariance could not be made positive definite: {cov_health(S)}") from exc
        info.update(repaired=True, method=f"cholesky_diagonal_loading({jit:g})")
    S2 = C2 * np.outer(d, d)
    S2 = (S2 + S2.T) / 2.0
    return S2, info


def _eigh_fallback(C):
    import scipy.linalg as sla
    for drv in ("evr", "ev", "evd"):
        try:
            return sla.eigh(C, driver=drv, check_finite=False)
        except Exception:
            continue
    raise np.linalg.LinAlgError("eigenvector computation failed for all drivers")


def ledoit_wolf_shrinkage(sample_cov, n_obs, return_delta=False):
    """
    Analytical Ledoit-Wolf covariance shrinkage (Oracle Approximating Shrinkage).
    Shrinks toward scaled identity matrix (constant-correlation target).

    Target: mu_target * I where mu_target = trace(S)/n (average variance)

    Formula: Sigma_lw = (1 - delta) * S + delta * mu_target * I

    delta estimated analytically:
        delta* = min(1, (n+2) / ((n+2) + T * rho))
    where rho = ||S - mu*I||_F^2 / ||S||_F^2 (relative misfit)

    This is the simplified Oracle LW (Ledoit-Wolf 2004, simplified form).
    Full OAS or sklearn's LedoitWolf would be more accurate but this is
    sufficient and keeps the dependency footprint minimal.

    Returns shrunk covariance matrix (guaranteed PD).
    """
    n = sample_cov.shape[0]
    T = n_obs

    # Shrinkage target: scaled identity
    mu_target = np.trace(sample_cov) / n

    # Relative misfit of sample cov from target
    if not np.isfinite(sample_cov).all():
        raise InsufficientDataError(f"covariance is not finite: {cov_health(sample_cov)}")
    diff = sample_cov - mu_target * np.eye(n)
    rho  = np.linalg.norm(diff, 'fro') ** 2 / (np.linalg.norm(sample_cov, 'fro') ** 2 + 1e-12)

    # Oracle LW delta
    delta = min(1.0, max(0.0, (n + 2) / ((n + 2) + T * rho)))

    shrunk = (1 - delta) * sample_cov + delta * mu_target * np.eye(n)

    # Guarantee strict PD; variances are preserved by repair_covariance
    shrunk, _ = repair_covariance(shrunk)

    logger.info("LW shrinkage delta=%.4f  (n=%d, T=%d, rho=%.4f)", delta, n, T, rho)
    return (shrunk, delta) if return_delta else shrunk


# ── COVARIANCE & STATS BUILDER ────────────────────────────────────────────────

MIN_OVERLAP_OBS = 60     # fewest overlapping valid returns allowed for any pair of funds


def build_cov_from_returns(returns_df, min_overlap=MIN_OVERLAP_OBS):
    """
    Build return estimates, covariance, vols and Sharpe ratios from a table of
    daily returns (NaN = no valid observation for that fund on that day).

    Covariance uses PAIRWISE-COMPLETE observations only: for funds i, j the
    covariance is computed from the days on which both have a valid return.
    No value is filled or imputed.  If any pair has fewer than `min_overlap`
    overlapping observations (or a fund has fewer than 2) the function raises
    InsufficientDataError instead of returning a NaN-contaminated matrix.

    Pipeline:
      1. Geometric CAGR per fund (log-return method, each fund's own valid days)
      2. Data-quality FLAGS (CAGR above threshold is capped; low volatility is
         flagged only - volatility is never altered)
      3. Simplified Ledoit-Wolf covariance shrinkage toward a scaled identity
      4. Return shrinkage toward the long-run market prior
      5. Downside deviation and Sortino ratio per fund

    Returns (ann_returns, cov_mat, vols, fund_sharpes, extra_stats).
    """
    fund_cols = list(returns_df.columns)
    n_funds = len(fund_cols)
    if n_funds < 2:
        raise InsufficientDataError("need at least two funds")

    valid = returns_df.notna().values.astype(int)
    n_obs = valid.sum(axis=0)
    if (n_obs < 2).any():
        bad = [fund_cols[i] for i in np.where(n_obs < 2)[0]]
        raise InsufficientDataError(f"funds with fewer than 2 valid returns: {bad}")
    overlap = valid.T @ valid
    low = [(fund_cols[i], fund_cols[j], int(overlap[i, j]))
           for i in range(n_funds) for j in range(i + 1, n_funds) if overlap[i, j] < min_overlap]
    if low:
        raise InsufficientDataError(
            f"{len(low)} fund pair(s) have fewer than {min_overlap} overlapping observations, "
            f"e.g. {low[:3]}")

    # ── Step 1: geometric CAGR per fund (own valid days only) ────────────
    raw_cagr_uncapped = np.array([_geometric_cagr(returns_df[c].dropna()) for c in fund_cols])
    raw_cagr = raw_cagr_uncapped.copy()

    # ── Step 2: pairwise-complete sample covariance (annualised) ─────────
    vals = returns_df.values.astype(float)
    if np.isinf(vals).any():
        bad = [fund_cols[j] for j in np.where(np.isinf(vals).any(axis=0))[0]]
        raise InsufficientDataError(f"infinite return values in: {bad}")
    sample_cov_raw = returns_df.cov(min_periods=min_overlap).values * 252
    if not np.isfinite(sample_cov_raw).all():
        raise InsufficientDataError("covariance is not finite after pairwise estimation: "
                                    f"{cov_health(sample_cov_raw)}")
    health_before = cov_health(sample_cov_raw)
    sample_cov_raw, repair_info = repair_covariance(sample_cov_raw)   # symmetrise + variance-preserving PD fix
    psd_adjusted = bool(repair_info["repaired"])      # pairwise estimation need not be PSD
    raw_vols = np.sqrt(np.diag(sample_cov_raw))

    flagged = []
    for i, col in enumerate(fund_cols):
        reasons = []
        if raw_cagr[i] > MAX_CAGR_THRESHOLD:
            reasons.append(f"CAGR {raw_cagr[i]*100:.1f}% > {MAX_CAGR_THRESHOLD*100:.0f}% (capped for estimation)")
            raw_cagr[i] = MAX_CAGR_THRESHOLD
        if raw_vols[i] < MIN_VOL_THRESHOLD:
            reasons.append(f"vol {raw_vols[i]*100:.1f}% < {MIN_VOL_THRESHOLD*100:.0f}% (flag only; volatility NOT altered)")
        if reasons:
            flagged.append((col, reasons))
            logger.warning("Quality flag - %s: %s", col, "; ".join(reasons))

    # ── Step 3: Ledoit-Wolf type covariance shrinkage ────────────────────
    T_eff = int(np.median(n_obs))
    cov_mat, lw_delta = ledoit_wolf_shrinkage(sample_cov_raw, T_eff, return_delta=True)
    vols = np.sqrt(np.diag(cov_mat))

    # ── Step 4: return shrinkage toward market prior ─────────────────────
    ann_returns = shrink_returns(raw_cagr, n_obs)

    # ── Step 5: downside deviation and Sortino ───────────────────────────
    semi_vols = np.zeros(n_funds)
    sortinos = np.zeros(n_funds)
    daily_rf = RF / 252
    for i, col in enumerate(fund_cols):
        r = returns_df[col].dropna().values
        down = r[r < daily_rf] - daily_rf
        semi_vols[i] = float(np.sqrt(np.mean(down ** 2) * 252)) if len(down) > 10 else vols[i]
        sortinos[i] = float((ann_returns[i] - RF) / semi_vols[i]) if semi_vols[i] > 0 else 0.0

    fund_sharpes = np.where(vols > 0, (ann_returns - RF) / vols, 0.0)

    logger.info("Returns: raw [%.1f%%-%.1f%%] -> shrunk [%.1f%%-%.1f%%]",
                raw_cagr.min() * 100, raw_cagr.max() * 100, ann_returns.min() * 100, ann_returns.max() * 100)
    logger.info("Vol: [%.1f%%-%.1f%%]  Sharpe: [%.2f-%.2f]", vols.min() * 100, vols.max() * 100,
                fund_sharpes.min(), fund_sharpes.max())

    extra_stats = {
        "semi_vols": semi_vols,
        "sortinos": sortinos,
        "raw_cagrs": raw_cagr,
        "raw_cagrs_uncapped": raw_cagr_uncapped,
        "raw_vols": raw_vols,
        "n_obs": n_obs,
        "min_pair_overlap": int(overlap[np.triu_indices(n_funds, 1)].min()),
        "psd_adjusted": psd_adjusted,
        "covariance_diagnostics": {"before_repair": health_before, "repair": repair_info,
                                   "n_obs_min": int(n_obs.min()), "n_obs_max": int(n_obs.max())},
        "lw_delta": float(lw_delta),
        "quality_flags": {col: reasons for col, reasons in flagged},
    }
    return ann_returns, cov_mat, vols, fund_sharpes, extra_stats


# ── VALUATION RULE FOR THE EQUITY WEIGHT ─────────────────────────────────────

def valuation_z(pe, pb, stats=None):
    """z-scores of P/E and P/B against reference statistics (expanding history, or the synopsis constants)."""
    st = stats or {"pe": SYNOPSIS_STATS["pe"], "pb": SYNOPSIS_STATS["pb"]}
    z_pe, z_pb = (pe - st["pe"][0]) / st["pe"][1], (pb - st["pb"][0]) / st["pb"][1]
    return z_pe, z_pb, (z_pe + z_pb) / 2


def equity_from_z(z, slope=EQUITY_SLOPE):
    return float(np.clip(EQUITY_NEUTRAL - slope * z, EQUITY_MIN, EQUITY_MAX))


# ── PORTFOLIO STATISTICS ──────────────────────────────────────────────────────

def portfolio_vol_from_weights(E, w_eq, D, G, cov_mat):
    """Volatility of p = [E*w, D, G] under the extended (funds, debt, gold) covariance. Returns (vol, sleeve_vol)."""
    w_eq = np.asarray(w_eq, float)
    S3 = asset_cov3(w_eq, cov_mat)
    return vol3(E, D, G, S3), float(np.sqrt(max(S3[0, 0], 0.0)))


def portfolio_return(E, w_eq, D, G, ann_returns, def_mu=None):
    """Expected portfolio return: fund-sleeve model return plus defensive means (historical or assumed; see source)."""
    mu = np.array([DEBT_RET, GOLD_RET]) if def_mu is None else np.asarray(def_mu, float)
    return E * float(np.dot(np.asarray(w_eq, float), ann_returns)) + D * mu[0] + G * mu[1]


def portfolio_semi_vol(E, w_eq, D, G, semi_vols):
    """First-order approximation: sigma_down_p ~ E * w'sigma_down (defensive legs ignored)."""
    return float(E * np.dot(w_eq, semi_vols))


def portfolio_sortino(E, w_eq, D, G, ann_returns, semi_vols, def_mu=None):
    ret = portfolio_return(E, w_eq, D, G, ann_returns, def_mu)
    s_vol = portfolio_semi_vol(E, w_eq, D, G, semi_vols)
    return float((ret - RF) / s_vol) if s_vol > 0 else 0.0


# ── OPTIMIZATION ─────────────────────────────────────────────────────────────

# ── FEASIBILITY HELPERS ──────────────────────────────────────────────────────

VOL_SOLVER_MARGIN = 1e-4     # solver targets (cap - margin) so rounding cannot breach the cap
VOL_FEAS_TOL      = 1e-9     # a returned portfolio must satisfy vol <= cap + this (numerical noise only)
WEIGHT_TOL        = 1e-6     # tolerance for bounds / sum-to-one on reported (rounded) weights
WEIGHT_DECIMALS   = 6


class OptimizationInfeasible(Exception):
    """No portfolio satisfying every constraint was found. Carries diagnostics."""

    def __init__(self, message, details=None):
        super().__init__(message)
        self.details = details or {}


def project_weights(w, lo=MIN_FUND_W, hi=MAX_FUND_W):
    """
    Euclidean projection of w onto {lo <= w_i <= hi, sum w_i = 1}, by bisection
    on a common shift. Raises OptimizationInfeasible if the bounds cannot sum to 1.
    """
    w = np.asarray(w, dtype=float)
    n = len(w)
    if n * lo > 1.0 + 1e-12 or n * hi < 1.0 - 1e-12:
        raise OptimizationInfeasible(
            f"weight bounds [{lo}, {hi}] cannot sum to 1 with {n} funds",
            {"n_funds": n, "min_funds": int(np.ceil(1.0 / hi)), "max_funds": int(np.floor(1.0 / lo))})
    a, b = lo - w.max() - 1.0, hi - w.min() + 1.0
    for _ in range(200):
        m = 0.5 * (a + b)
        if np.clip(w + m, lo, hi).sum() > 1.0:
            b = m
        else:
            a = m
    out = np.clip(w + 0.5 * (a + b), lo, hi)
    resid = 1.0 - out.sum()
    if abs(resid) > 1e-15:                       # spread residual over free weights
        free = np.where((out > lo + 1e-12) & (out < hi - 1e-12))[0]
        if len(free):
            out[free] += resid / len(free)
    return out


def check_portfolio(w, E, D, G, cov_mat, vol_cap=VOL_CAP):
    """Independent validation of every constraint. Returns a dict; ok only if ALL hold.
    Equity must lie in [50%, 90%]; debt and gold only need to be >= 0 (no minimum); E + D + G = 1."""
    w = np.asarray(w, dtype=float)
    vol, sleeve = portfolio_vol_from_weights(E, w, D, G, cov_mat)
    bounds_ok = bool(np.all(w >= MIN_FUND_W - WEIGHT_TOL) and np.all(w <= MAX_FUND_W + WEIGHT_TOL))
    sum_ok = bool(abs(w.sum() - 1.0) <= WEIGHT_TOL)
    alloc_ok = bool(abs(E + D + G - 1.0) <= 1e-9 and min(E, D, G) >= -1e-12)
    vol_ok = bool(vol <= vol_cap + VOL_FEAS_TOL)
    eq_ok = bool(EQUITY_MIN - 1e-6 <= E <= EQUITY_MAX + 1e-6)
    return {"ok": bounds_ok and sum_ok and alloc_ok and vol_ok and eq_ok, "equity_range_ok": eq_ok,
            "bounds_ok": bounds_ok, "sum_ok": sum_ok, "allocation_ok": alloc_ok, "vol_ok": vol_ok,
            "portfolio_vol": vol, "vol_cap": vol_cap, "min_weight": float(w.min()),
            "max_weight": float(w.max()), "weight_sum": float(w.sum())}


def _clean_alloc(E, D, G):
    """Non-negative, sums to exactly 1 at 6 decimals."""
    a = np.clip(np.array([E, D, G], float), 0.0, None)
    a = np.round(a / a.sum(), WEIGHT_DECIMALS)
    a[int(np.argmax(a))] += 1.0 - a.sum()
    return float(a[0]), float(a[1]), float(a[2])


def _round_alloc(E, D, G, dec=4):
    """Round to `dec` decimals so the three reported weights sum to exactly 1 (largest-remainder)."""
    a = np.array([E, D, G], float); q = 10 ** dec
    fl = np.floor(a * q); rem = int(round(q - fl.sum()))
    for i in np.argsort(-(a * q - fl))[:max(rem, 0)]:
        fl[i] += 1
    return tuple(float(x / q) for x in fl)


def _unpack(x, n):
    return np.asarray(x[:n], float), float(x[n]), float(x[n + 1]), float(x[n + 2])


def _joint_objective(x, n, mu_f, mu_d, prev, lambda_div=LAMBDA_DIVERSIFICATION, lambda_tc=LAMBDA_TURNOVER, tc_bps=TC_BPS,
                     criterion="sharpe", cov_mat=None):
    """Negative criterion plus the (soft) sleeve regularisers. criterion "sharpe": (return - RF) / volatility;
    "return": expected return. HHI and turnover act on the sleeve weights only."""
    s, E, D, G = _unpack(x, n)
    ret = E * float(s @ mu_f) + D * mu_d[0] + G * mu_d[1]
    crit = ret if criterion == "return" else (ret - RF) / max(_joint_vol(x, n, cov_mat), 1e-6)
    return -crit + lambda_div * float(s @ s) + lambda_tc * (tc_bps / 10000.0) * float(np.abs(s - prev).sum())


def _joint_vol(x, n, cov_mat):
    s, E, D, G = _unpack(x, n)
    return vol3(E, D, G, asset_cov3(s, cov_mat))


def _joint_starts(n, prev, seed):
    rng = np.random.default_rng(seed)
    sleeves = [np.ones(n) / n, prev] + [rng.dirichlet(np.ones(n) * 2) for _ in range(3)]
    allocs = [(0.3, 0.4, 0.3), (0.5, 0.3, 0.2), (0.7, 0.2, 0.1), (0.9, 0.05, 0.05), (0.2, 0.7, 0.1), (0.6, 0.1, 0.3)]
    starts = []
    for i, a in enumerate(allocs):
        for j, sl in enumerate(sleeves):
            if (i + j) % 2 == 0 or j == 0:
                starts.append(np.concatenate([project_weights(sl / sl.sum()), a]))
    return starts


def _solve_joint(mu_f, mu_d, cov_mat, prev, vol_cap, seed, objective="return", fix_E=None, max_starts=None):
    # objective: "sharpe" | "return" (both subject to vol <= vol_cap) | "min_variance"
    """
    Multi-start SLSQP over x = [sleeve weights (n), E, D, G].
      objective "return": maximise regularised expected return, subject to portfolio vol <= vol_cap (hard).
      objective "min_variance": minimise volatility (used for the fallback and the frontier's left end).
    Constraints: sum(sleeve) = 1, E + D + G = 1, 3% <= sleeve_i <= 25%, D, G >= 0 (no minimum for debt or gold),
    and E = fix_E when given (the valuation rule fixes the equity weight; D and G then split the remainder).
    Every candidate is rounded and re-validated by check_portfolio(); returns (s, E, D, G, obj) or None.
    """
    n = len(mu_f)
    target = vol_cap - VOL_SOLVER_MARGIN
    cons = [{'type': 'eq', 'fun': lambda x: np.sum(x[:n]) - 1.0},
            {'type': 'eq', 'fun': lambda x: x[n] + x[n + 1] + x[n + 2] - 1.0}]
    if fix_E is not None:
        cons.append({'type': 'eq', 'fun': lambda x: x[n] - fix_E})
    if objective in ("return", "sharpe"):
        cons.append({'type': 'ineq', 'fun': lambda x: target - _joint_vol(x, n, cov_mat)})
        fun = lambda x: _joint_objective(x, n, mu_f, mu_d, prev, criterion=objective, cov_mat=cov_mat)
    else:
        fun = lambda x: _joint_vol(x, n, cov_mat)
    bounds = [(MIN_FUND_W, MAX_FUND_W)] * n + [(0.0, 1.0)] * 3
    best = None
    starts = _joint_starts(n, prev, seed)
    if fix_E is not None:
        starts = [np.concatenate([x[:n], [fix_E, (1 - fix_E) * f, (1 - fix_E) * (1 - f)]]) for x, f in
                  zip(starts, np.resize([0.5, 0.2, 0.8, 0.35, 0.65], len(starts)))]
    if max_starts:
        starts = starts[:max_starts]
    for x0 in starts:
        try:
            r = minimize(fun, x0, method='SLSQP', bounds=bounds, constraints=cons,
                         options={'ftol': 1e-12, 'maxiter': 1500})
        except Exception:
            continue
        if not np.all(np.isfinite(r.x)):
            continue
        s, E, D, G = _unpack(r.x, n)
        try:
            s = np.round(project_weights(s), WEIGHT_DECIMALS)
        except OptimizationInfeasible:
            continue
        E, D, G = _clean_alloc(E, D, G)
        if not check_portfolio(s, E, D, G, cov_mat, vol_cap)["ok"]:
            continue
        val = fun(np.concatenate([s, [E, D, G]]))
        if best is None or val < best[4]:
            best = (s, E, D, G, val)
    return best


def _equity_ceiling_search(E_target, mu_f, mu_d, cov_mat, prev, vol_cap, seed):
    """
    Largest equity weight in [EQUITY_MIN, E_target] for which SOME fund / debt / gold allocation meets the volatility cap.
    Returns (E_feasible, minimum_vol_at_EQUITY_MIN or None). Feasibility is monotone in E (more equity, more risk),
    so bisection on the minimum achievable volatility is used.
    """
    def min_vol(E):
        m = _solve_joint(mu_f, mu_d, cov_mat, prev, 1.0, seed, objective="min_variance", fix_E=E, max_starts=3)
        return None if m is None else float(_joint_vol(np.concatenate([m[0], m[1:4]]), len(mu_f), cov_mat))
    ok = lambda v: v is not None and v <= vol_cap - 2 * VOL_SOLVER_MARGIN
    v_lo = min_vol(EQUITY_MIN)
    if not ok(v_lo):
        return None, v_lo
    lo, hi = EQUITY_MIN, E_target
    for _ in range(9):
        mid = (lo + hi) / 2
        if ok(min_vol(mid)):
            lo = mid
        else:
            hi = mid
    return lo, v_lo


def optimize_for_pe_pb(pe, pb, ann_returns, cov_mat, vols, fund_sharpes, funds,
                       vol_cap=VOL_CAP, prev_weights=None, semi_vols=None, val_stats=None, slope=EQUITY_SLOPE,
                       objective=OBJECTIVE_DEFAULT):
    """
    Main entry point.
      1. Equity weight from valuation: E_target = clip(0.70 - slope * z, 0.50, 0.90), z = mean of the P/E and P/B z-scores.
      2. Given E, ONE joint SLSQP chooses the fund weights (3% to 25% each, sum 1) and the debt / gold split of the
         remaining 1 - E (no minimum for either), maximising the Sharpe ratio (objective="sharpe", default) or the
         expected return (objective="return") subject to the full-covariance portfolio volatility <= vol_cap (hard).
      3. If E_target cannot meet the cap, E is lowered to the largest feasible value (never below 50%) and
         constraint_status = "equity_reduced_to_meet_vol_cap"; if even 50% cannot, OptimizationInfeasible is raised.
    Expected returns: fund returns are the shrunk historical CAGRs (ann_returns); debt and gold use their historical
    means (or flagged assumed constants when the dataset is absent).
    """
    n = len(funds)
    mu_f = np.asarray(ann_returns, float)
    cross = cross_of(cov_mat)
    mu_d = np.asarray(cross["mu"], float)
    st = val_stats or {"pe": SYNOPSIS_STATS["pe"], "pb": SYNOPSIS_STATS["pb"]}
    z_pe, z_pb, z = valuation_z(pe, pb, st)
    E_target = equity_from_z(z, slope)
    prev = np.ones(n) / n if prev_weights is None else np.asarray(prev_weights, float) / np.sum(prev_weights)
    if semi_vols is None:
        semi_vols = np.asarray(vols).copy()
    project_weights(np.ones(n) / n)
    seed = int(((pe or 0) * 100 + (pb or 0) * 10)) % (2 ** 31)

    sol, E_used, status = _solve_joint(mu_f, mu_d, cov_mat, prev, vol_cap, seed, objective=objective, fix_E=E_target), E_target, "target_met"
    if sol is None:
        E_used, v_min50 = _equity_ceiling_search(E_target, mu_f, mu_d, cov_mat, prev, vol_cap, seed)
        if E_used is None:
            raise OptimizationInfeasible(
                f"No allocation with equity of at least {EQUITY_MIN*100:.0f}% satisfies the {vol_cap*100:.0f}% volatility ceiling"
                + (f" (lowest achievable volatility at {EQUITY_MIN*100:.0f}% equity: {v_min50*100:.2f}%)." if v_min50 else "."),
                {"pe": pe, "pb": pb, "equity_target": E_target, "min_achievable_vol_at_50pct_equity": v_min50, "vol_cap": vol_cap})
        status = "equity_reduced_to_meet_vol_cap"
        sol = None
        for k in range(8):                      # step down a hair if the boundary point is numerically tight
            E_try = max(EQUITY_MIN, E_used - 0.002 * k)
            sol = _solve_joint(mu_f, mu_d, cov_mat, prev, vol_cap, seed, objective=objective, fix_E=E_try)
            if sol is not None:
                E_used = E_try
                break
        if sol is None:
            raise OptimizationInfeasible("could not resolve a feasible allocation near the volatility boundary",
                                         {"pe": pe, "pb": pb, "equity_target": E_target, "vol_cap": vol_cap})
    best_w, E_f, D_f, G_f = sol[0], sol[1], sol[2], sol[3]

    chk = check_portfolio(best_w, E_f, D_f, G_f, cov_mat, vol_cap)
    if not chk["ok"]:
        raise OptimizationInfeasible("final constraint validation failed", {"check": chk, "status": status})
    ret_final = portfolio_return(E_f, best_w, D_f, G_f, mu_f, mu_d)
    v_final, sl_vol = chk["portfolio_vol"], portfolio_vol_from_weights(E_f, best_w, D_f, G_f, cov_mat)[1]
    sharpe = float((ret_final - RF) / v_final) if v_final > 0 else 0.0
    sortino_p = portfolio_sortino(E_f, best_w, D_f, G_f, mu_f, semi_vols, mu_d)
    semi_vol_p = portfolio_semi_vol(E_f, best_w, D_f, G_f, semi_vols)
    Er, Dr, Gr = _round_alloc(E_f, D_f, G_f)
    at_zero = [k for k, v in (("debt", D_f), ("gold", G_f)) if v <= 1e-6]
    return {
        "pe": pe, "pb": pb, "z_pe": round(float(z_pe), 4), "z_pb": round(float(z_pb), 4), "z": round(float(z), 4),
        "z_basis": {k: (list(v) if isinstance(v, tuple) else v) for k, v in
                    (val_stats or {"method": "synopsis_constants", **SYNOPSIS_STATS}).items()},
        "equity_rule": f"E = clip({EQUITY_NEUTRAL:.2f} - {slope:.2f} z, {EQUITY_MIN:.2f}, {EQUITY_MAX:.2f})",
        "equity_target": round(E_target, 4),
        "equity": Er, "debt": Dr, "gold": Gr,
        "equity_range_ok": bool(EQUITY_MIN - 1e-6 <= E_f <= EQUITY_MAX + 1e-6),
        "constraint_status": status,
        "objective": objective,
        "allocation_method": "equity from the valuation rule; fund weights and the debt/gold split by joint SLSQP maximising "
                             + ("the Sharpe ratio" if objective == "sharpe" else "expected return") + " under the volatility cap (no minimum for debt or gold)",
        "expected_returns": {"equity_sleeve": float(best_w @ mu_f), "debt": float(mu_d[0]), "gold": float(mu_d[1])},
        "defensive_parameter_source": cross["source"],
        "debt_gold_at_zero": at_zero,
        "vol_ceiling_binding": bool(v_final >= vol_cap - 5 * VOL_SOLVER_MARGIN),
        "port_ret": round(ret_final, 4), "port_vol": round(v_final, 6),
        "sharpe": round(sharpe, 4), "eq_vol": round(sl_vol, 4),
        "fund_weights": [round(float(w), WEIGHT_DECIMALS) for w in best_w],
        "fund_sharpes": [round(float(s_), 3) for s_ in fund_sharpes],
        "fund_returns": [round(float(r), 4) for r in mu_f],
        "fund_vols": [round(float(v), 4) for v in vols],
        "constraint_ok": True, "status": status,
        "constraint_check": {k: (round(v, 8) if isinstance(v, float) else v) for k, v in chk.items()},
        "sortino": round(sortino_p, 4), "semi_vol": round(semi_vol_p, 4),
        "concentration_hhi": round(float(np.dot(best_w, best_w)), 4),
        "turnover": round(float(np.sum(np.abs(best_w - prev))), 4),
    }


# ── EFFICIENT FRONTIER ────────────────────────────────────────────────────────

def compute_frontier(ann_returns, cov_mat, n_points=9, vol_cap=VOL_CAP):
    """
    Return-risk points for the equity weights the valuation rule allows (50% to 90%): for each E the optimiser picks the
    fund weights and the debt/gold split that maximise expected return with the cap NOT applied, so the curve shows
    what the cap costs. Points above the cap are flagged and are not recommendable.
    """
    mu_f = np.asarray(ann_returns, float); n = len(mu_f)
    mu_d = np.asarray(cross_of(cov_mat)["mu"], float); prev = np.ones(n) / n
    out = []
    for E in np.linspace(EQUITY_MIN, EQUITY_MAX, n_points):
        sol = _solve_joint(mu_f, mu_d, cov_mat, prev, 1.0, 11, fix_E=float(E), max_starts=6)
        if sol is None:
            continue
        s, E_, D, G, _ = sol
        v = float(_joint_vol(np.concatenate([s, [E_, D, G]]), n, cov_mat))
        out.append({"v": round(v, 4), "r": round(portfolio_return(E_, s, D, G, mu_f, mu_d), 4), "E": round(E_, 3),
                    "D": round(D, 3), "G": round(G, 3), "within_vol_cap": bool(v <= vol_cap + VOL_FEAS_TOL)})
    return sorted(out, key=lambda x: x["v"])


# ── DIAGNOSTICS ───────────────────────────────────────────────────────────────

def compute_diagnostics(funds, ann_returns, cov_mat, vols, extra_stats):
    """Equal-weight comparison, correlations and fund Sortino (model basis)."""
    n = len(funds)
    cov = np.asarray(cov_mat, dtype=float)
    v = np.sqrt(np.diag(cov))
    corr = cov / np.outer(v, v)
    w_eq = np.ones(n) / n
    iu = np.triu_indices(n, 1)
    return {
        "funds": list(funds),
        "equal_weight": {"return": float(np.dot(w_eq, ann_returns)),
                         "vol": float(np.sqrt(w_eq @ cov @ w_eq)), "hhi": float(np.sum(w_eq ** 2))},
        "fund_sortino": [float(x) for x in extra_stats["sortinos"]],
        "correlation": [[round(float(x), 4) for x in row] for row in corr],
        "avg_pairwise_correlation": float(np.mean(corr[iu])),
        "min_pair_overlap_obs": extra_stats.get("min_pair_overlap"),
        "psd_adjusted": extra_stats.get("psd_adjusted"),
        "lw_delta": extra_stats.get("lw_delta"),
    }
