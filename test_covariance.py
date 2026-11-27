import numpy as np, pandas as pd, pytest
from fakes import FakeProvider, make_nav, raw_from, NAMES
import data_fetcher as DF, optimizer as O, universe as U, webapp as A

STARTS = ["2013-01-02"]*6 + ["2019-12-02", "2018-01-02", "2013-01-02", "2020-06-01", "2015-04-01", "2022-09-01"]

@pytest.fixture(scope="module")
def staggered_ds():
    """12 synthetic funds with staggered inception dates (like real schemes) to 2026-11-20."""
    prov = FakeProvider(n=1500)
    rng = np.random.default_rng(5)
    cidx = pd.bdate_range("2013-01-02", "2026-11-20"); common = pd.Series(rng.normal(0, .009, len(cidx)), index=cidx)
    for i, (lab, spec) in enumerate(U.SCHEME_SPECS.items()):
        idx = pd.bdate_range(STARTS[i], "2026-11-20")
        nav = make_nav(len(idx), seed=30 + i, start=STARTS[i], common=common.reindex(idx).values)
        prov.raw[spec["code"]] = raw_from(nav, NAMES[lab], spec["code"])
    return DF.fetch_nav_data(mf=prov)



def test_inf_and_overflow_return_rejected_with_clear_error():
    idx = pd.bdate_range("2020-01-01", periods=300); rng = np.random.default_rng(1)
    df = pd.DataFrame(rng.normal(0, .01, (300, 5)), index=idx)
    for val in (np.inf, 1e200):
        d = df.copy(); d.iloc[10, 2] = val
        with pytest.raises(O.InsufficientDataError):
            O.build_cov_from_returns(d)


def test_eigen_driver_failure_falls_back_to_scipy(monkeypatch):
    def boom(*a, **k): raise np.linalg.LinAlgError("Eigenvalues did not converge")
    monkeypatch.setattr(np.linalg, "eigvalsh", boom)
    ev, drv = O._eigvalsh_safe(np.diag([1., 2., 3.]))
    assert drv.startswith("scipy") and np.allclose(sorted(ev), [1, 2, 3])

def test_all_eigen_drivers_failing_uses_eigen_free_repair(monkeypatch):
    def boom(*a, **k): raise np.linalg.LinAlgError("Eigenvalues did not converge")
    monkeypatch.setattr(O, "_eigvalsh_safe", boom)
    S = np.array([[1, .9, .9], [.9, 1, -.9], [.9, -.9, 1.]]) * 0.04           # indefinite
    S2, info = O.repair_covariance(S)
    assert info["method"].startswith("cholesky") and np.allclose(np.diag(S2), np.diag(S))
    np.linalg.cholesky(S2)

def test_repair_preserves_variances_and_gives_pd():
    S = np.array([[1, .9, .9], [.9, 1, -.9], [.9, -.9, 1.]]) * np.outer([.2, .3, .15], [.2, .3, .15])
    assert np.linalg.eigvalsh(S).min() < 0
    S2, info = O.repair_covariance(S)
    assert info["repaired"] and np.allclose(np.diag(S2), np.diag(S))
    assert np.linalg.eigvalsh(S2).min() > 0 and np.allclose(S2, S2.T)

def test_pairwise_covariance_vols_consistent_with_returned_vols(staggered_ds):
    rd = staggered_ds["returns_df"]
    ann, cov, vols, shp, extra = O.build_cov_from_returns(rd)
    assert np.allclose(vols, np.sqrt(np.diag(cov)))
    # the repair step does not change the raw sample variances
    raw_var = rd.var(ddof=1).values * 252
    assert np.allclose(extra["raw_vols"] ** 2, raw_var, rtol=1e-9)
    assert np.isfinite(cov).all() and np.linalg.eigvalsh(cov).min() > 0




