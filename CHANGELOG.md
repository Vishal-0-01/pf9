# Changelog

## v5.1 (27 Nov 2026): optimiser corrections, G-Sec integrated
* Equity weight follows the author's valuation rule again: E = clip(0.70 - 0.10 z, 0.50, 0.90) (70% at fair value). The 3% floors for debt and gold are removed: the debt/gold split of the remainder and the fund weights are chosen by one joint SLSQP optimisation under the hard 10% volatility cap.
* The cap has priority: equity is lowered (not below 50%) and flagged `equity_reduced_to_meet_vol_cap`; if 50% cannot meet it the live API reports infeasibility (HTTP 422). Historical years where even 50% equity cannot meet the cap are flagged `cap_infeasible` and listed in `ceiling_breach_years`.
* Frontier now sweeps the equity weights 50% to 90%. `dynamic_target_only` removed; scenario sensitivities vary the z basis, valuation lag and slope.
* Reported weights sum to exactly 100% (4 decimals).
* Monte Carlo uses the application's recommended allocation, with `parameter_source` and `parameter_components` separating historical from assumed inputs.
* G-Sec: NSE retrieval was blocked in the build environment; `tools/fetch_gsec_nse.py` added (mock-tested only). The author supplied an Investing.com export (2016-04-05 to 2024-12-31); it is ingested (Saturday 2024-05-18 excluded, 2,158 observations, SHA-256 registered) and activates the scenario, allocation history, H1/H3 and the historical-debt Monte Carlo. Its status as the NSE TRI is unconfirmed.
* Added `tools/generate_results.py`, `docs/LIMITATIONS.md`, `tests/test_allocation_v51.py`, `tests/test_fetch_gsec.py`.

## v5 (27 Nov 2026): Vercel-native deployment
* One Vercel project replaces Render plus Vercel: root `app.py` entrypoint, Python modules in `backend/`, static page in `public/`, `vercel.json`, `.python-version`, pinned `requirements.txt`.
* Frontend uses relative `/api/...` paths (`API_BASE` is empty); the hard-coded Render URL is gone.
* Live NAV fetching removed from the code: no mftool, requests or flask-cors; `load_dataset()` reads only the bundled, checksummed file; the synthetic dataset and the `LIVE_NAV`, `ALLOW_SYNTHETIC`, `STRICT_UNIVERSE`, `CACHE_TTL_HOURS`, `DISABLE_CACHE` switches were removed. No environment variables are needed.
* State is built lazily in process memory on the first request (concurrent first requests wait for one build); the on-disk pickle cache and the start-up initialisation were removed.
* `/api/health` added (alias `/health` kept); unknown `/api/*` paths return JSON 404 listing the routes; unhandled errors return JSON 500; successful GET responses are CDN-cacheable.
* The 14-fund and 404 problems seen on the old Render deployment came from that deployment running an older build; the routes `/api/montecarlo`, `/api/datasets`, `/api/data-report` and the 12-fund universe are covered by tests (`tests/test_deployment.py`).
* `tools/ingest_raw.py` auto-detects a G-Sec level file in `data/raw/`; `tools/verify_bundle.py` emulates the function bundle.
* Removed unused synthetic parameters from `optimizer.py`.

## v4 (27 Nov 2026): real datasets
Added
* `tools/ingest_raw.py`: consolidates the year-wise Nifty files, validates (duplicates, gaps, dates, scale checks), keeps raw and cleaned separate, writes provenance with SHA-256 and a data-quality report.
* `assets.py`: three-asset layer (CovModel, cross-asset covariance from actual daily returns, SLSQP debt/gold split, point-in-time valuation statistics).
* Bundled checksummed fund NAV dataset served through the same identity validation as live data.
* Per-fund metrics on both own window and the 12-fund common window; Sortino and downside deviation.
* Scenario comparators: static 70/20/10, valuation-blind neutral-optimised, target-only diagnostic, Nifty 50 TRI; scenario sensitivities.
* Supplementary matched-return minimum-variance comparison for H2; two-sided "comparable return" test.
* `/api/data-report`.
Changed
* Universe reduced to the 12 funds in the supplied NAV file (ICICI and Kotak removed).
* Cash removed entirely from the optimiser, scenario, Monte Carlo, API and frontend. The synopsis static mix 70/18/10/2 becomes 70/20/10 (cash weight added to debt) and is labelled as such.
* Optimisation uses the common observation window (2023-02-23 to 2026-11-25).
* Evaluation window of the scenario is now determined by the data (common dates and a 252-observation covariance minimum), not fixed.
* Valuation z-scores are point-in-time expanding statistics (synopsis constants kept as a sensitivity).
* Cache version 5; the data fingerprint is part of the cache key.
Removed
* All walk-forward references from code, API and docs.
Known gaps
* G-Sec index levels missing: scenario, H1, H3 and the real-data sensitivities are NOT EVALUATED. Debt uses flagged assumed constants in the optimiser and Monte Carlo.

## Earlier versions
v1 to v3: universe and identity validation, covariance repair, scenario and Monte Carlo infrastructure, independent verification. Superseded by v4.

## Vercel deployment fix
- `vercel.json`: added `"framework": "flask"` (root cause of the `unused_function` build error with Framework Preset "Other"). `functions`, `maxDuration` 120 and `excludeFiles` unchanged.
- New `tools/verify_vercel_build.py` (local `vercel build` plus bundle emulation), regression assertions in `tests/test_deployment.py`, `DEPLOYMENT.md` corrected, `.gitignore` extended.
