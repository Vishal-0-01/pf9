Outputs of the application on the shipped real datasets (generated 27 Nov 2026 by tools/generate_results.py).
recommended_allocation.json     recommendation at the latest valuation (P/E 21.08, P/B 3.18): equity 72.5% (rule), debt 0%, gold 27.5%
allocation_sensitivity.json     allocation versus valuation (P/E, P/B)
frontier.json                   return and risk for equity weights 50% to 90% (cap not applied; flagged)
fund_evaluation_12_funds.csv    own-window and common-window metrics for the 12 funds
hypotheses_real_data.json       H1 directionally supported (not significant), H2 supported in-sample only, H3 supported
montecarlo_real_data.json       seeded Monte Carlo (10,000 paths, seed 20260825); debt, gold and equity inputs historical (equity mean shrunk, bull market)
historical_scenario.json        metrics, period, decisions, sensitivities (actual daily data)
allocation_history.json         yearly optimiser decisions
scenario_wealth_series.json     wealth and drawdown series
deployment_verification.json    output of tools/verify_bundle.py (clean-venv bundle emulation, network blocked)
Regenerate with tools/generate_results.py after any data change.
