# Data provenance

| Data | Source as stated | Raw file(s) | Cleaned file | Range | Rows |
|---|---|---|---|---|---|
| Fund NAVs (12 schemes) | TIGZIG daily NAV download (as stated by the project author) | `mfpro_nav_12funds_2026-11-26.csv` | `cleaned/fund_nav.csv` | 2013-01-01 to 2026-11-25 | 29,325 |
| Nifty 50 TRI (gross) | NSE Indices format, annual files; original download site not stated by the supplier | `nifty_2014.csv` ... `nifty_2024.csv` | `cleaned/nifty_tri.csv` | 2014-01-01 to 2024-12-31 | 2,721 |
| Nifty 50 P/E, P/B (monthly) | Downstox (as stated in the file header) | `nifty-50-pe-history.csv` | `cleaned/nifty_valuation.csv` | 1999-01-31 to 2026-11-26 | 335 |
| Gold (INR per troy oz) | User supplied; USD spot x INR/USD; original source not stated | `gold-price-history.csv` | `cleaned/gold.csv` | 2010-05-07 to 2026-11-19 | 3,536 |
| Debt (Nifty GS Composite) | Investing.com export supplied by the author (column `Price` = daily close). The NSE Total Return Index endpoint was blocked in the build environment, so it is NOT confirmed as the NSE TRI | `Nifty GS Compsite Historical Data.csv` (SHA-256 f9f445d5aa2a7137a9fac3b916772b9d735f7e31a085394fd2fb0d3f9aff011e) | `cleaned/debt.csv` | 2016-04-05 to 2024-12-31 | 2,158 (2,159 supplied; Saturday 2024-05-18 excluded) |

`data/provenance.json` holds, per cleaned dataset: file, source, retrieval date (2026-11-27), SHA-256, range, row count and the SHA-256 of each raw file it was built from.
The loader recomputes the checksum on every start and rejects a file that no longer matches. The deployment fingerprint (code, parameters and data hashes) is shown by `/api/health` (`code_fingerprint`). There is no disk cache; results are held in process memory only.
The bundled NAV file is served through a provider-shaped reader so the scheme-identity validation (scheme code, name, plan, option and category) is applied exactly as for any NAV source. The application has no live provider: nothing is fetched from AMFI, mfapi, mftool or any other service.

Model constants that are ASSUMPTIONS, not data: risk-free rate 6.5%, the 12% long-run equity prior used for return shrinkage, the equity rule (70% neutral, slope 0.10, band 50% to 90%, specified by the author), the 3% to 25% single-fund limits, and the assumed debt constants (return 7.5%, volatility 4%, correlation with equity -0.10), which are now only a fallback if the debt dataset is removed.

## G-Sec dataset validation (details in `data/DATA_QUALITY_REPORT.md`)
Schema: Date (dd-mm-yyyy), Price, Open, High, Low, Vol. (empty), Change %. No duplicates, nulls or non-positive values. The supplied file is newest-first; sorted ascending. **Saturday 2024-05-18** was excluded: it has Open = High = Low = Close = 2,695.96 (a flat print, no intraday range) and is not a trading day for a bond index; all other rows are Monday to Friday. Daily returns are simple returns between consecutive closes (2,157). The first observation is 2016-04-05; nothing earlier was created. Flags kept as supplied: zero-change days 2017-05-11 and 2023-09-05, and moves above 1% on 14 days including -2.1%/+2.1% on 2021-09-27/29 and -3.3%/+3.8% on 2022-09-12/13, which look like isolated erroneous prints but cannot be verified here.
The basis for each must be argued in the dissertation. They are flagged as assumed wherever they enter a result.
See `data/DATA_QUALITY_REPORT.md` for all checks and known issues.
