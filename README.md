# data/

* `raw/`      the original files exactly as supplied (never edited).
* `cleaned/`  consolidated and validated CSVs written by `tools/ingest_raw.py` (`nifty_tri`, `nifty_valuation`, `gold`, `fund_nav`; `debt` once supplied).
* `provenance.json`          source, retrieval date, row counts, date range and SHA-256 of every cleaned file and of the raw files it came from. The application re-computes each checksum on load and refuses a file that no longer matches.
* `data_quality_report.json` machine-readable validation results. Human-readable version: `DATA_QUALITY_REPORT.md`.

Rules: no value is interpolated, forward-filled or fabricated. A missing dataset is reported as missing and the dependent result is reported as NOT EVALUATED.

Rebuild (idempotent):
```
python tools/ingest_raw.py --raw data/raw --retrieved-on YYYY-MM-DD [--debt-file path/to/gsec_levels.csv]
```
The debt file needs a date column and an index level column (`Close`, `Index Value`, `Value`, or `Closing Index Value`).
To add the G-Sec levels: put the CSV (date and level columns, `G-SEC` in the file name, not the metadata 'about' sheet) in `raw/` and run the command above without `--debt-file`; it is auto-detected. Commit and push to redeploy.
G-Sec: the Investing.com export `Nifty GS Compsite Historical Data.csv` (2016-04-05 to 2024-12-31) was supplied and is ingested (`--debt-file`; weekend rows are removed and listed). It is not confirmed to be the NSE Total Return Index. To replace it with the official NSE series, download it on your own computer with `python tools/fetch_gsec_nse.py` (NSE Total Return Index endpoint; validates identity, coverage, duplicates and invalid values; writes the original responses, a cleaned CSV and a record with SHA-256 to `raw/`), then run `python tools/ingest_raw.py`. The build environment could not reach niftyindices.com (HTTP 403), so that tool is untested against the live endpoint.
