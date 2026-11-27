"""fetch_gsec_nse.py parsing / validation on MOCK responses only (the live endpoint was not reachable when built)."""
import json, os, sys
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import fetch_gsec_nse as F


def mock(rows, name="NIFTY COMPOSITE G-SEC INDEX", key="TotalReturnsIndex"):
    return {"d": json.dumps([{"IndexName": name, "Date": d, key: v} for d, v in rows])}


def test_parse_and_clean_ok(tmp_path):
    r = mock([("13 Oct 2015", "1000.5"), ("14 Oct 2015", "1001.0"), ("15 Oct 2015", "1,002.25")])
    s, rep = F.clean([F.parse_response(r)], "2015-10-13", "2015-10-15")
    assert len(s) == 3 and rep["interpolated_or_filled_values"] == 0 and rep["duplicate_date_rows"] == 0
    rec = F.save([{"response": r}], s, rep, str(tmp_path), "2026-11-27")
    assert rec["rows"] == 3 and len(rec["files"][F.CSV_NAME]["sha256"]) == 64
    assert pd.read_csv(tmp_path / F.CSV_NAME).shape[0] == 3


def test_wrong_index_refused():
    with pytest.raises(F.FetchError, match="refusing"):
        F.parse_response(mock([("13 Oct 2015", "1000")], name="NIFTY 10 YR BENCHMARK G-SEC INDEX"))


def test_price_index_payload_refused():
    with pytest.raises(F.FetchError, match="Total Return"):
        F.parse_response(mock([("13 Oct 2015", "1000")], key="Close"))


def test_invalid_and_conflicting_duplicates_abort():
    with pytest.raises(F.FetchError, match="invalid"):
        F.clean([F.parse_response(mock([("13 Oct 2015", "abc")]))], "2015-10-13", "2015-10-13")
    with pytest.raises(F.FetchError, match="conflicting"):
        F.clean([F.parse_response(mock([("13 Oct 2015", "1000"), ("13 Oct 2015", "1001")]))], "2015-10-13", "2015-10-13")


def test_identical_chunk_overlap_dropped_no_fill():
    f = F.parse_response(mock([("13 Oct 2015", "1000"), ("16 Oct 2015", "1003")]))
    s, rep = F.clean([f, f], "2015-10-13", "2015-10-16")
    assert len(s) == 2 and rep["duplicate_date_rows"] == 4      # 2 identical pairs -> reported, one copy kept; gap NOT filled


def test_empty_response_stops():
    with pytest.raises(F.FetchError):
        F.clean([F.parse_response({"d": "[]"})], "2015-10-13", "2015-10-16")
