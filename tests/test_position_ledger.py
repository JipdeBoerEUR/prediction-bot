"""Tests for position_ledger — the strategy-source record behind the
statarb exit pass and the live SL/TP monitor."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from position_ledger import (  # noqa: E402
    get_position_source,
    load_ledger,
    record_position,
    remove_position,
)


def _path(tmp_path):
    return str(tmp_path / "ledger.json")


def test_missing_file_returns_empty(tmp_path):
    assert load_ledger(_path(tmp_path)) == {}


def test_record_and_read_back(tmp_path):
    p = _path(tmp_path)
    record_position("aapl", "statarb", -1, path=p)
    ledger = load_ledger(p)
    assert ledger == {"AAPL": {"source": "statarb", "side": -1}}
    assert get_position_source("AAPL", path=p) == "statarb"
    # Ticker lookup is case-insensitive via upper-casing
    assert get_position_source("aapl", path=p) == "statarb"


def test_update_overwrites_entry(tmp_path):
    p = _path(tmp_path)
    record_position("MSFT", "statarb", 1, path=p)
    record_position("MSFT", "topic", 1, path=p)
    assert get_position_source("MSFT", path=p) == "topic"
    assert len(load_ledger(p)) == 1


def test_remove_position(tmp_path):
    p = _path(tmp_path)
    record_position("NVDA", "topic", 1, path=p)
    record_position("AMD", "statarb", -1, path=p)
    remove_position("NVDA", path=p)
    ledger = load_ledger(p)
    assert "NVDA" not in ledger
    assert "AMD" in ledger


def test_remove_unknown_ticker_is_noop(tmp_path):
    p = _path(tmp_path)
    record_position("TSLA", "topic", 1, path=p)
    remove_position("ZZZZ", path=p)   # must not raise
    assert get_position_source("TSLA", path=p) == "topic"


def test_unknown_ticker_source_is_none(tmp_path):
    assert get_position_source("GOOG", path=_path(tmp_path)) is None


def test_corrupt_file_returns_empty(tmp_path):
    p = _path(tmp_path)
    with open(p, "w", encoding="utf-8") as f:
        f.write("{not valid json")
    assert load_ledger(p) == {}


def test_non_dict_json_returns_empty(tmp_path):
    p = _path(tmp_path)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(["a", "list"], f)
    assert load_ledger(p) == {}
