# position_ledger.py — persistent record of which strategy opened each position.
#
# The bot runs multiple strategies (topic momentum, statarb mean-reversion)
# through one Alpaca account. Alpaca positions carry no strategy metadata, so
# without a ledger the 30-min statarb exit pass cannot tell a statarb book
# entry from a topic momentum trade (or from a manually opened position) —
# which previously caused it to force-sell topical winners.
#
# Deliberately dependency-free (stdlib only) so it is unit-testable in CI.
# Writes are atomic (tmp file + os.replace) so a crash mid-write cannot
# corrupt the ledger.

from __future__ import annotations

import json
import os
import tempfile
from typing import Dict, Optional

LEDGER_PATH = "bot_positions.json"


def load_ledger(path: str = LEDGER_PATH) -> Dict[str, dict]:
    """Return {TICKER: {"source": str, "side": int}}; empty dict if unreadable."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _save(ledger: Dict[str, dict], path: str = LEDGER_PATH) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".ledger.tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(ledger, f, indent=2, sort_keys=True)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def record_position(ticker: str, source: str, side: int,
                    path: str = LEDGER_PATH, **fields) -> None:
    """Record (or replace) a position entry.

    Extra keyword fields are stored alongside source/side — the monitor uses
    entry_price, stop_pct, trail_pct, opened_at and extreme_price. None-valued
    fields are dropped so absent data reads back as absent, not null.
    """
    ledger = load_ledger(path)
    entry = {"source": str(source), "side": int(side)}
    entry.update({k: v for k, v in fields.items() if v is not None})
    ledger[str(ticker).upper()] = entry
    _save(ledger, path)


def update_position(ticker: str, path: str = LEDGER_PATH, **fields) -> None:
    """Merge fields into an existing entry (create a bare one if missing).

    Used by the monitor to persist the trailing-stop extreme price across
    restarts without touching the rest of the entry.
    """
    ledger = load_ledger(path)
    key = str(ticker).upper()
    entry = ledger.get(key)
    if not isinstance(entry, dict):
        entry = {}
    entry.update({k: v for k, v in fields.items() if v is not None})
    ledger[key] = entry
    _save(ledger, path)


def remove_position(ticker: str, path: str = LEDGER_PATH) -> None:
    """Drop a ticker from the ledger once its position is closed."""
    ledger = load_ledger(path)
    if str(ticker).upper() in ledger:
        del ledger[str(ticker).upper()]
        _save(ledger, path)


def get_position_source(ticker: str, path: str = LEDGER_PATH) -> Optional[str]:
    """Return the strategy that opened this position, or None if unknown."""
    entry = load_ledger(path).get(str(ticker).upper())
    return entry.get("source") if isinstance(entry, dict) else None
