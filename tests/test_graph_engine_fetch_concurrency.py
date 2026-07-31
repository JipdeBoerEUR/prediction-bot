"""Regression guard: statarb.graph_engine's Alpaca fetch must stay concurrent.

The original implementation fetched tickers one at a time in a strictly
sequential `for sym in tickers:` loop — on a large universe (hundreds of
tickers) this made the one-time price fetch take an hour+ while the CPU sat
almost entirely idle (it's network-bound, not compute-bound). This test
mocks the Alpaca client to prove multiple symbols are genuinely in flight
at once, a slow symbol doesn't block the others, and correctness (which
symbols succeeded/failed, final DataFrame shape) is unaffected by fetching
concurrently instead of sequentially.

Uses small millisecond-scale sleeps so this stays fast in CI. Failure
handling is tested separately from timing: a failing symbol's retry backoff
is a fixed ~1s cost in the *code itself* (min(2**attempt, 10) with
attempt=0) regardless of how parallel the fetch is, so mixing it into a
speedup assertion just makes the timing math misleading.
"""
import os
import sys
import threading
import time
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ALPACA_API_KEY", "test_key")
os.environ.setdefault("ALPACA_SECRET_KEY", "test_secret")

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("alpaca", reason="alpaca-py not installed")

from statarb.graph_engine import _alpaca_fetch_close_prices  # noqa: E402

TICKERS = [f"T{i:03d}" for i in range(16)]
SLOW_SYMBOL = "T005"
FAIL_SYMBOL = "T010"
FAST_LATENCY = 0.01
SLOW_LATENCY = 0.15

_DATES = pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC")


def _make_fake_get_stock_bars(inflight, max_inflight, lock, fail_symbol=None):
    def fake_get_stock_bars(self, req):
        sym = req.symbol_or_symbols
        with lock:
            inflight.append(sym)
            max_inflight[0] = max(max_inflight[0], len(inflight))

        if sym == fail_symbol:
            time.sleep(FAST_LATENCY)
            with lock:
                inflight.remove(sym)
            raise RuntimeError("simulated permanent failure")

        time.sleep(SLOW_LATENCY if sym == SLOW_SYMBOL else FAST_LATENCY)

        idx = pd.MultiIndex.from_product([[sym], _DATES], names=["symbol", "timestamp"])
        df = pd.DataFrame({"close": range(len(_DATES))}, index=idx).astype(float)
        resp = MagicMock()
        resp.df = df
        resp.next_page_token = None

        with lock:
            inflight.remove(sym)
        return resp

    return fake_get_stock_bars


def test_fetch_is_actually_concurrent_and_fast():
    """No failures here — clean timing proof that fetching is parallel."""
    inflight: list = []
    max_inflight = [0]
    lock = threading.Lock()
    fake = _make_fake_get_stock_bars(inflight, max_inflight, lock, fail_symbol=None)

    with patch("alpaca.data.historical.stock.StockHistoricalDataClient.get_stock_bars", fake):
        t0 = time.time()
        prices = _alpaca_fetch_close_prices(
            tickers=TICKERS, start="2024-01-01", end="2024-01-02",
            interval="60m", feed="iex", max_retries=1, verbose=False, max_workers=8,
        )
        elapsed = time.time() - t0

    # Proof of real concurrency: strictly sequential fetching could never
    # have more than 1 request in flight at a time.
    assert max_inflight[0] >= 4, (
        f"only {max_inflight[0]} concurrent request(s) observed — fetch is not parallel"
    )
    assert prices.shape == (10, len(TICKERS))
    assert sorted(prices.columns) == sorted(TICKERS)

    # Strictly sequential would take (preflight + 16 symbols' latencies) —
    # 1 preflight + 15 fast + 1 slow ~= 0.01 + 0.15 + 0.15 = 0.31s.
    # 8-way concurrency should clearly beat that.
    naive_sequential = FAST_LATENCY + (len(TICKERS) - 1) * FAST_LATENCY + SLOW_LATENCY
    assert elapsed < naive_sequential * 0.75, (
        f"elapsed={elapsed:.3f}s not faster than naive sequential {naive_sequential:.3f}s"
    )


def test_fetch_excludes_failing_symbols_without_crashing():
    """Correctness only — a permanently-failing symbol must not be fatal,
    and must not appear in the result, regardless of fetch concurrency.
    No timing assertions: the retry backoff is a fixed ~1s cost baked into
    the retry logic itself, unrelated to how parallel the fetch is."""
    inflight: list = []
    max_inflight = [0]
    lock = threading.Lock()
    fake = _make_fake_get_stock_bars(inflight, max_inflight, lock, fail_symbol=FAIL_SYMBOL)

    with patch("alpaca.data.historical.stock.StockHistoricalDataClient.get_stock_bars", fake):
        prices = _alpaca_fetch_close_prices(
            tickers=TICKERS, start="2024-01-01", end="2024-01-02",
            interval="60m", feed="iex", max_retries=1, verbose=False, max_workers=8,
        )

    assert FAIL_SYMBOL not in prices.columns
    assert prices.shape == (10, len(TICKERS) - 1)
    assert sorted(prices.columns) == sorted(t for t in TICKERS if t != FAIL_SYMBOL)
