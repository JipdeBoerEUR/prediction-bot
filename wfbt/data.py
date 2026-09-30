"""Load the downloaded data into date × ticker panels.

Everything is aligned on the SPY trading calendar. A stock's prices are kept
for its whole history (so an open position can still be marked and exited
after it leaves the index), but the `member` panel says on which days it was
IN the S&P 500. Only members can enter the correlation graph or be traded.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MarketData:
    dates: pd.DatetimeIndex
    tickers: pd.Index
    open: np.ndarray        # (T, N) adjusted prices, NaN = no data that day
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    rets: np.ndarray        # (T, N) close-to-close simple returns
    member: np.ndarray      # (T, N) bool: in the S&P 500 on that day
    spy: np.ndarray         # (T,) SPY adjusted close
    spy_sma200: np.ndarray  # (T,) 200-day moving average of SPY
    vix: np.ndarray         # (T,) VIX close (forward-filled over VIX-only holidays)
    qa: dict                # data-quality facts for the report

    def pos(self, date) -> int:
        """Index of the first trading day on or after `date`."""
        return int(self.dates.searchsorted(pd.Timestamp(date)))


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_market_data(data_dir: str = "data", verify_hash: bool = True) -> MarketData:
    prices_path = os.path.join(data_dir, "prices.parquet")
    mem_path = os.path.join(data_dir, "membership.parquet")
    manifest_path = os.path.join(data_dir, "download_manifest.json")
    for p in (prices_path, mem_path):
        if not os.path.exists(p):
            raise SystemExit(f"{p} not found — run download_data.py first (see its docstring).")

    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        if verify_hash and manifest.get("prices_sha256") and manifest["prices_sha256"] != _sha256(prices_path):
            raise SystemExit("prices.parquet does not match the SHA-256 in download_manifest.json.")

    px = pd.read_parquet(prices_path)
    px["date"] = pd.to_datetime(px["date"])
    mem = pd.read_parquet(mem_path)

    def wide(col: str) -> pd.DataFrame:
        return px.pivot(index="date", columns="ticker", values=col).astype("float64")

    close_all = wide("close")
    if "SPY" not in close_all.columns:
        raise SystemExit("SPY missing from prices.parquet.")
    dates = close_all["SPY"].dropna().index          # NYSE trading calendar
    close_all = close_all.reindex(dates)

    spy = close_all["SPY"].to_numpy()
    vix = close_all["^VIX"].ffill().to_numpy() if "^VIX" in close_all else np.full(len(dates), np.nan)
    stock_cols = pd.Index(sorted(c for c in close_all.columns if c not in ("SPY", "^VIX")))

    close = close_all[stock_cols]
    open_ = wide("open").reindex(index=dates, columns=stock_cols)
    high = wide("high").reindex(index=dates, columns=stock_cols)
    low = wide("low").reindex(index=dates, columns=stock_cols)

    # Membership panel: True on days start <= d < end (end NaT = still member).
    member = pd.DataFrame(False, index=dates, columns=stock_cols)
    mem = mem.dropna(subset=["yf_symbol"])
    for sym, s, e in zip(mem["yf_symbol"], mem["start"], mem["end"], strict=True):
        if sym not in member.columns:
            continue
        in_iv = dates >= pd.Timestamp(s)
        if pd.notna(e):
            in_iv &= dates < pd.Timestamp(e)
        member.loc[in_iv, sym] = True

    # No filling of missing prices: a NaN return means "no trade that day".
    rets = close.pct_change(fill_method=None)

    r = rets.to_numpy()
    m = member.to_numpy()
    qa = {
        "n_trading_days": int(len(dates)),
        "first_date": str(dates[0].date()),
        "last_date": str(dates[-1].date()),
        "n_stocks_with_prices": int(len(stock_cols)),
        "avg_members_per_day": float(m.sum(axis=1).mean()),
        "avg_members_with_price_per_day": float((m & np.isfinite(close.to_numpy())).sum(axis=1).mean()),
        # NB: only counts members that have SOME prices. The true survivorship
        # figure (incl. members with no prices at all) is the manifest's
        # member_day_coverage.
        "loaded_member_days_with_price_share": float((m & np.isfinite(close.to_numpy())).sum() / max(m.sum(), 1)),
        # Big one-day moves are kept (clipping would be tampering with data) but
        # counted, since an adjusted-price glitch can look like one.
        "abs_daily_return_gt_50pct": int(np.nansum(np.abs(r) > 0.5)),
        "manifest": {k: manifest.get(k) for k in (
            "created_utc", "yfinance", "member_day_coverage", "n_index_members_in_span",
            "n_members_with_prices", "prices_sha256")},
    }

    spy_s = pd.Series(spy, index=dates)
    return MarketData(
        dates=dates, tickers=stock_cols,
        open=open_.to_numpy(), high=high.to_numpy(), low=low.to_numpy(), close=close.to_numpy(),
        rets=r, member=m,
        spy=spy, spy_sma200=spy_s.rolling(200, min_periods=200).mean().to_numpy(),
        vix=vix, qa=qa,
    )
