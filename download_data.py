# download_data.py
"""
Download everything the walk-forward backtest needs, ONCE, into data/.
================================================================================

Run this on a machine that can reach Yahoo Finance (e.g. your own laptop):

    python -m pip install -r requirements-backtest.txt
    python download_data.py                  # ~10-20 min, resumable

Outputs (all in data/):

    prices.parquet            Daily split- AND dividend-adjusted Open/High/Low/Close
                              (+ Volume) for every stock that was in the S&P 500
                              at any time in the span, plus SPY and ^VIX.
                              "Long" format: one row per (date, ticker).
    membership.parquet        When each stock was IN the index:
                              member_symbol, yf_symbol, start, end
                              (end is exclusive; NaT = still a member at --end).
    sp500_membership_raw.csv  Exact copy of the public source file (provenance).
    download_log.csv          Per-symbol result: rows, first/last date, and what
                              share of its index-member days have a price.
    download_manifest.json    Versions, arguments, source URL, SHA-256 hashes —
                              so anyone can verify the backtest used THIS data.

Why adjusted OHLC and not just adjusted close?
    The backtest fills orders at the next day's OPEN and checks stop-losses
    against each day's HIGH/LOW, so it needs all four prices. They are all
    adjusted by the same factor, so % returns are correct.

Honest limitations (repeat these in the report)
    * Yahoo Finance does NOT carry most delisted / acquired companies. Those
      index members get no prices and simply can't be traded → the backtest
      universe is still tilted towards survivors → results biased UPWARD.
      download_log.csv shows exactly which members are missing.
    * Ticker renames: the membership file uses the ticker valid at the time
      (FB until 2022-06-08, META after). RENAMES below maps a hand-checked list
      of old tickers to the current Yahoo symbol so the history is continuous.
      Renames not in that list show up as "missing" in the log.

After it finishes, push the data to the branch:

    git checkout cloud-backtest
    git add data/prices.parquet data/membership.parquet data/sp500_membership_raw.csv \
            data/download_log.csv data/download_manifest.json
    git commit -m "Add price + membership data"
    git push origin cloud-backtest
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import sys
import time
import urllib.request
from datetime import datetime, timezone
from typing import Dict, List

import numpy as np
import pandas as pd

MEMBERSHIP_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv"
)

# Old index ticker → current Yahoo symbol, for companies that simply CHANGED
# TICKER (same legal entity, continuous price history). Only renames I am
# confident about are listed; everything else is tried under its own symbol.
RENAMES: Dict[str, List[str]] = {
    "FB":    ["META"],   # Facebook → Meta Platforms (2022)
    "ANTM":  ["ELV"],    # Anthem → Elevance Health (2022)
    "ABC":   ["COR"],    # AmerisourceBergen → Cencora (2023)
    "RE":    ["EG"],     # Everest Re → Everest Group (2023)
    "PKI":   ["RVTY"],   # PerkinElmer → Revvity (2023)
    "CDAY":  ["DAY"],    # Ceridian → Dayforce (2024)
    "FLT":   ["CPAY"],   # FleetCor → Corpay (2024)
    "WLTW":  ["WTW"],    # Willis Towers Watson (2022)
    "HFC":   ["DINO"],   # HollyFrontier → HF Sinclair (2022)
    "HCP":   ["DOC"],    # HCP → Healthpeak (PEAK) → DOC (2024)
    "PEAK":  ["DOC"],
    "COG":   ["CTRA"],   # Cabot Oil & Gas → Coterra (2021)
    "BLL":   ["BALL"],   # Ball Corp (2022)
    "SYMC":  ["GEN"],    # Symantec → NortonLifeLock → Gen Digital
    "NLOK":  ["GEN"],
    "JEC":   ["J"],      # Jacobs Engineering (2019)
    "HRS":   ["LHX"],    # Harris → L3Harris (2019)
    "UTX":   ["RTX"],    # United Technologies → RTX (2020)
    "DWDP":  ["DD"],     # DowDuPont → DuPont (2019)
    "KORS":  ["CPRI"],   # Michael Kors → Capri (2018)
    "DISCA": ["WBD"],    # Discovery → Warner Bros. Discovery (2022)
    "ADS":   ["BFH"],    # Alliance Data → Bread Financial (2022)
    "GPS":   ["GAP"],    # Gap Inc (2024)
    "FI":    ["FISV", "FI"],   # Fiserv changed FISV→FI (2023) and back (2025)
    "BK":    ["BNY"],    # Bank of New York Mellon → BNY (2026; Yahoo has full history under BNY)
    "MMC":   ["MRSH"],   # Marsh & McLennan → MRSH (2026; full history under MRSH)
    "FISV":  ["FISV", "FI"],
}


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ── 1. Membership ────────────────────────────────────────────────────────────

def fetch_membership(out_dir: str) -> tuple[pd.DataFrame, str]:
    """Download the raw snapshot file and keep an exact copy for provenance."""
    raw_path = os.path.join(out_dir, "sp500_membership_raw.csv")
    print(f"[MEMBERSHIP] downloading {MEMBERSHIP_URL}")
    with urllib.request.urlopen(MEMBERSHIP_URL, timeout=60) as r:
        raw = r.read()
    with open(raw_path, "wb") as f:
        f.write(raw)
    snaps = pd.read_csv(io.BytesIO(raw))
    snaps["date"] = pd.to_datetime(snaps["date"])
    return snaps.sort_values("date").reset_index(drop=True), raw_path


def snapshots_to_intervals(snaps: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Turn "list of members on each change date" into membership intervals.

    A stock is a member from the first snapshot it appears in until the first
    later snapshot it is ABSENT from (exclusive end). Only intervals that
    overlap [start, end] are kept, clipped to that span.
    """
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    # Snapshot in force at `start` = last snapshot on or before it.
    first = snaps.index[snaps["date"] <= start_ts]
    i0 = int(first[-1]) if len(first) else 0
    rows = []
    open_since: Dict[str, pd.Timestamp] = {}
    for i in range(i0, len(snaps)):
        d = snaps.at[i, "date"]
        if d > end_ts:
            break
        members = set(str(snaps.at[i, "tickers"]).split(","))
        eff = max(d, start_ts)
        for t in members - open_since.keys():
            open_since[t] = eff
        for t in list(open_since.keys() - members):
            rows.append((t, open_since.pop(t), d))
    for t, s in open_since.items():
        rows.append((t, s, pd.NaT))              # still a member at --end
    iv = pd.DataFrame(rows, columns=["member_symbol", "start", "end"])
    return iv.sort_values(["member_symbol", "start"]).reset_index(drop=True)


# ── 2. Prices ────────────────────────────────────────────────────────────────

def yahoo_candidates(member_symbol: str) -> List[str]:
    """Yahoo symbols to try for one index ticker, in order."""
    if member_symbol in RENAMES:
        return RENAMES[member_symbol]
    return [member_symbol.replace(".", "-")]      # BRK.B → BRK-B


def _clean(df: pd.DataFrame) -> pd.DataFrame | None:
    if df is None or df.empty:
        return None
    df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    df = df.dropna(subset=["Close"])
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index = pd.DatetimeIndex(df.index).normalize()
    df.index.name = "date"
    return df if len(df) else None


def download_symbols(symbols: List[str], start: str, end: str, raw_dir: str,
                     chunk: int, pause: float) -> Dict[str, pd.DataFrame]:
    """Batch-download with a per-symbol parquet cache (re-running resumes)."""
    import yfinance as yf

    os.makedirs(raw_dir, exist_ok=True)
    out: Dict[str, pd.DataFrame] = {}
    todo = []
    for s in symbols:
        p = os.path.join(raw_dir, f"{s.replace('^', '_')}.parquet")
        if os.path.exists(p):
            out[s] = pd.read_parquet(p)
        elif os.path.exists(p + ".missing"):
            continue
        else:
            todo.append(s)
    print(f"[PRICES] {len(out)} cached, {len(todo)} to download")

    # Yahoo's `end` is exclusive → add one day so --end itself is included.
    end_excl = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    for k in range(0, len(todo), chunk):
        batch = todo[k:k + chunk]
        for attempt in range(4):
            try:
                df = yf.download(batch, start=start, end=end_excl, interval="1d",
                                 auto_adjust=True, actions=False, group_by="ticker",
                                 threads=True, progress=False)
                break
            except Exception as e:                    # rate limit / network blip
                wait = 5 * 2 ** attempt
                print(f"[PRICES] batch error ({e}); retry in {wait}s")
                time.sleep(wait)
        else:
            df = pd.DataFrame()
        for s in batch:
            sub = None
            if isinstance(df.columns, pd.MultiIndex) and s in df.columns.get_level_values(0):
                sub = _clean(df[s])
            p = os.path.join(raw_dir, f"{s.replace('^', '_')}.parquet")
            if sub is not None:
                sub.to_parquet(p)
                out[s] = sub
            else:
                open(p + ".missing", "w").close()     # remember: don't retry forever
        print(f"[PRICES] {min(k + chunk, len(todo))}/{len(todo)} done")
        time.sleep(pause)
    return out


# ── 3. Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--start", default="2012-01-01",
                    help="First price date (needs ~2y warm-up before the first train window)")
    ap.add_argument("--end", default="2026-09-30", help="Last price date (fixed for reproducibility)")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--chunk", type=int, default=40, help="Symbols per Yahoo request")
    ap.add_argument("--pause", type=float, default=1.5, help="Seconds between requests")
    ap.add_argument("--retry-missing", action="store_true",
                    help="Forget earlier 'missing' markers and try those symbols again")
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    raw_dir = os.path.join(a.out_dir, "_raw")
    if a.retry_missing and os.path.isdir(raw_dir):
        for f in os.listdir(raw_dir):
            if f.endswith(".missing"):
                os.remove(os.path.join(raw_dir, f))

    snaps, raw_path = fetch_membership(a.out_dir)
    iv = snapshots_to_intervals(snaps, a.start, a.end)
    members = sorted(iv["member_symbol"].unique())
    print(f"[MEMBERSHIP] {len(members)} distinct index members between {a.start} and {a.end}")

    # Try each member's candidate symbols in order; first one with data wins.
    wanted = sorted({c for m in members for c in yahoo_candidates(m)} | {"SPY", "^VIX"})
    got = download_symbols(wanted, a.start, a.end, raw_dir, a.chunk, a.pause)
    if "SPY" not in got:
        sys.exit("SPY could not be downloaded — Yahoo is unreachable or rate-limiting. Re-run later.")

    calendar = got["SPY"].index                     # NYSE trading days
    yf_of: Dict[str, str | None] = {}
    for m in members:
        yf_of[m] = next((c for c in yahoo_candidates(m) if c in got), None)
    iv["yf_symbol"] = iv["member_symbol"].map(yf_of)

    # Coverage: share of each member's index days that have a price.
    log_rows = []
    for m in members:
        sym = yf_of[m]
        m_iv = iv[iv["member_symbol"] == m]
        member_days = pd.DatetimeIndex([])
        for s, e in zip(m_iv["start"], m_iv["end"], strict=True):
            e = e if pd.notna(e) else calendar[-1] + pd.Timedelta(days=1)
            member_days = member_days.union(calendar[(calendar >= s) & (calendar < e)])
        if sym is None:
            cov, n, first, last = 0.0, 0, None, None
        else:
            px = got[sym]
            cov = float(member_days.isin(px.index).mean()) if len(member_days) else np.nan
            n, first, last = len(px), px.index[0].date(), px.index[-1].date()
        log_rows.append({"member_symbol": m, "yf_symbol": sym, "member_days": len(member_days),
                         "member_day_coverage": cov, "rows": n, "first": first, "last": last})
    log = pd.DataFrame(log_rows)
    log.to_csv(os.path.join(a.out_dir, "download_log.csv"), index=False)

    # Assemble the long price table (float32 + zstd keeps it well under 100 MB).
    used = sorted({s for s in yf_of.values() if s} | {"SPY", "^VIX"})
    frames = []
    for s in used:
        df = got[s].copy()
        df.columns = [c.lower() for c in df.columns]
        df["ticker"] = s
        frames.append(df.reset_index())
    prices = pd.concat(frames, ignore_index=True)
    for c in ["open", "high", "low", "close", "volume"]:
        prices[c] = prices[c].astype("float32")
    prices = prices[["date", "ticker", "open", "high", "low", "close", "volume"]]
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)
    prices_path = os.path.join(a.out_dir, "prices.parquet")
    prices.to_parquet(prices_path, compression="zstd", index=False)

    mem_path = os.path.join(a.out_dir, "membership.parquet")
    iv[["member_symbol", "yf_symbol", "start", "end"]].to_parquet(mem_path, index=False)

    # Member-days with a price, over all members — the key survivorship number.
    tot_days = log["member_days"].sum()
    cov_all = float((log["member_days"] * log["member_day_coverage"].fillna(0)).sum() / tot_days)
    missing = log.loc[log["yf_symbol"].isna(), "member_symbol"].tolist()

    import yfinance as yf
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "args": vars(a),
        "python": platform.python_version(),
        "pandas": pd.__version__, "numpy": np.__version__, "yfinance": yf.__version__,
        "membership_source": MEMBERSHIP_URL,
        "membership_raw_sha256": sha256_of(raw_path),
        "prices_sha256": sha256_of(prices_path),
        "membership_sha256": sha256_of(mem_path),
        "n_index_members_in_span": len(members),
        "n_members_with_prices": int(log["yf_symbol"].notna().sum()),
        "member_day_coverage": round(cov_all, 4),
        "missing_members": missing,
        "price_rows": int(len(prices)),
        "price_date_range": [str(prices["date"].min().date()), str(prices["date"].max().date())],
    }
    with open(os.path.join(a.out_dir, "download_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    mb = os.path.getsize(prices_path) / 1e6
    print("\n=== DONE ===")
    print(f"  members in span        : {len(members)}")
    print(f"  members with prices    : {manifest['n_members_with_prices']}")
    print(f"  member-day coverage    : {cov_all:.1%}   (the rest = mostly delisted names → survivorship bias)")
    print(f"  prices.parquet         : {len(prices):,} rows, {mb:.1f} MB")
    if mb > 95:
        print("  WARNING: >95 MB — GitHub rejects files >100 MB. Re-run with a later --start.")
    print("\nNext: git add data/prices.parquet data/membership.parquet data/sp500_membership_raw.csv "
          "data/download_log.csv data/download_manifest.json && git commit -m 'Add data' && "
          "git push origin cloud-backtest")


if __name__ == "__main__":
    main()
