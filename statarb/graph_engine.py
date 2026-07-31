"""
MarketGraph: Production-Grade Graph Signal Processing for Statistical Arbitrage
================================================================================
Combines robust data handling, vectorized GSP operations, and comprehensive analysis tools.

Strategy: Topological Arbitrage via Graph Diffusion
- Build correlation-based adjacency matrix W (undirected, sector-constrained)
- Compute diffusion operator P (row-stochastic random walk)
- Expected returns: r_hat[t] = r[t] @ P (neighbor-implied returns)
- Residual signal: e[t] = r[t] - r_hat[t] (mean-reversion opportunity)
- Trading logic: Large |e| → stock mispriced relative to topology → reversion trade

Author: Senior Quantitative Developer Team
"""

from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import warnings
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# Data provider helpers (imports are local to keep optional deps optional)
# ---------------------------------------------------------------------------

def _ts_to_utc_dt(ts: Optional[Union[str, pd.Timestamp]]) -> Optional[datetime]:
    """Convert YYYY-MM-DD strings or pd.Timestamp to timezone-aware UTC datetime."""
    if ts is None:
        return None
    if isinstance(ts, pd.Timestamp):
        t = ts
    else:
        t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return t.to_pydatetime()

def _normalize_fx_instrument(symbol: str) -> str:
    """Normalize FX tickers like EURUSD -> EUR/USD for Dukascopy."""
    s = symbol.strip().upper()
    if "/" in s:
        return s
    if "-" in s:
        return s.replace("-", "/")
    if "_" in s:
        return s.replace("_", "/")
    if len(s) == 6:
        return f"{s[:3]}/{s[3:]}"
    return s

def _dukascopy_interval(interval: str) -> str:
    """Map common interval strings to dukascopy-python interval constants."""
    interval = interval.lower().strip()
    from dukascopy_python import (
        INTERVAL_DAY_1,
        INTERVAL_HOUR_1,
        INTERVAL_HOUR_4,
        INTERVAL_MIN_30,
        INTERVAL_MIN_15,
        INTERVAL_MIN_10,
        INTERVAL_MIN_5,
        INTERVAL_MIN_1,
        INTERVAL_WEEK_1,
        INTERVAL_MONTH_1,
    )

    mapping = {
        "1d": INTERVAL_DAY_1,
        "day": INTERVAL_DAY_1,
        "1day": INTERVAL_DAY_1,
        "1h": INTERVAL_HOUR_1,
        "1hour": INTERVAL_HOUR_1,
        "hour": INTERVAL_HOUR_1,
        "4h": INTERVAL_HOUR_4,
        "4hour": INTERVAL_HOUR_4,
        "30m": INTERVAL_MIN_30,
        "30min": INTERVAL_MIN_30,
        "15m": INTERVAL_MIN_15,
        "15min": INTERVAL_MIN_15,
        "10m": INTERVAL_MIN_10,
        "10min": INTERVAL_MIN_10,
        "5m": INTERVAL_MIN_5,
        "5min": INTERVAL_MIN_5,
        "1m": INTERVAL_MIN_1,
        "1min": INTERVAL_MIN_1,
        "1w": INTERVAL_WEEK_1,
        "1wk": INTERVAL_WEEK_1,
        "week": INTERVAL_WEEK_1,
        "1mo": INTERVAL_MONTH_1,
        "1month": INTERVAL_MONTH_1,
        "month": INTERVAL_MONTH_1,
    }
    if interval in mapping:
        return mapping[interval]
    if interval.endswith("m"):
        try:
            n = int(interval[:-1])
        except ValueError:
            n = 0
        if n == 60:
            return INTERVAL_HOUR_1
        if n == 30:
            return INTERVAL_MIN_30
        if n == 15:
            return INTERVAL_MIN_15
        if n == 10:
            return INTERVAL_MIN_10
        if n == 5:
            return INTERVAL_MIN_5
        if n == 1:
            return INTERVAL_MIN_1
    if interval.endswith("h"):
        try:
            n = int(interval[:-1])
        except ValueError:
            n = 0
        if n == 1:
            return INTERVAL_HOUR_1
        if n == 4:
            return INTERVAL_HOUR_4
    raise ValueError(f"Unsupported interval for Dukascopy: {interval}")

def _infer_alpaca_timeframe(interval: str):
    """Map common string intervals to alpaca.data.timeframe.TimeFrame."""
    interval = interval.lower().strip()
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit  # type: ignore
    if interval in ("1d", "day", "1day"):
        return TimeFrame.Day
    if interval in ("1h", "60m", "60min", "1hour", "hour"):
        return TimeFrame.Hour
    if interval.endswith("m"):
        n = int(interval[:-1])
        return TimeFrame(n, TimeFrameUnit.Minute)
    if interval.endswith("min"):
        n = int(interval[:-3])
        return TimeFrame(n, TimeFrameUnit.Minute)
    raise ValueError(f"Unsupported interval for Alpaca: {interval}")

def _alpaca_fetch_close_prices(
    tickers: List[str],
    start: Optional[str],
    end: Optional[str],
    interval: str,
    feed: Optional[str],
    max_retries: int = 5,
    verbose: bool = True,
    max_workers: int = 8,
) -> pd.DataFrame:
    """
    Fetch close prices from Alpaca using *per-symbol* requests to avoid API truncation
    when requesting many symbols at once (limit applies per response and you must paginate
    using next_page_token otherwise).

    Symbols are fetched concurrently (max_workers threads) — this is network-bound
    (waiting on HTTP responses), not CPU-bound, so threading gives a large speedup
    even on machines whose cores otherwise sit idle during a sequential fetch. A
    failing symbol's retry backoff (time.sleep below) no longer blocks every other
    symbol behind it in line. max_workers is deliberately modest by default to stay
    well under Alpaca's per-account rate limits — raise it only if you know your
    account's limit comfortably allows more concurrent requests.
    """
    # Local imports (optional dependency)
    import os
    from alpaca.data.historical.stock import StockHistoricalDataClient  # type: ignore
    from alpaca.data.requests import StockBarsRequest  # type: ignore

    # Always reload .env so stale Windows/PowerShell session env vars never win.
    try:
        from dotenv import load_dotenv as _load_dotenv
        _load_dotenv(override=True)
    except ImportError:
        pass

    # Support both env vars and config.py constants (useful for local demo setups).
    cfg_key_id = None
    cfg_secret = None
    try:
        try:
            from statarb import config as sb_cfg  # type: ignore
        except Exception:
            import config as sb_cfg  # type: ignore
        cfg_key_id = getattr(sb_cfg, "ALPACA_API_KEY", None)
        cfg_secret = (
            getattr(sb_cfg, "ALPACA_SECRET_KEY", None)
            or getattr(sb_cfg, "ALPACA_API_SECRET", None)
        )
    except Exception:
        cfg_key_id = None
        cfg_secret = None

    key_id = (
        os.getenv("ALPACA_API_KEY")
        or os.getenv("ALPACA_KEY_ID")
        or os.getenv("APCA_API_KEY_ID")
        or cfg_key_id
    )
    secret = (
        os.getenv("ALPACA_API_SECRET")
        or os.getenv("ALPACA_SECRET_KEY")
        or os.getenv("APCA_API_SECRET_KEY")
        or cfg_secret
    )
    if not key_id or not secret:
        raise RuntimeError(
            "Missing Alpaca credentials. Set ALPACA_API_KEY + ALPACA_SECRET_KEY "
            "(or ALPACA_API_SECRET, APCA_API_KEY_ID, APCA_API_SECRET_KEY)."
        )

    client = StockHistoricalDataClient(key_id, secret)

    tf = _infer_alpaca_timeframe(interval)
    start_dt = _ts_to_utc_dt(start)
    end_dt = _ts_to_utc_dt(end)

    # Fail fast on auth/permissions/feed errors before iterating all symbols.
    preflight_symbol = (tickers[0] if tickers else "AAPL").upper()
    try:
        preflight_req = StockBarsRequest(
            symbol_or_symbols=preflight_symbol,
            timeframe=tf,
            start=start_dt,
            end=end_dt,
            limit=1,
            feed=feed,
        )
        client.get_stock_bars(preflight_req)
    except Exception as e:
        raise RuntimeError(
            "Alpaca preflight failed before bulk fetch. "
            f"symbol={preflight_symbol} feed={feed} interval={interval}. "
            "Check API key/secret pairing and account feed permissions. "
            f"Underlying error: {type(e).__name__}: {e}"
        ) from e

    def _fetch_one(sym: str) -> Tuple[str, Optional[pd.Series], Optional[str]]:
        """Fetch + paginate one symbol. Own client instance — cheap to build,
        avoids any doubt about concurrent use of a shared HTTP session."""
        thread_client = StockHistoricalDataClient(key_id, secret)
        page_token = None
        parts: List[pd.Series] = []
        sym_error: Optional[str] = None
        for _ in range(1000):  # pagination safety
            req = StockBarsRequest(
                symbol_or_symbols=sym,
                timeframe=tf,
                start=start_dt,
                end=end_dt,
                limit=10_000,
                feed=feed,
                page_token=page_token,
            )
            last_err: Optional[Exception] = None
            df = None
            for attempt in range(max_retries):
                try:
                    resp = thread_client.get_stock_bars(req)
                    df = getattr(resp, "df", None)
                    if df is None or len(df) == 0:
                        break
                    # df index may be MultiIndex (symbol, timestamp)
                    if isinstance(df.index, pd.MultiIndex):
                        try:
                            df_sym = df.xs(sym, level=0)
                        except Exception:
                            # Fallback: select rows matching symbol level
                            df_sym = df[df.index.get_level_values(0) == sym].copy()
                            df_sym.index = df_sym.index.get_level_values(-1)
                    else:
                        df_sym = df
                    s = df_sym["close"].astype(float).rename(sym)
                    parts.append(s)
                    page_token = getattr(resp, "next_page_token", None)
                    break
                except Exception as e:
                    last_err = e
                    sym_error = f"{type(e).__name__}: {e}"
                    time.sleep(min(2 ** attempt, 10))
            else:
                # retries exhausted
                if last_err:
                    return sym, None, f"{type(last_err).__name__}: {last_err}"
                break

            if df is None or len(df) == 0:
                break
            if not page_token:
                break

        if len(parts) == 0:
            return sym, None, sym_error

        s_all = pd.concat(parts).sort_index()
        # De-duplicate timestamps (keep last)
        s_all = s_all[~s_all.index.duplicated(keep="last")]
        return sym, s_all, None

    series_list: List[pd.Series] = []
    failed: set[str] = set()
    error_samples: Dict[str, str] = {}

    workers = max(1, min(int(max_workers), len(tickers)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_fetch_one, sym): sym for sym in tickers}
        for future in as_completed(futures):
            sym = futures[future]
            try:
                sym_out, series, err = future.result()
            except Exception as e:
                sym_out, series, err = sym, None, f"{type(e).__name__}: {e}"
            if series is not None:
                series_list.append(series)
            else:
                failed.add(sym_out)
                if err and sym_out not in error_samples:
                    error_samples[sym_out] = err

    if verbose:
        ok = len(series_list)
        print(f"[MarketGraph] Alpaca fetched close series: ok={ok} failed={len(failed)} feed={feed}")
        if failed and error_samples:
            # Print only a few examples to keep logs readable.
            sample_keys = list(error_samples.keys())[:5]
            samples = "; ".join(f"{k}: {error_samples[k]}" for k in sample_keys)
            print(f"[MarketGraph] Alpaca sample failures: {samples}")

    if len(series_list) == 0:
        detail = ""
        if error_samples:
            sample_keys = list(error_samples.keys())[:5]
            samples = "; ".join(f"{k}: {error_samples[k]}" for k in sample_keys)
            detail = f" Sample failures: {samples}"
        raise ValueError(f"Alpaca returned no price series (all symbols failed).{detail}")

    prices = pd.concat(series_list, axis=1).sort_index()

    # Normalize timezone handling (keep tz-naive UTC)
    if isinstance(prices.index, pd.DatetimeIndex) and prices.index.tz is not None:
        prices.index = prices.index.tz_convert("UTC").tz_localize(None)

    return prices

def _yfinance_fetch_close_prices(
    tickers: List[str],
    start: Optional[str],
    end: Optional[str],
    interval: str,
    auto_adjust: bool,
    verbose: bool = True,
) -> pd.DataFrame:
    """Fetch close prices via yfinance."""
    if verbose:
        print(f"[MarketGraph] yfinance downloading: tickers={len(tickers)} interval={interval} start={start} end={end}")
    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=auto_adjust,
        group_by="column",
        threads=True,
        progress=verbose,
    )
    if isinstance(data.columns, pd.MultiIndex):
        # Prefer Adj Close if present, else Close
        if ("Adj Close" in data.columns.get_level_values(0)):
            px = data["Adj Close"]
        elif ("Close" in data.columns.get_level_values(0)):
            px = data["Close"]
        else:
            # take first field
            px = data.xs(data.columns.get_level_values(0)[0], axis=1, level=0)
    else:
        # single ticker returns series-like frame
        px = data.rename(columns={"Adj Close": tickers[0], "Close": tickers[0]})
        if "Adj Close" in px.columns:
            px = px[["Adj Close"]].rename(columns={"Adj Close": tickers[0]})
        elif "Close" in px.columns:
            px = px[["Close"]].rename(columns={"Close": tickers[0]})
    px = px.sort_index()
    return px

def _dukascopy_fetch_close_prices(
    tickers: List[str],
    start: Optional[str],
    end: Optional[str],
    interval: str,
    verbose: bool = True,
) -> pd.DataFrame:
    """Fetch close prices via Dukascopy (FX)."""
    from dukascopy_python import fetch, OFFER_SIDE_BID

    start_dt = _ts_to_utc_dt(start)
    if start_dt is None:
        raise ValueError("Dukascopy requires a start date.")
    end_dt = _ts_to_utc_dt(end) or datetime.now(timezone.utc)
    if end_dt <= start_dt:
        raise ValueError("Dukascopy end date must be after start date.")

    interval_code = _dukascopy_interval(interval)

    series_list: List[pd.Series] = []
    failed: List[str] = []

    for sym in tickers:
        instrument = _normalize_fx_instrument(sym)
        try:
            df = fetch(
                instrument=instrument,
                interval=interval_code,
                offer_side=OFFER_SIDE_BID,
                start=start_dt,
                end=end_dt,
            )
        except Exception:
            failed.append(sym)
            continue

        if df is None or len(df) == 0 or "close" not in df.columns:
            failed.append(sym)
            continue

        s = df["close"].astype(float).rename(sym)
        s = s[~s.index.duplicated(keep="last")]
        series_list.append(s)

    if verbose:
        ok = len(series_list)
        print(
            f"[MarketGraph] dukascopy fetched close series: ok={ok} failed={len(failed)} "
            f"interval={interval} start={start} end={end}"
        )
        if failed:
            print(f"[MarketGraph] dukascopy failed tickers: {failed}")

    if len(series_list) == 0:
        raise ValueError("Dukascopy returned no price series (all symbols failed).")

    prices = pd.concat(series_list, axis=1).sort_index()

    if isinstance(prices.index, pd.DatetimeIndex) and prices.index.tz is not None:
        prices.index = prices.index.tz_convert("UTC").tz_localize(None)

    return prices

warnings.filterwarnings('ignore')


@dataclass
class MarketGraph:
    """
    Production-grade market graph constructor for GSP-based statistical arbitrage.
    
    Features:
    - Robust data ingestion with configurable missing data handling
    - Vectorized correlation and masking operations (no loops)
    - Sector-constrained edge construction
    - Graph diffusion operator for expected return calculation
    - Residual (mispricing) signal generation
    - Rich visualization with degree-based sizing and sector coloring
    - Comprehensive graph statistics and analysis tools
    
    Attributes
    ----------
    tickers : List[str]
        Stock ticker symbols
    start : Optional[str]
        Start date for data download (YYYY-MM-DD)
    end : Optional[str]
        End date for data download (YYYY-MM-DD)
    fill_limit : int
        Max consecutive days to forward fill missing data
    auto_adjust : bool
        Use yfinance auto-adjusted prices
    """
    
    tickers: List[str]
    start: Optional[str] = None
    end: Optional[str] = None
    fill_limit: int = 3
    auto_adjust: bool = False
    provider: str = "yfinance"
    alpaca_feed: Optional[str] = None
    max_nan_frac: float = 0.5
    min_obs: int = 10
    
    # Internal state (not set via __init__)
    prices_: pd.DataFrame = field(init=False, default_factory=pd.DataFrame)
    returns_: pd.DataFrame = field(init=False, default_factory=pd.DataFrame)
    sectors_: pd.Series = field(init=False, default_factory=pd.Series)
    adjacency_: pd.DataFrame = field(init=False, default_factory=pd.DataFrame)
    graph_: nx.Graph = field(init=False, default=None)
    dropped_tickers_: List[str] = field(init=False, default_factory=list)
    
    def __post_init__(self):
        """Validate initialization parameters."""
        if not self.tickers:
            raise ValueError("Tickers list cannot be empty")
        if self.fill_limit < 0:
            raise ValueError("fill_limit must be non-negative")
        
        print(f"MarketGraph initialized with {len(self.tickers)} tickers")
        if self.start:
            print(f"Date range: {self.start} to {self.end or 'present'}")
    
    # =========================================================================
    # DATA INGESTION & PREPROCESSING
    # =========================================================================
    
    def fetch_data(self, interval: str = "1d", verbose: bool = True) -> MarketGraph:
        """
        Download price data and compute log returns with robust missing data handling.

        Providers
        ---------
        - yfinance (default)
        - alpaca   (stocks; uses per-symbol requests to avoid truncation)
        - dukascopy (FX historical bars)
        """
        provider = (self.provider or "yfinance").lower().strip()

        if provider == "alpaca":
            prices = _alpaca_fetch_close_prices(
                tickers=self.tickers,
                start=self.start,
                end=self.end,
                interval=interval,
                feed=self.alpaca_feed or "iex",
                verbose=verbose,
            )
        elif provider == "dukascopy":
            prices = _dukascopy_fetch_close_prices(
                tickers=self.tickers,
                start=self.start,
                end=self.end,
                interval=interval,
                verbose=verbose,
            )
        else:
            prices = _yfinance_fetch_close_prices(
                tickers=self.tickers,
                start=self.start,
                end=self.end,
                interval=interval,
                auto_adjust=self.auto_adjust,
                verbose=verbose,
            )

        # Basic cleaning
        prices = prices.replace([np.inf, -np.inf], np.nan)
        prices = prices.where(prices > 0)  # avoid log(0) / log(negative)

        # If a few symbols have persistent gaps, full-panel alignment can collapse the history.
        # Drop symbols with too much missingness before aligning the panel.
        nan_frac = prices.isna().mean()
        to_drop = nan_frac[nan_frac > self.max_nan_frac].index.tolist()
        if to_drop:
            if verbose:
                print(f"[MarketGraph] Dropping {len(to_drop)} tickers with missingness > {self.max_nan_frac:.0%}")
            prices = prices.drop(columns=to_drop)

        # Forward fill up to fill_limit consecutive NaNs, then drop rows that are
        # ALL NaN (M7: was how="any" which removed entire dates if ANY ticker had NaN).
        prices = prices.ffill(limit=self.fill_limit).dropna(how="all")

        # Compute log returns — drop rows where ALL tickers are NaN
        returns = np.log(prices).diff().dropna(how="all")

        # Filter tickers with sufficient observations
        if returns.shape[0] < 2:
            raise ValueError("Insufficient data after cleaning (need at least 2 observations)")

        keep = [c for c in returns.columns if returns[c].dropna().shape[0] >= self.min_obs]
        dropped = [c for c in returns.columns if c not in keep]

        if len(keep) < 2:
            raise ValueError(
                f"Insufficient tickers after cleaning (need at least 2). "
                f"kept={len(keep)} dropped={len(dropped)}"
            )

        prices = prices[keep]
        returns = returns[keep]

        self.prices_ = prices
        self.returns_ = returns
        self.tickers = keep  # update
        self.dropped_tickers_ = dropped

        if verbose:
            print(f"[MarketGraph] provider={provider} interval={interval} | prices={prices.shape} returns={returns.shape}")

        return self

    def fetch_sectors(
        self,
        exclude_unknown: bool = False,
        verbose: bool = True
    ) -> MarketGraph:
        """
        Fetch sector metadata for each ticker using yfinance.
        
        Parameters
        ----------
        exclude_unknown : bool
            If True, remove tickers with missing/unknown sectors from dataset
        verbose : bool
            Print sector distribution
            
        Returns
        -------
        MarketGraph
            Self (for method chaining)
        """
        if self.returns_.empty:
            self.fetch_data(verbose=verbose)

        provider = (self.provider or "yfinance").lower().strip()
        if provider == "dukascopy":
            tickers = list(self.returns_.columns) if not self.returns_.empty else list(self.tickers)
            self.sectors_ = pd.Series({t: "Unknown" for t in tickers}, name="sector")
            return self
        
        if verbose:
            print("\nFetching sector information...")
        
        tickers = list(self.returns_.columns)
        sector_map: Dict[str, str] = {}
        
        for ticker in tickers:
            try:
                info = yf.Ticker(ticker).info or {}
                sector = info.get("sector") or "Unknown"
                sector_map[ticker] = sector
            except Exception as e:
                if verbose:
                    print(f"  Warning: Could not fetch sector for {ticker}: {e}")
                sector_map[ticker] = "Unknown"
        
        sectors = pd.Series(sector_map, name="sector")
        
        # Optionally filter out Unknown sectors
        if exclude_unknown:
            keep = sectors[sectors != "Unknown"].index
            n_removed = len(sectors) - len(keep)
            self.returns_ = self.returns_.loc[:, keep]
            self.prices_ = self.prices_.loc[:, keep]
            sectors = sectors.loc[keep]
            if verbose and n_removed > 0:
                print(f"  Removed {n_removed} tickers with unknown sectors")
        
        self.sectors_ = sectors
        
        # Print sector distribution
        if verbose:
            sector_counts = sectors.value_counts()
            print(f"\n{'Sector':<25} {'Count':>5}")
            print("-" * 32)
            for sector, count in sector_counts.items():
                print(f"{sector:<25} {count:>5}")
        
        return self
    
    @staticmethod
    def _create_sector_mask(sectors: pd.Series) -> pd.DataFrame:
        """
        Create binary sector mask matrix M where M[i,j] = 1 iff stocks share sector.
        
        Uses vectorized broadcasting for efficiency.
        
        Parameters
        ----------
        sectors : pd.Series
            Sector labels for each ticker
            
        Returns
        -------
        pd.DataFrame
            Binary mask matrix (N × N)
        """
        sectors_array = sectors.to_numpy(dtype=object)
        # Broadcasting: (N,1) == (1,N) → (N,N)
        mask = (sectors_array[:, None] == sectors_array[None, :]).astype(float)
        # O6: suppress Unknown-Unknown matches — tickers without a known sector
        # should NOT be linked to each other (prevents spurious cross-sector edges).
        unknown = sectors_array == "Unknown"
        both_unknown = unknown[:, None] | unknown[None, :]
        mask = np.where(both_unknown, 0.0, mask)
        return pd.DataFrame(mask, index=sectors.index, columns=sectors.index)
    
    # =========================================================================
    # GRAPH CONSTRUCTION
    # =========================================================================
    
    def build_graph(
        self,
        lookback_window: int = 22,
        sector_only: bool = True,
        threshold: float = 0.3,
        as_of: Optional[Union[str, pd.Timestamp]] = None,
        exclude_unknown_sectors: bool = False,
        verbose: bool = True
    ) -> nx.Graph:
        """
        Construct undirected market graph from windowed correlation matrix.
        
        Pipeline:
        1. Select return window (most recent lookback_window days)
        2. Compute correlation matrix (vectorized)
        3. Apply sector-only mask: W = W ⊙ M (optional)
        4. Threshold: correlations < threshold → 0 (noise filter)
        5. Remove self-loops: diagonal → 0
        6. Enforce symmetry: W = (W + W^T) / 2
        7. Convert to NetworkX undirected graph
        
        Parameters
        ----------
        lookback_window : int
            Number of recent trading days for correlation calculation
        sector_only : bool
            Only create edges between stocks in same sector
        threshold : float
            Minimum correlation to retain edge (noise filter)
        as_of : Optional[Union[str, pd.Timestamp]]
            End date for lookback window (default: most recent date)
        exclude_unknown_sectors : bool
            Remove tickers with unknown sectors before building graph
        verbose : bool
            Print graph statistics
            
        Returns
        -------
        nx.Graph
            Undirected weighted graph with tickers as nodes
        """
        # Ensure data is loaded
        if self.returns_.empty:
            self.fetch_data(verbose=verbose)
        
        if sector_only and self.sectors_.empty:
            self.fetch_sectors(exclude_unknown=exclude_unknown_sectors, verbose=verbose)
        
        # Select return window
        rets = self.returns_.copy()
        if as_of is not None:
            rets = rets.loc[:pd.Timestamp(as_of)]
        
        window = rets.tail(lookback_window)
        if window.shape[0] < 2:
            raise ValueError(
                f"Insufficient observations ({window.shape[0]}) for correlation. "
                f"Need at least 2, requested lookback={lookback_window}"
            )
        
        if verbose:
            print(f"\nBuilding graph with lookback={lookback_window} days...")
            print(f"  Window: {window.index[0].date()} to {window.index[-1].date()}")
        
        # Compute correlation matrix (vectorized)
        corr = window.corr()

        # Remove self-loops. pandas 3 copy-on-write makes .values read-only,
        # so zero the diagonal on an owned numpy copy.
        corr_vals = corr.to_numpy(copy=True)
        np.fill_diagonal(corr_vals, 0.0)
        corr = pd.DataFrame(corr_vals, index=corr.index, columns=corr.columns)
        
        # Apply sector mask if requested
        if sector_only:
            sectors = self.sectors_.reindex(corr.columns).fillna("Unknown")
            M = self._create_sector_mask(sectors)
            corr = corr * M
            if verbose:
                print("  ✓ Applied sector mask (edges only within same sector)")
        
        # Threshold filter (removes weak correlations AND negative correlations).
        # Diagonal is already 0 and 0 < threshold keeps it 0 through the filter.
        W = corr.where(corr >= threshold, 0.0)

        # Enforce symmetry (critical for undirected graph)
        W = (W + W.T) / 2.0
        
        self.adjacency_ = W
        
        # Build NetworkX graph
        G = nx.from_pandas_adjacency(W, create_using=nx.Graph)
        
        # Attach node attributes
        nx.set_node_attributes(G, {t: t for t in W.index}, name="ticker")
        if not self.sectors_.empty:
            sector_dict = self.sectors_.reindex(W.index).fillna("Unknown").to_dict()
            nx.set_node_attributes(G, sector_dict, name="sector")
        
        self.graph_ = G
        
        if verbose:
            n_nodes = G.number_of_nodes()
            n_edges = G.number_of_edges()
            max_edges = n_nodes * (n_nodes - 1) / 2
            density = n_edges / max_edges if max_edges > 0 else 0
            avg_degree = 2 * n_edges / n_nodes if n_nodes > 0 else 0
            
            print(f"\n{'Graph Statistics':-^50}")
            print(f"  Nodes: {n_nodes}")
            print(f"  Edges: {n_edges}")
            print(f"  Density: {density:.4f}")
            print(f"  Avg degree: {avg_degree:.2f}")
        
        return G
    
    # =========================================================================
    # GRAPH SIGNAL PROCESSING (GSP)
    # =========================================================================
    
    def diffusion_operator(self, eps: float = 1e-12) -> pd.DataFrame:
        """
        Construct row-stochastic diffusion operator P from adjacency W.
        
        Formulation:
            P[i,j] = W[i,j] / sum_k W[i,k]
        
        This creates a random-walk graph shift operator where each row sums to 1.
        For isolated nodes (zero degree), the row becomes all zeros.
        
        Parameters
        ----------
        eps : float
            Small value to avoid division by zero
            
        Returns
        -------
        pd.DataFrame
            Row-stochastic diffusion operator (N × N)
        """
        if self.adjacency_.empty:
            raise ValueError("Adjacency matrix is empty. Call build_graph() first.")
        
        W = self.adjacency_.copy()
        row_sum = W.sum(axis=1).to_numpy(dtype=float)
        
        # Inverse row sums (safe division)
        inv = np.where(row_sum > eps, 1.0 / row_sum, 0.0)
        
        # Row normalization: P = W / rowsum (vectorized)
        P = (W.T * inv).T
        
        return P
    
    def expected_returns(
        self,
        returns: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Compute graph diffusion-implied expected returns.
        
        Formula:
            r_hat[t] = r[t] @ P
        
        Where:
        - r[t] is the vector of actual returns at time t
        - P is the diffusion operator (row-stochastic)
        - r_hat[t] is the neighbor-weighted expected return
        
        Interpretation:
        Each stock's expected return is the weighted average of its neighbors'
        actual returns, where weights come from correlation strength.
        
        Parameters
        ----------
        returns : Optional[pd.DataFrame]
            Custom return panel (T × N). If None, uses self.returns_
            
        Returns
        -------
        pd.DataFrame
            Expected returns (same shape as input)
        """
        if self.adjacency_.empty:
            raise ValueError("Adjacency matrix is empty. Call build_graph() first.")
        
        P = self.diffusion_operator()
        
        if returns is None:
            r = self.returns_.reindex(columns=P.index)
        else:
            r = returns.reindex(columns=P.index)
        
        # Matrix multiplication: (T × N) @ (N × N) → (T × N)
        r_hat = pd.DataFrame(
            r.to_numpy() @ P.to_numpy(),
            index=r.index,
            columns=r.columns
        )
        
        return r_hat
    
    def residuals(
        self,
        returns: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        DEPRECATED — M5: This method uses random-walk diffusion (r @ P) which
        produces a DIFFERENT signal formula than SignalEngine.run_diffusion(),
        which uses Laplacian diffusion (solve (I-αL+regI)h=x, residual=x-h).

        Mixing both signal definitions in a single dataset corrupts BrainEngine
        training. Use SignalEngine.run_diffusion() or sim_engine functions for
        all residual computation.

        Kept for backward compatibility only — DO NOT use for new signal generation.
        """
        warnings.warn(
            "MarketGraph.residuals() uses random-walk diffusion and is deprecated. "
            "Use SignalEngine.run_diffusion() (Laplacian diffusion) instead to ensure "
            "consistent signal computation across training and live trading.",
            DeprecationWarning,
            stacklevel=2,
        )
        if returns is None:
            r = self.returns_.copy()
        else:
            r = returns.copy()

        r_hat = self.expected_returns(r)
        e = r.reindex(columns=r_hat.columns) - r_hat

        return e
    
    def get_neighbor_signal(
        self,
        ticker: str,
        as_of: Optional[Union[str, pd.Timestamp]] = None
    ) -> Dict[str, float]:
        """
        Get detailed GSP signal breakdown for a specific stock.
        
        Parameters
        ----------
        ticker : str
            Stock ticker symbol
        as_of : Optional[Union[str, pd.Timestamp]]
            Date for signal calculation (default: most recent)
            
        Returns
        -------
        Dict[str, float]
            Dictionary containing:
            - actual_return: Stock's actual return
            - expected_return: Neighbor-implied return
            - residual: Mispricing signal
            - signal: Trading direction (1=BUY, -1=SELL, 0=NEUTRAL)
        """
        if self.graph_ is None:
            raise ValueError("Graph not built. Call build_graph() first.")
        
        if ticker not in self.graph_:
            raise ValueError(f"{ticker} not in graph")
        
        # Get returns and expected returns
        r = self.returns_ if as_of is None else self.returns_.loc[:as_of]
        r_hat = self.expected_returns(r)
        e = self.residuals(r)
        
        # Get most recent values
        actual = r[ticker].iloc[-1]
        expected = r_hat[ticker].iloc[-1]
        residual = e[ticker].iloc[-1]
        
        # Generate trading signal
        if residual < -0.0001:  # Threshold to avoid noise
            signal = 1  # BUY
        elif residual > 0.0001:
            signal = -1  # SELL
        else:
            signal = 0  # NEUTRAL
        
        return {
            'actual_return': actual,
            'expected_return': expected,
            'residual': residual,
            'signal': signal,
            'signal_label': ['SELL', 'NEUTRAL', 'BUY'][signal + 1]
        }
    
    # =========================================================================
    # ANALYSIS & STATISTICS
    # =========================================================================
    
    def get_graph_statistics(self) -> pd.Series:
        """
        Compute comprehensive graph topology statistics.
        
        Returns
        -------
        pd.Series
            Dictionary of graph metrics
        """
        if self.graph_ is None:
            raise ValueError("Graph not built. Call build_graph() first.")
        
        G = self.graph_
        
        stats = {
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'density': nx.density(G),
            'avg_degree': np.mean([d for _, d in G.degree()]),
            'max_degree': max([d for _, d in G.degree()]) if G.number_of_nodes() > 0 else 0,
            'avg_clustering': nx.average_clustering(G),
            'num_components': nx.number_connected_components(G),
        }
        
        # Optional metrics (may fail for disconnected graphs)
        try:
            stats['avg_shortest_path'] = nx.average_shortest_path_length(G)
        except:
            stats['avg_shortest_path'] = np.nan
        
        try:
            stats['diameter'] = nx.diameter(G)
        except:
            stats['diameter'] = np.nan
        
        return pd.Series(stats, name='graph_statistics')
    
    def analyze_residuals(
        self,
        n_days: int = 30
    ) -> pd.DataFrame:
        """
        Analyze residual statistics for signal quality assessment.
        
        Parameters
        ----------
        n_days : int
            Number of recent days to analyze
            
        Returns
        -------
        pd.DataFrame
            Statistics per ticker: mean, std, sharpe, skew, kurtosis
        """
        e = self.residuals().tail(n_days)
        
        stats = pd.DataFrame({
            'mean': e.mean(),
            'std': e.std(),
            'sharpe': e.mean() / e.std(),
            'skew': e.skew(),
            'kurtosis': e.kurtosis(),
            'max': e.max(),
            'min': e.min()
        })
        
        return stats.sort_values('sharpe', ascending=False)
    
    # =========================================================================
    # VISUALIZATION
    # =========================================================================
    
    def plot_market_graph(
        self,
        figsize: Tuple[int, int] = (16, 12),
        seed: int = 42,
        base_node_size: float = 300.0,
        degree_scale: float = 800.0,
        edge_alpha: float = 0.35,
        label_fontsize: int = 9,
        show_labels: bool = True,
        save_path: Optional[str] = None
    ) -> None:
        """
        Create publication-quality network visualization.
        
        Features:
        - Node color: Sector membership
        - Node size: Weighted degree (connectedness)
        - Edge thickness: Correlation strength
        - Spring layout: Force-directed positioning
        
        Parameters
        ----------
        figsize : Tuple[int, int]
            Figure dimensions (width, height) in inches
        seed : int
            Random seed for layout reproducibility
        base_node_size : float
            Minimum node size
        degree_scale : float
            Scaling factor for degree-based sizing
        edge_alpha : float
            Edge transparency (0-1)
        label_fontsize : int
            Font size for ticker labels
        show_labels : bool
            Display ticker labels on nodes
        save_path : Optional[str]
            Path to save figure (PNG, PDF, etc.)
        """
        if self.graph_ is None or len(self.graph_.nodes) == 0:
            self.build_graph()
        
        G = self.graph_
        
        print("\nGenerating graph visualization...")
        
        # Layout (force-directed with edge weights)
        pos = nx.spring_layout(G, seed=seed, weight='weight', k=0.5, iterations=50)
        
        # Get node sectors
        sectors = nx.get_node_attributes(G, 'sector')
        if not sectors:
            sectors = {n: 'Unknown' for n in G.nodes}
        
        nodes = list(G.nodes)
        sector_series = pd.Series(sectors).reindex(nodes).fillna('Unknown')
        unique_sectors = sorted(sector_series.unique())
        
        # Create sector color map
        n_sectors = len(unique_sectors)
        cmap = plt.cm.get_cmap('tab20', max(n_sectors, 1))
        sector_to_color = {sec: cmap(i) for i, sec in enumerate(unique_sectors)}
        node_colors = [sector_to_color[sector_series.loc[n]] for n in nodes]
        
        # Calculate node sizes proportional to weighted degree
        wdeg = np.array([G.degree(n, weight='weight') for n in nodes], dtype=float)
        if wdeg.max() > 0:
            wdeg_norm = wdeg / wdeg.max()
        else:
            wdeg_norm = np.zeros_like(wdeg)
        
        node_sizes = base_node_size + degree_scale * wdeg_norm
        
        # Calculate edge widths proportional to weight
        weights = np.array([d.get('weight', 0.0) for _, _, d in G.edges(data=True)])
        if len(weights) > 0 and weights.max() > weights.min():
            widths = 0.5 + 4.5 * (weights - weights.min()) / (weights.max() - weights.min())
        else:
            widths = np.full(len(weights), 2.0)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Draw edges
        nx.draw_networkx_edges(
            G, pos,
            width=widths,
            alpha=edge_alpha,
            edge_color='gray',
            ax=ax
        )
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos,
            node_size=node_sizes,
            node_color=node_colors,
            linewidths=0.5,
            edgecolors='black',
            ax=ax
        )
        
        # Draw labels
        if show_labels:
            nx.draw_networkx_labels(
                G, pos,
                font_size=label_fontsize,
                font_weight='bold',
                ax=ax
            )
        
        # Create legend
        legend_elements = [
            Patch(facecolor=sector_to_color[sec], label=sec, edgecolor='black')
            for sec in unique_sectors
        ]
        ax.legend(
            handles=legend_elements,
            loc='upper left',
            bbox_to_anchor=(1, 1),
            title='Sectors',
            title_fontsize=12,
            frameon=True,
            fancybox=True,
            shadow=True
        )
        
        # Title and formatting
        ax.set_title(
            'Market Graph: Topological Structure for Statistical Arbitrage\n'
            'Node size ∝ weighted degree | Edge width ∝ correlation | Color = sector',
            fontsize=14,
            fontweight='bold',
            pad=20
        )
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Figure saved to {save_path}")
        
        plt.show()
    
    def plot_residual_heatmap(
        self,
        n_days: int = 30,
        figsize: Tuple[int, int] = (14, 8),
        cmap: str = 'RdYlGn_r',
        save_path: Optional[str] = None
    ) -> None:
        """
        Visualize recent residuals as a heatmap for signal analysis.
        
        Parameters
        ----------
        n_days : int
            Number of recent days to display
        figsize : Tuple[int, int]
            Figure dimensions
        cmap : str
            Matplotlib colormap name
        save_path : Optional[str]
            Path to save figure
        """
        e = self.residuals().tail(n_days)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        im = ax.imshow(e.T, aspect='auto', cmap=cmap, interpolation='nearest')
        
        # Formatting
        ax.set_xlabel('Trading Day', fontsize=12, fontweight='bold')
        ax.set_ylabel('Ticker', fontsize=12, fontweight='bold')
        ax.set_title(
            f'Residual Signals (Last {n_days} Days)\n'
            'Red = Overbought (SELL) | Green = Oversold (BUY)',
            fontsize=14,
            fontweight='bold',
            pad=15
        )
        
        ax.set_yticks(range(len(e.columns)))
        ax.set_yticklabels(e.columns, fontsize=8)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Residual', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Heatmap saved to {save_path}")
        
        plt.show()


# =============================================================================
# DEMONSTRATION & TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print(" MarketGraph: Topological Arbitrage via Graph Signal Processing")
    print("=" * 70)
    
    # Sample portfolio: Tech, Finance, Energy stocks
    tickers = [
        # Technology
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
        'ADBE', 'CRM', 'ORCL', 'CSCO', 'INTC',
        # Financials
        'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C',
        # Energy
        'XOM', 'CVX', 'COP', 'SLB',
        # Consumer
        'WMT', 'HD', 'MCD', 'NKE', 'COST'
    ]
    
    # Initialize MarketGraph
    mg = MarketGraph(
        tickers=tickers,
        start='2023-01-01',
        end='2024-12-01',
        fill_limit=3
    )
    
    # Execute full pipeline
    print("\n" + "=" * 70)
    print(" STEP 1: Data Ingestion")
    print("=" * 70)
    mg.fetch_data(verbose=True)
    
    print("\n" + "=" * 70)
    print(" STEP 2: Sector Metadata")
    print("=" * 70)
    mg.fetch_sectors(exclude_unknown=False, verbose=True)
    
    print("\n" + "=" * 70)
    print(" STEP 3: Graph Construction")
    print("=" * 70)
    mg.build_graph(lookback_window=22, sector_only=True, threshold=0.3, verbose=True)
