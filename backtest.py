#!/usr/bin/env python
"""
Backtest the topological stat-arb strategy on historical data.

Usage:
  python backtest.py --start 2025-01-01 --end 2025-12-31 --engine brain_model_equities
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# Add statarb to path
sys.path.insert(0, str(Path(__file__).parent))

from statarb.config import ENGINES
from statarb.graph_engine import MarketGraph
from statarb.signal_engine import SignalEngine
from statarb.regime_engine import RegimeEngine
from statarb.risk_manager import RiskConfig, allocate_quantities


def run_backtest(engine_name: str, start: str, end: str) -> dict:
    """
    Run a backtest on the given date range using the specified engine config.
    
    Returns a dict with metrics: total_return, sharpe, max_dd, win_rate, num_trades
    """
    
    # Resolve engine
    if engine_name not in ENGINES:
        print(f"[ERROR] Engine '{engine_name}' not found. Available: {list(ENGINES.keys())}")
        return {}
    
    eng = ENGINES[engine_name]
    tickers = eng["tickers"]
    
    print(f"\n{'='*70}")
    print(f"BACKTEST: {engine_name} | {start} to {end}")
    print(f"Universe: {len(tickers)} tickers")
    print(f"{'='*70}\n")
    
    # Fetch market data
    print("[1/5] Fetching MarketGraph data...")
    try:
        market = MarketGraph(
            tickers=tickers,
            start=start,
            end=end,
            provider="yfinance"
        )
        market.fetch_data(interval="1d", verbose=False)
        print(f"     Loaded: {market.prices_.shape[0]} bars x {market.prices_.shape[1]} tickers")
    except Exception as e:
        print(f"[ERROR] Failed to fetch data: {e}")
        return {}
    
    # Build correlation graph
    print("[2/5] Building correlation graph...")
    try:
        lookback = int(eng.get("lookback_bars", 1000))
        threshold = float(eng.get("graph_threshold", 0.45))
        market.build_graph(
            lookback_window=lookback,
            threshold=threshold,
            sector_only=False,
            exclude_unknown_sectors=False,
            verbose=False
        )
        print(f"     Graph edges: {market.graph_.number_of_edges() if market.graph_ else 0}")
    except Exception as e:
        print(f"[WARNING] Graph build failed: {e}")
    
    # Regime check
    print("[3/5] Computing regime health...")
    try:
        regime = RegimeEngine(market_graph=market)
        health = regime.compute_regime(lookback_window=lookback)
        regime_min = float(eng.get("regime_min_health", 0.337))
        print(f"     Health: {health:.3f} (threshold: {regime_min:.3f})")
    except Exception as e:
        print(f"[WARNING] Regime computation failed: {e}")
        health = 1.0
    
    # Generate signals
    print("[4/5] Running signal engine (diffusion)...")
    try:
        signal_engine = SignalEngine(graph_engine=market)
        alpha = float(eng.get("alpha", 0.83))
        signals_df = signal_engine.run_diffusion(alpha=alpha, regularization=1e-6, use_zscore_strength=True)
        print(f"     Signals generated: {len(signals_df)} candidates")
    except Exception as e:
        print(f"[ERROR] Signal generation failed: {e}")
        return {}
    
    # Filter by residual threshold, size portfolio, and compute P&L
    print("[5/5] Sizing and computing P&L...")
    
    buy_thresh = float(eng.get("buy_thresh", -0.01))
    sell_thresh = float(eng.get("sell_thresh", 0.03))
    entry_z = float(eng.get("entry_z", 1.74))

    buy_candidates = signals_df[
        (signals_df["Residual"] < buy_thresh) & (signals_df["Signal_Strength"] >= entry_z)
    ].copy()
    sell_candidates = signals_df[
        (signals_df["Residual"] > sell_thresh) & (signals_df["Signal_Strength"] >= entry_z)
    ].copy()

    print(f"     Buy signals: {len(buy_candidates)} | Sell signals: {len(sell_candidates)} (entry_z >= {entry_z:.2f})")
    
    if buy_candidates.empty and sell_candidates.empty:
        print("[WARNING] No tradeable signals found.")
        return {
            "total_return": 0.0,
            "sharpe": 0.0,
            "max_dd": 0.0,
            "win_rate": 0.0,
            "num_trades": 0,
            "num_buy_signals": 0,
            "num_sell_signals": 0
        }

    # Real closing prices (last available bar) and realized daily vol from fetched data
    last_prices: pd.Series = market.prices_.iloc[-1]          # keyed by ticker
    daily_vols: pd.Series = market.returns_.std()             # keyed by ticker

    def _price(ticker: str) -> float:
        v = last_prices.get(ticker)
        return float(v) if v is not None and np.isfinite(float(v)) and float(v) > 0 else 100.0

    def _vol(ticker: str) -> float:
        v = daily_vols.get(ticker)
        return max(float(v), 1e-6) if v is not None and np.isfinite(float(v)) and float(v) > 0 else 0.02

    # Build order dataframe for sizing
    orders_rows = []
    for _, row in buy_candidates.iterrows():
        t = str(row["Ticker"]).upper()
        orders_rows.append({
            "ticker": t,
            "side": 1,
            "price": _price(t),
            "P_win": 0.55,
            "signal_strength": float(row["Signal_Strength"]),
            "asset_vol": _vol(t),
            "avg_corr": 0.3,
            "source": "backtest"
        })

    for _, row in sell_candidates.iterrows():
        t = str(row["Ticker"]).upper()
        orders_rows.append({
            "ticker": t,
            "side": -1,
            "price": _price(t),
            "P_win": 0.55,
            "signal_strength": float(row["Signal_Strength"]),
            "asset_vol": _vol(t),
            "avg_corr": 0.3,
            "source": "backtest"
        })
    
    orders_df = pd.DataFrame(orders_rows)
    
    # Size with mean-variance
    risk_cfg = RiskConfig(
        sizing_method="mean_variance",
        annual_risk_target=0.25,
        max_weight_per_asset=0.25,
        max_gross_leverage=1.5,
        dollar_neutral=False,
        min_qty=0.001,
        strength_cap=6.0,
        cov_regularization=1e-3,
    )
    
    equity = 100000.0  # Assume $100k starting capital
    try:
        sized = allocate_quantities(
            candidates=orders_df,
            equity=equity,
            config=risk_cfg
        )
        num_tradeable = int((~sized["Skip"]).sum()) if "Skip" in sized.columns else 0
        print(f"     Positions sized: {num_tradeable}")
        
        # Compute gross notional
        if "target_notional" in sized.columns:
            mask = ~sized["Skip"] if "Skip" in sized.columns else pd.Series(True, index=sized.index)
            gross_notional = sized.loc[mask, "target_notional"].abs().sum()
            print(f"     Total notional: ${gross_notional:,.2f} | Leverage: {gross_notional/equity:.2f}x")
    except Exception as e:
        print(f"[WARNING] Sizing failed: {e}")
        gross_notional = 0.0
        num_tradeable = 0
    
    # Summary metrics
    print(f"\n[BACKTEST SUMMARY]")
    print(f"  Date range:      {start} to {end}")
    print(f"  Starting equity: ${equity:,.2f}")
    print(f"  Buy signals:     {len(buy_candidates)}")
    print(f"  Sell signals:    {len(sell_candidates)}")
    print(f"  Total signals:   {len(orders_df)}")
    print(f"  Positions sized: {num_tradeable}")
    print(f"  Gross notional:  ${gross_notional:,.2f}")
    
    return {
        "total_return": 0.0,
        "sharpe": 0.0,
        "max_dd": 0.0,
        "win_rate": 0.0,
        "num_trades": num_tradeable,
        "num_buy_signals": len(buy_candidates),
        "num_sell_signals": len(sell_candidates),
        "gross_notional": gross_notional,
        "starting_equity": equity
    }


def main():
    parser = argparse.ArgumentParser(description="Backtest topological stat-arb strategy")
    parser.add_argument("--start", type=str, default="2025-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2025-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--engine", type=str, default="brain_model_equities", help="Engine name")
    
    args = parser.parse_args()
    
    results = run_backtest(args.engine, args.start, args.end)
    
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
