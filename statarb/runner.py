# statarb/runner.py
"""
Wrapper that runs the Topological Stat-Arb ML pipeline from within the
unified prediction-bot daemon.

Usage:
    from statarb.runner import run_statarb_scan
    run_statarb_scan()   # called by the daemon scheduler
"""

import os
import sys
import traceback


def run_statarb_scan():
    """
    Execute one full cycle of the Topological Stat-Arb ML pipeline.
    Gracefully skips if Alpaca keys are missing or dependencies are unavailable.
    """
    print("\n" + "=" * 60)
    print("=== STAT-ARB ENGINE (Topological ML) ===")
    print("=" * 60)

    # Read Alpaca keys from statarb config (hardcoded for demo use)
    from statarb.config import ALPACA_API_KEY, ALPACA_SECRET_KEY
    api_key = ALPACA_API_KEY
    secret_key = ALPACA_SECRET_KEY
    if not api_key or not secret_key:
        print("[STAT-ARB] Skipped: Alpaca keys not configured in statarb/config.py.")
        return

    # Temporarily add the statarb directory to sys.path so that
    # internal imports like `import config` resolve to statarb/config.py
    statarb_dir = os.path.dirname(os.path.abspath(__file__))
    original_path = sys.path.copy()
    sys.path.insert(0, statarb_dir)

    try:
        # These imports happen inside the function so they use the modified sys.path
        # and resolve `import config` to statarb/config.py
        if "config" in sys.modules:
            # Remove any previously cached 'config' module from the insider bot
            del sys.modules["config"]

        from statarb.execution import ExecutionEngine
        from statarb.graph_engine import MarketGraph
        from statarb.regime_engine import RegimeEngine
        from statarb.signal_engine import SignalEngine

        # Now import config from statarb dir
        import config as cfg

        import numpy as np
        import pandas as pd
        from datetime import date

        paper = bool(getattr(cfg, "PAPER", True))
        dry_run = bool(getattr(cfg, "DRY_RUN", False))

        trader = ExecutionEngine(api_key=api_key, secret_key=secret_key, paper=paper)

        # Get account equity
        eq = trader.get_account_equity() if not dry_run else None
        equity = float(eq) if (eq is not None and np.isfinite(eq) and eq > 0) else float(getattr(cfg, "DEFAULT_EQUITY", 100_000))
        print(f"[STAT-ARB] Account equity: ${equity:,.2f}")
        print(f"[STAT-ARB] Paper mode: {paper} | Dry run: {dry_run}")

        engines = getattr(cfg, "ENGINES", None)
        if not isinstance(engines, dict) or not engines:
            print("[STAT-ARB] No ENGINES configured in statarb/config.py. Skipping.")
            return

        # Run each engine
        for name, ecfg in engines.items():
            tickers = list(ecfg.get("tickers", []))
            if not tickers:
                print(f"[STAT-ARB] Engine '{name}' has no tickers, skipping.")
                continue

            print(f"\n[STAT-ARB] Running engine: {name} | {len(tickers)} tickers")

            history_days = int(getattr(cfg, "HISTORY_DAYS", 365))
            end = pd.Timestamp(date.today())
            start = end - pd.Timedelta(days=history_days)

            mg = MarketGraph(
                tickers=tickers,
                start=str(start.date()),
                end=str(end.date()),
                provider=str(ecfg.get("provider", getattr(cfg, "PROVIDER", "yfinance"))),
                alpaca_feed=ecfg.get("alpaca_feed", getattr(cfg, "ALPACA_FEED", None)),
            )
            mg.fetch_data(verbose=False)

            lookback = int(ecfg.get("lookback", getattr(cfg, "LOOKBACK_DAYS", 22)))
            threshold = float(ecfg.get("graph_threshold", getattr(cfg, "GRAPH_THRESHOLD", 0.30)))
            sector_only = bool(ecfg.get("sector_only", False))

            if sector_only:
                try:
                    mg.fetch_sectors(exclude_unknown=False, verbose=False)
                except Exception:
                    sector_only = False

            mg.build_graph(
                lookback_window=lookback,
                threshold=threshold,
                sector_only=sector_only,
                exclude_unknown_sectors=False,
                verbose=False,
            )

            # Regime check
            health = 1.0
            if bool(ecfg.get("use_regime_filter", False)):
                try:
                    reg = RegimeEngine(market_graph=mg)
                    health = float(reg.compute_regime(lookback_window=lookback))
                except Exception:
                    health = float("nan")

            regime_min = float(ecfg.get("regime_min_health", getattr(cfg, "REGIME_MIN_HEALTH", 0.25)))
            regime_ok = pd.notna(health) and (health >= regime_min)
            print(f"[STAT-ARB] Regime health: {health:.4f} | OK: {regime_ok}")

            # Generate signals
            se = SignalEngine(graph_engine=mg)
            df_signals = se.run_diffusion(
                alpha=float(ecfg.get("alpha", getattr(cfg, "ALPHA", 0.8))),
                regularization=float(ecfg.get("regularization", getattr(cfg, "REGULARIZATION", 1e-6))),
                use_zscore_strength=True,
            )

            buy_thresh = float(getattr(cfg, "BUY_THRESH", -0.01))
            sell_thresh = float(getattr(cfg, "SELL_THRESH", 0.01))

            buy_candidates = df_signals[df_signals["Residual"] < buy_thresh].sort_values("Residual", ascending=True)
            sell_candidates = df_signals[df_signals["Residual"] > sell_thresh].sort_values("Residual", ascending=False)

            if not regime_ok:
                buy_candidates = buy_candidates.iloc[0:0]

            print(f"[STAT-ARB] Buy candidates: {len(buy_candidates)} | Sell candidates: {len(sell_candidates)}")

            # Get existing positions
            positions = trader.get_positions()
            held = {t: int(q) for t, q in positions.items() if int(q) > 0}

            max_positions = int(getattr(cfg, "MAX_POSITIONS", 10))
            slots = max(0, max_positions - len(held))

            top_k = int(ecfg.get("top_k_entries", 2))
            buys = buy_candidates.head(top_k)

            # Build orders
            sell_rows = []
            for _, row in sell_candidates.iterrows():
                t = str(row["Ticker"]).strip().upper()
                if t in held:
                    sell_rows.append({"ticker": t, "side": "SELL", "Qty": held[t], "residual": float(row["Residual"])})

            buy_rows = []
            for _, row in buys.iterrows():
                if len(buy_rows) >= slots:
                    break
                t = str(row["Ticker"]).strip().upper()
                if t not in held:
                    buy_rows.append({"ticker": t, "side": "BUY", "Qty": int(getattr(cfg, "QTY_SHARES", 10)), "residual": float(row["Residual"])})

            all_orders = pd.DataFrame(sell_rows + buy_rows)
            if all_orders.empty:
                print("[STAT-ARB] No orders to submit today.")
                continue

            print("\n[STAT-ARB] Orders to submit:")
            print(all_orders.to_string(index=False))

            report = trader.execute_trades(
                all_orders,
                cancel_existing=False,
                skip_if_open_order=True,
                dry_run=dry_run,
            )
            print(f"[STAT-ARB] Execution: submitted={report.submitted}, skipped={report.skipped}, errors={report.errors}")

            # --- SEND EMAIL ALERT FOR PRICING MISMATCHES ---
            try:
                # Add root prediction-bot directory to sys.path if not there
                bot_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                if bot_root not in sys.path:
                    sys.path.append(bot_root)
                    
                from notifier import send_statarb_alert
                send_statarb_alert(all_orders)
            except Exception as alert_err:
                print(f"[STAT-ARB] Failed to trigger email alert: {alert_err}")

    except ImportError as e:
        print(f"[STAT-ARB] Missing dependency: {e}")
        print("[STAT-ARB] Run: .\\venv\\Scripts\\pip install alpaca-py scipy scikit-learn joblib networkx")
    except Exception as e:
        print(f"[STAT-ARB] Error during execution: {e}")
        traceback.print_exc()
    finally:
        # Restore original sys.path
        sys.path = original_path
        # Re-import the insider bot's config if needed
        if "config" in sys.modules:
            del sys.modules["config"]

    print("\n[STAT-ARB] Scan complete.")
