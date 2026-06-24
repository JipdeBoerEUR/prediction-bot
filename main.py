# main.py — Unified Multi-Strategy Orchestrator
# Primary: Topical Trading (BERTopic news narrative momentum)
# Secondary: Topological Stat-Arb (Laplacian diffusion on correlation graph)
#
# Flow per scan:
#   [1] Market Gate   → macro kill-switch + RegimeEngine health >= 0.25
#   [2] Dual Scouts   → asyncio.gather(topic_scout, statarb) concurrently
#   [3] Normalize     → UnifiedCandidate schema
#   [4] AI Audit      → TopicSentiment | FinBERT | LSTM | RAG (concurrent per ticker)
#   [5] P_win         → BrainEngine batch predict_proba + TradeLearner multiplier
#   [6] Risk & Size   → allocate_quantities(mean_variance, equity=$account, high-risk)
#   [7] Execute       → ExecutionEngine (Alpaca) + notify + log

from __future__ import annotations

import asyncio
import math
import os
import re
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval     import IntervalTrigger
from apscheduler.triggers.cron         import CronTrigger

# ── Statarb module ───────────────────────────────────────────────────────────
from statarb.regime_engine import RegimeEngine
from statarb.graph_engine  import MarketGraph
from statarb.signal_engine import SignalEngine
from statarb.brain_engine  import BrainEngine, BrainConfig, build_trade_features
from statarb.risk_manager  import RiskConfig, allocate_quantities
from statarb.execution     import ExecutionEngine
import statarb.config as cfg

# ── AI modules ───────────────────────────────────────────────────────────────
from ai.sentiment_ai      import analyze_sentiment
from ai.sec_analyzer      import analyze_sec_filings
from ai.earnings_analyzer import analyze_earnings_sentiment
from ai.rag_library       import generate_rag_risk_report
from ai.price_predictor   import predict_price_movement
from ai.trade_learner     import TradeLearner

# ── Topical Trading modules (new primary scout) ───────────────────────────────
from ai.news_aggregator  import aggregate_headlines, get_ticker_headlines
from ai.topic_engine     import TopicEngine, TopicCandidate
from ai.topic_exposure   import build_exposure_map

# ── Root modules ─────────────────────────────────────────────────────────────
from macro_filter       import is_market_crashing
from news_scanner       import scan_news
from notifier           import send_trade_alert, send_statarb_alert
from logger             import log_evaluation
from portfolio_monitor  import evaluate_portfolio

# Retired → kept on disk for reference:
#   sec_listener.py       → superseded by TopicEngine (topical trading primary scout)
#   fundamental_engine.py → still available but not required by topic candidates
#   scoring_engine.py     → insider conviction scoring (retired from primary flow)

# Retired — kept on disk for reference:
#   kelly_size.py   → superseded by statarb/risk_manager.allocate_quantities
#   execution.py    → superseded by statarb/execution.ExecutionEngine
#   statarb/runner.py → superseded by run_unified_scan() below

warnings.filterwarnings("ignore")

_executor = ThreadPoolExecutor()   # shared pool for blocking I/O

_MIN_PWIN        = 0.55
_REGIME_MIN      = cfg.REGIME_MIN_HEALTH   # 0.25
_ENTRY_Z         = cfg.ENGINES["equities"].get("entry_z", cfg.ENTRY_Z)
_MODEL_PATH      = cfg.ENGINES["equities"]["model_path"]
_MIN_QTY             = 0.001 if getattr(cfg, "FORCE_FRACTIONAL_SHARES", True) else 1
_MAX_POSITIONS_PER_SCAN = 5       # hard cap on new positions opened in a single scan


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 0 — OrchestratorContext & Initialization
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class OrchestratorContext:
    exec_engine:    ExecutionEngine
    regime_engine:  RegimeEngine
    brain_engine:   BrainEngine
    trade_learner:  TradeLearner
    market_graph:   MarketGraph
    account_equity:      float = 1000.0
    health_score:        float = float("nan")
    allocated_this_scan: float = 0.0   # cumulative notional committed in current scan
    positions_this_scan: int   = 0     # new positions opened in current scan


def initialize_components() -> OrchestratorContext:
    """Build and warm all singleton components once at daemon startup."""
    print("\n[INIT] Initializing Unified Orchestrator…")

    # Execution engine
    exec_engine = ExecutionEngine(
        api_key    = cfg.ALPACA_API_KEY,
        secret_key = cfg.ALPACA_SECRET_KEY,
        paper      = cfg.PAPER,
    )
    equity = exec_engine.get_account_equity() or 1000.0
    if equity < 500:
        print(f"[INIT] ⚠  Account equity ${equity:.2f}. "
              "Fractional-share mode enabled (Alpaca); stocks >$200 will be sized sub-share.")

    # Regime engine (correlation matrices passed per-scan)
    regime_engine = RegimeEngine()

    # Brain engine
    brain_engine = BrainEngine(model_type=cfg.BRAIN_MODEL_TYPE)
    if os.path.exists(_MODEL_PATH):
        try:
            brain_engine.load_model(_MODEL_PATH)
            print(f"[INIT] BrainEngine model loaded: {_MODEL_PATH}")
        except Exception as e:
            print(f"[INIT] BrainEngine model load failed ({e}). Heuristic P_win active.")
    else:
        print(f"[INIT] No BrainEngine model at {_MODEL_PATH}. Heuristic P_win active.")

    # Trade learner
    trade_learner = TradeLearner()

    # Market graph — warm up with full equities universe
    eng = cfg.ENGINES["equities"]
    market_graph = MarketGraph(
        tickers  = eng["tickers"],
        start    = eng.get("start", "2023-01-01"),
        end      = eng.get("end", None),
        provider = "yfinance",
    )
    print("[INIT] Fetching MarketGraph data (equities universe, 1d bars)…")
    try:
        market_graph.fetch_data(interval="1d", verbose=False)
        market_graph.fetch_sectors(verbose=False)
        print(f"[INIT] MarketGraph ready: {len(market_graph.tickers)} tickers.")
    except Exception as e:
        print(f"[INIT] MarketGraph fetch failed ({e}). StatArb scout will skip this cycle.")

    return OrchestratorContext(
        exec_engine    = exec_engine,
        regime_engine  = regime_engine,
        brain_engine   = brain_engine,
        trade_learner  = trade_learner,
        market_graph   = market_graph,
        account_equity = equity,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 1 — Market Gate
# ═══════════════════════════════════════════════════════════════════════════════

def check_market_health(ctx: OrchestratorContext) -> bool:
    """
    Two-stage gate. Returns False (abort scan) if either fails.
      Stage 1 — macro_filter: SPY < 200-SMA or VIX > 30.
      Stage 2 — RegimeEngine: topological health score < REGIME_MIN_HEALTH (0.25).
    """
    # Stage 1: macro kill-switch
    is_crash, reason = is_market_crashing()
    if is_crash:
        print(f"\n[GATE-1] MACRO KILL-SWITCH: {reason}")
        return False
    print(f"[GATE-1] Macro OK — {reason}")

    # Stage 2: topological regime
    try:
        if (not hasattr(ctx.market_graph, "returns_")
                or ctx.market_graph.returns_ is None
                or ctx.market_graph.returns_.empty):
            print("[GATE-2] No MarketGraph returns available. Blocking scan (fail-safe).")
            ctx.health_score = float("nan")
            return False

        corr_matrix = ctx.market_graph.returns_.tail(22).corr().values
        health = float(ctx.regime_engine.compute_regime(corr_matrix))
        ctx.health_score = health
        status = "HEALTHY" if health >= _REGIME_MIN else "STRESSED"
        print(f"[GATE-2] Regime health: {health:.4f} | {status}")

        if health < _REGIME_MIN:
            print(f"[GATE-2] REGIME BLOCK — health {health:.4f} < {_REGIME_MIN}. Aborting scan.")
            return False
    except Exception as e:
        print(f"[GATE-2] Regime check error ({e}). Blocking scan (fail-safe).")
        ctx.health_score = float("nan")
        return False

    return True


# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
# Module-level TopicEngine singleton (persists across scans)
_topic_engine_singleton: Optional[TopicEngine] = None
_prior_headlines: List[str] = []


def _get_topic_engine() -> TopicEngine:
    global _topic_engine_singleton
    if _topic_engine_singleton is None:
        tc = cfg.TOPICAL_TRADING
        _topic_engine_singleton = TopicEngine(
            min_cluster_size=tc.get("bertopic_min_cluster", 3),
            max_topics=tc.get("max_topics_per_scan", 8),
            lookback_hours=tc.get("topic_lookback_hours", 6),
            prior_headlines=_prior_headlines,
        )
    return _topic_engine_singleton


# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
# Layer 2 \u2014 Dual Scouts
# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550

async def scout_topic_candidates(ctx: OrchestratorContext) -> List[dict]:
    """Engine A (PRIMARY) \u2014 BERTopic news topic discovery."""
    global _prior_headlines

    loop    = asyncio.get_running_loop()
    tcfg    = cfg.TOPICAL_TRADING
    tickers = cfg.ENGINES["equities"]["tickers"]

    print("\n[SCOUT-A] Topical Trading Scout \u2014 aggregating headlines\u2026")

    try:
        headlines = await loop.run_in_executor(
            _executor,
            lambda: aggregate_headlines(lookback_hours=tcfg.get("topic_lookback_hours", 6)),
        )
    except Exception as e:
        print(f"[SCOUT-A] Headline aggregation failed: {e}")
        return []

    if not headlines:
        print("[SCOUT-A] No headlines fetched.")
        return []

    print(f"[SCOUT-A] {len(headlines)} headlines. Running BERTopic\u2026")

    engine = _get_topic_engine()
    engine.prior_headlines = _prior_headlines

    try:
        topics = await loop.run_in_executor(
            _executor, lambda: engine.discover_topics(headlines, use_bertopic=True)
        )
    except Exception as e:
        print(f"[SCOUT-A] BERTopic failed ({e}). Trying keyword fallback\u2026")
        try:
            topics = await loop.run_in_executor(
                _executor, lambda: engine.discover_topics(headlines, use_bertopic=False)
            )
        except Exception as e2:
            print(f"[SCOUT-A] Fallback failed ({e2}). Returning empty.")
            _prior_headlines = [h["title"] for h in headlines]
            return []

    if not topics:
        print("[SCOUT-A] No topics discovered.")
        _prior_headlines = [h["title"] for h in headlines]
        return []

    print(f"[SCOUT-A] {len(topics)} topics discovered. Mapping to tickers\u2026")

    try:
        topic_candidates = await loop.run_in_executor(
            _executor,
            lambda: engine.score_topic_candidates(
                topics, tickers,
                min_topic_momentum    = tcfg.get("min_topic_momentum", 0.30),
                min_topic_sentiment   = tcfg.get("min_topic_sentiment", 0.05),
                min_exposure_score    = tcfg.get("min_exposure_score", 0.15),
                max_tickers_per_topic = tcfg.get("max_tickers_per_topic", 3),
            ),
        )
    except Exception as e:
        print(f"[SCOUT-A] Ticker mapping failed: {e}")
        _prior_headlines = [h["title"] for h in headlines]
        return []

    _prior_headlines = [h["title"] for h in headlines]

    min_composite = tcfg.get("min_composite_score", 0.20)
    results = []
    for tc_obj in topic_candidates:
        if tc_obj.composite_score < min_composite:
            continue
        results.append({
            "ticker":          tc_obj.ticker,
            "topic_label":     tc_obj.topic_label,
            "topic_momentum":  tc_obj.topic_momentum,
            "topic_sentiment": tc_obj.topic_sentiment,
            "exposure_score":  tc_obj.exposure_score,
            "topic_composite": tc_obj.composite_score,
            "headline_count":  tc_obj.headline_count,
            "topic_headlines": tc_obj.headlines,
        })

    print(f"[SCOUT-A] {len(results)} topic candidates above composite threshold.")
    return results





async def scout_statarb_candidates(ctx: OrchestratorContext) -> List[dict]:
    """
    Engine B — refresh MarketGraph → build_graph → SignalEngine diffusion
               → filter abs(Residual) by ENTRY_Z threshold.
    """
    loop = asyncio.get_running_loop()

    if (not hasattr(ctx.market_graph, "returns_")
            or ctx.market_graph.returns_ is None
            or ctx.market_graph.returns_.empty):
        print("[SCOUT-B] MarketGraph has no data. Skipping stat-arb scout.")
        return []

    try:
        eng = cfg.ENGINES["equities"]

        # Refresh prices in thread pool
        await loop.run_in_executor(_executor, lambda: ctx.market_graph.fetch_data("1d", False))

        # Build correlation graph
        await loop.run_in_executor(
            _executor,
            lambda: ctx.market_graph.build_graph(
                lookback_window = cfg.LOOKBACK_DAYS,
                sector_only     = cfg.SECTOR_ONLY,
                threshold       = eng.get("graph_threshold", cfg.GRAPH_THRESHOLD),
                verbose         = False,
            ),
        )

        # Run Laplacian diffusion
        sig_engine = SignalEngine(ctx.market_graph)
        signals_df = await loop.run_in_executor(
            _executor, lambda: sig_engine.run_diffusion(alpha=0.5)
        )

        filtered = signals_df[signals_df["Signal_Strength"] >= _ENTRY_Z].copy()
        print(f"[SCOUT-B] Diffusion done. {len(filtered)}/{len(signals_df)} "
              f"candidates pass ENTRY_Z={_ENTRY_Z}.")
        return filtered.to_dict("records")

    except Exception as e:
        print(f"[SCOUT-B] Stat-arb scout failed: {e}. Skipping.")
        return []


async def run_dual_scouts(ctx: OrchestratorContext) -> Tuple[List[dict], List[dict]]:
    """asyncio.gather both scouts — they run fully concurrently."""
    topic_raws, statarb_raws = await asyncio.gather(
        scout_topic_candidates(ctx),
        scout_statarb_candidates(ctx),
    )
    return topic_raws, statarb_raws


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 3 — UnifiedCandidate Schema & Normalization
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class UnifiedCandidate:
    # Identity
    ticker:           str
    source:           str         # 'topic' | 'statarb'
    side:             int         # +1 BUY, -1 SELL
    price:            float = 0.0

    # Signal geometry
    residual:         float = 0.0
    signal_strength:  float = 0.0
    abs_residual:     float = 0.0
    residual_z:       float = 0.0

    # Technical / market features (BrainEngine input)
    rsi14:            float = 50.0
    spy_above_200dma: float = 1.0
    vix_close:        float = 20.0
    market_vol:       float = 0.01
    asset_vol:        float = 0.02
    health_score:     float = 0.5
    avg_corr:         float = 0.30
    degree:           float = 0.0
    wdegree:          float = 0.0
    sector:           str   = "Unknown"

    # ── Topical Trading fields (primary scout) ────────────────────────────────
    topic_label:      str   = ""     # e.g. "AI chip shortage"
    topic_momentum:   float = 0.0    # BERTopic headline velocity score [0,1]
    topic_sentiment:  float = 0.0    # FinBERT mean sentiment on topic corpus [-1,+1]
    exposure_score:   float = 0.0    # How strongly ticker is exposed to topic [0,1]
    topic_composite:  float = 0.0    # Combined signal score (momentum×sentiment×exposure)
    topic_headlines:  List[str] = field(default_factory=list)  # Raw topic headlines

    # Legacy / StatArb metadata (kept for backward compat with stat-arb normalize path)
    insider_score:    float = 0.0
    title:            str   = ""
    value_str:        str   = ""
    in_universe:      bool  = False  # True if ticker is in MarketGraph universe

    # Filled in Layer 4
    rag_report:           str   = ""
    ai_sentiment_score:   float = 0.0
    ai_news_score:        float = 0.0   # VADER pre-filter score from news_scanner
    ai_price_bullish:     bool  = False   # fail-closed default: treat as bearish until LSTM confirms
    ai_predicted_return:  float = 0.0

    # Filled in Layer 5
    p_win_raw:   float = 0.0
    p_win_final: float = 0.0

    # Flow control
    skip:        bool = False
    skip_reason: str  = ""


def _compute_rsi(prices: pd.Series, period: int = 14) -> float:
    """Wilder RSI for the last bar of a price series."""
    try:
        delta = prices.diff().dropna()
        gain  = delta.clip(lower=0)
        loss  = (-delta).clip(lower=0)
        ag = gain.ewm(com=period - 1, min_periods=period).mean().iloc[-1]
        al = loss.ewm(com=period - 1, min_periods=period).mean().iloc[-1]
        return 100.0 if al == 0 else float(100.0 - 100.0 / (1.0 + ag / al))
    except Exception:
        return 50.0


def _market_context(ctx: OrchestratorContext) -> dict:
    """Extract scalar market-wide features (same value for every candidate in a scan)."""
    result = {"spy_above_200dma": 1.0, "vix_close": 20.0, "market_vol": 0.01}
    try:
        if hasattr(ctx.market_graph, "prices_") and ctx.market_graph.prices_ is not None:
            px = ctx.market_graph.prices_
            if "SPY" in px.columns:
                spy = px["SPY"].dropna()
                sma = spy.rolling(200).mean().iloc[-1]
                result["spy_above_200dma"] = 1.0 if spy.iloc[-1] > sma else 0.0
        vix = yf.Ticker("^VIX").fast_info.get("lastPrice", 20.0)
        result["vix_close"] = float(vix)
        if hasattr(ctx.market_graph, "returns_") and ctx.market_graph.returns_ is not None:
            result["market_vol"] = float(ctx.market_graph.returns_.tail(22).std().mean())
    except Exception:
        pass
    return result


def _topology_from_graph(
    ticker: str,
    ctx: OrchestratorContext,
) -> Tuple[float, float, float]:
    """Return (avg_corr, degree, wdegree) from MarketGraph adjacency, or sentinels."""
    try:
        adj = ctx.market_graph.adjacency_
        if adj is not None and ticker in adj.columns:
            row = adj[ticker]
            pos = row[row > 0]
            return (
                float(pos.mean()) if not pos.empty else 0.30,
                float((row > 0).sum()),
                float(row.sum()),
            )
    except Exception:
        pass
    return 0.30, 0.0, 0.0


def normalize_topic_candidate(
    raw: dict,
    ctx: OrchestratorContext,
    mkt: dict,
) -> UnifiedCandidate:
    """Convert a raw topic dict (from Scout A) into a UnifiedCandidate."""
    ticker = raw["ticker"]

    # Price + per-ticker technicals
    price, rsi14, asset_vol = 0.0, 50.0, 0.02
    try:
        fast  = yf.Ticker(ticker).fast_info
        price = float(fast.get("lastPrice", 0.0))
        hist  = yf.Ticker(ticker).history(period="30d", interval="1d")
        if len(hist) >= 15:
            rets      = hist["Close"].pct_change().dropna()
            asset_vol = float(rets.std())
            rsi14     = _compute_rsi(hist["Close"])
    except Exception:
        pass

    # Topology — use real values if ticker is in the stat-arb universe
    in_universe = (
        hasattr(ctx.market_graph, "returns_")
        and ctx.market_graph.returns_ is not None
        and ticker in ctx.market_graph.returns_.columns
    )
    avg_corr, degree, wdegree = _topology_from_graph(ticker, ctx) if in_universe else (0.30, 0.0, 0.0)

    # Sector
    sector = "Unknown"
    try:
        sector = yf.Ticker(ticker).info.get("sector", "Unknown") or "Unknown"
    except Exception:
        pass

    # Topic composite drives signal_strength (0–1 normalised)
    topic_composite = float(raw.get("topic_composite", 0.0))
    signal_strength  = min(topic_composite, 1.0)

    return UnifiedCandidate(
        ticker           = ticker,
        source           = "topic",
        side             = 1,           # topical momentum is long-biased
        price            = price,
        residual         = 0.0,
        signal_strength  = signal_strength,
        abs_residual     = 0.0,
        residual_z       = 0.0,
        rsi14            = rsi14,
        spy_above_200dma = mkt["spy_above_200dma"],
        vix_close        = mkt["vix_close"],
        market_vol       = mkt["market_vol"],
        asset_vol        = asset_vol,
        health_score     = ctx.health_score if math.isfinite(ctx.health_score) else 0.5,
        avg_corr         = avg_corr,
        degree           = degree,
        wdegree          = wdegree,
        sector           = sector,
        in_universe      = in_universe,
        # Topical fields
        topic_label      = raw.get("topic_label", ""),
        topic_momentum   = float(raw.get("topic_momentum", 0.0)),
        topic_sentiment  = float(raw.get("topic_sentiment", 0.0)),
        exposure_score   = float(raw.get("exposure_score", 0.0)),
        topic_composite  = topic_composite,
        topic_headlines  = raw.get("topic_headlines", []),
    )


def normalize_statarb_candidate(
    row: dict,
    ctx: OrchestratorContext,
    mkt: dict,
) -> UnifiedCandidate:
    """Convert a SignalEngine output row (from Scout B) into a UnifiedCandidate."""
    ticker   = row["Ticker"]
    residual = float(row["Residual"])
    strength = float(row["Signal_Strength"])
    side     = 1 if residual < 0 else -1   # underperformed peers → BUY mean-reversion

    price = 0.0
    try:
        price = float(yf.Ticker(ticker).fast_info.get("lastPrice", 0.0))
    except Exception:
        pass

    avg_corr, degree, wdegree = _topology_from_graph(ticker, ctx)

    # Per-ticker technicals from MarketGraph data
    rsi14, asset_vol = 50.0, 0.02
    try:
        if hasattr(ctx.market_graph, "prices_") and ticker in ctx.market_graph.prices_.columns:
            rsi14 = _compute_rsi(ctx.market_graph.prices_[ticker].dropna())
        if hasattr(ctx.market_graph, "returns_") and ticker in ctx.market_graph.returns_.columns:
            asset_vol = float(ctx.market_graph.returns_[ticker].tail(22).std())
    except Exception:
        pass

    # Sector from graph node attributes
    sector = "Unknown"
    try:
        if ctx.market_graph.graph_ and ticker in ctx.market_graph.graph_.nodes:
            sector = ctx.market_graph.graph_.nodes[ticker].get("sector", "Unknown") or "Unknown"
    except Exception:
        pass

    return UnifiedCandidate(
        ticker           = ticker,
        source           = "statarb",
        side             = side,
        price            = price,
        residual         = residual,
        signal_strength  = strength,
        abs_residual     = abs(residual),
        residual_z       = math.copysign(strength, residual),
        rsi14            = rsi14,
        spy_above_200dma = mkt["spy_above_200dma"],
        vix_close        = mkt["vix_close"],
        market_vol       = mkt["market_vol"],
        asset_vol        = asset_vol,
        health_score     = ctx.health_score if math.isfinite(ctx.health_score) else 0.5,
        avg_corr         = avg_corr,
        degree           = degree,
        wdegree          = wdegree,
        sector           = sector,
        insider_score    = 0.0,
        in_universe      = True,    # statarb candidates are always in the universe
    )


async def build_unified_candidates(
    topic_raws:   List[dict],
    statarb_raws: List[dict],
    ctx: OrchestratorContext,
) -> List[UnifiedCandidate]:
    """
    Normalize both raw lists concurrently, merge, and deduplicate by ticker.
    Topic candidates take priority when the same ticker appears in both engines.
    Zero-price candidates are discarded immediately.
    Each per-ticker normalization runs in the thread-pool so yfinance calls
    are all in-flight at the same time instead of executing one-by-one.
    """
    loop = asyncio.get_running_loop()
    mkt  = await loop.run_in_executor(_executor, _market_context, ctx)

    topic_results = await asyncio.gather(
        *[loop.run_in_executor(_executor, normalize_topic_candidate, r, ctx, mkt)
          for r in topic_raws],
        return_exceptions=True,
    )
    statarb_results = await asyncio.gather(
        *[loop.run_in_executor(_executor, normalize_statarb_candidate, r, ctx, mkt)
          for r in statarb_raws],
        return_exceptions=True,
    )

    topic_cands  = [c for c in topic_results  if isinstance(c, UnifiedCandidate)]
    statarb_cands = [c for c in statarb_results if isinstance(c, UnifiedCandidate)]

    seen: Dict[str, UnifiedCandidate] = {}
    for c in topic_cands:
        seen[c.ticker] = c
    for c in statarb_cands:
        if c.ticker not in seen:
            seen[c.ticker] = c

    candidates = [c for c in seen.values() if c.price > 0]
    print(f"[NORMALIZE] {len(candidates)} unique candidates "
          f"({len(topic_cands)} topic, {len(statarb_cands)} statarb, "
          f"{len(topic_cands) + len(statarb_cands) - len(candidates)} duplicates/zero-price dropped).")
    return candidates


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 4 — Unified AI Audit
# ═══════════════════════════════════════════════════════════════════════════════

async def audit_candidate_ai(
    candidate:  UnifiedCandidate,
    market_ctx: dict | None = None,
) -> UnifiedCandidate:
    """
    Run all AI gates for a single candidate.
    Gate order (all concurrent where possible):
      [topic source]
        1. Topic Momentum gate — skip if topic momentum too weak.
        2. FinBERT re-score   — deep NLP on ticker-specific headlines.
        3. LSTM price forecast — regime-gated directional 5-day prediction.
        4. RAG deep research   — informational, no hard gate.
      [statarb source]
        1. NewsScanner (VADER) — fast pre-filter.
        2. FinBERT sentiment   — deep NLP.
        3. LSTM price forecast.
        4. RAG deep research.
    market_ctx keys: vix, spy_above_200dma, health_score  (all optional).
    Mutates and returns the candidate.
    """
    if candidate.skip:
        return candidate

    loop   = asyncio.get_running_loop()
    ticker = candidate.ticker
    is_topic   = candidate.source == "topic"
    is_statarb = candidate.source == "statarb"

    # ── Gate 1: Headline sentiment pre-filter ────────────────────────────────
    # Topic candidates: use BERTopic-computed FinBERT sentiment (already scored).
    # StatArb candidates: run fast VADER news scan as cheap pre-filter.
    if is_topic:
        # Gate 1A: topic sentiment gate (already computed upstream by TopicEngine)
        if candidate.topic_sentiment < 0.0:   # clearly bearish narrative
            candidate.skip        = True
            candidate.skip_reason = f"Topic sentiment negative ({candidate.topic_sentiment:.2f})"
            print(f"  [AI] {ticker} REJECTED — topic sentiment {candidate.topic_sentiment:.2f} "
                  f"(topic='{candidate.topic_label}')")
            return candidate
        candidate.ai_news_score = candidate.topic_sentiment
        print(f"  [AI] {ticker} topic gate OK (sentiment={candidate.topic_sentiment:.2f}, "
              f"momentum={candidate.topic_momentum:.2f}, topic='{candidate.topic_label}')")
    else:
        # Gate 1B: VADER news scan for statarb
        try:
            news_r = await loop.run_in_executor(_executor, scan_news, ticker)
        except Exception as e:
            news_r = {"is_passed": False, "score": 0.0,
                      "reason": f"News scan exception (fail-safe): {e}"}

        candidate.ai_news_score = float(news_r.get("score", 0.0))
        if not news_r.get("is_passed", False):
            candidate.skip        = True
            candidate.skip_reason = f"News pre-filter blocked ({candidate.ai_news_score:.4f})"
            log_evaluation(ticker, 0.0, "NO TRADE",
                           f"News Toxic (VADER={candidate.ai_news_score:.4f})")
            print(f"  [AI] {ticker} REJECTED — news VADER score {candidate.ai_news_score:.4f}: "
                  f"{news_r.get('reason', '')}")
            return candidate

        print(f"  [AI] {ticker} news OK (VADER={candidate.ai_news_score:.4f}) — "
              f"running deep AI audit…")

    # ── Gates 2-4: Deep AI audit (all I/O-bound, run concurrently) ───────────
    try:
        if is_topic:
            print(f"  [AI] {ticker} (topic): FinBERT re-score + LSTM…")
            sentiment_r, lstm_r = await asyncio.gather(
                loop.run_in_executor(_executor, analyze_sentiment,                   ticker),
                loop.run_in_executor(_executor, predict_price_movement, ticker, market_ctx),
            )
            sec_r      = {"is_passed": True, "score": 0.0, "full_text": "", "reason": "N/A"}
            earnings_r = {"is_passed": True, "score": 0.0, "reason": "N/A"}
        else:
            # statarb: same pipeline as before
            print(f"  [AI] {ticker} (statarb): sentiment + LSTM…")
            sentiment_r, lstm_r = await asyncio.gather(
                loop.run_in_executor(_executor, analyze_sentiment,                   ticker),
                loop.run_in_executor(_executor, predict_price_movement, ticker, market_ctx),
            )
            sec_r      = {"is_passed": True, "score": 0.0, "full_text": "", "reason": "N/A"}
            earnings_r = {"is_passed": True, "score": 0.0, "reason": "N/A"}

    except Exception as e:
        candidate.skip        = True
        candidate.skip_reason = f"AI gather exception: {e}"
        return candidate

    # ── Gate 2: FinBERT news sentiment ───────────────────────────────────────
    candidate.ai_sentiment_score = float(sentiment_r.get("score", 0.0))
    if not sentiment_r.get("is_passed", False):
        candidate.skip        = True
        candidate.skip_reason = f"Sentiment fail ({candidate.ai_sentiment_score:.2f})"
        log_evaluation(ticker, 0.0, "NO TRADE",
                       f"Sentiment Risk ({candidate.ai_sentiment_score:.2f})")
        print(f"  [AI] {ticker} REJECTED — sentiment {candidate.ai_sentiment_score:.2f}")
        return candidate

    # ── Gate 3: LSTM directional forecast ────────────────────────────────────
    candidate.ai_price_bullish    = bool(lstm_r.get("is_bullish", False))
    candidate.ai_predicted_return = float(lstm_r.get("predicted_return", 0.0))
    direction_ok = (
        (candidate.side ==  1 and     candidate.ai_price_bullish) or
        (candidate.side == -1 and not candidate.ai_price_bullish)
    )
    if not direction_ok:
        candidate.skip        = True
        candidate.skip_reason = (f"LSTM direction mismatch "
                                  f"(bullish={candidate.ai_price_bullish}, side={candidate.side})")
        log_evaluation(ticker, 0.0, "NO TRADE",
                       f"LSTM Mismatch (conf={lstm_r.get('confidence', 0):.2f})")
        print(f"  [AI] {ticker} REJECTED — LSTM direction mismatch")
        return candidate

    # ── Gate 4: RAG deep research — informational, no hard gate ──────────────
    try:
        candidate.rag_report = await loop.run_in_executor(
            _executor, generate_rag_risk_report, ticker, ""
        )
    except Exception:
        candidate.rag_report = "RAG unavailable."

    topic_info = (f" topic='{candidate.topic_label}' momentum={candidate.topic_momentum:.2f}"
                  if is_topic else "")
    print(f"  [AI] {ticker} PASSED all gates —{topic_info} "
          f"news={candidate.ai_news_score:.4f}, "
          f"sentiment={candidate.ai_sentiment_score:.2f}, "
          f"LSTM_bullish={candidate.ai_price_bullish}")
    return candidate



async def apply_ai_audit(
    candidates: List[UnifiedCandidate],
    ctx: OrchestratorContext,
) -> List[UnifiedCandidate]:
    """Audit all non-skipped candidates concurrently via asyncio.gather.

    Builds a shared market_ctx dict from the first candidate's macro fields
    (all candidates share the same macro environment in a single scan) and
    passes it to each audit so the LSTM MarketGate can modulate feature weights.
    """
    active  = [c for c in candidates if not c.skip]
    skipped = [c for c in candidates if     c.skip]

    # Build shared market context for LSTM gating
    # Prefer candidate-level macro data (already fetched); fallback to ctx
    _ref = active[0] if active else None
    market_ctx: dict = {
        "vix":              float(getattr(_ref, "vix_close",        20.0) if _ref else 20.0),
        "spy_above_200dma": float(getattr(_ref, "spy_above_200dma",  1.0) if _ref else 1.0),
        "health_score":     float(ctx.health_score if math.isfinite(ctx.health_score) else 0.5),
    }
    print(f"\n[AI-AUDIT] MarketGate context: VIX={market_ctx['vix']:.1f}, "
          f"SPY>200dma={int(market_ctx['spy_above_200dma'])}, "
          f"health={market_ctx['health_score']:.2f}")
    print(f"[AI-AUDIT] Running unified AI audit on {len(active)} candidate(s)…")
    audited = await asyncio.gather(
        *[audit_candidate_ai(c, market_ctx=market_ctx) for c in active],
        return_exceptions=True,
    )

    result = []
    for item in audited:
        if isinstance(item, UnifiedCandidate):
            result.append(item)
        elif isinstance(item, Exception):
            print(f"[AI-AUDIT] Unhandled task exception: {item}")

    return result + skipped


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 5 — P_win Derivation
# ═══════════════════════════════════════════════════════════════════════════════

def _build_features_row(candidate: UnifiedCandidate, ctx: OrchestratorContext) -> pd.DataFrame:
    """
    Build a 1-row feature DataFrame for BrainEngine.predict_proba().

    In-universe path  → delegates to build_trade_features() for real topology.
    Out-of-universe   → manual row with sentinel topology values.
    """
    if candidate.in_universe and hasattr(ctx.market_graph, "returns_"):
        try:
            actual_return = 0.0
            if candidate.ticker in ctx.market_graph.returns_.columns:
                actual_return = float(ctx.market_graph.returns_[candidate.ticker].iloc[-1])

            synthetic_signals = pd.DataFrame([{
                "Ticker":          candidate.ticker,
                "Actual_Return":   actual_return,
                "Expected_Return": actual_return - candidate.residual,
                "Residual":        candidate.residual,
                "Signal_Strength": candidate.signal_strength,
            }])
            features = build_trade_features(
                synthetic_signals,
                market_graph = ctx.market_graph,
                as_of        = pd.Timestamp.now(tz="UTC"),
                health_score = ctx.health_score if math.isfinite(ctx.health_score) else 0.5,
            )
            return features
        except Exception as e:
            print(f"  [PWIN] build_trade_features failed for {candidate.ticker} ({e}). "
                  "Using manual fallback.")

    # Manual feature vector (out-of-universe or fallback)
    return pd.DataFrame([{
        "residual":         candidate.residual,
        "abs_residual":     candidate.abs_residual,
        "signal_strength":  candidate.signal_strength,
        "residual_z":       candidate.residual_z,
        "rsi14":            candidate.rsi14,
        "spy_above_200dma": candidate.spy_above_200dma,
        "vix_close":        candidate.vix_close,
        "market_vol":       candidate.market_vol,
        "asset_vol":        candidate.asset_vol,
        "health_score":     candidate.health_score,
        "avg_corr":         candidate.avg_corr,
        "degree":           candidate.degree,
        "wdegree":          candidate.wdegree,
        "side":             float(candidate.side),
        "sector":           candidate.sector,
    }])


def derive_all_pwin(
    candidates: List[UnifiedCandidate],
    ctx: OrchestratorContext,
) -> List[UnifiedCandidate]:
    """
    Batch P_win derivation:
      1. Synthesize a feature row per candidate.
      2. Single batch call to BrainEngine.predict_proba (CPU-efficient).
      3. Apply TradeLearner confidence multiplier per candidate.
      4. Gate: p_win_final < 0.55 → skip.
    Falls back to bounded heuristics if BrainEngine model is not fitted.
    """
    active  = [c for c in candidates if not c.skip]
    if not active:
        return candidates

    print(f"\n[PWIN] Deriving P_win for {len(active)} candidate(s)…")

    feature_rows = [_build_features_row(c, ctx) for c in active]

    # ── BrainEngine batch predict ─────────────────────────────────────────────
    if ctx.brain_engine.fitted_:
        try:
            batch_df = pd.concat(feature_rows, ignore_index=True)
            probas   = ctx.brain_engine.predict_proba(batch_df)
            for c, p in zip(active, probas):
                c.p_win_raw = float(p)
        except Exception as e:
            print(f"  [PWIN] BrainEngine.predict_proba error ({e}). Switching to heuristics.")
            ctx.brain_engine.fitted_ = False

    # ── Heuristic fallback ────────────────────────────────────────────────────
    if not ctx.brain_engine.fitted_:
        for c in active:
            if c.source == "insider":
                # insider_score on 0–10 scale → maps to 0.40–0.70
                c.p_win_raw = min(0.70, 0.40 + 0.10 * c.insider_score)
            else:
                # signal_strength on 0–∞ → bounded via tanh → 0.50–0.65
                c.p_win_raw = 0.50 + 0.15 * math.tanh(c.signal_strength)

    # ── TradeLearner confidence multiplier + final gate ───────────────────────
    for c in active:
        try:
            mult = float(ctx.trade_learner.get_confidence_multiplier(
                c.ticker,
                {"insider_score": c.insider_score, "reasons": [c.source, c.sector]},
            ))
        except Exception:
            mult = 1.0

        c.p_win_final = min(0.95, c.p_win_raw * mult)

        if c.p_win_final < _MIN_PWIN:
            c.skip        = True
            c.skip_reason = f"P_win {c.p_win_final:.3f} < {_MIN_PWIN}"
            print(f"  [PWIN] {c.ticker} REJECTED — P_win={c.p_win_final:.3f}")
        else:
            print(f"  [PWIN] {c.ticker} OK — "
                  f"P_win={c.p_win_final:.3f} (raw={c.p_win_raw:.3f}, mult={mult:.2f})")

    return candidates


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 6 — Risk & Sizing
# ═══════════════════════════════════════════════════════════════════════════════

def build_orders_dataframe(candidates: List[UnifiedCandidate]) -> pd.DataFrame:
    """Assemble the DataFrame required by allocate_quantities."""
    rows = []
    for c in candidates:
        if c.skip or c.price <= 0:
            continue
        rows.append({
            "ticker":         c.ticker,
            "price":          c.price,
            "side":           c.side,
            "P_win":          c.p_win_final,
            "signal_strength": c.signal_strength,
            "asset_vol":      max(c.asset_vol, 1e-6),
            "avg_corr":       c.avg_corr,
            # preserve source for notification routing
            "source":         c.source,
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def run_risk_sizing(
    candidates: List[UnifiedCandidate],
    ctx: OrchestratorContext,
) -> pd.DataFrame:
    """Run mean-variance portfolio allocation for the $50 high-risk account."""
    orders_df = build_orders_dataframe(candidates)
    if orders_df.empty:
        print("[SIZING] No tradeable candidates.")
        return pd.DataFrame()

    # High-risk RiskConfig — fractional shares enabled via min_qty=0.001
    risk_cfg = RiskConfig(
        sizing_method        = "mean_variance",
        annual_risk_target   = 0.25,        # 25% annual vol target (high-risk)
        max_weight_per_asset = 0.25,        # $250 max notional at $1,000 equity
        max_gross_leverage   = 1.5,         # up to $1,500 gross exposure
        dollar_neutral       = False,        # long-biased account
        min_qty              = _MIN_QTY,    # 0.001 for fractional, 1 otherwise
        strength_cap         = 6.0,
        cov_regularization   = 1e-3,
    )

    free_equity = ctx.account_equity - ctx.allocated_this_scan
    if free_equity <= 0:
        print(f"[SIZING] No free capital (committed=${ctx.allocated_this_scan:.2f} / "
              f"equity=${ctx.account_equity:.2f}). Sizing skipped.")
        return pd.DataFrame()

    print(f"\n[SIZING] mean-variance allocation — "
          f"{len(orders_df)} candidates, free equity=${free_equity:.2f} "
          f"(${ctx.allocated_this_scan:.2f} already committed this scan)…")
    try:
        sized = allocate_quantities(
            candidates = orders_df,
            equity     = free_equity,
            config     = risk_cfg,
        )
        n_tradeable = int((~sized["Skip"]).sum()) if "Skip" in sized.columns else len(sized)
        print(f"[SIZING] {n_tradeable}/{len(sized)} positions tradeable.")
        return sized
    except Exception as e:
        print(f"[SIZING] allocate_quantities error: {e}")
        return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 7 — Execution, Notifications, Logging
# ═══════════════════════════════════════════════════════════════════════════════

def run_execution(
    sized_orders: pd.DataFrame,
    ctx: OrchestratorContext,
) -> Optional[Any]:
    """Submit non-skipped orders to Alpaca via ExecutionEngine."""
    if sized_orders.empty:
        return None

    skip_col = "Skip" if "Skip" in sized_orders.columns else None
    to_exec  = sized_orders[~sized_orders[skip_col]].copy() if skip_col else sized_orders.copy()

    if to_exec.empty:
        print("[EXEC] No orders after sizing filter.")
        return None

    # --- Buying-power guard: reject orders that exceed remaining cash ---
    buying_power = ctx.exec_engine.get_buying_power()
    if buying_power is None:
        # Could not fetch BP — fall back to estimated free equity from tracker
        buying_power = max(0.0, ctx.account_equity - ctx.allocated_this_scan)
        print(f"[EXEC] Could not fetch live buying power; using estimated free equity "
              f"${buying_power:.2f} as cap.")
    else:
        print(f"[EXEC] Live buying power: ${buying_power:.2f}")

    affordable: List[dict] = []
    remaining_bp = buying_power
    for _, row in to_exec.iterrows():
        needed = float(row.get("Qty", 0)) * float(row.get("price", 0))
        side   = row.get("side", 1)
        # SELL orders free up cash — always allow them
        if str(side).upper() in {"SELL", "-1"} or int(side) == -1 if str(side).lstrip('-').isdigit() else False:
            affordable.append(row.to_dict())
            continue
        if needed <= 0 or remaining_bp >= needed:
            affordable.append(row.to_dict())
            remaining_bp = max(0.0, remaining_bp - needed)
        else:
            print(f"[EXEC] Skipping {row.get('ticker','?')} — notional ${needed:.2f} "
                  f"exceeds remaining buying power ${remaining_bp:.2f}.")

    if not affordable:
        print("[EXEC] All orders rejected by buying-power check.")
        return None

    to_exec = pd.DataFrame(affordable)
    print(f"[EXEC] Submitting {len(to_exec)} order(s) (DRY_RUN={cfg.DRY_RUN})…")
    try:
        report = ctx.exec_engine.execute_trades(to_exec, dry_run=cfg.DRY_RUN)
        print(f"[EXEC] submitted={report.submitted} | skipped={report.skipped} | errors={report.errors}")
        return report
    except Exception as e:
        print(f"[EXEC] Execution error: {e}")
        return None


def dispatch_notifications(
    sized_orders: pd.DataFrame,
    candidates_map: Dict[str, UnifiedCandidate],
    take_profit_pct: float = 0.15,
    stop_loss_pct:   float = 0.07,
) -> None:
    """Route alerts: insider → send_trade_alert, statarb → send_statarb_alert."""
    if sized_orders.empty:
        return

    skip_col = "Skip" if "Skip" in sized_orders.columns else None
    executed = sized_orders[~sized_orders[skip_col]].copy() if skip_col else sized_orders.copy()
    if executed.empty:
        return

    statarb_rows = []
    for _, row in executed.iterrows():
        c = candidates_map.get(row.get("ticker", ""))
        if c and c.source == "insider":
            try:
                send_trade_alert(
                    ticker          = c.ticker,
                    title           = c.title,
                    value           = c.value_str,
                    current_price   = c.price,
                    shares          = row.get("Qty", 0),
                    allocated       = float(row.get("target_notional", 0.0)),
                    take_profit_pct = take_profit_pct,
                    stop_loss_pct   = stop_loss_pct,
                    rag_report      = c.rag_report,
                )
            except Exception as e:
                print(f"[NOTIFY] send_trade_alert failed for {c.ticker}: {e}")
        else:
            statarb_rows.append(row)

    if statarb_rows:
        try:
            df = pd.DataFrame(statarb_rows)
            if "side" in df.columns:
                df["side"] = df["side"].map(lambda x: "BUY" if x == 1 else "SELL")
            send_statarb_alert(df)
        except Exception as e:
            print(f"[NOTIFY] send_statarb_alert failed: {e}")


def dispatch_logging(
    sized_orders: pd.DataFrame,
    candidates_map: Dict[str, UnifiedCandidate],
) -> None:
    """Append all executed trades to trade_history.csv for TradeLearner feedback."""
    if sized_orders.empty:
        return

    skip_col = "Skip" if "Skip" in sized_orders.columns else None
    executed = sized_orders[~sized_orders[skip_col]] if skip_col else sized_orders

    for _, row in executed.iterrows():
        c = candidates_map.get(row.get("ticker", ""))
        if not c:
            continue
        signal  = "TRADE" if c.source == "insider" else "STATARB_TRADE"
        reasons = (f"source={c.source} | strategy={c.source} | "
                   f"p_win={c.p_win_final:.3f} | strength={c.signal_strength:.4f} | "
                   f"sector={c.sector}")
        log_evaluation(c.ticker, c.insider_score, signal, reasons)


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 8 — Orchestrator & Daemon
# ═══════════════════════════════════════════════════════════════════════════════

def run_statarb_exits(ctx: OrchestratorContext) -> None:
    """
    Signal-based stat-arb exit pass (runs every 30 min, independent of the main scan).

    For every held position that appears in the equities universe, checks whether
    the current diffusion residual has turned positive (mean-reversion complete).
    If so, submits a market SELL order directly — bypassing the AI audit gates
    since it is a closing, not an opening, trade.
    """
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"\n[STATARB-EXIT] Running exit pass ({ts})…")

    eng = cfg.ENGINES.get("equities", {})
    sell_thresh = float(eng.get("sell_thresh", getattr(cfg, "SELL_THRESH", 0.01)))

    # Current held positions from Alpaca
    try:
        positions = ctx.exec_engine.get_positions()   # {ticker: qty}
    except Exception as e:
        print(f"[STATARB-EXIT] Could not fetch positions: {e}. Skipping.")
        return

    if not positions:
        print("[STATARB-EXIT] No open positions — nothing to close.")
        return

    # Latest diffusion residuals from the shared MarketGraph
    try:
        sig_engine  = SignalEngine(ctx.market_graph)
        signals_df  = sig_engine.run_diffusion(alpha=0.5)
    except Exception as e:
        print(f"[STATARB-EXIT] SignalEngine failed: {e}. Skipping.")
        return

    residuals = dict(zip(signals_df["Ticker"], signals_df["Residual"]))

    sell_rows: List[dict] = []
    for ticker, qty in positions.items():
        residual = residuals.get(ticker)
        if residual is None:
            continue    # not in universe — leave to portfolio_monitor stop-loss
        if residual > sell_thresh:
            print(f"[STATARB-EXIT] {ticker}: residual={residual:.4f} > "
                  f"sell_thresh={sell_thresh:.4f} → queuing SELL {qty}")
            sell_rows.append({"ticker": ticker, "side": "SELL", "Qty": qty})

    if not sell_rows:
        print("[STATARB-EXIT] No exit signals this pass.")
        return

    orders_df = pd.DataFrame(sell_rows)
    print(f"[STATARB-EXIT] Submitting {len(orders_df)} SELL order(s)…")
    try:
        report = ctx.exec_engine.execute_trades(orders_df, dry_run=cfg.DRY_RUN)
        print(f"[STATARB-EXIT] submitted={report.submitted} | "
              f"skipped={report.skipped} | errors={report.errors}")
    except Exception as e:
        print(f"[STATARB-EXIT] Execution error: {e}")


async def _run_unified_scan_async(ctx: OrchestratorContext) -> None:
    """Full async pipeline for one scan cycle."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'='*60}\n=== UNIFIED SCAN | {ts} ===\n{'='*60}")

    # [1] Market Gate
    if not check_market_health(ctx):
        print("=== SCAN ABORTED (Market Gate) ===\n")
        return

    # Refresh account equity + reset per-scan capital tracker
    live_equity = ctx.exec_engine.get_account_equity()
    if live_equity:
        ctx.account_equity = live_equity
    ctx.allocated_this_scan = 0.0
    ctx.positions_this_scan = 0
    print(f"\n[INFO] Account equity: ${ctx.account_equity:.2f}")

    # [2] Dual Scouts
    insider_raws, statarb_raws = await run_dual_scouts(ctx)
    print(f"\n[SCOUTS] Insider OK: {len(insider_raws)} | StatArb OK: {len(statarb_raws)}")

    if not insider_raws and not statarb_raws:
        print("No candidates from either engine. Scan complete.\n")
        return

    # [3] Normalize → UnifiedCandidate list (concurrent per-ticker yfinance calls)
    candidates = await build_unified_candidates(insider_raws, statarb_raws, ctx)
    if not candidates:
        print("No valid candidates after normalization. Scan complete.\n")
        return

    candidates_map = {c.ticker: c for c in candidates}

    # [4] Unified AI Audit
    candidates = await apply_ai_audit(candidates, ctx)

    # [5] P_win derivation + gate
    candidates = derive_all_pwin(candidates, ctx)

    passed = [c for c in candidates if not c.skip]
    print(f"\n[PIPELINE] {len(passed)}/{len(candidates)} candidates survived all gates.")
    if not passed:
        print("No candidates survived. Scan complete.\n")
        return

    # Cap to remaining position slots for this scan
    remaining_slots = _MAX_POSITIONS_PER_SCAN - ctx.positions_this_scan
    if len(passed) > remaining_slots:
        print(f"[SIZING] Capping {len(passed)} candidates to {remaining_slots} slot(s) "
              f"(scan limit={_MAX_POSITIONS_PER_SCAN}).")
        passed = passed[:remaining_slots]

    # [6] Risk & Sizing — uses free equity (equity minus already-committed notional)
    sized_orders = run_risk_sizing(passed, ctx)
    if sized_orders.empty:
        print("Sizing returned no orders. Scan complete.\n")
        return

    # Update per-scan capital tracker from sizing output
    if "Skip" in sized_orders.columns and "target_notional" in sized_orders.columns:
        tradeable_mask = ~sized_orders["Skip"]
        ctx.allocated_this_scan += float(
            sized_orders.loc[tradeable_mask, "target_notional"].abs().sum()
        )
        ctx.positions_this_scan += int(tradeable_mask.sum())
        print(f"[SIZING] Capital committed this scan: "
              f"${ctx.allocated_this_scan:.2f} / ${ctx.account_equity:.2f} "
              f"({ctx.positions_this_scan} new position(s))")

    # [7] Execute → Notify → Log
    run_execution(sized_orders, ctx)
    dispatch_notifications(sized_orders, candidates_map)
    dispatch_logging(sized_orders, candidates_map)

    print(f"\n=== SCAN COMPLETE | {datetime.now().strftime('%H:%M:%S')} ===\n")


def run_unified_scan(ctx: OrchestratorContext) -> None:
    """Synchronous wrapper — each call gets its own event loop."""
    asyncio.run(_run_unified_scan_async(ctx))


def run_portfolio_monitor(ctx: OrchestratorContext) -> None:
    """Lightweight job — delegates to portfolio_monitor.evaluate_portfolio()."""
    print(f"\n[PORTFOLIO] Monitor running ({datetime.now().strftime('%H:%M:%S')})…")
    try:
        evaluate_portfolio(exec_engine=ctx.exec_engine, dry_run=cfg.DRY_RUN)
    except Exception as e:
        print(f"[PORTFOLIO] Error: {e}")


def _is_market_hours() -> bool:
    """True if current time is within NYSE trading hours (Mon–Fri 09:25–16:05 ET).
    Uses America/New_York so EST/EDT transitions are handled automatically.
    """
    try:
        from zoneinfo import ZoneInfo
    except ImportError:
        from backports.zoneinfo import ZoneInfo  # Python <3.9 fallback
    now_et = datetime.now(ZoneInfo("America/New_York"))
    if now_et.weekday() >= 5:
        return False
    open_  = now_et.replace(hour=9,  minute=25, second=0, microsecond=0)
    close_ = now_et.replace(hour=16, minute=5,  second=0, microsecond=0)
    return open_ <= now_et <= close_


def main() -> None:
    print("\n" + "=" * 62)
    print("  UNIFIED MULTI-STRATEGY QUANT BOT — STARTING")
    print("  Strategies : Insider Catalyst + Topological Stat-Arb")
    print("  AI Stack   : FinBERT | LSTM | RAG/Gemini | BrainEngine")
    print("  Risk Engine: mean-variance (statarb/risk_manager.py)")
    print("  Execution  : Alpaca (statarb/execution.py)")
    print("=" * 62 + "\n")

    ctx = initialize_components()

    # Train TradeLearner if enough history exists
    history_file = "trade_history.csv"
    if os.path.exists(history_file):
        try:
            n = sum(1 for _ in open(history_file)) - 1
            if n >= 20:
                print(f"\n[LEARNER] Training on {n} historical trades…")
                print(f"[LEARNER] {ctx.trade_learner.train()}")
            else:
                print(f"[LEARNER] {n} rows in history (need ≥20). Training deferred.")
        except Exception as e:
            print(f"[LEARNER] Training skipped: {e}")

    # Immediate startup scan
    print("\n--- STARTUP SCAN ---")
    run_unified_scan(ctx)
    run_portfolio_monitor(ctx)

    # ── APScheduler — each job runs in its own background thread ─────────────
    def _guarded_scan() -> None:
        if _is_market_hours():
            run_unified_scan(ctx)
        else:
            print("[SCHED] Outside market hours — scan skipped.")

    def _guarded_monitor() -> None:
        if _is_market_hours():
            run_portfolio_monitor(ctx)

    def _guarded_statarb_exits() -> None:
        if _is_market_hours():
            run_statarb_exits(ctx)

    scheduler = BackgroundScheduler(
        job_defaults={"coalesce": True, "max_instances": 1, "misfire_grace_time": 120}
    )
    scheduler.add_job(_guarded_scan,           CronTrigger(hour=15, minute=30, timezone="America/New_York"),  id="unified_scan")
    scheduler.add_job(_guarded_monitor,        IntervalTrigger(minutes=15),  id="portfolio_monitor")
    scheduler.add_job(_guarded_statarb_exits,  IntervalTrigger(minutes=30),  id="statarb_exits")
    scheduler.start()

    print("\n[DAEMON] Active. Unified scan at 3:30 PM ET | Portfolio monitor every 15 min "
          "| StatArb exits every 30 min.")
    print("[DAEMON] Press Ctrl+C to stop.\n")

    try:
        while True:
            time.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
        print("\n[DAEMON] Gracefully terminated.")


if __name__ == "__main__":
    main()
