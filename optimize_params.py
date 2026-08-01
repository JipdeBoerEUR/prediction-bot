"""optimize_params.py
===================
Bayesian hyperparameter search for the Topological Arbitrage strategy.

Uses Optuna (TPE sampler) to find the combination of dataset-generation and
signal parameters that maximises a composite trading objective.

Objective (maximised):
    score = Sharpe(net_pnl per event) * win_rate * log(1 + n_events)

  - Sharpe   : risk-adjusted signal quality
  - win_rate : raw accuracy check (prevents degenerate all-short / all-long)
  - log(n)   : penalises configs that produce almost no events (too selective)

Parameters searched:
    entry_z            [1.2, 3.0]   signal detection threshold
    hold_bars          [24, 168]    bars to hold  (24h – 1 week at 60m)
    graph_threshold    [0.15, 0.55] min correlation to form a graph edge
    lookback_bars      [252, 2000]  correlation window (bars)
    cost_bps           [0, 15]      round-trip transaction cost (bps)
    regime_min_health  [0.10, 0.50] market regime gate
    alpha              [0.5, 0.99]  diffusion coefficient in GSP solver
    exit_z             [0.10, 0.80] mean-reversion exit threshold

Usage:
    # Install Optuna once:
    pip install optuna optuna-dashboard

    # Run 100 trials (default), one process:
    python optimize_params.py --engine equities --n-trials 100

    # Resume a previous study:
    python optimize_params.py --engine equities --n-trials 50 --storage sqlite:///optuna.db

    # Visual dashboard (after running):
    optuna-dashboard sqlite:///optuna.db
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── make sure statarb/ is importable when called from repo root ──────────────
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "statarb"))

# ── Load .env before importing anything that reads credentials ───────────────
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass


# Broad exploration bounds (phase 1)
PARAM_BOUNDS_PHASE1 = {
    "entry_z": (1.2, 3.0),
    "hold_bars": (24, 168),
    "graph_threshold": (0.15, 0.55),
    "lookback_bars": (252, 2000),
    "cost_bps": (0.0, 15.0),
    "regime_min_health": (0.10, 0.50),
    "alpha": (0.50, 0.99),
    "exit_z": (0.10, 0.80),
}

# Refined bounds (phase 2) — tightened from actual top-10 phase-1 trial clusters.
# Data basis (top-10 completed trials):
#   entry_z          min=1.20  max=2.55  mean=1.80   top3≈[1.75, 1.75, 1.71]
#   hold_bars        min=25    max=167   mean=97.2   top3≈[104, 104, 98]
#   graph_threshold  min=0.31  max=0.45  mean=0.40   top3≈[0.45, 0.45, 0.45]
#   lookback_bars    min=750   max=1902  mean=1161   top3≈[800, 750, 800]
#   cost_bps         min=0.00  max=0.63  mean=0.38   top3≈[0.004, 0.003, 0.63]
#   regime_min_health min=0.24 max=0.50  mean=0.39   top3≈[0.35, 0.50, 0.38]
#   alpha            min=0.79  max=0.97  mean=0.83   top3≈[0.82, 0.81, 0.81]
#   exit_z           min=0.30  max=0.79  mean=0.41   top3≈[0.30, 0.35, 0.30]
PARAM_BOUNDS_PHASE2 = {
    "entry_z":           (1.2,  2.6),    # evidence: top trials cluster 1.2–2.6
    "hold_bars":         (24,   120),    # evidence: top-3 ≈ 98–104; keep some range
    "graph_threshold":   (0.30, 0.46),  # evidence: top-3 tight near 0.44–0.45
    "lookback_bars":     (700,  1950),   # evidence: top-3 ≈ 750–800; allow broader resilience
    "cost_bps":          (0.0,  1.0),    # evidence: virtually all top trials < 0.65; cut 15→1
    "regime_min_health": (0.22, 0.50),  # evidence: full range still active
    "alpha":             (0.78, 0.97),  # evidence: top-3 all ≈ 0.81; small upside margin
    "exit_z":            (0.28, 0.50),  # evidence: top-3 all ≈ 0.30–0.35
}

# Ultra-tight bounds (phase 3) — from top-10 phase-2 trial clusters.
# Phase-2 best: trial 81 score=13.54 (7.6 pts better than phase-1 best=5.93).
# Data basis (top-10 P2 completed trials):
#   entry_z          min=1.21  max=2.16  mean=1.86   top3≈[1.22, 2.16, 1.24]
#   hold_bars        min=98    max=115   mean=108.2  top3≈[108, 98, 109]
#   graph_threshold  min=0.44  max=0.46  mean=0.45   top3≈[0.46, 0.44, 0.46] ← very tight!
#   lookback_bars    min=950   max=1100  mean=1035   top3≈[950, 1050, 950]
#   cost_bps         min=0.08  max=0.44  mean=0.15   top3≈[0.44, 0.12, 0.13] ← tighter
#   regime_min_health min=0.31 max=0.49  mean=0.39   top3≈[0.35, 0.48, 0.32]
#   alpha            min=0.80  max=0.89  mean=0.85   top3≈[0.80, 0.85, 0.80] ← tighter
#   exit_z           min=0.29  max=0.32  mean=0.30   top3≈[0.29, 0.32, 0.29] ← very tight!
PARAM_BOUNDS_PHASE3 = {
    "entry_z":           (1.117, 2.254),
    # Integer parameters MUST have integer bounds — trial.suggest_int rejects
    # floats, which aborted every --phase3 run before the first trial.
    # lookback_bars bounds are also aligned to the sampler's step=50.
    "hold_bars":         (96, 117),
    "graph_threshold":   (0.438, 0.462),   # extremely tight (0.44–0.46)
    "lookback_bars":     (950, 1100),
    "cost_bps":          (0.042, 0.472),
    "regime_min_health": (0.297, 0.503),
    "alpha":             (0.789, 0.894),
    "exit_z":            (0.282, 0.321),   # extremely tight (0.28–0.32)
}


def _sample_params(trial, bounds: Dict[str, tuple]) -> Dict[str, float | int]:
    """Sample one trial from the provided bounds."""
    return {
        "entry_z": trial.suggest_float("entry_z", bounds["entry_z"][0], bounds["entry_z"][1]),
        "hold_bars": trial.suggest_int(
            "hold_bars", int(round(bounds["hold_bars"][0])), int(round(bounds["hold_bars"][1]))
        ),
        "graph_threshold": trial.suggest_float(
            "graph_threshold", bounds["graph_threshold"][0], bounds["graph_threshold"][1]
        ),
        "lookback_bars": trial.suggest_int(
            "lookback_bars",
            int(round(bounds["lookback_bars"][0])),
            int(round(bounds["lookback_bars"][1])),
            step=50,
        ),
        "cost_bps": trial.suggest_float("cost_bps", bounds["cost_bps"][0], bounds["cost_bps"][1]),
        "regime_min_health": trial.suggest_float(
            "regime_min_health", bounds["regime_min_health"][0], bounds["regime_min_health"][1]
        ),
        "alpha": trial.suggest_float("alpha", bounds["alpha"][0], bounds["alpha"][1]),
        "exit_z": trial.suggest_float("exit_z", bounds["exit_z"][0], bounds["exit_z"][1]),
    }


# ---------------------------------------------------------------------------
# Inline backtest: given a labelled events DataFrame, compute metrics
# ---------------------------------------------------------------------------

def _compute_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Return Sharpe, win_rate, n_events, profit_factor from an events frame."""
    if df.empty or "pnl_net" not in df.columns:
        return {"sharpe": -99.0, "win_rate": 0.0, "n_events": 0, "profit_factor": 0.0}

    pnl = df["pnl_net"].dropna().values.astype(float)
    n = len(pnl)
    if n < 10:
        return {"sharpe": -99.0, "win_rate": 0.0, "n_events": n, "profit_factor": 0.0}

    mean_pnl = float(np.mean(pnl))
    std_pnl  = float(np.std(pnl))
    sharpe   = mean_pnl / (std_pnl + 1e-12) * np.sqrt(n)   # information ratio

    wins     = pnl[pnl > 0]
    losses   = pnl[pnl < 0]
    win_rate = float(np.mean(pnl > 0))
    profit_factor = float(wins.sum() / (-losses.sum() + 1e-12)) if len(losses) > 0 else float(wins.sum())

    return {
        "sharpe":        sharpe,
        "win_rate":      win_rate,
        "n_events":      n,
        "profit_factor": profit_factor,
        "mean_pnl":      mean_pnl,
    }


def _objective_score(metrics: Dict[str, float]) -> float:
    """Composite score Optuna maximises."""
    s    = metrics["sharpe"]
    wr   = metrics["win_rate"]
    n    = metrics["n_events"]
    pf   = metrics["profit_factor"]

    if n < 20:          # too few events → unreliable
        return -99.0
    if wr < 0.30:       # below 30% win-rate → skip
        return -99.0

    # Reward high Sharpe, reasonable win-rate, and sufficient event count.
    # log(n) prevents the optimiser from simply maximising selectivity at the
    # cost of never trading.
    score = s * wr * np.log1p(n) * min(pf, 5.0)
    return float(score)


# ---------------------------------------------------------------------------
# In-memory dataset generation (no CSV write)
# ---------------------------------------------------------------------------

def _generate_events_in_memory(
    trial_params: Dict,
    engine: str,
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    sectors: Optional[pd.Series],
    macro: pd.DataFrame,
    compute_n_jobs: int = 1,
) -> pd.DataFrame:
    """Re-run the event labelling loop with trial_params, return raw events DataFrame."""
    import sim_engine as sim
    from dataset_builder_v2 import (
        BuildConfig,
        _avg_corr,
        _market_proxy_return,
        _realized_vol,
        _asset_vol,
        _compute_regime_health,
        _degrees_from_W,
        _compute_rsi,
        _apply_cost,
    )

    bcfg = BuildConfig(
        entry_z           = trial_params["entry_z"],
        hold_bars         = trial_params["hold_bars"],
        exec_lag_bars     = 1,
        lookback_bars     = trial_params["lookback_bars"],
        cost_bps          = trial_params["cost_bps"],
        sector_only       = False,
        graph_threshold   = trial_params["graph_threshold"],
        alpha             = trial_params["alpha"],
        regularization    = 1e-6,
        use_regime_filter = True,
        regime_min_health = trial_params["regime_min_health"],
        vol_lookback      = 20,
        max_events_per_bar= 3,
    )

    exit_z         = trial_params["exit_z"]
    trail_stop_pct = 0.012        # fixed — not being optimised here
    rsi_period     = 14

    rsi_df  = _compute_rsi(prices, period=rsi_period).reindex(returns.index)
    macro_r = macro.reindex(returns.index).ffill()

    returns_clean = returns.dropna(how="all")
    prices_clean  = prices.reindex(returns_clean.index)

    # Residual panel
    trade_dates = returns_clean.index[bcfg.lookback_bars - 1:]
    try:
        resid_panel = sim.compute_residual_panel(
            returns_clean,
            lookback_days   = bcfg.lookback_bars,
            sectors         = sectors,
            sector_only     = bcfg.sector_only,
            graph_threshold = bcfg.graph_threshold,
            alpha           = bcfg.alpha,
            regularization  = bcfg.regularization,
            trade_dates     = trade_dates,
            n_jobs          = compute_n_jobs,
            backend         = "threading",
        )
    except Exception:
        return pd.DataFrame()

    resid_panel = resid_panel.reindex(returns_clean.index)
    resid_std   = resid_panel.std(axis=1)
    resid_z_panel = resid_panel.div(resid_std, axis=0)

    mkt_proxy = _market_proxy_return(returns_clean)

    idx      = returns_clean.index
    abs_entry_z = abs(bcfg.entry_z)
    last_i   = len(idx) - 1 - (bcfg.exec_lag_bars + bcfg.hold_bars)

    rows = []
    for i in range(bcfg.lookback_bars - 1, max(bcfg.lookback_bars, last_i + 1)):
        d        = idx[i]
        resid_z  = resid_z_panel.iloc[i]
        if resid_z.isna().all():
            continue

        long_cand  = resid_z[resid_z <= -abs_entry_z].sort_values(ascending=True)
        short_cand = resid_z[resid_z >= abs_entry_z].sort_values(ascending=False)
        if long_cand.empty and short_cand.empty:
            continue

        # Regime gate
        window = returns_clean.iloc[i - bcfg.lookback_bars + 1: i + 1]
        health = _compute_regime_health(window)
        if not np.isfinite(health) or health < bcfg.regime_min_health:
            continue

        market_vol = _realized_vol(mkt_proxy, i, bcfg.vol_lookback)
        av         = _asset_vol(returns_clean, i, bcfg.vol_lookback)
        avg_corr   = _avg_corr(window)

        W = sim.build_adjacency_from_returns(
            window,
            sectors     = sectors,
            sector_only = bcfg.sector_only,
            threshold   = bcfg.graph_threshold,
        )
        deg, wdeg = _degrees_from_W(W)

        longs  = list(long_cand.index[: bcfg.max_events_per_bar])
        shorts = list(short_cand.index[: bcfg.max_events_per_bar])

        entry_i   = i + bcfg.exec_lag_bars
        max_exit_i = entry_i + bcfg.hold_bars
        entry_dt  = idx[entry_i]
        resid_vec = resid_panel.iloc[i]

        for ticker, side in [(t, 1) for t in longs] + [(t, -1) for t in shorts]:
            if ticker not in prices_clean.columns:
                continue
            ep = prices_clean.iloc[entry_i][ticker]
            if pd.isna(ep) or ep <= 0:
                continue

            exit_i_final = max_exit_i
            best_price   = float(ep)
            for j in range(entry_i + 1, max_exit_i + 1):
                pj = prices_clean.iloc[j][ticker] if j < len(prices_clean) else np.nan
                if pd.isna(pj) or pj <= 0:
                    continue
                if side == 1:
                    if pj > best_price:
                        best_price = float(pj)
                    if trail_stop_pct > 0 and pj <= best_price * (1 - trail_stop_pct):
                        exit_i_final = j; break
                else:
                    if pj < best_price:
                        best_price = float(pj)
                    if trail_stop_pct > 0 and pj >= best_price * (1 + trail_stop_pct):
                        exit_i_final = j; break
                rz_j = resid_z_panel.iloc[j].get(ticker, np.nan) if j < len(resid_z_panel) else np.nan
                if np.isfinite(rz_j) and abs(float(rz_j)) <= exit_z:
                    exit_i_final = j; break

            if exit_i_final >= len(prices_clean):
                continue
            xp = prices_clean.iloc[exit_i_final][ticker]
            if pd.isna(xp) or xp <= 0:
                continue

            gross = (float(xp) - float(ep)) * side
            entry_notional = float(ep)
            exit_notional  = float(xp)
            net = gross - _apply_cost(entry_notional, bcfg.cost_bps) - _apply_cost(exit_notional, bcfg.cost_bps)

            rsi_val   = float(rsi_df.loc[d, ticker])   if (ticker in rsi_df.columns and d in rsi_df.index)   else float("nan")
            spy_above = float(macro_r.loc[d, "spy_above_200dma"]) if ("spy_above_200dma" in macro_r.columns and d in macro_r.index) else float("nan")
            vix_val   = float(macro_r.loc[d, "vix_close"])        if ("vix_close" in macro_r.columns and d in macro_r.index)        else float("nan")

            rows.append({
                "engine":          engine,
                "signal_dt":       d,
                "entry_dt":        entry_dt,
                "ticker":          ticker,
                "side":            int(side),
                "label_win":       int(net > 0),
                "pnl_net":         float(net),
                "entry_price":     float(ep),
                "exit_price":      float(xp),
                "residual_z":      float(resid_z.loc[ticker]),
                "signal_strength": abs(float(resid_z.loc[ticker])),
                "market_vol":      float(market_vol),
                "asset_vol":       float(av.get(ticker, np.nan)),
                "health_score":    float(health),
                "avg_corr":        float(avg_corr),
                "rsi14":           rsi_val,
                "spy_above_200dma": spy_above,
                "vix_close":       vix_val,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Data loading (done once, shared across all trials)
# ---------------------------------------------------------------------------

def _load_shared_data(engine: str, refresh_cache: bool = False):
    """Load prices/returns/sectors/macro from disk cache if present, else fetch and save."""
    import config as cfg
    from graph_engine import MarketGraph
    from dataset_builder_v2 import _fetch_macro_series

    engines = getattr(cfg, "ENGINES", {})
    e       = dict(engines.get(engine, {}))
    if not e:
        # Convenience: allow passing model stem (e.g. "brain_model_equities") as engine alias.
        alias = None
        for k, v in engines.items():
            model_path = str(v.get("model_path", ""))
            if model_path and Path(model_path).stem == engine:
                alias = k
                break

        if alias is not None:
            print(f"[optimize] Engine alias '{engine}' mapped to cfg.ENGINES['{alias}']")
            engine = alias
            e = dict(engines.get(engine, {}))
        else:
            available = ", ".join(sorted(engines.keys())) or "<none>"
            raise ValueError(
                f"Engine '{engine}' not found in cfg.ENGINES. Available engines: {available}"
            )

    tickers     = list(e.get("tickers", []))
    start       = e.get("start", "2020-01-01")
    end         = e.get("end", None)
    interval    = e.get("interval", "60m")
    provider    = e.get("provider", "alpaca")
    alpaca_feed = e.get("alpaca_feed", "iex")

    cache_dir    = Path("data") / "optimize_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    prices_path  = cache_dir / f"{engine}_prices.parquet"
    returns_path = cache_dir / f"{engine}_returns.parquet"
    sectors_path = cache_dir / f"{engine}_sectors.csv"
    macro_path   = cache_dir / f"{engine}_macro.parquet"

    cache_exists = all(p.exists() for p in [prices_path, returns_path, macro_path])

    if cache_exists and not refresh_cache:
        print(f"[optimize] Loading cached data from {cache_dir}  (use --refresh-cache to re-download)")
        prices  = pd.read_parquet(prices_path)
        returns = pd.read_parquet(returns_path)
        macro   = pd.read_parquet(macro_path)
        sectors = None
        if sectors_path.exists():
            sectors = pd.read_csv(sectors_path, index_col=0).squeeze("columns")
        print(f"[optimize] Prices shape: {prices.shape}  Returns shape: {returns.shape}")
        return prices, returns, sectors, macro

    print(f"[optimize] Fetching shared price data …  engine={engine} tickers={len(tickers)}")
    mg = MarketGraph(tickers=tickers, start=start, end=end, provider=provider, alpaca_feed=alpaca_feed)
    mg.fetch_data(interval=interval, verbose=True)

    sectors = None
    try:
        mg.fetch_sectors(exclude_unknown=False, verbose=False)
        sectors = mg.sectors_
        if isinstance(sectors, dict):
            sectors = pd.Series(sectors)
    except Exception:
        pass

    macro = _fetch_macro_series(start=start, end=end)
    if isinstance(macro.index, pd.DatetimeIndex) and macro.index.tz is not None:
        macro.index = macro.index.tz_convert("UTC").tz_localize(None)

    # Save to cache
    mg.prices_.to_parquet(prices_path)
    mg.returns_.to_parquet(returns_path)
    macro.to_parquet(macro_path)
    if sectors is not None:
        sectors.to_csv(sectors_path)
    print(f"[optimize] Data cached to {cache_dir}")

    print(f"[optimize] Prices shape: {mg.prices_.shape}  Returns shape: {mg.returns_.shape}")
    return mg.prices_.copy(), mg.returns_.copy(), sectors, macro


# ---------------------------------------------------------------------------
# Optuna study
# ---------------------------------------------------------------------------

def _make_objective(engine, prices, returns, sectors, macro, compute_n_jobs: int, bounds: Dict[str, tuple]):
    """Return a closure that Optuna calls for each trial."""
    trial_num = [0]

    def objective(trial):
        trial_num[0] += 1
        params = _sample_params(trial, bounds)
        try:
            df      = _generate_events_in_memory(
                params,
                engine,
                prices,
                returns,
                sectors,
                macro,
                compute_n_jobs=compute_n_jobs,
            )
            metrics = _compute_metrics(df)
            score   = _objective_score(metrics)
        except Exception as exc:
            print(f"  [trial {trial_num[0]}] ERROR: {exc}")
            return -99.0

        print(
            f"  [trial {trial_num[0]:3d}] "
            f"score={score:7.3f}  sharpe={metrics['sharpe']:6.2f}  "
            f"wr={metrics['win_rate']:.2%}  n={metrics['n_events']:5d}  "
            f"entry_z={params['entry_z']:.2f}  hold={params['hold_bars']}  "
            f"thr={params['graph_threshold']:.2f}  exit_z={params['exit_z']:.2f}"
        )
        return score

    return objective


def run_optimization(
    engine: str,
    n_trials: int,
    storage: Optional[str] = None,
    compute_n_jobs: int = 1,
    parallel_trials: int = 1,
    refresh_cache: bool = False,
    phase2: bool = False,
    phase3: bool = False,
    study_name: Optional[str] = None,
) -> None:
    try:
        import optuna
    except ImportError:
        print("[optimize] Optuna not installed.  Run:  pip install optuna optuna-dashboard")
        sys.exit(1)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Load data once (from disk cache if available)
    prices, returns, sectors, macro = _load_shared_data(engine, refresh_cache=refresh_cache)

    if phase3:
        bounds = PARAM_BOUNDS_PHASE3
        phase_label = "3"
    elif phase2:
        bounds = PARAM_BOUNDS_PHASE2
        phase_label = "2"
    else:
        bounds = PARAM_BOUNDS_PHASE1
        phase_label = "1"
    
    study_name = study_name or f"statarb_{engine}_phase{phase_label}"
    sampler    = optuna.samplers.TPESampler(seed=42)

    if storage:
        study = optuna.create_study(
            study_name   = study_name,
            storage      = storage,
            sampler      = sampler,
            direction    = "maximize",
            load_if_exists = True,
        )
    else:
        study = optuna.create_study(sampler=sampler, direction="maximize")

    objective = _make_objective(
        engine,
        prices,
        returns,
        sectors,
        macro,
        compute_n_jobs=compute_n_jobs,
        bounds=bounds,
    )

    # compute_n_jobs=-1 means "use every core" INSIDE each trial's own
    # residual computation. Combined with parallel_trials > 1 (multiple
    # trials running at once), that multiplies out to far more threads than
    # the machine has — e.g. parallel_trials=4 with compute_n_jobs=-1 on a
    # 16-thread CPU launches up to 64 competing threads, which causes
    # oversubscription/thrashing (slower, not faster) rather than the extra
    # throughput you'd expect from more parallelism.
    cpu_count = os.cpu_count() or 1
    effective_compute_jobs = cpu_count if compute_n_jobs in (-1, 0) else max(1, compute_n_jobs)
    total_threads_requested = parallel_trials * effective_compute_jobs
    if parallel_trials > 1 and total_threads_requested > cpu_count * 1.5:
        print(
            f"[optimize] WARNING: parallel_trials={parallel_trials} x "
            f"compute_n_jobs={compute_n_jobs} (effectively {effective_compute_jobs}/trial) "
            f"requests ~{total_threads_requested} threads, but this machine has "
            f"{cpu_count} logical CPUs. This will likely oversubscribe and run SLOWER, "
            f"not faster. Suggested: keep parallel_trials * compute_n_jobs ≈ {cpu_count} "
            f"(e.g. --parallel-trials {max(1, cpu_count // 4)} --compute-n-jobs 4)."
        )

    print(
        f"\n[optimize] Starting {n_trials} Optuna trials  (engine={engine}, "
        f"phase={phase_label}, compute_n_jobs={compute_n_jobs}, "
        f"parallel_trials={parallel_trials}, study={study_name})\n"
    )
    print("[optimize] Active bounds:")
    for k, (lo, hi) in bounds.items():
        print(f"  - {k}: [{lo}, {hi}]")

    try:
        study.optimize(
            objective,
            n_trials=n_trials,
            show_progress_bar=False,
            n_jobs=parallel_trials,
        )
    except KeyboardInterrupt:
        # Without this, Ctrl+C skipped straight past the results/reporting
        # code below — every completed trial was safely in `storage` (Optuna
        # persists each one as it finishes), but the convenient best-trial
        # summary and best_params_<engine>.txt file were never written,
        # forcing a full extra run just to see results from what had already
        # completed. Fall through to the same reporting code instead.
        n_done = len(study.trials)
        print(f"\n[optimize] Interrupted — {n_done} trial(s) completed and saved. "
              f"Reporting best result so far…")

    # ── Results ──────────────────────────────────────────────────────────────
    completed = [t for t in study.trials if t.value is not None]
    if not completed:
        print("[optimize] No trials completed yet — nothing to report. "
              "Re-run with --storage sqlite:///optuna.db to resume.")
        return

    best = study.best_trial
    print("\n" + "=" * 65)
    print("BEST TRIAL")
    print("=" * 65)
    print(f"  Score : {best.value:.4f}")
    for k, v in best.params.items():
        print(f"  {k:<22s}: {v}")

    # Top-10 trials
    top10 = sorted(study.trials, key=lambda t: t.value if t.value is not None else -999, reverse=True)[:10]
    print("\nTop-10 trials:")
    header = f"{'#':>4}  {'score':>8}  {'entry_z':>7}  {'hold_bars':>9}  {'graph_thr':>9}  {'exit_z':>6}  {'cost_bps':>8}  {'alpha':>6}"
    print(header)
    for t in top10:
        p = t.params
        print(
            f"  {t.number:>4}  {t.value:>8.3f}  "
            f"{p.get('entry_z',0):>7.2f}  "
            f"{p.get('hold_bars',0):>9d}  "
            f"{p.get('graph_threshold',0):>9.2f}  "
            f"{p.get('exit_z',0):>6.2f}  "
            f"{p.get('cost_bps',0):>8.1f}  "
            f"{p.get('alpha',0):>6.2f}"
        )

    # Importance
    try:
        imp = optuna.importance.get_param_importances(study)
        print("\nParameter importance:")
        for k, v in sorted(imp.items(), key=lambda x: -x[1]):
            bar = "█" * int(v * 30)
            print(f"  {k:<22s} {v:.3f}  {bar}")
    except Exception:
        pass

    # Save best params as a config snippet
    out_path = Path("data") / f"best_params_{engine}.txt"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        f.write(f"# Best params for engine={engine}  score={best.value:.4f}\n")
        for k, v in best.params.items():
            f.write(f"{k.upper()} = {v}\n")
    print(f"\n[optimize] Best params saved to {out_path}")

    if storage:
        print(f"[optimize] View dashboard:  optuna-dashboard {storage}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Bayesian param optimisation for Topological Arbitrage")
    p.add_argument("--engine",   default="equities", help="Engine name (cfg.ENGINES key)")
    p.add_argument("--n-trials", default=80, type=int, help="Number of Optuna trials (default 80)")
    p.add_argument(
        "--compute-n-jobs",
        default=1,
        type=int,
        help="Threads for per-trial residual computation (default 1, use -1 for all cores)",
    )
    p.add_argument(
        "--parallel-trials",
        default=1,
        type=int,
        help="Number of Optuna trials to run concurrently (default 1)",
    )
    p.add_argument("--storage",  default=None,
                   help="Optuna storage URL e.g. sqlite:///optuna.db  (enables resume + dashboard)")
    p.add_argument("--refresh-cache", action="store_true",
                   help="Force re-download of price data even if a local cache exists")
    p.add_argument("--phase2", action="store_true",
                   help="Use narrowed parameter bounds from phase-1 results")
    p.add_argument("--phase3", action="store_true",
                   help="Use ultra-tight bounds from phase-2 results (highest precision)")
    p.add_argument("--study-name", default=None,
                   help="Optional Optuna study name override")
    args = p.parse_args()
    run_optimization(
        engine=args.engine,
        n_trials=args.n_trials,
        storage=args.storage,
        compute_n_jobs=args.compute_n_jobs,
        parallel_trials=args.parallel_trials,
        refresh_cache=args.refresh_cache,
        phase2=args.phase2,
        phase3=args.phase3,
        study_name=args.study_name,
    )
