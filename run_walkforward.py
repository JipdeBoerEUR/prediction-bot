# run_walkforward.py
"""
Walk-forward optimisation + out-of-sample validation of the graph stat-arb
strategy. Produces the CSV/JSON files that make_report.py turns into
results/REPORT.md.
================================================================================

THE ONE RULE: parameters for a test year are chosen using ONLY the three years
BEFORE it. The headline result is the out-of-sample (OOS) test years stitched
together.

    train 2016-2018 → pick params → trade 2019 (unseen)
    train 2017-2019 → pick params → trade 2020 (unseen)
    ...
    train 2023-2025 → pick params → trade 2026-01 … 2026-09 (unseen)

How the parameters are picked in each training window
    Optuna's TPE sampler. The first 12 trials are random ("coarse" — spread
    over the whole search space), the next 20 concentrate where the good
    trials were ("fine"). Score = Sharpe ratio of daily returns over the
    3 training years, net of BASE costs; configurations with < 30 trades
    score -1 (too few trades to judge). A MedianPruner stops a trial early
    when its Sharpe after training year 1 or 2 is below the median of
    earlier trials at the same point.

Parallel + checkpointed
    8 windows run in 4 processes (one per core). Every finished trial is
    written immediately to results/optuna/window_<k>.log (Optuna
    JournalStorage). Re-running this script resumes where it stopped.
    A hard deadline (--budget-min) stops optimisation; whatever trials
    finished are used and the count is reported.

Usage
    python run_walkforward.py --time-single        # time ONE backtest, then exit
    python run_walkforward.py --budget-min 150     # full run
    python run_walkforward.py --eval-only          # re-evaluate saved studies
"""

from __future__ import annotations

import os

# One BLAS thread per process: we parallelise across processes instead,
# and 4 processes × 4 BLAS threads on 4 cores would just thrash.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import multiprocessing as mp  # noqa: E402
import platform  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402
from dataclasses import replace  # noqa: E402

import numpy as np  # noqa: E402
import optuna  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from wfbt import metrics as M  # noqa: E402
from wfbt.data import MarketData, load_market_data  # noqa: E402
from wfbt.signals import zscore_panel  # noqa: E402
from wfbt.sim import BASE_COSTS, STRESS_COSTS, Params, Segment, simulate  # noqa: E402

SEED = 42
optuna.logging.set_verbosity(optuna.logging.WARNING)
OUT = "results"
OPTUNA_DIR = os.path.join(OUT, "optuna")

# ── Walk-forward windows ─────────────────────────────────────────────────────
TRAIN_YEARS = 3
TEST_PERIODS = [(f"{y}-01-01", f"{y}-12-31") for y in range(2019, 2026)] + [("2026-01-01", "2026-09-30")]

# ── Search space (every value is on a grid so "neighbours" are well defined) ─
# Finance logic for each range is written up in REPORT.md "Search space".
SPACE = {
    "lookback":  [126, 252, 504],                       # 6m / 1y / 2y correlation window
    "threshold": [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60],
    "alpha":     [0.25, 0.5, 1.0, 2.0],                  # how hard peers pull
    "z_window":  [40, 60, 120],                          # ~2 / 3 / 6 months of residual history
    "entry_z":   [1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0],
    "exit_z":    [0.0, 0.25, 0.5, 0.75, 1.0],
    "max_hold":  [5, 10, 15, 20],                        # 1-4 weeks: short-term reversal horizon
    "stop_k":    [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],    # stop distance in daily σ
}
N_STARTUP = 12          # random "coarse" trials before TPE takes over
MIN_TRADES = 30         # fewer trades in 3 years → can't judge → score −1

_MD: MarketData | None = None     # loaded once in the parent, shared with forked workers


def suggest(trial: optuna.Trial) -> Params:
    """Optuna picks an INDEX into each grid (0 … len-1). Integers are treated as
    ordered, so TPE knows entry_z=2.0 lies between 1.75 and 2.25 — which it
    would not if the values were unordered categories."""
    return decode({k: trial.suggest_int(k, 0, len(v) - 1) for k, v in SPACE.items()})


def decode(idx: dict) -> Params:
    return Params(**{k: SPACE[k][int(i)] for k, i in idx.items()})


def window_bounds(md: MarketData, k: int) -> dict:
    ts, te = (pd.Timestamp(x) for x in TEST_PERIODS[k])
    tr_s = ts - pd.DateOffset(years=TRAIN_YEARS)
    return {
        "train_start": md.pos(tr_s), "train_end": md.pos(ts) - 1,
        "test_start": md.pos(ts), "test_end": int(md.dates.searchsorted(te, side="right")) - 1,
    }


def run_period(md: MarketData, p: Params, start: int, end: int, costs=BASE_COSTS, checkpoint=None):
    z = zscore_panel(md, p.lookback, p.threshold, p.alpha, p.z_window, start, end)
    return simulate(md, [Segment(start, end, p, z)], costs, checkpoint=checkpoint)


# ── Optimisation (one process per window) ────────────────────────────────────

def _storage(k: int):
    os.makedirs(OPTUNA_DIR, exist_ok=True)
    backend = optuna.storages.journal.JournalFileBackend(os.path.join(OPTUNA_DIR, f"window_{k}.log"))
    return optuna.storages.JournalStorage(backend)


def _study(k: int) -> optuna.Study:
    return optuna.create_study(
        study_name=f"window_{k}", storage=_storage(k), direction="maximize", load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=SEED + k, n_startup_trials=N_STARTUP, multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=N_STARTUP, n_warmup_steps=0),
    )


def optimise_window(args: tuple) -> dict:
    k, n_trials, deadline = args
    md = _MD
    random.seed(SEED + k)
    np.random.seed(SEED + k)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    b = window_bounds(md, k)
    study = _study(k)
    done = [t for t in study.trials if t.state in (optuna.trial.TrialState.COMPLETE,
                                                  optuna.trial.TrialState.PRUNED)]
    remaining = n_trials - len(done)

    def objective(trial: optuna.Trial) -> float:
        p = suggest(trial)

        def ckpt(step: int, eq: pd.Series) -> None:
            s = M.sharpe(M.daily_returns(eq))
            trial.report(s if math.isfinite(s) else -1.0, step)
            if trial.should_prune():
                raise optuna.TrialPruned()

        res = run_period(md, p, b["train_start"], b["train_end"], checkpoint=ckpt)
        n = len(res.trades)
        m = M.perf(res.equity)
        trial.set_user_attr("n_trades", n)
        trial.set_user_attr("cagr", m.get("cagr"))
        trial.set_user_attr("max_drawdown", m.get("max_drawdown"))
        s = m.get("sharpe", float("nan"))
        trial.set_user_attr("sharpe_raw", s)
        if n < MIN_TRADES or not math.isfinite(s):
            return -1.0
        return s

    t0 = time.time()
    if remaining > 0 and time.time() < deadline:
        study.optimize(objective, n_trials=remaining, timeout=max(1.0, deadline - time.time()),
                       gc_after_trial=False)
    states = pd.Series([t.state.name for t in study.trials]).value_counts().to_dict()
    msg = f"[OPT] window {k} ({TEST_PERIODS[k][0][:4]}): {states} in {time.time() - t0:.0f}s"
    print(msg, flush=True)
    with open(os.path.join(OUT, "progress.log"), "a") as f:
        f.write(msg + "\n")
    return {"window": k, "states": states}


# ── Evaluation helpers ───────────────────────────────────────────────────────

def ew_universe_equity(md: MarketData, start: int, end: int, capital: float) -> pd.Series:
    """Equal-weight buy-and-hold of the SAME point-in-time universe, rebalanced
    daily, no costs. Day t's return = average return of stocks that were index
    members at the previous close. (Also missing delisted names → also biased up.)"""
    r = md.rets[start:end + 1]
    mem = md.member[start - 1:end]
    rr = np.where(mem & np.isfinite(r), r, np.nan)
    daily = np.nanmean(rr, axis=1)
    daily[0] = 0.0
    return pd.Series(capital * np.cumprod(1 + np.nan_to_num(daily)), index=md.dates[start:end + 1])


def spy_equity(md: MarketData, start: int, end: int, capital: float) -> pd.Series:
    s = md.spy[start:end + 1]
    return pd.Series(capital * s / s[0], index=md.dates[start:end + 1])


def best_params(study: optuna.Study) -> tuple[Params | None, optuna.trial.FrozenTrial | None]:
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not done:
        return None, None
    bt = max(done, key=lambda t: t.value)
    return decode(bt.params), bt


def neighbours(p: Params) -> list[tuple[str, Params]]:
    """Every parameter set one grid step away in exactly one dimension."""
    out = []
    for k, grid in SPACE.items():
        i = grid.index(getattr(p, k))
        for j in (i - 1, i + 1):
            if 0 <= j < len(grid):
                out.append((f"{k}={grid[j]}", replace(p, **{k: grid[j]})))
    return out


def _neighbour_job(args: tuple) -> dict:
    k, label, p = args
    md = _MD
    b = window_bounds(md, k)
    tr = run_period(md, p, b["train_start"], b["train_end"])
    te = run_period(md, p, b["test_start"], b["test_end"])
    return {"window": k, "test_year": TEST_PERIODS[k][0][:4], "variant": label,
            "train_sharpe": M.perf(tr.equity).get("sharpe"), "train_trades": len(tr.trades),
            "test_sharpe": M.perf(te.equity).get("sharpe"),
            "test_return": M.perf(te.equity).get("total_return"), "test_trades": len(te.trades),
            **{f"p_{kk}": vv for kk, vv in p.to_dict().items()}}


def evaluate(md: MarketData, workers: int, meta: dict) -> None:
    windows, trials_rows, segs = [], [], []
    total = {"COMPLETE": 0, "PRUNED": 0, "FAIL": 0, "RUNNING": 0}
    for k in range(len(TEST_PERIODS)):
        study = _study(k)
        for t in study.trials:
            total[t.state.name] = total.get(t.state.name, 0) + 1
            trials_rows.append({"window": k, "test_year": TEST_PERIODS[k][0][:4], "trial": t.number,
                                "state": t.state.name, "value": t.value,
                                **(decode(t.params).to_dict() if len(t.params) == len(SPACE) else {}),
                                **t.user_attrs})
        p, bt = best_params(study)
        b = window_bounds(md, k)
        row = {"window": k, "train": f"{md.dates[b['train_start']].date()}→{md.dates[b['train_end']].date()}",
               "test": f"{md.dates[b['test_start']].date()}→{md.dates[b['test_end']].date()}",
               "n_trials": len(study.trials),
               "n_complete": sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials),
               "n_pruned": sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials)}
        if p is None:
            row["status"] = "NO COMPLETED TRIALS — window skipped"
            windows.append(row)
            continue
        # Re-run the chosen params on train to get the IS return distribution (for DSR).
        tr = run_period(md, p, b["train_start"], b["train_end"])
        r_tr = M.daily_returns(tr.equity)
        sr_trials = np.array([t.user_attrs.get("sharpe_raw", np.nan) for t in study.trials
                              if t.state == optuna.trial.TrialState.COMPLETE]) / math.sqrt(252)
        dsr = M.deflated_sharpe(M.sharpe(r_tr) / math.sqrt(252), sr_trials, len(r_tr),
                                float(stats.skew(r_tr)), float(stats.kurtosis(r_tr, fisher=False)))
        row.update({"status": "ok", "best_trial": bt.number, "train_sharpe": bt.value,
                    "train_trades": bt.user_attrs.get("n_trades"),
                    "train_dsr": dsr["dsr"], "train_sr0_ann": dsr.get("sr0_ann"),
                    **{f"p_{kk}": vv for kk, vv in p.to_dict().items()}})
        windows.append(row)
        z = zscore_panel(md, p.lookback, p.threshold, p.alpha, p.z_window, b["test_start"], b["test_end"])
        segs.append(Segment(b["test_start"], b["test_end"], p, z))

    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame(trials_rows).to_csv(os.path.join(OUT, "all_trials.csv"), index=False)
    if not segs:
        raise SystemExit("No window has a completed trial — nothing to evaluate.")

    # ── Stitched out-of-sample run, two cost levels ─────────────────────────
    t0 = time.time()
    oos = {name: simulate(md, segs, c) for name, c in (("base", BASE_COSTS), ("stress", STRESS_COSTS))}
    meta["oos_sim_seconds"] = time.time() - t0
    s0, s1 = segs[0].start, segs[-1].end
    cap = 100_000.0
    spy = spy_equity(md, s0, s1, cap)
    ew = ew_universe_equity(md, s0, s1, cap)
    daily = pd.DataFrame({
        "strategy_base": oos["base"].equity, "strategy_stress": oos["stress"].equity,
        "spy_buy_hold": spy, "ew_universe": ew,
        "gross_exposure": oos["base"].gross_exposure, "net_exposure": oos["base"].net_exposure,
    })
    daily.index.name = "date"
    daily.to_csv(os.path.join(OUT, "oos_daily.csv"))
    for name, res in oos.items():
        res.trades.to_csv(os.path.join(OUT, f"oos_trades_{name}.csv"), index=False)

    # Per-window OOS (sub-period) numbers from the stitched curves.
    sub = []
    for w, sg in zip([w for w in windows if w.get("status") == "ok"], segs, strict=True):
        # From the previous close (so the first test day's return counts) to the last test day.
        first = max(sg.start - 1, s0) - s0
        rowsub = {"window": w["window"], "test": w["test"], "train_sharpe": w["train_sharpe"]}
        for col in ("strategy_base", "strategy_stress", "spy_buy_hold", "ew_universe"):
            m = M.perf(daily[col].iloc[first:sg.end - s0 + 1])
            rowsub[f"{col}_return"] = m.get("total_return")
            rowsub[f"{col}_sharpe"] = m.get("sharpe")
            rowsub[f"{col}_maxdd"] = m.get("max_drawdown")
        tb = oos["base"].trades
        rowsub["trades"] = int(((tb.entry_date >= md.dates[sg.start]) & (tb.entry_date <= md.dates[sg.end])).sum()) if len(tb) else 0
        sub.append(rowsub)
    pd.DataFrame(sub).to_csv(os.path.join(OUT, "oos_by_window.csv"), index=False)
    pd.DataFrame(windows).to_csv(os.path.join(OUT, "windows.csv"), index=False)

    # ── Robustness: neighbouring parameters ──────────────────────────────────
    jobs = []
    for w in windows:
        if w.get("status") != "ok":
            continue
        p = Params(**{kk[2:]: w[kk] for kk in w if kk.startswith("p_")})
        jobs.append((w["window"], "chosen", p))
        jobs += [(w["window"], lab, q) for lab, q in neighbours(p)]
    t0 = time.time()
    with mp.get_context("fork").Pool(workers) as pool:
        nb = pool.map(_neighbour_job, jobs, chunksize=1)
    meta["neighbour_seconds"] = time.time() - t0
    meta["neighbour_evaluations"] = len(jobs)
    pd.DataFrame(nb).to_csv(os.path.join(OUT, "neighbours.csv"), index=False)

    # ── Headline metrics ─────────────────────────────────────────────────────
    summary = {"meta": meta, "data_qa": md.qa, "trial_counts": total,
               "total_trials": int(sum(total.values())), "windows": len(TEST_PERIODS),
               "windows_evaluated": len(segs),
               "costs": {"base": BASE_COSTS.__dict__, "stress": STRESS_COSTS.__dict__}}
    for name, res in oos.items():
        eq = res.equity
        summary[f"oos_{name}"] = {
            **M.perf(eq), **M.trade_stats(res.trades),
            "turnover_annual_one_way": M.turnover(res.traded_notional, eq),
            "avg_gross_exposure": float(res.gross_exposure.mean()),
            "avg_net_exposure": float(res.net_exposure.mean()),
            "costs_paid": res.costs, "vs_spy": M.capm(eq, spy),
        }
    summary["spy_buy_hold"] = M.perf(spy)
    summary["ew_universe"] = M.perf(ew)
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(json.dumps({k: summary[k] for k in ("total_trials", "trial_counts")}, indent=1))
    print("OOS base:", {k: summary["oos_base"][k] for k in ("cagr", "sharpe", "max_drawdown", "n_trades")})


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    global _MD
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--trials-per-window", type=int, default=32)
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument("--budget-min", type=float, default=150.0,
                    help="Hard wall-clock limit for the OPTIMISATION phase (minutes)")
    ap.add_argument("--time-single", action="store_true", help="Time one backtest and exit")
    ap.add_argument("--eval-only", action="store_true")
    a = ap.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    os.makedirs(OUT, exist_ok=True)
    t_load = time.time()
    _MD = md = load_market_data(a.data_dir)
    load_s = time.time() - t_load
    print(f"[DATA] {len(md.tickers)} stocks × {len(md.dates)} days loaded in {load_s:.1f}s; "
          f"{md.qa['avg_members_with_price_per_day']:.0f} members with prices per day on average")

    meta = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "optuna": optuna.__version__, "seed": SEED, "cpu_count": os.cpu_count(),
            "workers": a.workers, "trials_per_window": a.trials_per_window,
            "budget_min": a.budget_min, "data_load_seconds": load_s,
            "test_periods": TEST_PERIODS, "train_years": TRAIN_YEARS, "space": SPACE,
            "n_startup_trials": N_STARTUP, "min_trades": MIN_TRADES}

    if a.time_single:
        b = window_bounds(md, 0)
        p = Params()
        t0 = time.time()
        z = zscore_panel(md, p.lookback, p.threshold, p.alpha, p.z_window, b["train_start"], b["train_end"])
        t1 = time.time()
        res = simulate(md, [Segment(b["train_start"], b["train_end"], p, z)], BASE_COSTS)
        t2 = time.time()
        timing = {"period": f"{md.dates[b['train_start']].date()}→{md.dates[b['train_end']].date()}",
                  "params": p.to_dict(), "signal_seconds": t1 - t0, "sim_seconds": t2 - t1,
                  "total_seconds": t2 - t0, "n_trades": len(res.trades),
                  "note": "timing run with default params on window-0 TRAIN data; not a result"}
        with open(os.path.join(OUT, "timing.json"), "w") as f:
            json.dump(timing, f, indent=2)
        print(json.dumps(timing, indent=1))
        return

    if not a.eval_only:
        deadline = time.time() + a.budget_min * 60
        t0 = time.time()
        jobs = [(k, a.trials_per_window, deadline) for k in range(len(TEST_PERIODS))]
        with mp.get_context("fork").Pool(a.workers) as pool:
            pool.map(optimise_window, jobs, chunksize=1)
        meta["optimisation_seconds"] = time.time() - t0
        meta["deadline_hit"] = time.time() >= deadline
    evaluate(md, a.workers, meta)


if __name__ == "__main__":
    main()
