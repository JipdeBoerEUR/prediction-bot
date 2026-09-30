# make_report.py
"""
Build results/REPORT.md and charts from the files run_walkforward.py saved.
Re-running this never re-runs a backtest — it only reads results/*.csv/json.

If results/NARRATIVE.md exists (the hand-written plain-language summary), it
is inserted near the top of the report.
"""

from __future__ import annotations

import json
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = "results"

# Colours (validated categorical palette, light surface) — see dataviz notes.
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
C_STRAT, C_SPY, C_EW = "#2a78d6", "#eb6834", "#1baf7a"


def pct(x, d=1):
    return "—" if x is None or not np.isfinite(x) else f"{x * 100:+.{d}f}%"


def pctu(x, d=1):
    return "—" if x is None or not np.isfinite(x) else f"{x * 100:.{d}f}%"


def f2(x):
    return "—" if x is None or not np.isfinite(x) else f"{x:.2f}"


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def chart_equity(daily: pd.DataFrame, boundaries: list, path: str) -> None:
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6.8), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1.3]}, facecolor=SURFACE)
    series = [("strategy_base", "Strategy (base costs)", C_STRAT, "-"),
              ("strategy_stress", "Strategy (stress costs)", C_STRAT, "--"),
              ("spy_buy_hold", "SPY buy & hold", C_SPY, "-"),
              ("ew_universe", "Equal-weight S&P 500 members", C_EW, "-")]
    for col, lab, c, ls in series:
        g = daily[col] / daily[col].iloc[0] * 100
        a1.plot(g.index, g.values, color=c, ls=ls, lw=2 if ls == "-" else 1.6, label=lab)
        a1.annotate(f"{g.iloc[-1]:.0f}", (g.index[-1], g.iloc[-1]), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=8, color=INK)
    for b in boundaries[1:]:
        for ax in (a1, a2):
            ax.axvline(b, color=GRID, lw=0.8, zorder=0)
    _style(a1)
    a1.set_ylabel("Growth of 100 (log scale)", color=INK2, fontsize=9)
    a1.set_yscale("log")
    a1.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=(1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 6.0, 7.0, 8.0, 9.0)))
    a1.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}"))
    a1.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    a1.legend(loc="upper left", frameon=False, fontsize=8, labelcolor=INK)
    a1.set_title("Out-of-sample equity (stitched walk-forward test windows; grey lines = re-optimisation dates)",
                 fontsize=10, color=INK, loc="left")
    for col, lab, c in (("strategy_base", "Strategy (base)", C_STRAT), ("spy_buy_hold", "SPY", C_SPY)):
        dd = daily[col] / daily[col].cummax() - 1
        a2.plot(dd.index, dd.values * 100, color=c, lw=1.4, label=lab)
    _style(a2)
    a2.set_ylabel("Drawdown %", color=INK2, fontsize=9)
    a2.legend(loc="lower left", frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def chart_windows(byw: pd.DataFrame, path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.2), facecolor=SURFACE)
    x = np.arange(len(byw))
    w = 0.26
    for i, (col, lab, c) in enumerate((("strategy_base_return", "Strategy (base costs)", C_STRAT),
                                       ("ew_universe_return", "Equal-weight members", C_EW),
                                       ("spy_buy_hold_return", "SPY", C_SPY))):
        ax.bar(x + (i - 1) * (w + 0.02), byw[col] * 100, width=w, color=c, label=lab, edgecolor=SURFACE, lw=1)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(x, [t.split("→")[0][:4] + ("*" if t.endswith("09-30") else "") for t in byw["test"]])
    _style(ax)
    ax.set_ylabel("Return in test window %", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, ncol=3, loc="upper left")
    ax.set_title("Out-of-sample return per test window (* = partial year)", fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def chart_neighbours(nb: pd.DataFrame, path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.2), facecolor=SURFACE)
    years = sorted(nb.test_year.unique())
    for i, y in enumerate(years):
        d = nb[nb.test_year == y]
        other = d[d.variant != "chosen"]
        jitter = np.linspace(-0.18, 0.18, max(len(other), 1))
        ax.scatter(i + jitter[:len(other)], other.test_sharpe, s=22, color="#a9a8a2",
                   edgecolor=SURFACE, lw=0.8, zorder=2, label="1-step neighbour" if i == 0 else None)
        ch = d[d.variant == "chosen"]
        ax.scatter([i] * len(ch), ch.test_sharpe, s=70, color=C_STRAT, edgecolor=SURFACE, lw=2,
                   zorder=3, label="Chosen parameters" if i == 0 else None)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(range(len(years)), years)
    _style(ax)
    ax.set_ylabel("Out-of-sample Sharpe", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left")
    ax.set_title("Robustness: test-window Sharpe of the chosen parameters vs every 1-step neighbour",
                 fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def convergence(trials: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k, d in trials.groupby("window"):
        d = d.sort_values("trial")
        comp = d[d.state == "COMPLETE"]
        best = comp.value.max() if len(comp) else np.nan
        half = d.trial.max() // 2
        best_half = comp[comp.trial <= half].value.max() if len(comp[comp.trial <= half]) else np.nan
        last_improve = int(comp.loc[comp.value.cummax().diff().fillna(1) > 0, "trial"].max()) if len(comp) else None
        rows.append({"window": k, "test_year": d.test_year.iloc[0], "trials": len(d),
                     "pruned": int((d.state == "PRUNED").sum()),
                     "best_at_half": best_half, "best_final": best,
                     "last_improvement_at_trial": last_improve})
    return pd.DataFrame(rows)


def main() -> None:
    s = json.load(open(os.path.join(OUT, "summary.json")))
    daily = pd.read_csv(os.path.join(OUT, "oos_daily.csv"), index_col=0, parse_dates=True)
    byw = pd.read_csv(os.path.join(OUT, "oos_by_window.csv"))
    win = pd.read_csv(os.path.join(OUT, "windows.csv"))
    nb = pd.read_csv(os.path.join(OUT, "neighbours.csv"))
    trials = pd.read_csv(os.path.join(OUT, "all_trials.csv"))
    timing = json.load(open(os.path.join(OUT, "timing.json"))) if os.path.exists(os.path.join(OUT, "timing.json")) else {}
    narrative = open(os.path.join(OUT, "NARRATIVE.md")).read() if os.path.exists(os.path.join(OUT, "NARRATIVE.md")) else \
        "_Plain-language summary not written yet._"

    boundaries = [pd.Timestamp(t.split("→")[0]) for t in byw["test"]]
    chart_equity(daily, boundaries, os.path.join(OUT, "equity_drawdown.png"))
    chart_windows(byw, os.path.join(OUT, "oos_by_window.png"))
    chart_neighbours(nb, os.path.join(OUT, "neighbours.png"))
    conv = convergence(trials)
    conv.to_csv(os.path.join(OUT, "convergence.csv"), index=False)

    b, st = s["oos_base"], s["oos_stress"]
    spy, ew = s["spy_buy_hold"], s["ew_universe"]
    meta, qa = s["meta"], s["data_qa"]
    man = qa.get("manifest", {}) or {}

    def perf_row(name, m):
        return (f"| {name} | {pct(m.get('cagr'))} | {pctu(m.get('ann_vol'))} | {f2(m.get('sharpe'))} | "
                f"{f2(m.get('sortino'))} | {pct(m.get('max_drawdown'))} | {f2(m.get('calmar'))} | "
                f"{pct(m.get('total_return'))} |")

    def trade_row(name, m):
        return (f"| {name} | {m.get('n_trades', 0):,} ({m.get('n_long', 0):,} L / {m.get('n_short', 0):,} S) | "
                f"{pctu(m.get('win_rate'), 0)} | {m.get('avg_holding_days', float('nan')):.1f} | "
                f"{m.get('turnover_annual_one_way', float('nan')):.1f}× | {pctu(m.get('avg_gross_exposure'), 0)} | "
                f"{pct(m.get('avg_net_exposure'), 0)} | {f2(m.get('profit_factor'))} |")

    cb = b.get("vs_spy", {})
    cs = st.get("vs_spy", {})

    # Chosen-parameter table
    pcols = [c for c in win.columns if c.startswith("p_")]
    ptab = ["| Test window | Train Sharpe (IS) | Trades (IS) | DSR | " + " | ".join(c[2:] for c in pcols) + " |",
            "|---" * (4 + len(pcols)) + "|"]
    for _, r in win.iterrows():
        if r.get("status") != "ok":
            ptab.append(f"| {r['test']} | {r.get('status')} |" + " |" * (2 + len(pcols)))
            continue
        ptab.append(f"| {r['test']} | {f2(r['train_sharpe'])} | {int(r['train_trades'])} | {f2(r['train_dsr'])} | "
                    + " | ".join(str(r[c]) for c in pcols) + " |")

    wtab = ["| Test window | Strategy base | Strategy stress | EW members | SPY | Strat. Sharpe (OOS) | Train Sharpe (IS) | Strat. MaxDD | Trades |",
            "|---|---|---|---|---|---|---|---|---|"]
    for _, r in byw.iterrows():
        wtab.append(f"| {r['test']} | {pct(r['strategy_base_return'])} | {pct(r['strategy_stress_return'])} | "
                    f"{pct(r['ew_universe_return'])} | {pct(r['spy_buy_hold_return'])} | "
                    f"{f2(r['strategy_base_sharpe'])} | {f2(r['train_sharpe'])} | {pct(r['strategy_base_maxdd'])} | {int(r['trades'])} |")
    n_pos_w = int((byw.strategy_base_return > 0).sum())
    n_beat_ew = int((byw.strategy_base_return > byw.ew_universe_return).sum())

    ntab = ["| Test year | Chosen OOS Sharpe | Neighbours | Median neighbour OOS Sharpe | Neighbours with OOS Sharpe > 0 | Median neighbour IS Sharpe |",
            "|---|---|---|---|---|---|"]
    for y, d in nb.groupby("test_year"):
        ch = d[d.variant == "chosen"].test_sharpe
        o = d[d.variant != "chosen"]
        ntab.append(f"| {y} | {f2(ch.iloc[0]) if len(ch) else '—'} | {len(o)} | {f2(o.test_sharpe.median())} | "
                    f"{(o.test_sharpe > 0).mean():.0%} | {f2(o.train_sharpe.median())} |")

    ctab = ["| Test year | Trials | Pruned | Best IS Sharpe after half the trials | Best IS Sharpe final | Last improvement at trial # |",
            "|---|---|---|---|---|---|"]
    for _, r in conv.iterrows():
        ctab.append(f"| {r.test_year} | {r.trials} | {r.pruned} | {f2(r.best_at_half)} | {f2(r.best_final)} | {r.last_improvement_at_trial} |")

    tc = s["trial_counts"]
    reasons = ", ".join(f"{k} {v:,}" for k, v in (b.get("exit_reasons") or {}).items())
    first_d, last_d = daily.index[0].date(), daily.index[-1].date()

    md = f"""# Walk-Forward Backtest Report — Graph Stat-Arb Strategy (price-only)

> **Backtest, not live trading.** Every number below is **out-of-sample**: each test
> window was traded with parameters chosen only on the 3 years before it, and the
> test windows are stitched together. Period **{first_d} → {last_d}**, universe =
> **point-in-time S&P 500 members with Yahoo Finance prices**, long/short, costs
> stated below. **Survivorship bias remains** (delisted members are missing), so
> results are biased upward. The news-sentiment filter of the live bot is **not
> backtested** (no historical headline data).

## 1. Headline — out-of-sample, {first_d} → {last_d}

| Book | CAGR | Volatility | Sharpe | Sortino | Max drawdown | Calmar | Total return |
|---|---|---|---|---|---|---|---|
{perf_row("Strategy — base costs", b)}
{perf_row("Strategy — stress costs", st)}
{perf_row("Benchmark: equal-weight S&P 500 members (same universe)", ew)}
{perf_row("Benchmark: SPY buy & hold", spy)}

| Book | Trades | Win rate (net) | Avg holding (days) | Turnover (one-way/yr) | Avg gross exposure | Avg net exposure | Profit factor |
|---|---|---|---|---|---|---|---|
{trade_row("Strategy — base costs", b)}
{trade_row("Strategy — stress costs", st)}

Market exposure (daily-return regression on SPY): base beta **{f2(cb.get('beta'))}**,
annualised alpha **{pct(cb.get('alpha_ann'))}** (t = {f2(cb.get('alpha_t_stat'))});
stress beta {f2(cs.get('beta'))}, alpha {pct(cs.get('alpha_ann'))} (t = {f2(cs.get('alpha_t_stat'))}).
t-stat of the mean daily return, base: **{f2(b.get('t_stat_mean_return'))}** (|t| < 2 ≈ not distinguishable from zero).

Costs paid (base): trading ${b['costs_paid']['trading']:,.0f}, short borrow ${b['costs_paid']['borrow']:,.0f}
on $100,000 starting capital. Exit reasons (base): {reasons}.

![Equity and drawdown](equity_drawdown.png)

## 2. Plain-language summary

{narrative}

## 3. Method

**Strategy.** Every 21 trading days, link S&P 500 members whose daily returns
were correlated ≥ `threshold` over the last `lookback` days. Each day, compute
every stock's *peer-implied* return by graph smoothing, `h = (I + αL)⁻¹x`, and the
residual `x − h`. Standardise the residual against that stock's previous
`z_window` residuals → z-score. At the close: **buy** the most negative
z ≤ −entry_z, **short** the most positive z ≥ +entry_z (max 5 long + 5 short,
10% of equity each), filled at the **next open**. Exit when |z| is back inside
`exit_z` (same z units as entry), on a stop of `stop_k` × daily σ (clamped 4–15%,
trailing once in profit), or after `max_hold` days. New entries only when SPY >
its 200-day average and VIX < 30.

**Data.** Yahoo Finance daily prices, split- and dividend-adjusted, downloaded
{man.get('created_utc', '?')} (yfinance {man.get('yfinance', '?')}); S&P 500 membership history from
the public `fja05680/sp500` dataset. A stock can only enter the graph or be
traded on days it was an index member. {man.get('n_index_members_in_span', '?')} stocks were members at
some point in the span; {man.get('n_members_with_prices', '?')} have Yahoo prices; **member-day coverage
{pctu(man.get('member_day_coverage') or float('nan'))}** — the missing rest are mostly
delisted/acquired companies. On average {qa.get('avg_members_with_price_per_day', float('nan')):.0f} members
with prices per day. Days with |return| > 50% (kept, not clipped): {qa.get('abs_daily_return_gt_50pct')}.
SHA-256 of prices.parquet: `{man.get('prices_sha256', '?')}`.

**Costs** (every fill, per side, plus a borrow fee on shorts):

| Level | Commission | Slippage (½ spread + impact) | Total per side | Short borrow |
|---|---|---|---|---|
| Base | {s['costs']['base']['commission_bps']:.0f} bp | {s['costs']['base']['slippage_bps']:.0f} bp | {s['costs']['base']['commission_bps'] + s['costs']['base']['slippage_bps']:.0f} bp | {s['costs']['base']['borrow_bps']:.0f} bp/yr |
| Stress | {s['costs']['stress']['commission_bps']:.0f} bp | {s['costs']['stress']['slippage_bps']:.0f} bp | {s['costs']['stress']['commission_bps'] + s['costs']['stress']['slippage_bps']:.0f} bp | {s['costs']['stress']['borrow_bps']:.0f} bp/yr |

Parameters were optimised under **base** costs only; the stress run re-trades the
same chosen parameters with higher costs. Cash earns 0% (no T-bill data), which
also means Sharpe uses a 0% risk-free rate.

**Walk-forward.** {s['windows']} windows: train 3 calendar years → test the next
year (2026 = Jan–Sep). Parameters re-chosen each window; positions carry over at
the switch instead of being force-closed.

**Search space** (grid, so "neighbour" is well-defined; ranges chosen from finance logic *before* seeing results):

| Parameter | Values | Why this range |
|---|---|---|
| lookback (correlation window, days) | {meta['space']['lookback']} | 6 months–2 years: shorter = noisy correlations, longer = stale relationships |
| threshold (min correlation for a link) | {meta['space']['threshold']} | < 0.3 links are mostly noise; > 0.6 leaves few links outside same-industry pairs |
| alpha (peer pull) | {meta['space']['alpha']} | log-spaced: peers barely matter → peers dominate |
| z_window (residual history, days) | {meta['space']['z_window']} | ~2–6 months of the stock's own residual volatility |
| entry_z | {meta['space']['entry_z']} | < 1.5 fires constantly on noise; > 3 is too rare to trade |
| exit_z | {meta['space']['exit_z']} | 0 = wait for full reversion; 1 = take profit early |
| max_hold (days) | {meta['space']['max_hold']} | short-term reversal literature: effects last ~1–4 weeks |
| stop_k (stop in daily σ) | {meta['space']['stop_k']} | 2σ = tight (stopped on noise), 5σ = loose |

**Fixed, not optimised:** 5 long + 5 short slots at 10% each, rebuild every 21 days,
regime filter (SPY 200-DMA, VIX 30), costs. **Sentiment threshold: not in the search
space** — no historical sentiment data exists.

**Optimiser.** Optuna TPE sampler (seed {meta['seed']} + window), per window: {meta['n_startup_trials']} random
"coarse" trials, then TPE "fine" trials concentrated near the best so far.
Objective = Sharpe of daily net returns over the 3 training years (base costs);
< {meta['min_trades']} trades → score −1. MedianPruner stops a trial early if its Sharpe after training
year 1 or 2 is below the median of earlier trials. 8 windows run in parallel on
{meta['workers']} cores; every trial is checkpointed to `results/optuna/window_<k>.log`.

**Trials run (multiple-testing count): {s['total_trials']:,}** — complete {tc.get('COMPLETE', 0):,},
pruned {tc.get('PRUNED', 0):,}, failed {tc.get('FAIL', 0):,}. Plus {meta.get('neighbour_evaluations', 0):,}
robustness evaluations (neighbours, below) that were **not** used to choose anything.
Optimisation wall-clock: {meta.get('optimisation_seconds', float('nan')) / 60:.1f} min
(budget {meta['budget_min']:.0f} min; deadline hit: {meta.get('deadline_hit')}).
One backtest (3 years, default params): {timing.get('total_seconds', float('nan')):.1f} s
(signals {timing.get('signal_seconds', float('nan')):.1f} s + simulation {timing.get('sim_seconds', float('nan')):.1f} s).

## 4. Results by sub-period (each test window, out-of-sample)

{chr(10).join(wtab)}

Strategy positive in **{n_pos_w} of {len(byw)}** test windows; beat the equal-weight
universe in **{n_beat_ew} of {len(byw)}**.

![Per-window returns](oos_by_window.png)

## 5. Chosen parameters per window (stability check)

Parameters that jump around from window to window are a sign the optimiser is
fitting noise. **DSR** = Deflated Sharpe Ratio of the chosen in-sample result
given all trials in that window (probability the true IS Sharpe > 0 after the
selection effect; < 0.95 = the in-sample "best" is not statistically convincing).

{chr(10).join(ptab)}

## 6. Robustness — neighbouring parameters

For each window, every parameter was moved one grid step up/down (one at a time)
and re-traded on the same test window. A real edge should survive small changes;
a lone spike surrounded by bad neighbours is overfitting.

{chr(10).join(ntab)}

![Neighbour robustness](neighbours.png)

## 7. Convergence

{chr(10).join(ctab)}

If the best in-sample score was still improving near the last trial, the search
had **not converged** — more trials might find a better *in-sample* score (which,
per sections 5–6, would not necessarily help out-of-sample).

## 8. Limitations — read before quoting any number

1. **Survivorship bias (upward).** Only {pctu(man.get('member_day_coverage') or float('nan'))} of index member-days have prices;
   the missing members are mostly companies that were acquired, went bankrupt or
   were delisted. Losers are under-represented, so both the strategy and the
   equal-weight benchmark are flattered. Membership filtering removes the
   "picked today's winners" bias but not this.
2. **Sentiment filter not backtested.** The live bot's VADER/FinBERT headline gates,
   LSTM forecast and ML "brain" model are all absent. This is the price-only core.
3. **Not the live bot's exact configuration** (different correlation window, α,
   no sector mask, different z definition) — this is a research backtest of the
   strategy idea.
4. **Costs are modelled, not measured.** Flat bps per side; no market-impact curve,
   no hard-to-borrow names, no borrow recalls, no short-sale restrictions. Fills at
   the open assume the open price was obtainable.
5. **0% risk-free rate** in Sharpe/Sortino and 0% on cash.
6. **Daily bars:** stop-losses use the day's high/low with pessimistic ordering.
7. **Multiple testing:** {s['total_trials']:,} trials across windows. The OOS design protects
   the headline from direct selection bias, but the *design* (search space, costs,
   windows) was chosen by a human who knows market history 2019–2026.
8. **Adjusted prices:** dividend-adjusted, so dividends are treated as reinvested
   for longs; shorts implicitly pay them (correct direction).

## 9. Reproduce

```
python -m pip install -r requirements-backtest.txt
python download_data.py                 # or use the committed data/ files (check SHA-256)
python run_walkforward.py --time-single
python run_walkforward.py --budget-min {meta['budget_min']:.0f}
python make_report.py
```
Seeds: numpy/random {meta['seed']}; Optuna TPE seed {meta['seed']}+window. Resuming an interrupted
run continues the saved studies but the sampler's random state restarts, so a
resumed run is not bit-identical to an uninterrupted one.
Versions: Python {meta['python']}, numpy {meta['numpy']}, pandas {meta['pandas']}, Optuna {meta['optuna']}.

Files: `oos_daily.csv` (equity curves + exposure), `oos_trades_base.csv` /
`oos_trades_stress.csv` (every trade), `oos_by_window.csv`, `windows.csv`
(chosen params), `all_trials.csv` (every Optuna trial), `neighbours.csv`,
`convergence.csv`, `summary.json`, `timing.json`, `optuna/` (checkpoints).
"""
    with open(os.path.join(OUT, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print(f"[REPORT] {OUT}/REPORT.md written")


if __name__ == "__main__":
    main()
