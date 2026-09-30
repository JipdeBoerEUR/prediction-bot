"""Performance statistics. Every formula is written out so it can be explained.

Conventions
-----------
* Daily simple returns r_t = equity_t / equity_{t-1} − 1; 252 trading days/yr.
* Sharpe = mean(r) / std(r) × √252 with a 0% risk-free rate (no T-bill data in
  this environment; stated in the report — this flatters Sharpe in 2023-25
  when T-bills paid ~5%).
* Sortino = mean(r) / downside deviation × √252, downside deviation =
  √mean(min(r, 0)²) over ALL days (the standard definition).
* Max drawdown = worst peak-to-trough fall of the equity curve.
* CAGR = (end / start)^(252 / days) − 1.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252


def daily_returns(equity: pd.Series) -> pd.Series:
    return equity.pct_change().dropna()


def sharpe(r: pd.Series) -> float:
    sd = r.std()
    return float(r.mean() / sd * math.sqrt(TRADING_DAYS)) if sd and sd > 0 else float("nan")


def perf(equity: pd.Series) -> dict:
    equity = equity.dropna()
    if len(equity) < 3:
        return {}
    r = daily_returns(equity)
    days = len(r)
    total = equity.iloc[-1] / equity.iloc[0] - 1.0
    cagr = (1 + total) ** (TRADING_DAYS / days) - 1 if total > -1 else -1.0
    vol = r.std() * math.sqrt(TRADING_DAYS)
    dd_dev = math.sqrt((np.minimum(r, 0) ** 2).mean())
    sortino = r.mean() / dd_dev * math.sqrt(TRADING_DAYS) if dd_dev > 0 else float("nan")
    dd = equity / equity.cummax() - 1.0
    mdd = float(dd.min())
    return {
        "start": str(equity.index[0].date()), "end": str(equity.index[-1].date()),
        "years": days / TRADING_DAYS,
        "total_return": float(total), "cagr": float(cagr), "ann_vol": float(vol),
        "sharpe": sharpe(r), "sortino": float(sortino), "max_drawdown": mdd,
        "calmar": float(cagr / abs(mdd)) if mdd < 0 else float("nan"),
        # t-stat of the mean daily return (≈ Sharpe × √years): is it
        # distinguishable from zero?
        "t_stat_mean_return": float(r.mean() / (r.std() / math.sqrt(days))) if r.std() > 0 else float("nan"),
    }


def trade_stats(trades: pd.DataFrame) -> dict:
    if trades is None or trades.empty:
        return {"n_trades": 0}
    w = trades[trades.pnl_net > 0]
    lo = trades[trades.pnl_net <= 0]
    return {
        "n_trades": int(len(trades)),
        "n_long": int((trades.side == 1).sum()), "n_short": int((trades.side == -1).sum()),
        "win_rate": float(len(w) / len(trades)),
        "avg_trade_ret_gross": float(trades.ret_gross.mean()),
        "avg_win_ret_gross": float(w.ret_gross.mean()) if len(w) else float("nan"),
        "avg_loss_ret_gross": float(lo.ret_gross.mean()) if len(lo) else float("nan"),
        "profit_factor": float(w.pnl_net.sum() / -lo.pnl_net.sum()) if len(lo) and lo.pnl_net.sum() < 0 else float("nan"),
        "avg_holding_days": float(trades.days_held.mean()),
        "exit_reasons": trades.reason.value_counts().to_dict(),
    }


def turnover(traded_notional: pd.Series, equity: pd.Series) -> float:
    """Annual one-way turnover: $ bought+sold per year ÷ 2 ÷ average equity.
    1.0 = the whole book replaced once a year."""
    years = len(equity) / TRADING_DAYS
    return float(traded_notional.sum() / 2.0 / equity.mean() / years) if years > 0 else float("nan")


def capm(strategy_eq: pd.Series, bench_eq: pd.Series) -> dict:
    """OLS of strategy daily returns on benchmark daily returns:
    r_s = a + b·r_b + e. Beta = market exposure; alpha = return NOT explained
    by the market (annualised). Plain OLS t-stat (no autocorrelation correction)."""
    rs, rb = daily_returns(strategy_eq).align(daily_returns(bench_eq), join="inner")
    if len(rs) < 30:
        return {}
    res = stats.linregress(rb.to_numpy(), rs.to_numpy())
    se_a = res.intercept_stderr
    return {"beta": float(res.slope), "alpha_ann": float(res.intercept * TRADING_DAYS),
            "alpha_t_stat": float(res.intercept / se_a) if se_a > 0 else float("nan"),
            "correlation": float(res.rvalue)}


def deflated_sharpe(sr_best_daily: float, sr_trials_daily: np.ndarray, n_obs: int,
                    skew: float, kurt: float) -> dict:
    """Deflated Sharpe Ratio (Bailey & López de Prado, 2014).

    If you try N parameter sets, the BEST in-sample Sharpe is inflated by luck.
    SR0 = the Sharpe you'd expect from the best of N trials if the true Sharpe
    were zero. DSR = probability that the best trial's true Sharpe > 0 after
    accounting for that selection. Inputs are DAILY (non-annualised) Sharpes.
    """
    sr_trials_daily = np.asarray([s for s in sr_trials_daily if np.isfinite(s)])
    N = len(sr_trials_daily)
    if N < 2 or n_obs < 3:
        return {"n_trials": N, "sr0_daily": float("nan"), "dsr": float("nan")}
    g = 0.5772156649
    var = sr_trials_daily.var(ddof=1)
    sr0 = math.sqrt(var) * ((1 - g) * stats.norm.ppf(1 - 1 / N) + g * stats.norm.ppf(1 - 1 / (N * math.e)))
    denom = math.sqrt(max(1e-12, 1 - skew * sr_best_daily + (kurt - 1) / 4 * sr_best_daily ** 2))
    dsr = stats.norm.cdf((sr_best_daily - sr0) * math.sqrt(n_obs - 1) / denom)
    return {"n_trials": N, "sr0_daily": float(sr0), "sr0_ann": float(sr0 * math.sqrt(TRADING_DAYS)),
            "dsr": float(dsr)}
