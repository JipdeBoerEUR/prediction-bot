"""Correctness tests for the walk-forward engine (wfbt/).

These use small made-up arrays ONLY to check the mechanics (maths identity,
no look-ahead, fill timing, cost accounting). They produce no performance
numbers — every reported result comes from real downloaded prices.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from statarb.sim_engine import build_adjacency_from_returns, solve_diffusion_residual  # noqa: E402
from wfbt import signals  # noqa: E402
from wfbt.data import MarketData  # noqa: E402
from wfbt.sim import Costs, Params, Segment, simulate  # noqa: E402


def _toy_md(T=400, N=15, seed=0, rets=None):
    rng = np.random.default_rng(seed)
    if rets is None:
        common = rng.normal(0, 0.01, (T, 1))
        rets = common + rng.normal(0, 0.01, (T, N))
    close = 100 * np.cumprod(1 + np.nan_to_num(rets), axis=0)
    close[np.isnan(rets)] = np.nan
    r = pd.DataFrame(close).pct_change(fill_method=None).to_numpy()
    dates = pd.bdate_range("2015-01-01", periods=T)
    spy = np.linspace(100, 200, T)          # always above its 200-DMA
    return MarketData(
        dates=dates, tickers=pd.Index([f"S{i}" for i in range(N)]),
        open=close.copy(), high=close * 1.001, low=close * 0.999, close=close,
        rets=r, member=np.ones((T, N), bool),
        spy=spy, spy_sma200=pd.Series(spy).rolling(200, min_periods=200).mean().to_numpy(),
        vix=np.full(T, 15.0), qa={},
    )


# ── signals ──────────────────────────────────────────────────────────────────

def test_batched_residuals_match_production_solver():
    md = _toy_md()
    win = md.rets[100:352]
    W = signals.build_weights(win, 0.3)
    W_prod = build_adjacency_from_returns(pd.DataFrame(win), threshold=0.3).to_numpy()
    np.testing.assert_allclose(W, W_prod, atol=1e-12)
    X = md.rets[352:360]
    R = signals.residuals_for_block(X, W, alpha=0.7)
    for i in range(len(X)):
        np.testing.assert_allclose(R[i], solve_diffusion_residual(X[i], W, alpha=0.7), atol=1e-10)


def test_residuals_with_gap_use_subgraph_like_production():
    md = _toy_md()
    W = signals.build_weights(md.rets[100:352], 0.3)
    X = md.rets[352:354].copy()
    X[1, 3] = np.nan
    R = signals.residuals_for_block(X, W, alpha=0.7)
    v = np.isfinite(X[1])
    expected = solve_diffusion_residual(X[1, v], W[np.ix_(v, v)], alpha=0.7)
    np.testing.assert_allclose(R[1, v], expected, atol=1e-10)
    assert np.isnan(R[1, 3])


def test_no_lookahead_future_data_does_not_change_past_z():
    md = _toy_md()
    signals._CACHE.clear()
    z1 = signals.zscore_panel(md, 126, 0.3, 0.5, 40, 300, 380).copy()
    md.rets = md.rets.copy()
    md.rets[351:] = md.rets[351:] * -5 + 0.03       # scramble everything after day 350
    signals._CACHE.clear()
    z2 = signals.zscore_panel(md, 126, 0.3, 0.5, 40, 300, 380)
    np.testing.assert_allclose(z1[:51], z2[:51], rtol=1e-12, atol=1e-12)  # days 300..350 unchanged
    assert not np.allclose(np.nan_to_num(z1[51:]), np.nan_to_num(z2[51:]))


def test_same_day_signal_is_identical_regardless_of_window_start():
    md = _toy_md()
    signals._CACHE.clear()
    a = signals.zscore_panel(md, 126, 0.3, 0.5, 40, 250, 390)
    b = signals.zscore_panel(md, 126, 0.3, 0.5, 40, 320, 390)
    np.testing.assert_allclose(a[70:], b, rtol=1e-12, atol=1e-12)  # batch-size rounding only


def test_non_members_excluded_from_graph():
    md = _toy_md()
    md.member[:, 0] = False
    signals._CACHE.clear()
    z = signals.zscore_panel(md, 126, 0.3, 0.5, 40, 300, 320)
    assert np.isnan(z[:, 0]).all()


# ── simulator ────────────────────────────────────────────────────────────────

def _one_signal_md():
    """Flat prices; a single z signal on day 250 for stock 0."""
    T, N = 300, 12
    md = _toy_md(T, N, rets=np.zeros((T, N)))
    md.open[251, 0] = 101.0          # the NEXT day's open differs from day-250 close
    md.close[251:, 0] = 101.0
    md.open[252:, 0] = 101.0
    md.high[251:, 0] = 101.0
    md.low[251:, 0] = 101.0
    z = np.zeros((T - 240, N))
    z[250 - 240, 0] = -3.0            # signal at the CLOSE of day 250
    return md, z


def test_entry_fills_at_next_open_and_costs_are_charged():
    md, z = _one_signal_md()
    p = Params(entry_z=2.0, exit_z=0.5, max_hold=100, stop_k=3.0)
    costs = Costs(commission_bps=1, slippage_bps=4, borrow_bps=0)
    res = simulate(md, [Segment(240, 299, p, z)], costs, capital=100_000)
    # Day 251: z on day 251 is 0 → exit queued, filled at day-252 open.
    t = res.trades.iloc[0]
    assert t.entry_date == md.dates[251] and t.entry_px == 101.0
    assert t.exit_date == md.dates[252] and t.reason == "RESIDUAL_REVERTED"
    # 10% of 100k bought at 101 and sold at 101 → P&L = −2 × 5 bps × 10k.
    assert abs(t.pnl_net - (-2 * 0.0005 * 10_000)) < 1e-6
    assert abs(res.equity.iloc[-1] - (100_000 - 10.0)) < 1e-6


def test_no_trade_when_regime_is_off():
    md, z = _one_signal_md()
    md.vix[:] = 35.0                   # VIX >= 30 blocks entries
    res = simulate(md, [Segment(240, 299, Params(), z)], Costs(), capital=100_000)
    assert res.trades.empty and res.equity.iloc[-1] == 100_000


def test_stop_loss_fills_at_gap_open_when_worse():
    md, z = _one_signal_md()
    # After entry at 101 on day 251, gap down to 80 on day 253 (stop ~4%).
    for arr in (md.open, md.close, md.high, md.low):
        arr[253:, 0] = 80.0
    z[:] = 0.0
    z[250 - 240, 0] = -3.0
    z[251 - 240:, 0] = -3.0            # never reverts → only the stop can exit
    res = simulate(md, [Segment(240, 299, Params(max_hold=100), z)], Costs(0, 0, 0))
    t = res.trades.iloc[0]
    assert t.reason == "STOP_LOSS" and t.exit_px == 80.0 and t.exit_date == md.dates[253]


def test_short_pays_borrow():
    md, z = _one_signal_md()
    z[250 - 240, 0] = +3.0
    z[251 - 240:, 0] = +3.0            # stays stretched
    res = simulate(md, [Segment(240, 260, Params(max_hold=100), z)],
                   Costs(0, 0, borrow_bps=252.0), capital=100_000)
    # 10k short × 252bps/yr ÷ 252 days = 1.00 $/day for days 251..260 → ~$10
    assert 9.5 < res.costs["borrow"] < 10.5
