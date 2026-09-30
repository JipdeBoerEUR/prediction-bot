"""Signal panel: for every day and stock, how unusually far did the stock lag
(or lead) its correlated peers today?

Plain-language pipeline (repeated every `rebuild_every` trading days):

  1. GRAPH. Take the stocks that are S&P 500 members today and have a price on
     >= 90% of the last `lookback` days. Correlate their daily returns over
     those days (strictly BEFORE today). Link two stocks when their
     correlation >= `threshold`; the link weight is the correlation.

  2. PEER-IMPLIED RETURN. For each day's return vector x, solve
         (I + alpha * L) h = x          L = D - W  (graph Laplacian)
     h is a smoothed version of x: each stock's return pulled towards its
     peers'. alpha = how hard the peers pull. This is the SAME formula as the
     live engine (statarb.sim_engine.solve_diffusion_residual); tests check
     that the fast batched solve here gives identical numbers.

  3. RESIDUAL r = x - h. Negative = the stock did worse than its peers implied.

  4. Z-SCORE. z_t = (r_t - mean) / std, where mean and std are taken over the
     stock's PREVIOUS `z_window` residuals (t-z_window … t-1), not including
     today. So z = -2 means "today's lag is 2 of this stock's typical
     residual moves below its recent average".

No look-ahead: the graph at day t only uses returns up to t-1, the z-score at
day t uses residuals up to t, and the trade happens at the open of t+1.
Tests in tests/test_wfbt.py check this by changing future data and
confirming past signals do not move.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Tuple

import numpy as np
import pandas as pd
import scipy.linalg as la

from .data import MarketData

REGULARIZATION = 1e-6     # same tiny ridge as the live solver
MIN_NAMES = 10            # don't build a graph with fewer stocks than this
MIN_HISTORY_SHARE = 0.90  # a stock needs prices on >= 90% of the lookback days

_CACHE: "OrderedDict[Tuple, np.ndarray]" = OrderedDict()
_CACHE_MAX = 48


def build_weights(window: np.ndarray, threshold: float) -> np.ndarray:
    """Correlation graph (mirrors sim_engine.build_adjacency_from_returns)."""
    if np.isfinite(window).all():
        corr = np.corrcoef(window, rowvar=False)          # fast path, no gaps
    else:                                                 # pairwise-complete, like pandas .corr()
        corr = pd.DataFrame(window).corr(min_periods=max(20, window.shape[0] // 2)).to_numpy(copy=True)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 0.0)
    W = np.where(corr >= threshold, corr, 0.0)
    return np.clip((W + W.T) / 2.0, 0.0, None)


def _system(W: np.ndarray, alpha: float) -> np.ndarray:
    L = np.diag(W.sum(axis=1)) - W
    n = W.shape[0]
    return np.eye(n) + alpha * L + REGULARIZATION * np.eye(n)


def residuals_for_block(X: np.ndarray, W: np.ndarray, alpha: float) -> np.ndarray:
    """Residuals x - h for every row of X (days × names) under one graph W.

    Days where every name has a return share one LU factorisation (fast).
    Days with gaps are solved on the sub-graph of names that traded — exactly
    what the live engine does (it drops NaN names before solving).
    """
    out = np.full_like(X, np.nan)
    finite = np.isfinite(X)
    full = finite.all(axis=1)
    if full.any():
        lu = la.lu_factor(_system(W, alpha))
        H = la.lu_solve(lu, X[full].T)
        out[full] = X[full] - H.T
    for i in np.where(~full)[0]:
        v = finite[i]
        if v.sum() < MIN_NAMES:
            continue
        Wv = W[np.ix_(v, v)]
        h = la.solve(_system(Wv, alpha), X[i, v], assume_a="sym")
        out[i, v] = X[i, v] - h
    return out


def rebuild_positions(n_days: int, lookback: int, every: int) -> np.ndarray:
    """Graph rebuild days, on a FIXED grid (every `every` days from day 0).

    A fixed grid means the signal for a given date is the same no matter which
    train/test window asks for it — windows can't get different signals for
    the same day.
    """
    first = int(np.ceil(lookback / every) * every)
    return np.arange(first, n_days, every)


def zscore_panel(md: MarketData, lookback: int, threshold: float, alpha: float,
                 z_window: int, start: int, end: int, rebuild_every: int = 21) -> np.ndarray:
    """Residual z-scores for day positions [start, end] (inclusive).

    Returns a (end-start+1, N) array aligned with md.tickers; NaN where a stock
    was not in the graph or lacked history.
    """
    key = (lookback, round(threshold, 4), alpha, z_window, start, end, rebuild_every)
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key]

    T, N = md.rets.shape
    Z = np.full((end - start + 1, N), np.nan)
    grid = rebuild_positions(T, lookback, rebuild_every)
    # Blocks: [grid[k], grid[k+1]) — only those overlapping [start, end].
    for k, s in enumerate(grid):
        e = grid[k + 1] if k + 1 < len(grid) else T      # exclusive
        if e <= start or s > end:
            continue
        win = md.rets[s - lookback:s]                        # strictly before s
        enough = np.isfinite(win).sum(axis=0) >= MIN_HISTORY_SHARE * lookback
        cols = np.where(md.member[s] & enough)[0]
        if len(cols) < MIN_NAMES:
            continue
        W = build_weights(win[:, cols], threshold)

        first_needed = max(s, start)
        last_needed = min(e - 1, end)
        h0 = max(0, first_needed - z_window)                 # z history
        X = md.rets[h0:last_needed + 1][:, cols]
        R = residuals_for_block(X, W, alpha)

        # Prior-only rolling stats: shift(1) so day t is compared with t-1…t-z.
        Rdf = pd.DataFrame(R)
        prior = Rdf.shift(1).rolling(z_window, min_periods=z_window // 2)
        mu, sd = prior.mean().to_numpy(), prior.std().to_numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            zb = (R - mu) / sd
        zb[~np.isfinite(zb)] = np.nan

        rows = np.arange(first_needed, last_needed + 1)
        Z[np.ix_(rows - start, cols)] = zb[rows - h0]

    _CACHE[key] = Z
    if len(_CACHE) > _CACHE_MAX:
        _CACHE.popitem(last=False)
    return Z
