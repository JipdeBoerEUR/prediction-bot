"""sim_engine.py

Low-level computation building blocks for Topological Arbitrage simulation and
dataset building. All functions are pure (no shared state) so they work safely
inside joblib parallel workers.

Provides
--------
build_adjacency_from_returns  : correlation-based W matrix (N x N)
solve_diffusion_residual       : Laplacian diffusion residual vector
compute_residual_panel         : residual vectors for every bar in a date list
compute_degree_panel           : node degree vectors for every bar in a date list
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

try:
    import scipy.linalg as la  # type: ignore

    _HAVE_SCIPY = True
except Exception:
    la = None  # type: ignore
    _HAVE_SCIPY = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sector_mask(sectors: pd.Series) -> pd.DataFrame:
    """Binary mask M[i,j]=1 iff tickers share the same *known* sector.

    Tickers whose sector is "Unknown" are treated as having no sector and will
    not form edges with any other ticker, including other Unknowns.
    """
    s = sectors.to_numpy(dtype=object)
    mask = (s[:, None] == s[None, :]).astype(float)
    # Suppress Unknown-Unknown matches
    unknown = s == "Unknown"
    both_unknown = unknown[:, None] | unknown[None, :]
    mask = np.where(both_unknown, 0.0, mask)
    return pd.DataFrame(mask, index=sectors.index, columns=sectors.index)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_adjacency_from_returns(
    window_returns: pd.DataFrame,
    *,
    sectors: Optional[pd.Series] = None,
    sector_only: bool = False,
    threshold: float = 0.3,
) -> pd.DataFrame:
    """Build a correlation-based adjacency matrix W.

    Parameters
    ----------
    window_returns:
        (T x N) log-return panel for the lookback window.
    sectors:
        Per-ticker sector labels.  Required when *sector_only=True*.
    sector_only:
        Only retain edges between tickers in the same sector.
    threshold:
        Minimum correlation required to retain an edge (noise filter).

    Returns
    -------
    pd.DataFrame
        (N x N) symmetric, non-negative adjacency matrix.
    """
    N = window_returns.shape[1]
    if window_returns.shape[0] < 2 or N < 2:
        return pd.DataFrame(
            np.zeros((N, N)),
            index=window_returns.columns,
            columns=window_returns.columns,
        )

    corr = window_returns.corr()
    np.fill_diagonal(corr.values, 0.0)

    if sector_only and sectors is not None:
        s = sectors.reindex(corr.columns).fillna("Unknown")
        M = _sector_mask(s)
        corr = corr * M

    W = corr.where(corr >= threshold, 0.0)
    np.fill_diagonal(W.values, 0.0)
    W = (W + W.T) / 2.0  # enforce exact symmetry
    return W.clip(lower=0.0)


def solve_diffusion_residual(
    x: np.ndarray,
    W: np.ndarray,
    *,
    alpha: float = 0.5,
    regularization: float = 1e-6,
) -> np.ndarray:
    """Laplacian diffusion: solve (I + alpha*L + reg*I)h = x, return x - h.

    Parameters
    ----------
    x:
        (N,) actual returns at this bar.
    W:
        (N x N) adjacency matrix aligned with *x*.
    alpha:
        Diffusion coefficient in (0, 1).
    regularization:
        Small diagonal ridge for numerical stability with sparse/disconnected graphs.

    Returns
    -------
    np.ndarray
        (N,) residual vector  r = x - h.
    """
    n = len(x)
    D = np.diag(W.sum(axis=1))
    L = D - W
    # (I + αL) is SPD → stable solve, true smoothing. See signal_engine.py for
    # why the previous (I − αL) form produced sign-flipped residuals.
    A = np.eye(n) + alpha * L + regularization * np.eye(n)

    if _HAVE_SCIPY:
        try:
            h = la.solve(A, x)  # type: ignore[misc]
        except np.linalg.LinAlgError:
            h = np.linalg.pinv(A) @ x
    else:
        try:
            h = np.linalg.solve(A, x)
        except np.linalg.LinAlgError:
            h = np.linalg.pinv(A) @ x

    return x - h


def compute_residual_panel(
    returns: pd.DataFrame,
    *,
    lookback_days: int,
    sectors: Optional[pd.Series] = None,
    sector_only: bool = False,
    graph_threshold: float = 0.3,
    alpha: float = 0.5,
    regularization: float = 1e-6,
    trade_dates: Optional[Sequence] = None,
    n_jobs: int = 1,
    backend: str = "threading",
) -> pd.DataFrame:
    """Compute residual vectors for every bar in *trade_dates*.

    Parameters
    ----------
    returns:
        Full (T x N) log-return panel.
    lookback_days:
        Rolling window length for correlation / adjacency calculation.
    trade_dates:
        Subset of ``returns.index`` to compute.  Defaults to all dates.

    Returns
    -------
    pd.DataFrame
        Same shape as *returns*; NaN for bars skipped or before lookback is
        available.
    """
    idx = returns.index
    dates = list(idx) if trade_dates is None else list(trade_dates)

    def _bar(date):
        loc = idx.get_loc(date)
        if not isinstance(loc, int):
            loc = int(np.where(idx == date)[0][0])
        if loc < lookback_days - 1:
            return date, None
        window = returns.iloc[max(0, loc - lookback_days + 1) : loc + 1]
        x_full = returns.iloc[loc].to_numpy(dtype=float)
        valid = np.isfinite(x_full)
        if valid.sum() < 2:
            return date, None
        tickers_v = returns.columns[valid]
        W = build_adjacency_from_returns(
            window[tickers_v],
            sectors=sectors,
            sector_only=sector_only,
            threshold=graph_threshold,
        )
        r = solve_diffusion_residual(
            x_full[valid], W.to_numpy(dtype=float),
            alpha=alpha, regularization=regularization,
        )
        row = pd.Series(np.nan, index=returns.columns)
        row[tickers_v] = r
        return date, row

    use_parallel = n_jobs != 1 and len(dates) > 1
    if use_parallel:
        try:
            from joblib import Parallel, delayed  # type: ignore

            raw = Parallel(n_jobs=n_jobs, prefer=backend)(
                delayed(_bar)(d) for d in dates
            )
        except Exception:
            raw = [_bar(d) for d in dates]
    else:
        raw = [_bar(d) for d in dates]

    panel = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)
    for date, row in raw:
        if row is not None:
            panel.loc[date] = row
    return panel


def compute_degree_panel(
    returns: pd.DataFrame,
    *,
    lookback_days: int,
    sectors: Optional[pd.Series] = None,
    sector_only: bool = False,
    graph_threshold: float = 0.3,
    trade_dates: Optional[Sequence] = None,
    n_jobs: int = 1,
    backend: str = "threading",
) -> pd.DataFrame:
    """Compute node degree (number of edges) for every bar in *trade_dates*.

    Returns
    -------
    pd.DataFrame
        Same shape as *returns*; NaN for skipped bars.
    """
    idx = returns.index
    dates = list(idx) if trade_dates is None else list(trade_dates)

    def _bar(date):
        loc = idx.get_loc(date)
        if not isinstance(loc, int):
            loc = int(np.where(idx == date)[0][0])
        if loc < lookback_days - 1:
            return date, None
        window = returns.iloc[max(0, loc - lookback_days + 1) : loc + 1]
        W = build_adjacency_from_returns(
            window,
            sectors=sectors,
            sector_only=sector_only,
            threshold=graph_threshold,
        )
        Wv = W.to_numpy()
        np.fill_diagonal(Wv, 0.0)
        deg = (Wv > 0).sum(axis=1).astype(float)
        row = pd.Series(np.nan, index=returns.columns)
        row[W.columns] = deg
        return date, row

    use_parallel = n_jobs != 1 and len(dates) > 1
    if use_parallel:
        try:
            from joblib import Parallel, delayed  # type: ignore

            raw = Parallel(n_jobs=n_jobs, prefer=backend)(
                delayed(_bar)(d) for d in dates
            )
        except Exception:
            raw = [_bar(d) for d in dates]
    else:
        raw = [_bar(d) for d in dates]

    panel = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)
    for date, row in raw:
        if row is not None:
            panel.loc[date] = row
    return panel
