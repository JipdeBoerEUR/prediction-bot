# regime_engine.py
"""
RegimeEngine: Topological Data Analysis (TDA) for Market Regime / Stress Detection
================================================================================

This module implements persistent homology features on a market correlation
structure to infer a single scalar "Market Health Score" suitable for regime
gating (Trade ON vs Trade OFF).

Core Pipeline (Persistent Homology):
    1) Convert correlation matrix C to distance matrix D via:
           D = sqrt(2 * (1 - C))
    2) Build a Vietoris–Rips filtration over D (precomputed distances)
    3) Compute persistence diagrams for H0 and H1
    4) Aggregate diagram complexity into a scalar score using (preferred):
           - gtda.diagrams.Amplitude  (Wasserstein amplitude)
       with fallbacks to:
           - gudhi RipsComplex + "total persistence" proxy
           - simple spectral/dispersion proxy (no PH)

Interpretation (heuristic):
    - Higher complexity (higher amplitude) => diversified structure => "healthier" market
    - Collapse to low complexity => panic/high global correlation => "stress" market

Dependencies:
    Preferred: giotto-tda
    Fallback:  gudhi
    Minimal:   numpy, pandas, matplotlib (plotting is optional)

Notes:
    - Designed to accept either a raw correlation matrix (numpy array / DataFrame)
      or a MarketGraph-like object providing returns_ (pandas DataFrame).

References:
    - See the three prompt variants merged into this implementation:
      version 1, 2, 3 (user-provided).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

ArrayLike2D = Union[np.ndarray, pd.DataFrame]

# -----------------------------------------------------------------------------
# Preferred backend: giotto-tda
# -----------------------------------------------------------------------------
try:
    from gtda.homology import VietorisRipsPersistence  # type: ignore
    from gtda.diagrams import Amplitude  # type: ignore
    from gtda.plotting import plot_diagram  # type: ignore

    _GTDA_AVAILABLE = True
except Exception:
    VietorisRipsPersistence = None  # type: ignore
    Amplitude = None  # type: ignore
    plot_diagram = None  # type: ignore
    _GTDA_AVAILABLE = False

# -----------------------------------------------------------------------------
# Secondary fallback: gudhi
# -----------------------------------------------------------------------------
try:
    import gudhi  # type: ignore

    _GUDHI_AVAILABLE = True
except Exception:
    gudhi = None  # type: ignore
    _GUDHI_AVAILABLE = False


def _to_numpy_corr(corr: ArrayLike2D) -> np.ndarray:
    """Validate and convert correlation input to a float numpy array (N, N)."""
    if isinstance(corr, pd.DataFrame):
        C = corr.to_numpy(dtype=float)
    elif isinstance(corr, np.ndarray):
        C = corr.astype(float, copy=False)
    else:
        raise TypeError("correlation_matrix must be a numpy array or pandas DataFrame.")

    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError(f"correlation_matrix must be square (N x N). Got shape {C.shape}.")

    # Symmetrize to be robust to minor numerical asymmetry.
    C = 0.5 * (C + C.T)

    # Clip to valid correlation bounds and enforce diagonal.
    C = np.clip(C, -1.0, 1.0)
    np.fill_diagonal(C, 1.0)

    return C


def correlation_to_distance(C: np.ndarray) -> np.ndarray:
    """
    Convert correlation matrix to Euclidean distance matrix:
        D = sqrt(2 * (1 - C))

    Ensures:
      - 0 diagonal
      - symmetry
      - numerical stability (guard against tiny negatives under the sqrt)
    """
    inside = np.maximum(0.0, 2.0 * (1.0 - C))
    D = np.sqrt(inside)
    np.fill_diagonal(D, 0.0)
    D = 0.5 * (D + D.T)
    return D


def _total_persistence(intervals: np.ndarray) -> float:
    """Sum of finite lifetimes as a lightweight amplitude proxy."""
    if intervals.size == 0:
        return 0.0
    lifetimes = intervals[:, 1] - intervals[:, 0]
    lifetimes = lifetimes[np.isfinite(lifetimes)]
    lifetimes = lifetimes[lifetimes > 0]
    return float(lifetimes.sum()) if lifetimes.size else 0.0


def _effective_rank(C: np.ndarray) -> float:
    """Effective rank of a PSD-like matrix via eigenvalue entropy."""
    evals = np.linalg.eigvalsh(0.5 * (C + C.T))
    evals = np.maximum(evals, 0.0)
    s = float(evals.sum())
    if s <= 0:
        return 0.0
    p = evals / s
    p = p[p > 0]
    ent = float(-(p * np.log(p)).sum())
    return float(np.exp(ent))


@dataclass
class RegimeEngine:
    """
    TDA-based regime detector using persistent homology.

    Parameters
    ----------
    market_graph:
        Optional MarketGraph-like object. If provided, and correlation_matrix is None,
        compute_regime will compute correlations from market_graph.returns_.
    homology_dimensions:
        Homology dimensions to compute (default: (0, 1)).
    amplitude_metric:
        gtda Amplitude metric. For the requested behavior, use "wasserstein".
    p:
        Wasserstein p parameter (default 2).
    n_jobs:
        Parallel workers for gtda (default -1).
    gudhi_max_edge_length:
        For gudhi fallback: max edge length in RipsComplex. If None, a robust default
        is estimated from the distance matrix.
    """

    market_graph: Optional[Any] = None
    homology_dimensions: Sequence[int] = (0, 1)
    amplitude_metric: str = "wasserstein"
    p: int = 2
    n_jobs: int = -1
    gudhi_max_edge_length: Optional[float] = None

    backend_: str = field(init=False, default="none")
    vr_: Any = field(init=False, default=None)
    amp_: Any = field(init=False, default=None)

    last_distance_: Optional[np.ndarray] = field(init=False, default=None)
    last_diagrams_: Optional[np.ndarray] = field(init=False, default=None)  # gtda format: (1, n_points, 3)
    last_score_: Optional[float] = field(init=False, default=None)

    def __post_init__(self) -> None:
        if _GTDA_AVAILABLE:
            self.backend_ = "gtda"
            self.vr_ = VietorisRipsPersistence(
                metric="precomputed",
                homology_dimensions=tuple(self.homology_dimensions),
                n_jobs=self.n_jobs,
            )
            # gtda.diagrams.Amplitude: return shape (n_samples, n_dims)
            self.amp_ = Amplitude(metric=self.amplitude_metric, n_jobs=self.n_jobs)
        elif _GUDHI_AVAILABLE:
            self.backend_ = "gudhi"
        else:
            self.backend_ = "simple"

    def _corr_from_market_graph(
        self,
        lookback_window: int = 22,
        as_of: Optional[Union[str, pd.Timestamp]] = None,
        min_assets: int = 3,
        missing: str = "drop",
    ) -> np.ndarray:
        """
        Compute a correlation matrix from self.market_graph.returns_ (pandas DataFrame).

        missing:
            - "drop": drop columns with any NaNs in the window (default)
            - "ffill": forward-fill then drop remaining NaNs
        """
        if self.market_graph is None:
            raise ValueError("No market_graph provided; correlation_matrix must be passed explicitly.")

        if not hasattr(self.market_graph, "returns_"):
            raise ValueError("market_graph must expose a pandas DataFrame attribute returns_.")

        rets = self.market_graph.returns_
        if not isinstance(rets, pd.DataFrame) or rets.empty:
            raise ValueError("market_graph.returns_ must be a non-empty pandas DataFrame.")

        if as_of is None:
            window = rets.tail(int(lookback_window))
        else:
            ts = pd.Timestamp(as_of)
            rets2 = rets.loc[:ts]
            if rets2.empty:
                raise ValueError(f"No returns available on or before {ts}.")
            window = rets2.tail(int(lookback_window))

        if missing == "ffill":
            window = window.ffill()

        # Keep a complete panel
        window = window.dropna(axis=1, how="any")

        if window.shape[1] < min_assets:
            raise ValueError(f"Not enough assets after NaN handling (got {window.shape[1]}, need >= {min_assets}).")

        return _to_numpy_corr(window.corr())

    def compute_regime(
        self,
        correlation_matrix: Optional[ArrayLike2D] = None,
        *,
        lookback_window: int = 22,
        as_of: Optional[Union[str, pd.Timestamp]] = None,
        min_assets: int = 3,
        aggregate: str = "mean",
    ) -> float:
        """
        Compute the Market Health Score (float). Higher = more topological complexity.

        If correlation_matrix is None, compute it from market_graph.returns_.

        aggregate:
            - "mean": mean score across homology dimensions (default)
            - "sum" : sum score across homology dimensions
        """
        aggregate = aggregate.lower().strip()
        if aggregate not in {"mean", "sum"}:
            raise ValueError("aggregate must be 'mean' or 'sum'.")

        if correlation_matrix is None:
            C = self._corr_from_market_graph(lookback_window=lookback_window, as_of=as_of, min_assets=min_assets)
        else:
            C = _to_numpy_corr(correlation_matrix)

        if C.shape[0] < min_assets:
            raise ValueError(f"Correlation matrix too small (N={C.shape[0]}). Need >= {min_assets}.")

        D = correlation_to_distance(C)
        self.last_distance_ = D

        # ------------------------------------------------------------------
        # Backend: giotto-tda (preferred)
        # ------------------------------------------------------------------
        if self.backend_ == "gtda":
            # gtda expects (n_samples, n_points, n_points) when metric="precomputed"
            X = D[None, :, :]
            diagrams = self.vr_.fit_transform(X)  # (1, n_features, 3)
            self.last_diagrams_ = diagrams

            scores = self.amp_.fit_transform(diagrams)  # (1, n_dims)
            s = scores[0]
            score = float(np.mean(s) if aggregate == "mean" else np.sum(s))

            self.last_score_ = score
            return score

        # ------------------------------------------------------------------
        # Backend: gudhi fallback
        # ------------------------------------------------------------------
        if self.backend_ == "gudhi":
            max_len = self.gudhi_max_edge_length
            if max_len is None:
                off = D[~np.eye(D.shape[0], dtype=bool)]
                max_len = float(np.percentile(off, 80)) if off.size else 1.0

            rips = gudhi.RipsComplex(distance_matrix=D, max_edge_length=max_len)  # type: ignore[attr-defined]
            st = rips.create_simplex_tree(max_dimension=max(self.homology_dimensions) + 1)
            st.compute_persistence()

            parts = []
            for dim in self.homology_dimensions:
                intervals = np.asarray(st.persistence_intervals_in_dimension(dim), dtype=float)
                parts.append(_total_persistence(intervals))

            score = float(np.mean(parts) if aggregate == "mean" else np.sum(parts))
            self.last_score_ = score
            self.last_diagrams_ = None
            return score

        # ------------------------------------------------------------------
        # Backend: simple proxy (no PH)
        # ------------------------------------------------------------------
        off = C[~np.eye(C.shape[0], dtype=bool)]
        off = off[np.isfinite(off)]
        disp = float(np.std(off)) if off.size else 0.0
        score = float(_effective_rank(C) * disp)
        self.last_score_ = score
        self.last_diagrams_ = None
        return score

    def plot_persistence(self) -> None:
        """
        Plot the persistence diagram (H0 and H1 points).

        Requested behavior:
          - Uses gtda.plotting.plot_diagram, plotting points with (birth, death, dim).
        """
        if self.backend_ != "gtda":
            raise RuntimeError(
                f"plot_persistence requires giotto-tda. Current backend is '{self.backend_}'. "
                "Install giotto-tda (preferred) to enable persistent diagram plotting."
            )
        if self.last_diagrams_ is None:
            raise ValueError("No persistence diagram available. Run compute_regime() first.")

        fig = plot_diagram(self.last_diagrams_[0])
        # Most gtda versions return a plotly figure; attempt to show.
        try:
            fig.show()
        except Exception:
            # If running in a non-interactive context, just return the fig implicitly.
            return


if __name__ == "__main__":
    # Minimal smoke test on synthetic correlation matrices.
    rng = np.random.default_rng(7)
    n = 12

    # "Healthy": more diverse correlations
    C1 = rng.uniform(0.1, 0.6, size=(n, n))
    C1 = 0.5 * (C1 + C1.T)
    np.fill_diagonal(C1, 1.0)

    # "Stress": high global correlation (collapse)
    C2 = rng.uniform(0.85, 0.99, size=(n, n))
    C2 = 0.5 * (C2 + C2.T)
    np.fill_diagonal(C2, 1.0)

    engine = RegimeEngine()
    s1 = engine.compute_regime(C1)
    s2 = engine.compute_regime(C2)

    print(f"Backend: {engine.backend_}")
    print(f"Healthy score: {s1:.6f}")
    print(f"Stress  score: {s2:.6f}")
