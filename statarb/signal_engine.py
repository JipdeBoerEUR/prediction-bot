# signal_engine.py
"""
SignalEngine: Graph Signal Processing (GSP) Logic for Statistical Arbitrage
================================================================================
Implements Laplacian Diffusion to generate trading signals based on market topology.

Core Logic (Laplacian Diffusion):
    h = (I - alpha * L)^(-1) * x
    r = x - h

Where:
    x: Actual returns vector (selected date; default: most recent)
    L: Graph Laplacian (combinatorial) = D - W
    h: Expected returns (neighbor-implied)
    r: Residual (signal)

Interpretation:
    - r < 0  => underperformed neighbors => BUY (mean-reversion)
    - r > 0  => outperformed neighbors  => SELL (mean-reversion)

This module is designed to work with the MarketGraph implementation in graph_engine.py.

Author: Senior Quantitative Developer Team
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Prefer SciPy for a more stable linear solve; fall back to NumPy if unavailable.
try:
    import scipy.linalg as la  # type: ignore
    _HAVE_SCIPY = True
except Exception:
    la = None  # type: ignore
    _HAVE_SCIPY = False


@dataclass
class SignalEngine:
    """
    Computes topological arbitrage signals using Graph Laplacian Diffusion.

    Parameters
    ----------
    graph_engine:
        An initialized MarketGraph instance (from graph_engine.py) with:
          - adjacency_ : pd.DataFrame (N x N)
          - returns_   : pd.DataFrame (T x N)

        Requirement: graph_engine.build_graph(...) must have been run so adjacency_ is set.
    """

    graph_engine: Any  # typed as Any to avoid circular import constraints

    def __post_init__(self) -> None:
        """Validate that the graph engine has the necessary data."""
        if not hasattr(self.graph_engine, "adjacency_") or self.graph_engine.adjacency_.empty:
            raise ValueError("MarketGraph has no adjacency matrix. Run build_graph() first.")
        if not hasattr(self.graph_engine, "returns_") or self.graph_engine.returns_.empty:
            raise ValueError("MarketGraph has no return data. Run fetch_data() first.")

    def run_diffusion(
        self,
        alpha: float = 0.5,
        regularization: float = 1e-6,
        as_of: Optional[Union[str, pd.Timestamp]] = None,
        use_zscore_strength: bool = True,
        min_assets: int = 2,
    ) -> pd.DataFrame:
        """
        Execute Laplacian diffusion to extract residual signals for a given date.

        Solves:
            (I - alpha * L + regularization * I) h = x

        Parameters
        ----------
        alpha:
            Diffusion coefficient. Practical range (0, 1) typically.
        regularization:
            Small diagonal ridge to improve conditioning for sparse/disconnected graphs.
        as_of:
            Date for which to compute signals. If None, uses most recent returns row.
        use_zscore_strength:
            If True, Signal_Strength = |zscore(residuals)|. Else |residual|.
        min_assets:
            Minimum number of tickers required after alignment.

        Returns
        -------
        pd.DataFrame
            Columns:
              - Ticker
              - Actual_Return
              - Expected_Return
              - Residual
              - Signal_Strength
            Sorted by Residual ascending (most negative first => BUY side).
        """
        if not (0 < alpha < 1):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")

        W_df: pd.DataFrame = self.graph_engine.adjacency_.copy()
        rets: pd.DataFrame = self.graph_engine.returns_.copy()

        if as_of is None:
            x_series = rets.iloc[-1]
            used_date = rets.index[-1]
        else:
            used_date = pd.Timestamp(as_of)
            # Use the last available row up to as_of
            rets2 = rets.loc[:used_date]
            if rets2.empty:
                raise ValueError(f"No returns available on or before {used_date}.")
            x_series = rets2.iloc[-1]
            used_date = rets2.index[-1]

        # Align tickers: latest returns available AND present in adjacency
        valid_tickers = x_series.dropna().index.intersection(W_df.index)

        if len(valid_tickers) < min_assets:
            raise ValueError(
                f"Not enough overlapping tickers between returns and adjacency "
                f"(got {len(valid_tickers)}, need >= {min_assets})."
            )

        # Subset to aligned universe
        x = x_series.loc[valid_tickers].to_numpy(dtype=float)
        W = W_df.loc[valid_tickers, valid_tickers].to_numpy(dtype=float)

        # Combinatorial Laplacian: L = D - W
        D = np.diag(np.sum(W, axis=1))
        L = D - W

        n = len(valid_tickers)
        I = np.eye(n)

        # System matrix
        A = I - (alpha * L) + (regularization * I)

        # Solve for expected returns h
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

        residuals = x - h

        if use_zscore_strength:
            # M4: use rolling historical std (last lookback_window bars) rather than
            # cross-sectional std of this single bar — gives a stable, non-leaking denominator.
            # We reuse the same adjacency snapshot W for every bar in the window.
            hist_window = rets[valid_tickers].loc[:used_date].tail(252).dropna(how="any")
            hist_resids: list[float] = []
            if len(hist_window) >= 10:
                n_v = len(valid_tickers)
                I_v = np.eye(n_v)
                D_v = np.diag(W.sum(axis=1))
                L_v = D_v - W
                A_v = I_v - (alpha * L_v) + (regularization * I_v)
                for row_vals in hist_window.to_numpy():
                    if not np.all(np.isfinite(row_vals)):
                        continue
                    try:
                        if _HAVE_SCIPY:
                            h_tmp = la.solve(A_v, row_vals)  # type: ignore[misc]
                        else:
                            h_tmp = np.linalg.solve(A_v, row_vals)
                        hist_resids.extend((row_vals - h_tmp).tolist())
                    except (np.linalg.LinAlgError, ValueError):
                        pass
            sig_std = float(np.std(hist_resids)) if len(hist_resids) >= 2 else float(np.std(residuals))
            strength = np.abs(residuals / sig_std) if sig_std > 1e-12 else np.zeros_like(residuals)
        else:
            strength = np.abs(residuals)

        df = pd.DataFrame(
            {
                "Ticker": valid_tickers,
                "Actual_Return": x,
                "Expected_Return": h,
                "Residual": residuals,
                "Signal_Strength": strength,
            }
        ).sort_values("Residual", ascending=True).reset_index(drop=True)

        # Store metadata (non-breaking)
        df.attrs["as_of"] = used_date

        return df

    @staticmethod
    def get_signal(df: pd.DataFrame, ticker: str) -> Optional[pd.Series]:
        """
        Retrieve a single ticker's signal row from a diffusion output DataFrame.

        Parameters
        ----------
        df:
            Output of run_diffusion().
        ticker:
            Symbol to query.

        Returns
        -------
        Optional[pd.Series]
            Row if present, else None.
        """
        if df.empty:
            return None
        out = df[df["Ticker"] == ticker]
        return None if out.empty else out.iloc[0]

    @staticmethod
    def plot_signals(df: pd.DataFrame, top_n: int = 5, title: Optional[str] = None) -> None:
        """
        Plot top undervalued (buy) and overvalued (sell) signals.

        Parameters
        ----------
        df:
            Output from run_diffusion().
        top_n:
            Number of tickers shown on each side.
        title:
            Optional plot title.
        """
        if df.empty:
            raise ValueError("df is empty.")
        if top_n <= 0:
            raise ValueError("top_n must be positive.")

        max_n = len(df) // 2
        if max_n <= 0:
            raise ValueError("Not enough tickers to plot both buy and sell sides.")
        top_n = min(top_n, max_n)

        top_buy = df.head(top_n).copy()
        top_sell = df.tail(top_n).copy()
        plot_data = pd.concat([top_buy, top_sell], axis=0)

        colors = ["seagreen" if r < 0 else "crimson" for r in plot_data["Residual"].to_numpy(dtype=float)]

        fig_h = max(6, int(0.6 * len(plot_data)) + 2)
        plt.figure(figsize=(10, fig_h))
        bars = plt.barh(plot_data["Ticker"], plot_data["Residual"], color=colors, alpha=0.85)
        plt.axvline(0, color="black", linewidth=0.8, linestyle="--")
        plt.grid(axis="x", linestyle=":", alpha=0.6)

        as_of = df.attrs.get("as_of", None)
        if title is None:
            title = f"Top {top_n} Mean-Reversion Signals (Laplacian Diffusion)"
            if as_of is not None:
                title += f" | as_of={pd.Timestamp(as_of).date()}"
        plt.title(title, fontsize=13, fontweight="bold")
        plt.xlabel("Residual (Actual - Expected)\n<-- Undervalued (BUY) | Overvalued (SELL) -->")

        # Annotate
        for bar in bars:
            w = bar.get_width()
            y = bar.get_y() + bar.get_height() / 2
            ha = "left" if w > 0 else "right"
            x_pos = w * 1.03 if w != 0 else 0.0
            plt.text(x_pos, y, f"{w:.5f}", va="center", ha=ha, fontsize=8)

        plt.tight_layout()
        plt.show()


# =============================================================================
# Integration test (expects graph_engine.py in the same project)
# =============================================================================
if __name__ == "__main__":
    # Example usage:
    #   1) Build the graph
    #   2) Run diffusion on latest returns
    from graph_engine import MarketGraph  # local import to keep module modular

    tickers = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "JPM", "BAC", "XOM", "CVX"]

    mg = MarketGraph(tickers=tickers, start="2023-01-01", end=None, fill_limit=3)
    mg.fetch_data(verbose=True)
    mg.fetch_sectors(exclude_unknown=False, verbose=True)
    mg.build_graph(lookback_window=22, sector_only=True, threshold=0.3, verbose=True)

    engine = SignalEngine(graph_engine=mg)
    signals = engine.run_diffusion(alpha=0.5, regularization=1e-6, use_zscore_strength=True)

    print("\nSignals (sorted by Residual):")
    print(signals.to_string(index=False, formatters={
        "Actual_Return": "{:,.6f}".format,
        "Expected_Return": "{:,.6f}".format,
        "Residual": "{:,.6f}".format,
        "Signal_Strength": "{:,.2f}".format,
    }))

    engine.plot_signals(signals, top_n=3)
