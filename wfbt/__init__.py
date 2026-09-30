"""wfbt — walk-forward backtest of the graph stat-arb strategy.

Modules
-------
data     load data/prices.parquet + data/membership.parquet into aligned panels
signals  correlation graph → peer-implied return → residual → z-score panel
sim      day-by-day trading simulator (next-open fills, stops, costs)
metrics  performance statistics (Sharpe, drawdown, deflated Sharpe, ...)

The driver scripts are run_walkforward.py (optimise + validate) and
make_report.py (REPORT.md + charts) in the repo root.
"""
