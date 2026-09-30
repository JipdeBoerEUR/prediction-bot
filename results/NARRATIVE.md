**Bottom line: out of sample, the strategy does not beat its costs, and it
does not beat either benchmark.** Over 2019-01 → 2026-09, with parameters
re-chosen each year on the previous 3 years only, it returned **−1.8% a year
(Sharpe −0.16)** at base costs and **−15.8% a year** at stress costs. Holding
the same stocks equal-weighted earned +14.3% a year; SPY earned +17.4%. This
is a negative result. It is reported as it came out, with no re-tuning after
seeing it.

**What worked**

- **There is a small real-looking signal before costs.** Re-trading the exact
  same trades with zero costs gives +5.9% a year, Sharpe 0.69 (t ≈ 1.9, just
  short of the usual significance bar). The average trade earns **+8.5 bps**
  before costs. Stocks that lagged their correlated peers did tend to catch
  up, mostly on the long side (+15.6 bps per long trade vs +1.4 bps per
  short).
- **It really is market-neutral.** Beta to SPY is 0.01 and the correlation is
  0.02. It kept its losses small in the 2022 bear market (−1.1% vs SPY
  −18.2%), partly because the regime filter kept it out of the market on
  18.5% of all days.
- **The engineering held up.** There is no look-ahead (tested). Membership is
  point-in-time. Costs are modelled at two levels. The results are
  reproducible from committed data with SHA-256 hashes. 256 Optuna trials ran
  in 3.4 minutes on 4 cores.

**What did not work**

- **Costs kill it.** The strategy holds positions for only 1.7 days on
  average, so it turns the book over ~76× a year. A round trip costs 10 bps
  at base, more than the 8.5 bps average gross gain, and base costs totalled
  about $57k on $100k of capital over the period. Break-even is about
  4 bps per side, which is only realistic for an institution with very cheap
  execution.
- **The edge is not stable over time.** Only 3 of 8 test years were positive
  (2020, 2023, 2024). In every year, all ~12 neighbouring parameter sets had
  the **same sign** as the chosen one. The outcome was driven by the market
  regime of that year, not by the parameter choice: parameters barely matter,
  but the year does.
- **The in-sample optimisation was not convincing.** The chosen parameters
  averaged Sharpe 0.34 in training. The Deflated Sharpe Ratio (which corrects
  for trying 32 combinations) was at most 0.44 and never close to 0.95. In
  the first window, even the *best* training result was negative, and the
  strategy still traded. The parameters also jumped around from window to
  window: lookback 126 → 504 → 252, and alpha from 2.0 down to 0.25.
- **The search had not converged in-sample.** In 6 of 8 windows the best
  training score was still improving after trial 16 (latest at trial 31). More
  trials would likely raise the *in-sample* Sharpe. Given the neighbour and
  DSR evidence above, I did not expect that to fix the out-of-sample result,
  and I did not run more trials to find out.
- **The last 1.5 years were bad.** 2025 returned −9.6% and 2026 year-to-date
  −17.1%, with a −31% maximum drawdown in mid-2026. This happened even before
  costs (−1.1% and −11.7%).

**How to read the upward biases.** 14% of index member-days have no price data,
mostly companies that were delisted or acquired. The price-only strategy
therefore saw a friendlier universe than reality. Because the result is
negative even with that tailwind, the conclusion "does not survive costs"
holds.
