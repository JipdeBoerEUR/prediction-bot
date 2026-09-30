# CV bullets and interview sheet

All figures come from `results/REPORT.md` / `results/summary.json`
(walk-forward backtest, **not live trading**). Out-of-sample period
2019-01-02 → 2026-09-30. Universe: point-in-time S&P 500 members with Yahoo
Finance prices (86% of member-days; delisted names missing). Costs: base = 5 bp
per side + 0.5%/yr short borrow; stress = 15 bp per side + 2%/yr borrow.

## CV bullet options (pick one or two; all are strictly true)

**A. Engineering / method**
> Built a walk-forward backtesting and optimisation pipeline in Python (Optuna
> TPE with pruning, parallel, checkpointed) for a graph-based statistical-
> arbitrage strategy on point-in-time S&P 500 constituents. It ran 256 trials
> across 8 rolling 3-year-train / 1-year-test windows, and reports only
> out-of-sample results (2019–Sep 2026, backtest).

**B. Result, stated honestly**
> Backtested (out-of-sample, 2019–Sep 2026, S&P 500 point-in-time universe) a
> market-neutral long/short stat-arb strategy (beta 0.01). It earned +5.9%/yr
> (Sharpe 0.69) before costs but −1.8%/yr (Sharpe −0.16) after 5 bp/side costs
> and borrow fees. I identified ~76× annual turnover as the cause and
> documented the negative result.

**C. Research rigour / bias control**
> Audited a trading bot's backtest and fixed look-ahead and selection biases:
> a full-period data filter, a z-score entry/exit unit mismatch, and
> transaction costs being tuned by the optimiser. Added point-in-time index
> membership, a ticker-reuse data screen, two cost levels, the Deflated Sharpe
> Ratio and neighbour-parameter robustness tests.

Don'ts: don't say "profitable", "live", "alpha", or quote the zero-cost
+5.9% without "before costs".

---

## One-page interview sheet

**30-second pitch.** "I tested whether a stock that under-performs its
correlated peers tends to catch up the next few days. I built a correlation
network of S&P 500 stocks, estimated each stock's 'expected' return from its
neighbours, and traded the biggest deviations long/short. I optimised
parameters only on past data and tested on the following unseen year, eight
times, from 2019 to 2026. Before costs there was a small edge, Sharpe 0.7.
After realistic costs it lost 1.8% a year, because the positions only lasted
about 1.7 days and turnover ate the edge. The main things I learned were bias
control and that costs, not the signal, decide these strategies."

**Method in 5 steps.**
1. *Data.* Daily adjusted prices, 2012–2026, from Yahoo. S&P 500 membership
   history from a public dataset, so a stock is only tradable while it was
   actually in the index. I screened out reused tickers (e.g. "EP" is a
   different company today) and one bad spin-off adjustment (DHR 2016).
2. *Signal.* Every 21 days, link stocks whose 1-year return correlation ≥
   threshold. Each day, solve `(I + αL)h = x` (L = graph Laplacian) to get the
   peer-implied return h. Residual = actual − h. Z-score it against the
   stock's previous 40–120 residuals.
3. *Trading.* At the close, buy z ≤ −entry and short z ≥ +entry, filled at the
   next open. Max 5 long + 5 short positions at 10% each. Exit when |z| <
   exit_z, on a volatility-scaled stop, or after max_hold days. No new trades
   when SPY is below its 200-day average or VIX ≥ 30.
4. *Optimisation.* Optuna TPE: 12 random + 20 guided trials per window,
   maximising in-sample Sharpe net of base costs. Trials below the median
   after year 1 or 2 are pruned. 8 windows run on 4 cores, 3.4 minutes total.
5. *Validation.* Stitched out-of-sample equity at two cost levels, compared
   with SPY and an equal-weight portfolio of the same universe. Also results by
   year, ±1-step neighbour parameters, and the Deflated Sharpe Ratio.

**Numbers to know by heart.** Out-of-sample: −1.8%/yr, Sharpe −0.16, max
drawdown −31%, 5,777 trades, 55% net win rate, 1.7-day average hold, ~76×
turnover, beta 0.01. Before costs: +5.9%/yr, Sharpe 0.69 (t ≈ 1.9). Gross gain
8.5 bp per trade vs 10 bp round-trip cost. SPY: +17.4%/yr. 256 trials. Positive
in 3 of 8 years.

**Likely questions and short answers.**
- *Why is the out-of-sample result credible?* Parameters for each test year
  were chosen using only the 3 prior years. The graph uses only past returns.
  Trades fill at the next open. A unit test changes future prices and checks
  that past signals don't move.
- *What about overfitting / multiple testing?* 256 trials in total. The best
  in-sample Sharpe per window had a Deflated Sharpe Ratio of at most 0.44, so
  none of the in-sample "winners" was statistically convincing. Out of sample,
  that showed up.
- *Is the result driven by one lucky parameter set?* No. For every year, all
  one-step neighbours had the same sign as the chosen set. The year mattered,
  the parameters didn't.
- *Survivorship bias?* Partly fixed (point-in-time membership). Delisted
  companies (14% of member-days) are still missing, which flatters the
  results. The conclusion is negative anyway, so the bias doesn't change it.
- *Why long/short, and what about sentiment?* Long/short removes market
  direction (beta 0.01). The live bot's news-sentiment filter could not be
  backtested because no historical headline archive exists. I said so rather
  than simulate one.
- *Why use Sharpe with a 0% risk-free rate?* No T-bill data was available. For
  a strategy with a negative Sharpe this doesn't change the conclusion; I
  would use T-bill excess returns with proper data.

**Limitations.** Survivorship (delisted names missing). Flat-bps cost model
(no market-impact curve, no hard-to-borrow names). Open-price fills assumed
achievable. Daily bars only. The research configuration differs from the live
bot. A 0% risk-free rate. The search space and design were chosen by someone
who knows 2019–2026 market history.

**What I'd do next** (pre-registered before testing, to avoid tuning on the
test set):
1. **Cut turnover.** Enter only on larger z, hold longer / exit at z = 0,
   rebalance weekly, and optimise *net of stress costs* with a turnover
   penalty.
2. **Sit out when training is unconvincing.** Don't trade a window whose best
   in-sample Sharpe is < 0 or whose DSR is low, as in 2019.
3. **Better data.** CRSP via WRDS (a university licence) for delisted stocks;
   a historical news dataset (e.g. FNSPID) to test the sentiment filter.
4. **Sector-neutral graph** (links only within an industry) and a proper
   market-impact model.
