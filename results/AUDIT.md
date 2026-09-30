# Step 0: Recon and Bias Audit (branch `cloud-backtest`)

_Written 2026-09-30, before any optimisation or backtest runs. **No backtest
numbers exist yet.** The price-data source is blocked in the cloud
environment (see §2)._

---

## 1. What the strategy actually does (plain language)

The repo contains **two** strategies. They are easy to mix up:

| | Stat-arb sleeve ("Scout B") | Topic / news sleeve ("Scout A") |
|---|---|---|
| Idea | A stock that lagged the stocks it normally moves with will tend to catch up (mean reversion) | Stocks tied to a fast-growing, positive news theme keep rising (momentum) |
| Inputs | Daily prices only | Live RSS headlines, BERTopic clusters, FinBERT sentiment |
| Backtestable? | **Yes**, with price history | **No**: there is no historical archive of the headlines it uses |

The backtest (`backtest_walkforward.py`) covers **only the stat-arb sleeve**.

### Stat-arb, step by step

1. **"Pair" selection is really peer-group selection.** This is *not*
   classic two-stock pairs trading. Every ~21 trading days the code computes
   the correlation of daily returns between all stocks over the last 252 days.
   Any two stocks with correlation ≥ `graph_threshold` (0.45) are linked, and
   the link strength equals the correlation. The result is a network (graph)
   of related stocks.
2. **Expected return from peers.** Each day the code asks, for every stock,
   "given what its linked peers did today, what should it have done?" It
   answers with a graph-smoothing formula, `h = (I + αL)⁻¹ x`: a
   correlation-weighted blend of the stock's own return and its peers'
   returns. `α` (0.8) sets how strongly peers pull.
3. **Residual.** `residual = actual return − peer-implied return`. A negative
   residual means the stock lagged its peers today.
4. **Z-score.** The residual is standardised against the stock's own last 60
   days of residuals: `z = (residual − mean) / std`. So `z = −2` means "an
   unusually large lag for this stock".
5. **Entry.** At the close, if `z ≤ −entry_z` (−1.75) the stock is bought at
   the **next day's open**. There are at most 5 positions of 20% of equity
   each. The long-only mode is the default; the "dollar-neutral" flag also
   shorts `z ≥ +entry_z`.
6. **Exits** (whichever comes first):
   - **Mean reversion done:** z is back within `exit_z` (0.30) of zero, so the
     position is closed at the next open.
   - **Stop-loss:** 3 × the stock's daily volatility (clamped to 4–15%),
     checked against the day's low.
   - **Trailing stop:** once in profit, the position is closed if it gives
     back one stop-width from its best price.
   - **Max hold:** 15 trading days.
7. **Regime filter:** new entries only when SPY is above its 200-day average
   and VIX is below 30.

### Where is the "sentiment filter"?

In the **live** bot (`main.py`), stat-arb candidates also pass through a
VADER headline check, a FinBERT sentiment check, an LSTM direction forecast
and a gradient-boosted "brain" model. Those gates read
`yfinance.Ticker(...).news`, which returns only the **latest ~10
headlines**, not history. None of these gates are in the backtest, and
the repo holds no historical sentiment data (`chroma_db/` holds 3 placeholder
"TESTCO3" 10-K snippets). **The sentiment filter cannot be backtested with
what is in the repo.** See §2.

---

## 2. Data availability: BLOCKED

| Data | Source in code | Status in this cloud session |
|---|---|---|
| Daily OHLC, ~300 US large caps | `yfinance` → `query1/query2.finance.yahoo.com`, `fc.yahoo.com` | ❌ **403: blocked by the environment's network policy** |
| SPY, ^VIX | same | ❌ blocked |
| Alternatives tried | stooq.com, data.alpaca.markets, alphavantage.co, api.tiingo.com, financialmodelingprep.com, huggingface.co, kaggle.com, sec.gov, fred.stlouisfed.org | ❌ all blocked |
| Historical news / sentiment | none in repo | ❌ **does not exist**; it would need an external dataset |
| Risk-free rate (T-bills, for Sharpe) | none | ❌ FRED blocked |

Per the brief, no synthetic or simulated data was used. No backtest was run.

**To unblock prices:** allow `query1.finance.yahoo.com`,
`query2.finance.yahoo.com` and `fc.yahoo.com` in the environment's network
settings, or switch to broader network access. Another option is to run the
optimisation on your own machine, where yfinance works.

**To backtest sentiment:** you would need a point-in-time historical
headline dataset. Examples are FNSPID (Hugging Face, ~1999–2023, free),
GDELT, or a paid feed such as RavenPack. Without one, the honest scope is
"price-only stat-arb sleeve; sentiment gate not backtested".

---

## 3. Bias and correctness audit

Severity: 🔴 invalidates headline numbers · 🟠 materially biases numbers · 🟡 minor.

| # | Issue | Where | Sev. | Status |
|---|---|---|---|---|
| A1 | **Z-score entry/exit unit mismatch (the one you found).** Live entry: `\|z\| ≥ ENTRY_Z` (z-units). Live exit: raw residual `> SELL_THRESH = 0.03`, i.e. a 3% one-day *out*-performance. That threshold is in different units and almost never triggers, so statarb longs were effectively held until the stop-loss. **It was not fixed on `main`.** The backtest itself was already consistent (z in, z out). | `main.py` `run_statarb_exits` | 🔴 (live) | ✅ **Fixed.** New shared rule `trade_utils.statarb_signal_exit(side, z, exit_z)` is used by both live and backtest. `SignalEngine` now also returns a signed `Residual_Z`. There are 5 unit tests. |
| A2 | **"Out-of-sample" period is contaminated.** The backtest defaults (`entry_z=1.75, exit_z=0.30, graph_threshold=0.45, alpha=0.8`) came from `optimize_params.py` runs on data from 2020 onwards. That data **includes** the 2024+ period the report labels "out-of-sample". Any number from an earlier `BACKTEST_REPORT.md` is in-sample. | `backtest_walkforward.py` defaults, `optimize_params.py` | 🔴 | ⚠️ **Flagged.** Resolved by design in Step 3: parameters get chosen on each train window only. |
| A3 | **Optimiser tuned the transaction cost.** `cost_bps` was a *search parameter* (0–15 bps), so Optuna chose ~0.04–0.47 bps, which is near-free trading. `COST_BPS=0.08` in the config is the result. Costs are an assumption, not a free parameter. | `optimize_params.py` | 🔴 | ⚠️ **Flagged.** The new optimiser (Step 2) will hold costs fixed. `optimize_params.py` is left untouched and should not be cited. |
| A4 | **The optimiser has no train/test split.** It scores every trial on the whole history, then narrowed the ranges three times ("phase 1→2→3") around the best trials. That is classic overfitting to one sample. | `optimize_params.py` | 🔴 | ⚠️ **Flagged.** Replaced in Step 2 by per-window optimisation. |
| B1 | **Survivorship bias.** The universe is a hand-written list of *today's* large caps ("compiled from general knowledge"). Stocks that went bust or were acquired before today are absent. Stocks that only *became* large caps later (PLTR, ABNB, SNOW…) are present from their IPO. For a long-biased strategy, this inflates returns. The buy-and-hold benchmark of the same universe is inflated the same way. yfinance has no delisted tickers, so this **cannot be fully fixed** with the current data source. | `statarb/config.example.py`, data layer | 🟠 | ⚠️ **Flagged.** Partial mitigation proposed (§4). It must be stated in the report and on the CV. |
| B2 | **Full-period coverage filter = look-ahead.** Tickers were kept only if they had prices on ≥70% of *all* days in the span. That decision uses information from the end of the backtest: names that listed late or disappeared early were dropped. | `backtest_walkforward.load_data` | 🟠 | ✅ **Fixed.** No full-period filter. Missing data is handled point-in-time (per graph window and per day). |
| C1 | **Costs incomplete.** The backtest used a single flat 5 bps per side, with no short-borrow fee. | `backtest_walkforward.py` | 🟠 | ✅ **Fixed.** Costs are now `--commission-bps` (1), `--slippage-bps` (4) and `--borrow-bps` (50/yr, accrued daily on shorts). Costs paid are reported. Step 3 will run two cost levels. |
| D1 | **The backtest is not the live signal path**, although the README says it is. Live uses a 22-day correlation window (backtest: 252), α=0.5 hard-coded (backtest: 0.8), a same-sector mask (backtest: none), and a z-score = residual ÷ one pooled std across all stocks (backtest: per-stock 60-day mean/std). Live also adds the sentiment, LSTM and brain gates. | `main.py` vs `backtest_walkforward.py` | 🟠 | ⚠️ **Flagged.** Results will be described as "a graph-based stat-arb strategy", **not** "my live bot's performance". |
| D2 | **Sentiment filter not backtestable** (see §2). | – | 🟠 | ⚠️ **Flagged / data missing.** |
| E1 | A queued exit was silently dropped if the stock had no open price the next day (halt or data gap). | `backtest_walkforward.py` | 🟡 | ✅ **Fixed.** The exit stays queued. |
| E2 | The z-score's rolling mean/std includes today's residual. This is **not** look-ahead (it is known at the close), but it slightly damps z. | `backtest_walkforward.py` | 🟡 | Flagged, left as is. |
| E3 | Sharpe/Sortino use a 0% risk-free rate. T-bills paid ~5% in 2023–24, so the Sharpe of a strategy holding cash is overstated then. | `compute_metrics` | 🟡 | Flagged. It will be stated in the report (FRED is blocked). |
| E4 | `--dollar-neutral` does **not** enforce dollar neutrality. It just allows shorts to share the 5 slots. | `backtest_walkforward.py` | 🟡 | Flagged. The name will be clarified in the report. |
| E5 | yfinance `auto_adjust=True` back-adjusts prices for splits and dividends. Returns are correct, but past *price levels* are not the traded prices. It does not affect % P&L. | data layer | 🟡 | OK, noted. |

### Checks that passed (no look-ahead found)

- The correlation graph at day *t* uses returns up to *t−1* only
  (`rets.iloc[t−lookback:t]`).
- Signals are computed at the close of *t* and filled at the **open of *t+1***.
- The regime filter uses SPY and VIX at the close of *t*.
- The stop width uses volatility up to *t−1*.
- Stops are checked against the day's high/low with pessimistic ordering. The
  stop is checked *before* the trailing high-water mark is updated, and
  gap-opens fill at the worse open price.
- Position size uses the prior close's equity.

---

## 4. What I need from you to continue

1. **Price data:** allow the Yahoo Finance hosts listed above (or give the
   environment broader network access), then tell me to continue. The
   other option is to run it locally.
2. **Sentiment:** confirm the scope is **price-only stat-arb**, with the
   sentiment filter clearly listed as "not backtested". The alternative is to
   provide a historical news dataset.
3. **Survivorship (optional but recommended):** I can download the public
   point-in-time S&P 500 membership history (fja05680/sp500 on GitHub, which
   *is* reachable). Each stock would then be tradable only on dates it was
   actually in the index. That removes the "picked today's winners"
   selection. It cannot add back delisted names that yfinance lacks, and that
   residual bias would be stated.
4. **Time budget:** the brief says "[X hours]". Please give a number.

---

## 5. Files changed on `cloud-backtest`

- `trade_utils.py`: new `statarb_signal_exit()` (z-unit exit, shared).
- `statarb/signal_engine.py`: also returns the signed `Residual_Z`.
- `main.py`: live statarb exit uses z-units and `EXIT_Z`, not raw `SELL_THRESH`.
- `statarb/config.example.py`: `exit_z` added to the engine config.
- `backtest_walkforward.py`: coverage-filter look-ahead removed; cost model
  split into commission, slippage and borrow; queued exits kept; shared exit rule.
- `tests/test_exit_logic.py`: 5 new tests for the exit rule.
- `requirements-backtest.txt`: exactly pinned minimal backtest environment.
