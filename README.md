# Sea15 Trading Strategy Review

## Overview

This repository contains a suite of scripts for backtesting, analyzing, and live trading a "Gap" trading strategy. The strategy targets NASDAQ stocks priced between $1 and $50 with significant pre-market gaps.

## Project Structure

*   `Sea15_live.py`: The live trading execution script. It uses Alpaca for execution and Financial Modeling Prep (FMP) for data.
*   `Sea15_MC_BT.py`: Monte Carlo Backtest simulation. It runs multiple simulations to estimate the range of possible outcomes, assuming a hard Stop Loss.
*   `Sea15_BT.py`: Standard historical backtest logic.
*   `Sea15_MC-friction_BT.py`: Monte Carlo backtest that includes slippage and commission costs to model "friction".
*   `Sea15_SL-SensTest.py`: Sensitivity analysis script to test different Stop Loss percentages (e.g., 5%, 10%, 15%, No SL).

## Strategy Logic

**Universe Selection:**
*   Exchange: NASDAQ
*   Price: $1.00 - $50.00
*   Volume: > 10,000 (applies to previous day or latest available data in screener)

**Entry Signals (Daily):**
*   **Short Setup:**
    *   Gap Up between +5% and +13%.
    *   Gap calculated as `(Open - PrevClose) / PrevClose` (Backtest) or `(CurrentPrice - PrevClose) / PrevClose` (Live).
*   **Long Setup:**
    *   Gap Down of -5% or more (<= -0.05).

**Position Sizing:**
*   Risk per trade: $1,000.
*   Shares calculation: `Risk / (EntryPrice * StopBuffer)`.
*   With a 5% Stop Buffer, this sizes the position such that a 5% move against the entry equals $1,000 risk.

## Critical Review & Findings

### 1. 🚨 CRITICAL: Missing Stop Loss in Live Execution

There is a **major design flaw** in `Sea15_live.py` regarding risk management.

*   **Backtest Assumption:** All backtests (`Sea15_MC_BT.py`, etc.) explicitly model a hard Stop Loss (default 5%). They check if the intraday `High` (for shorts) or `Low` (for longs) breached the stop price and cap the loss at `-STOP_LOSS_PCT`.
*   **Live Reality:** The `Sea15_live.py` script **does NOT submit a Stop Loss order**.
    *   It calculates share size based on a 5% risk, effectively taking a position size of ~$20,000 to risk $1,000.
    *   It executes a standard `MarketOrderRequest`.
    *   It **only** exits at the end of the day (`job_exit_flatten` at 12:55 PT).
*   **Consequence:** If a stock moves more than 5% against the position (e.g., a short squeeze spikes 20% or 50%), the strategy will suffer **uncapped losses** far exceeding the $1,000 risk target. A 50% move on a $20,000 position is a $10,000 loss (10R). The backtest results are invalid for the current live implementation because they assume protection that does not exist.

### 2. Execution Timing & Look-Ahead Discrepancy

*   **Backtest:** Assumes entry exactly at the `Open` price.
*   **Live:** Waits 15 seconds (`SECONDS_DELAY`) after market open before fetching quotes and submitting market orders.
*   **Risk:**
    *   The price 15 seconds after open can be significantly different from the Open price, especially for volatile gappers.
    *   Slippage is not accounted for in the standard backtest (though `Sea15_MC-friction_BT.py` attempts to model it).
    *   Gap calculation in Live (`CurrentPrice` vs `PrevClose`) may differ from Backtest (`Open` vs `PrevClose`), potentially selecting different stocks.

### 3. Survivorship Bias in Backtesting

*   The backtests rely on `get_nasdaq_tickers()` which fetches the *current* list of tickers from the FMP Screener.
*   It then downloads 3 years (`LOOKBACK_DAYS = 1095`) of history for these *current* tickers.
*   **Risk:** This ignores stocks that were valid candidates 1, 2, or 3 years ago but have since been delisted or went bankrupt. This often inflates backtest performance as it excludes the "worst case" failures that no longer exist.

### 4. Security & Operational Risks

*   **API Keys:** API keys for FMP and Alpaca are **hardcoded** in the source files. This is a severe security risk. Keys should be loaded from environment variables.
*   **Timezone Dependency:** The live script uses hardcoded strings (e.g., "06:30") and relies on the system time. If the server is not set to Pacific Time, the schedule will fire at the wrong time (e.g., "06:30 UTC" is in the middle of the night for US markets).
*   **Error Handling:** `fetch_batch_quotes` suppresses all exceptions. If the data feed fails, the script will silently return empty lists and do nothing, potentially missing trades without alert.
*   **Capital Requirements:** The script attempts to submit up to 10 trades (5 long, 5 short). With a 5% stop sizing, each trade is ~$20,000 notional. Total buying power required is ~$200,000. The script does not check for available buying power before submitting orders.

## Recommendations

1.  **Implement Stop Losses:** Modify `Sea15_live.py` to submit `Bracket` orders (Entry + TakeProfit/StopLoss) or immediately submit a separate Stop Loss order upon fill.
2.  **Secure Credentials:** Move API keys to environment variables or a `.env` file.
3.  **Align Execution:** Decide whether to execute immediately at Open (to match backtest) or update the backtest to simulate "Open + 15s" entry prices (using 1-minute bar data if available).
4.  **Survivorship Bias:** Acknowledge that historical results are likely optimistic due to survivorship bias.
