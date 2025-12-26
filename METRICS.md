# Sea15 Backtesting Metrics

This document explains the key performance indicators (KPIs) calculated in the Sea15 backtesting engine (`Sea15_BT.py`) and how they are derived.

## Financial & Risk Metrics

### 1. CAGR (Compound Annual Growth Rate)
The mean annual growth rate of the strategy over the simulation period.

*   **Formula:** `((Ending Equity / Starting Capital) ^ (1 / Years)) - 1`
*   **Significance:** Measures the geometric progression ratio that provides a constant rate of return over the time period.

### 2. Vol (Ann) (Annualized Volatility)
The standard deviation of daily returns, annualized.

*   **Formula:** `StdDev(Daily Returns) * Sqrt(252)`
*   **Significance:** Measures the risk or dispersion of returns. Higher volatility implies higher risk.

### 3. Sharpe Ratio
A measure of risk-adjusted return.

*   **Formula:** `(Mean(Daily Excess Returns) / StdDev(Daily Returns)) * Sqrt(252)`
    *   *Excess Returns* = Daily Return - (Risk Free Rate / 252)
*   **Significance:** Indicates how much excess return is received for the extra volatility endured for holding a riskier asset.

### 4. Sortino Ratio
A variation of the Sharpe Ratio that differentiates harmful volatility from total overall volatility.

*   **Formula:** `(Mean(Excess Returns) * 252) / Downside Deviation`
    *   *Downside Deviation* = Annualized StdDev of negative returns only.
*   **Significance:** Focuses on the downside risk, which is more relevant to investors than upside volatility.

### 5. Max DD % (Maximum Drawdown)
The maximum observed loss from a peak to a trough of the equity curve, before a new peak is attained.

*   **Formula:** `Min((Equity - Running Max Equity) / Running Max Equity)`
*   **Significance:** An indicator of downside risk over a specified time period.

### 6. Recovery Factor
A measure of how fast the strategy recovers from drawdowns relative to the net profit generated.

*   **Formula:** `Total Net Profit / Max Drawdown ($)`
*   **Significance:** Higher values indicate the strategy earns significantly more than its deepest drawdown.

### 7. Beta
A measure of the volatility—or systematic risk—of a portfolio in comparison to the benchmark (QQQ).

*   **Formula:** `Covariance(Strategy Returns, Benchmark Returns) / Variance(Benchmark Returns)`
*   **Significance:** A beta of 1 indicates the strategy moves with the market. < 1 means less volatile, > 1 means more volatile.

### 8. Alpha (Ann)
Jensen's Alpha represents the excess return of the strategy over the expected return based on its Beta and the market return (CAPM).

*   **Formula:** `Annualized Strategy Return - (Risk Free Rate + Beta * (Annualized Benchmark Return - Risk Free Rate))`
*   **Significance:** Measures the value added by the active management of the strategy. Positive alpha indicates outperformance.

### 9. Treynor Ratio
A risk-adjusted measure of return based on systematic risk (Beta).

*   **Formula:** `(Annualized Strategy Return - Risk Free Rate) / Beta`
*   **Significance:** Similar to Sharpe, but uses Beta (market risk) instead of total volatility.

### 10. Information Ratio
Measures the portfolio returns beyond the returns of the benchmark, compared to the volatility of those excess returns.

*   **Formula:** `(Annualized Strategy Return - Annualized Benchmark Return) / Tracking Error`
    *   *Tracking Error* = Annualized StdDev of (Strategy Returns - Benchmark Returns)
*   **Significance:** Assesses the consistency of the active manager.

### 11. Calmar Ratio
A comparison of the average annual compounded rate of return and the maximum drawdown risk.

*   **Formula:** `CAGR / Abs(Max Drawdown %)`
*   **Significance:** Used to evaluate the return relative to the worst-case downside risk.

### 12. Omega Ratio
The probability-weighted ratio of gains versus losses for a given threshold (0%).

*   **Formula:** `Sum(Positive Returns) / Abs(Sum(Negative Returns))`
*   **Significance:** Captures all higher moments of the return distribution (skewness, kurtosis), unlike Sharpe which assumes normality.

### 13. VaR 95% (Value at Risk)
The maximum loss expected over a day with 95% confidence.

*   **Formula:** 5th Percentile of Daily Returns distribution.
*   **Significance:** "We are 95% confident that the daily loss will not exceed X%."

### 14. CVaR 95% (Conditional Value at Risk / Expected Shortfall)
The average expected loss given that the loss is greater than the VaR threshold.

*   **Formula:** `Mean(Returns <= VaR 95%)`
*   **Significance:** Measures the "tail risk" – how bad things get when the 5% worst-case scenario happens.

## Trade Statistics

### 15. Win Rate %
The percentage of trades that resulted in a positive net profit.

*   **Formula:** `Count(Wins) / Total Trades`

### 16. Stop-out %
The percentage of trades that were closed due to hitting the hard stop-loss level.

*   **Formula:** `Count(Outcome == 'Stopped Out') / Total Trades`

### 17. Profit Factor
The ratio of gross profit to gross loss.

*   **Formula:** `Sum(Gross Wins) / Abs(Sum(Gross Losses))`
*   **Significance:** A value > 1.0 indicates a profitable system. > 1.5 is generally considered good.

### 18. Avg Win % / Avg Loss %
The average percentage return of winning and losing trades, respectively.

*   **Formula:** `Mean(Winning Trade PnLs)` / `Mean(Losing Trade PnLs)`

### 19. Payoff Ratio
The ratio of the average win to the absolute average loss.

*   **Formula:** `Avg Win % / Abs(Avg Loss %)`
*   **Significance:** Shows how much larger an average win is compared to an average loss.

### 20. Expectancy %
The expected average return per trade based on win rate and payoff.

*   **Formula:** `(Win Rate * Avg Win %) + (Loss Rate * Avg Loss %)`
*   **Significance:** The statistical edge of the strategy per trade.

### 21. Max Consec Wins / Losses
The longest streak of consecutive winning or losing trades.

*   **Formula:** Maximum count of uninterrupted sequences of wins or losses.
