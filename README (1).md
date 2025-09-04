# Investment Strategy Backtesting and Performance Analysis

This project implements and evaluates a sophisticated investment strategy that uses a combination of technical indicators, machine learning models, and deep reinforcement learning for asset allocation. The goal is to backtest this strategy on historical stock market data to assess its profitability and risk-adjusted returns.

## Project Objective

The primary objective is to build a robust system that can:

1.  Generate trading signals using technical indicators (Keltner Channel, Bollinger Bands, MACD).
2.  Select the top 25% performing stocks each quarter using predictive models (LSTM, SVM).
3.  Determine optimal asset allocation using a Deep Reinforcement Learning (DRL) ensemble.
4.  Evaluate the strategy's performance by calculating total returns, maximum drawdowns, and key performance ratios like the Sharpe ratio.

## Methodology

1.  **Data Acquisition:** Download historical price data for NASDAQ and NSE indices and their constituent stocks using the `yfinance` library.
2.  **Feature Engineering:** Calculate the specified technical indicators to serve as features for the machine learning models and as direct trading signals.
3.  **Stock Selection:** Employ LSTM and SVM models to predict which stocks are likely to be in the top quartile of performers for the upcoming quarter.
4.  **Asset Allocation:** Use a DRL ensemble model to decide the optimal capital allocation across the selected stocks.
5.  **Backtesting:** Simulate the execution of the trading strategy on historical data, tracking the portfolio's value over time.
6.  **Performance Analysis:** Calculate and visualize key metrics, including annual returns, volatility, maximum drawdown, and the Sharpe ratio.

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd investment_analysis_project
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the pipeline:**
    ```bash
    python src/main.py
    ```
