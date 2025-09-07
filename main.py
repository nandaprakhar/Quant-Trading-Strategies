import os
import pandas as pd
from functools import partial

# Import our custom modules
from data_loader import download_stock_data
from feature_engineering import calculate_technical_indicators
from stock_selection import prepare_data_for_ml, train_svm_model, select_top_stocks
from drl_allocator import get_drl_allocations
from backtester import run_backtest
from performance_analyzer import calculate_performance_metrics, plot_performance


def main():
    """
    Main function to orchestrate the investment strategy backtesting.
    """
    print("Starting the investment analysis pipeline...")

    # --- Configuration ---
    nasdaq_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN',
                      'NVDA', 'TSLA', 'META', 'AVGO', 'PEP', 'COST']
    nifty_tickers = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS',
                     'ICICIBANK.NS', 'HINDUNILVR.NS', 'SBIN.NS', 'ITC.NS', 'BHARTIARTL.NS', 'LICI.NS']
    ALL_TICKERS = nasdaq_tickers + nifty_tickers

    START_DATE = '2021-01-01'
    END_DATE = '2023-12-31'
    BACKTEST_START_DATE = '2022-01-01'  # Allow one year for indicator history
    INITIAL_CAPITAL = 100000.0
    DATA_FILEPATH = 'data/stock_prices.csv'

    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # --- Step 1: Data Acquisition ---
    if not os.path.exists(DATA_FILEPATH):
        download_stock_data(ALL_TICKERS, START_DATE, END_DATE, DATA_FILEPATH)
    else:
        print("Data already exists. Skipping download.")

    data = pd.read_csv(DATA_FILEPATH, index_col='Date', parse_dates=True)
    print("Step 1: Data loaded successfully.")

    # --- Step 2: Feature Engineering ---
    data_with_indicators = data.groupby('Ticker').apply(
        calculate_technical_indicators).reset_index(level=0, drop=True)
    print("Step 2: Feature engineering complete.")

    # --- Step 3: Stock Selection (ML Models) ---
    ml_data = prepare_data_for_ml(data_with_indicators.copy())
    svm_model, data_scaler = train_svm_model(ml_data)

    # Create a partial function for the backtester to use
    # This "freezes" the arguments for the model and scaler
    stock_selector_func = partial(
        select_top_stocks, data=ml_data, model=svm_model, scaler=data_scaler)
    print("Step 3: Stock selection model trained.")

    # --- Step 4: Asset Allocation (DRL) ---
    # Create a partial function for the allocator
    allocator_func = partial(get_drl_allocations, data=ml_data)
    print("Step 4: Asset allocator is ready.")

    # --- Step 5: Backtesting ---
    portfolio_history = run_backtest(
        data=data_with_indicators,
        start_date=BACKTEST_START_DATE,
        end_date=END_DATE,
        initial_capital=INITIAL_CAPITAL,
        stock_selector=stock_selector_func,
        allocator=allocator_func
    )
    print("Step 5: Backtest complete.")

    # --- Step 6: Performance Analysis ---
    if not portfolio_history.empty:
        calculate_performance_metrics(portfolio_history)
        plot_performance(portfolio_history)
        portfolio_history.to_csv('results/portfolio_history.csv')
        print("Step 6: Performance analysis complete. Results saved.")
    else:
        print("Backtest did not generate any results to analyze.")

    print("\nPipeline finished successfully.")


if __name__ == "__main__":
    main()
