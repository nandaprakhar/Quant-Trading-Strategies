import pandas as pd
import numpy as np


def run_backtest(data, start_date, end_date, initial_capital, stock_selector, allocator):
    """
    Runs a quarterly rebalancing backtest on the investment strategy.

    Args:
        data (pd.DataFrame): The full dataset of all stocks with indicators.
        start_date (str): The start date of the backtest.
        end_date (str): The end date of the backtest.
        initial_capital (float): The starting capital for the portfolio.
        stock_selector (function): The function to select top stocks.
        allocator (function): The function to determine asset allocations.

    Returns:
        pd.DataFrame: A DataFrame containing the daily portfolio value.
    """
    print("Starting backtest...")

    # Get the quarterly rebalancing dates
    rebalance_dates = pd.date_range(start=start_date, end=end_date, freq='Q')

    portfolio_value = initial_capital
    portfolio = {}  # Holds current stock positions

    # Create a DataFrame to track daily portfolio value
    portfolio_history = pd.DataFrame(
        index=pd.date_range(start=start_date, end=end_date))
    portfolio_history['Portfolio_Value'] = np.nan

    # Main backtesting loop
    for i, quarter_end in enumerate(rebalance_dates):
        if i + 1 >= len(rebalance_dates):
            break

        start_of_quarter = quarter_end + pd.Timedelta(days=1)
        end_of_quarter = rebalance_dates[i+1]

        print(
            f"\n--- Rebalancing for quarter: {start_of_quarter.date()} to {end_of_quarter.date()} ---")

        # 1. Select top stocks for the upcoming quarter
        selected_tickers = stock_selector(quarter_end.strftime('%Y-%m-%d'))

        # 2. Get asset allocations
        allocations = allocator(
            selected_tickers, quarter_end.strftime('%Y-%m-%d'))

        # 3. Execute trades (simplified: sell all, then buy new)
        # In a real system, you'd account for transaction costs, slippage, etc.
        portfolio = {}
        for ticker, weight in allocations.items():
            investment_amount = portfolio_value * weight

            # Get the price at the start of the quarter
            start_price = data.loc[(data.index == start_of_quarter) & (
                data['TICKER'] == ticker), 'OPEN'].iloc[0]

            num_shares = investment_amount / start_price
            portfolio[ticker] = num_shares

        # 4. Simulate the quarter
        for current_date in pd.date_range(start=start_of_quarter, end=end_of_quarter):
            if current_date not in data.index:
                continue

            current_day_value = 0
            for ticker, shares in portfolio.items():
                # Get the closing price for the current day
                try:
                    current_price = data.loc[(data.index == current_date) & (
                        data['TICKER'] == ticker), 'CLOSE'].iloc[0]
                    current_day_value += shares * current_price
                except IndexError:
                    # Handle cases where a stock might not have data for a specific day
                    pass

            if current_day_value > 0:
                portfolio_history.loc[current_date,
                                      'Portfolio_Value'] = current_day_value

        # Update portfolio value for the next rebalancing
        portfolio_value = portfolio_history.loc[end_of_quarter,
                                                'Portfolio_Value']

    portfolio_history.dropna(inplace=True)
    print("\nBacktest finished.")
    return portfolio_history


if __name__ == '__main__':
    # This module is not meant to be run standalone.
    # It will be called by the main.py orchestrator.
    print("This is the backtesting module. It should be imported and used by main.py.")
