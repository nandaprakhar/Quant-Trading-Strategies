import yfinance as yf
import pandas as pd
from tqdm import tqdm


def download_stock_data(tickers, start_date, end_date, filepath):
    """
    Downloads historical stock price data for a list of tickers and saves it to a CSV file.

    Args:
        tickers (list): A list of stock ticker symbols.
        start_date (str): The start date for the data in 'YYYY-MM-DD' format.
        end_date (str): The end date for the data in 'YYYY-MM-DD' format.
        filepath (str): The path to save the CSV file.
    """
    print(
        f"Downloading data for {len(tickers)} tickers from {start_date} to {end_date}...")

    all_data = []
    for ticker in tqdm(tickers, desc="Downloading Tickers"):
        try:
            data = yf.download(ticker, start=start_date,
                               end=end_date, progress=False)
            if not data.empty:
                data['Ticker'] = ticker
                all_data.append(data)
        except Exception as e:
            print(f"Could not download data for {ticker}: {e}")

    if not all_data:
        print("No data downloaded. Exiting.")
        return

    df = pd.concat(all_data)
    df.to_csv(filepath)
    print(f"Data successfully downloaded and saved to {filepath}")


if __name__ == '__main__':
    # Example usage:
    # Define tickers for NASDAQ-100 and NIFTY-50 (example subset)
    nasdaq_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
    nifty_tickers = ['RELIANCE.NS', 'TCS.NS',
                     'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS']
    all_tickers = nasdaq_tickers + nifty_tickers

    # Define date range
    START_DATE = '2020-01-01'
    END_DATE = '2023-12-31'

    # Define output path
    DATA_FILEPATH = 'data/stock_prices.csv'

    download_stock_data(all_tickers, START_DATE, END_DATE, DATA_FILEPATH)
