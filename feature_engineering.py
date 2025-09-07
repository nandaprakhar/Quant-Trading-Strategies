import pandas as pd
import pandas_ta as ta


def calculate_technical_indicators(df):
    """
    Calculates technical indicators for the given stock data.

    Args:
        df (pd.DataFrame): DataFrame with stock prices (must contain 'High', 'Low', 'Close', 'Volume').

    Returns:
        pd.DataFrame: The original DataFrame with added indicator columns.
    """
    print("Calculating technical indicators...")

    # Calculate Keltner Channels
    df.ta.kc(append=True)

    # Calculate Bollinger Bands
    df.ta.bbands(append=True)

    # Calculate MACD
    df.ta.macd(append=True)

    # Clean up column names for consistency if needed (pandas_ta can create long names)
    df.columns = [col.upper() for col in df.columns]

    df.dropna(inplace=True)

    print("Technical indicators calculated successfully.")
    return df


def generate_signals(df):
    """
    Generates trading signals based on the calculated technical indicators.

    This is a simplified example. A real strategy would be more complex.

    Args:
        df (pd.DataFrame): DataFrame with technical indicators.

    Returns:
        pd.DataFrame: The DataFrame with an added 'SIGNAL' column (1 for Buy, -1 for Sell, 0 for Hold).
    """
    print("Generating trading signals...")

    # Example Signal Logic: Buy when price closes above upper Bollinger Band, Sell when below lower.
    df['SIGNAL'] = 0
    df.loc[df['CLOSE'] > df['BBU_20_2.0'], 'SIGNAL'] = 1  # Buy Signal
    df.loc[df['CLOSE'] < df['BBL_20_2.0'], 'SIGNAL'] = -1  # Sell Signal

    print("Trading signals generated.")
    return df


if __name__ == '__main__':
    # Example usage:
    # Load some sample data (assuming it's been downloaded)
    try:
        data = pd.read_csv('data/stock_prices.csv',
                           index_col='Date', parse_dates=True)

        # Process data for a single ticker for demonstration
        aapl_data = data[data['Ticker'] == 'AAPL'].copy()

        # Calculate indicators
        aapl_with_indicators = calculate_technical_indicators(aapl_data)

        # Generate signals
        aapl_with_signals = generate_signals(aapl_with_indicators)

        print("\nSample data with indicators and signals:")
        print(aapl_with_signals.tail())

    except FileNotFoundError:
        print("Please run data_loader.py first to download the stock data.")
