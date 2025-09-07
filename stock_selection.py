import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def prepare_data_for_ml(df):
    """
    Prepares the data for machine learning by creating features and a target variable.

    Args:
        df (pd.DataFrame): DataFrame with stock prices and indicators.

    Returns:
        pd.DataFrame: The DataFrame ready for ML, with features (X) and target (y).
    """
    print("Preparing data for ML...")
    # Create a target variable: 1 if the next day's close is higher, 0 otherwise.
    df['TARGET'] = (df['CLOSE'].shift(-1) > df['CLOSE']).astype(int)

    # For simplicity, we'll use the indicator values as features.
    # Drop non-feature columns and the last row with NaN target.
    df.dropna(inplace=True)

    return df


def train_svm_model(df):
    """
    Trains a Support Vector Machine (SVM) model to predict stock performance.

    Args:
        df (pd.DataFrame): The prepared DataFrame with features and target.

    Returns:
        model: The trained SVM model.
        scaler: The scaler used to normalize the data.
    """
    print("Training SVM model...")

    features = [col for col in df.columns if col not in [
        'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'TICKER', 'TARGET', 'SIGNAL']]
    X = df[features]
    y = df['TARGET']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = SVC(kernel='rbf', probability=True)
    model.fit(X_train_scaled, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test_scaled)
    print(f"SVM Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")

    return model, scaler


def select_top_stocks(data, model, scaler, quarter_end_date):
    """
    Uses the trained model to predict and select the top 25% of stocks for the next quarter.

    Args:
        data (pd.DataFrame): The full dataset of all stocks.
        model: The trained ML model.
        scaler: The fitted scaler.
        quarter_end_date (str): The date to make predictions from (e.g., '2023-03-31').

    Returns:
        list: A list of ticker symbols for the selected top stocks.
    """
    print(f"Selecting top stocks for the quarter after {quarter_end_date}...")

    # Get the latest data for each stock on the specified date
    latest_data = data.loc[data.index == quarter_end_date].copy()

    if latest_data.empty:
        print(
            f"No data available for {quarter_end_date}. Cannot select stocks.")
        return []

    features = [col for col in latest_data.columns if col not in [
        'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'TICKER', 'TARGET', 'SIGNAL']]
    X_latest = latest_data[features]

    X_latest_scaled = scaler.transform(X_latest)

    # Predict the probability of the price going up
    latest_data['UP_PROBABILITY'] = model.predict_proba(X_latest_scaled)[:, 1]

    # Select the top 25% of stocks based on this probability
    top_25_percentile = latest_data['UP_PROBABILITY'].quantile(0.75)
    top_stocks = latest_data[latest_data['UP_PROBABILITY']
                             >= top_25_percentile]

    selected_tickers = top_stocks['TICKER'].tolist()
    print(f"Selected {len(selected_tickers)} stocks: {selected_tickers}")

    return selected_tickers


if __name__ == '__main__':
    # Example usage:
    try:
        data = pd.read_csv('data/stock_prices.csv',
                           index_col='Date', parse_dates=True)

        # Calculate indicators for all stocks
        data_with_indicators = data.groupby('Ticker').apply(
            calculate_technical_indicators).reset_index(level=0, drop=True)

        # Prepare data for ML
        ml_data = prepare_data_for_ml(data_with_indicators)

        # Train the model
        svm_model, data_scaler = train_svm_model(ml_data)

        # Select top stocks for a specific quarter
        select_top_stocks(ml_data, svm_model, data_scaler,
                          quarter_end_date='2023-03-31')

    except FileNotFoundError:
        print("Please run data_loader.py first to download the stock data.")
