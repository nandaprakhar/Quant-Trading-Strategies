import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_performance_metrics(portfolio_history):
    """
    Calculates and prints key performance metrics for the backtest results.

    Args:
        portfolio_history (pd.DataFrame): DataFrame with a 'Portfolio_Value' column.

    Returns:
        dict: A dictionary containing the calculated performance metrics.
    """
    print("\n--- Performance Analysis ---")

    if portfolio_history.empty:
        print("Portfolio history is empty. Cannot calculate metrics.")
        return {}

    # Calculate daily returns
    daily_returns = portfolio_history['Portfolio_Value'].pct_change().dropna()

    # --- Average Annual Return ---
    total_return = (portfolio_history['Portfolio_Value']
                    [-1] / portfolio_history['Portfolio_Value'][0]) - 1
    num_days = len(portfolio_history)
    annualization_factor = 252 / num_days
    average_annual_return = (1 + total_return) ** annualization_factor - 1

    # --- Annual Volatility ---
    annual_volatility = daily_returns.std() * np.sqrt(252)

    # --- Sharpe Ratio ---
    # Assuming a risk-free rate of 0 for simplicity
    risk_free_rate = 0.0
    sharpe_ratio = (average_annual_return - risk_free_rate) / annual_volatility

    # --- Maximum Drawdown ---
    cumulative_returns = (1 + daily_returns).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns / peak) - 1
    max_drawdown = drawdown.min()

    # Print the results
    print(f"Average Annual Return: {average_annual_return:.2%}")
    print(f"Annual Volatility: {annual_volatility:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Maximum Drawdown: {max_drawdown:.2%}")

    metrics = {
        'sharpe_ratio': sharpe_ratio,
        'average_annual_return': average_annual_return,
        'max_drawdown': max_drawdown
    }

    return metrics


def plot_performance(portfolio_history):
    """
    Plots the portfolio value over time and the drawdown.

    Args:
        portfolio_history (pd.DataFrame): DataFrame with portfolio history.
    """
    sns.set_style('whitegrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Plot Portfolio Value
    ax1.plot(portfolio_history.index,
             portfolio_history['Portfolio_Value'], label='Portfolio Value', color='navy')
    ax1.set_title('Portfolio Value Over Time')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend()

    # Plot Drawdown
    daily_returns = portfolio_history['Portfolio_Value'].pct_change().dropna()
    cumulative_returns = (1 + daily_returns).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns / peak) - 1

    ax2.plot(drawdown.index, drawdown,
             label='Drawdown', color='red', alpha=0.7)
    ax2.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.1)
    ax2.set_title('Portfolio Drawdown')
    ax2.set_ylabel('Drawdown')
    ax2.set_xlabel('Date')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('results/performance_summary.png')
    print("\nPerformance summary plot saved to results/performance_summary.png")
    plt.show()


if __name__ == '__main__':
    # Example usage:
    # Create some dummy portfolio history data
    dates = pd.date_range(start='2021-01-01', end='2023-12-31')
    initial_value = 100000
    dummy_returns = np.random.normal(loc=0.0005, scale=0.01, size=len(dates))
    dummy_values = initial_value * (1 + dummy_returns).cumprod()
    dummy_history = pd.DataFrame(
        {'Portfolio_Value': dummy_values}, index=dates)

    # Calculate and plot performance
    calculate_performance_metrics(dummy_history)
    plot_performance(dummy_history)
