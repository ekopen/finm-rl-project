import yfinance as yf
import pandas as pd
import os

def fetch_single_asset(
    ticker,
    start="2000-01-01",
    end="2025-01-01",
    interval="1d",
    save_path=None
):
    """
    Download OHLCV data for a single ticker.
    """
    print(f"Downloading data for {ticker}...")
    df = yf.download(ticker, start=start, end=end, interval=interval)

    # Ensure standard column format
    df = df.rename(columns=str.lower)  # open, high, low, close, volume
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path)
        print(f"Saved to {save_path}")

    return df


def fetch_multiple_assets(
    tickers,
    start="2000-01-01",
    end="2025-01-01",
    interval="1d",
    save_path=None
):
    """
    Download OHLCV data for multiple tickers.
    Returns a dict: {ticker: dataframe}
    """
    data = {}

    for t in tickers:
        print(f"Downloading {t}...")
        df = yf.download(t, start=start, end=end, interval=interval)
        df = df.rename(columns=str.lower)
        data[t] = df

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        for t, df in data.items():
            df.to_csv(os.path.join(save_path, f"{t}.csv"))
        print(f"Saved all data to {save_path}")

    return data


def load_from_csv(path):
    """Simple helper to load previously saved CSV."""
    return pd.read_csv(path, index_col=0, parse_dates=True)


def get_close_prices(df):
    """Return just the Close price series for convenience."""
    return df["close"].copy()
