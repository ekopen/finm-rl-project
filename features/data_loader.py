"""
High-level data loading interface for experiments.

This module sits on top of `data.data_loader` and:
- standardizes to lowercase OHLCV columns
- ensures a DateTime index and sorted dates
"""

from __future__ import annotations

import pandas as pd

from data.data_loader import fetch_single_asset


def load_price_data(
    ticker: str,
    start: str | None = None,
    end: str | None = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Load OHLCV price data for a single ticker.

    Columns are lowercase: ``open, high, low, close, volume`` and the index
    is a sorted ``DatetimeIndex``.
    """
    # Use broad defaults if dates are not provided; callers can override.
    start_arg = start or "2000-01-01"
    end_arg = end or "2025-01-01"
    df = fetch_single_asset(ticker, start=start_arg, end=end_arg, interval=interval)

    # If yfinance returns a MultiIndex (Price, Ticker), flatten to a single
    # column index for the single requested ticker.
    if isinstance(df.columns, pd.MultiIndex):
        # Typical shape: levels ['Price', 'Ticker'], with one ticker column.
        if df.columns.nlevels == 2:
            # Drop the ticker level, leaving e.g. close, high, low, open, volume.
            df = df.droplevel(1, axis=1)
        else:
            # Fallback: just take the top level.
            df = df.copy()
            df.columns = df.columns.get_level_values(0)

    # Ensure datetime index and ordering.
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    return df


def train_test_split_by_date(
    df: pd.DataFrame,
    train_end_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simple train/test split based on a cutoff date.

    Parameters
    ----------
    df : DataFrame
        Price or feature dataframe. The index will be coerced to a
        ``DatetimeIndex`` and sorted before splitting.
    train_end_date : str
        Last date (inclusive) to consider part of the training set.
    """
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    cutoff = pd.to_datetime(train_end_date)
    train = df[df.index <= cutoff].copy()
    test = df[df.index > cutoff].copy()
    return train, test


