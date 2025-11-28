from __future__ import annotations

import numpy as np
import pandas as pd


def buy_and_hold(prices: np.ndarray) -> float:
    """
    Minimal buy & hold baseline returning final PnL.

    Parameters
    ----------
    prices : np.ndarray
        Array of price levels over time.
    """
    return float(prices[-1] / prices[0] - 1.0)


def run_buy_and_hold(df: pd.DataFrame, initial_cash: float = 1.0) -> np.ndarray:
    """
    Run a buy-and-hold strategy and return the equity curve.

    Parameters
    ----------
    df : DataFrame
        Must contain a ``close`` column (lowercase). If only ``Close`` is
        present it will be used as a fallback.
    initial_cash : float
        Starting equity value.
    """
    if "close" in df.columns:
        prices = df["close"].to_numpy(dtype=float)
    elif "Close" in df.columns:
        prices = df["Close"].to_numpy(dtype=float)
    else:
        raise KeyError("DataFrame must contain a 'close' or 'Close' column.")

    if prices.shape[0] < 2:
        raise ValueError("Need at least two price points for buy-and-hold.")

    returns = prices[1:] / prices[:-1] - 1.0  # daily simple returns

    # Equity[0] = initial_cash, then compound returns.
    equity = np.empty(prices.shape[0], dtype=float)
    equity[0] = initial_cash
    for t, r in enumerate(returns, start=1):
        equity[t] = equity[t - 1] * (1.0 + r)

    return equity

