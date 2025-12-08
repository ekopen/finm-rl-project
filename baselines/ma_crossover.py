"""
Moving-average crossover baseline strategy.

Implements a simple long/flat strategy based on a fast vs slow moving average
of the close price.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def run_ma_crossover(
    df: pd.DataFrame,
    fast: int = 20,
    slow: int = 50,
    initial_cash: float = 1.0,
    allow_short: bool = False,
) -> tuple[np.ndarray, pd.Index]:
    """
    Run a simple moving-average crossover strategy on a price series.

    Parameters
    ----------
    df : DataFrame
        Should contain at least a \"close\" or \"Close\" column.
    fast : int
        Fast moving-average window length.
    slow : int
        Slow moving-average window length.
    initial_cash : float
        Starting equity value.
    allow_short : bool
        If True, go short when fast < slow (instead of flat).
        If False, go flat when fast < slow (default for backward compatibility).

    Returns
    -------
    equity : np.ndarray
        Equity curve over time.
    index : pd.Index
        Corresponding index from the filtered dataframe.
    """
    if "close" in df.columns:
        close_series = df["close"]
    elif "Close" in df.columns:
        close_series = df["Close"]
    else:
        raise KeyError("DataFrame must contain a 'close' or 'Close' column.")

    # Build a minimal working dataframe to avoid any column-name surprises.
    data = pd.DataFrame(index=df.index)
    data["close"] = close_series
    data["fast"] = data["close"].rolling(fast).mean()
    data["slow"] = data["close"].rolling(slow).mean()

    # Drop periods where MAs are not defined.
    data = data.dropna()

    close_vals = data["close"].to_numpy(dtype=float)
    fast_vals = data["fast"].to_numpy(dtype=float)
    slow_vals = data["slow"].to_numpy(dtype=float)

    if close_vals.shape[0] < 2:
        raise ValueError("Not enough data after MA warmup for crossover strategy.")

    # Position: 1 when fast > slow, else 0 (flat) or -1 (short if allow_short=True).
    if allow_short:
        position = np.where(fast_vals > slow_vals, 1.0, -1.0)
    else:
        position = (fast_vals > slow_vals).astype(float)

    # Returns over each interval [t-1, t].
    returns = close_vals[1:] / close_vals[:-1] - 1.0

    equity = np.empty_like(close_vals, dtype=float)
    equity[0] = initial_cash
    for t, r in enumerate(returns, start=1):
        # Use position at t-1 for interval [t-1, t].
        equity[t] = equity[t - 1] * (1.0 + position[t - 1] * r)

    return equity, data.index


