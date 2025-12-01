"""
Helpers for labeling market regimes (bull/bear, high/low volatility, etc.).

Only function signatures are provided at this scaffolding stage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _get_close_series(df: pd.DataFrame) -> pd.Series:
    """Return a lowercase close series regardless of original casing."""
    if "close" in df.columns:
        close = df["close"]
    elif "Close" in df.columns:
        close = df["Close"]
    else:
        raise KeyError("Dataframe must contain a 'close' or 'Close' column.")
    return close.astype(float)


def label_bull_bear(df: pd.DataFrame, ma_window: int = 200) -> pd.Series:
    """Classify each date as bull (price above MA) or bear (below MA)."""
    if ma_window <= 0:
        raise ValueError("ma_window must be positive.")

    close = _get_close_series(df)
    long_ma = close.rolling(ma_window, min_periods=1).mean().bfill().ffill()

    labels = np.where(close >= long_ma, "bull", "bear")
    return pd.Series(labels, index=close.index, name="bull_bear")


def label_volatility(
    df: pd.DataFrame,
    window: int = 20,
    quantile: float = 0.7,
) -> pd.Series:
    """Label each date as high or low volatility via rolling realized vol."""
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be within (0, 1).")
    if window <= 1:
        raise ValueError("window must be > 1 for volatility estimation.")

    close = _get_close_series(df)
    returns = close.pct_change().fillna(0.0)
    realized_vol = returns.rolling(window).std().bfill().ffill()

    threshold = realized_vol.quantile(quantile)
    labels = np.where(realized_vol >= threshold, "high_vol", "low_vol")
    return pd.Series(labels, index=close.index, name="vol_regime")


