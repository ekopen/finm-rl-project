"""
Feature-construction utilities for trading environments.

This module currently provides a simple feature set; richer logic will be
added in later steps.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_simple_features(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Construct a minimal feature set from a price dataframe.

    Based on the lowercase ``close`` column, this adds:
    - ``ret``: daily close-to-close returns
    - ``ma``: rolling mean of ``close`` over ``window``
    - ``vol``: rolling std of ``ret`` over ``window``
    """
    if "close" not in df.columns:
        if "Close" in df.columns:
            base = df.copy()
            base["close"] = base["Close"]
        else:
            raise KeyError("Expected a 'close' column in the input dataframe.")
    else:
        base = df.copy()

    # Daily returns.
    base["ret"] = base["close"].pct_change().fillna(0.0)

    # Rolling mean of price; fill initial NaNs sensibly.
    ma = base["close"].rolling(window).mean()
    ma = ma.bfill().ffill()
    base["ma"] = ma.astype(np.float32)

    # Rolling volatility of returns.
    vol = base["ret"].rolling(window).std().fillna(0.0)
    base["vol"] = vol.astype(np.float32)

    return base


def make_rich_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Placeholder for a richer feature set (multi-horizon momentum/vol, etc.).

    Implementation will follow in later steps once the simple pipeline is
    working and evaluated.
    """
    raise NotImplementedError("make_rich_features is not implemented yet.")


