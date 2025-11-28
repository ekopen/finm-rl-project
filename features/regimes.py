"""
Helpers for labeling market regimes (bull/bear, high/low volatility, etc.).

Only function signatures are provided at this scaffolding stage.
"""

from __future__ import annotations

import pandas as pd


def label_bull_bear(df: pd.DataFrame, ma_window: int = 200) -> pd.Series:
    """
    Label each date as bull or bear market.

    The concrete implementation (e.g., based on a long moving average) will
    be added in a later step.
    """
    raise NotImplementedError("label_bull_bear is not implemented yet.")


def label_volatility(
    df: pd.DataFrame,
    window: int = 20,
    quantile: float = 0.7,
) -> pd.Series:
    """
    Label each date as high or low volatility.

    Implementation will be added later using realized volatility estimates.
    """
    raise NotImplementedError("label_volatility is not implemented yet.")


