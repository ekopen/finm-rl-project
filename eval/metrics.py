"""
Metric utilities for evaluating trading strategies and PPO agents.
"""

from __future__ import annotations

import numpy as np


def _to_equity_array(equity: np.ndarray) -> np.ndarray:
    eq = np.asarray(equity, dtype=float)
    if eq.ndim != 1:
        raise ValueError("Equity curve must be 1D.")
    return eq


def _equity_to_returns(equity: np.ndarray) -> np.ndarray:
    eq = _to_equity_array(equity)
    if eq.size < 2:
        return np.zeros(0, dtype=float)
    # Simple returns from equity.
    return eq[1:] / eq[:-1] - 1.0


def compute_total_return(equity: np.ndarray) -> float:
    """
    Compute total return from an equity curve: final / initial - 1.
    """
    eq = _to_equity_array(equity)
    if eq.size == 0 or eq[0] == 0:
        return 0.0
    return float(eq[-1] / eq[0] - 1.0)


def compute_annualized_return(
    equity: np.ndarray,
    periods_per_year: int,
) -> float:
    """
    Compute annualized return given an equity curve and sampling frequency.

    Uses geometric growth: (final / initial) ** (periods_per_year / T) - 1,
    where T is the number of periods.
    """
    eq = _to_equity_array(equity)
    if eq.size < 2 or eq[0] <= 0 or eq[-1] <= 0:
        return 0.0
    total_return_factor = eq[-1] / eq[0]
    T = eq.size - 1
    if T <= 0:
        return 0.0
    ann = total_return_factor ** (periods_per_year / T) - 1.0
    return float(ann)


def compute_volatility(
    equity: np.ndarray,
    periods_per_year: int,
) -> float:
    """
    Compute annualized volatility from an equity curve.

    Based on the standard deviation of simple returns.
    """
    rets = _equity_to_returns(equity)
    if rets.size == 0:
        return 0.0
    vol = float(rets.std(ddof=1))
    return vol * np.sqrt(periods_per_year)


def compute_sharpe(
    equity: np.ndarray,
    rf: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Compute (simple) Sharpe ratio from an equity curve.

    Parameters
    ----------
    equity : np.ndarray
        Equity curve.
    rf : float
        Annual risk-free rate (in the same units as returns).
    periods_per_year : int
        Number of periods per year (e.g., 252 for daily data).
    """
    ann_ret = compute_annualized_return(equity, periods_per_year)
    ann_vol = compute_volatility(equity, periods_per_year)
    if ann_vol <= 0.0:
        return 0.0
    return float((ann_ret - rf) / ann_vol)


def compute_max_drawdown(equity: np.ndarray) -> float:
    """
    Compute maximum drawdown (as a positive number) from an equity curve.
    """
    eq = _to_equity_array(equity)
    if eq.size == 0:
        return 0.0
    peaks = np.maximum.accumulate(eq)
    # Drawdowns as negative numbers (0 at peaks).
    drawdowns = eq / peaks - 1.0
    max_dd = float(drawdowns.min())  # most negative drawdown
    return abs(max_dd)


def compute_hit_rate(positions: np.ndarray, returns: np.ndarray) -> float:
    """
    Compute hit rate: fraction of times position * next_return > 0.

    Assumes positions[t] is held over interval [t, t+1], so we compare
    positions[:-1] with returns[1:].
    """
    pos = np.asarray(positions, dtype=float)
    rets = np.asarray(returns, dtype=float)
    if pos.ndim != 1 or rets.ndim != 1:
        raise ValueError("positions and returns must be 1D.")
    if pos.size < 2 or rets.size < 2:
        return 0.0

    aligned_pos = pos[:-1]
    aligned_rets = rets[1:]
    mask = np.isfinite(aligned_pos) & np.isfinite(aligned_rets)
    if not np.any(mask):
        return 0.0
    aligned_pos = aligned_pos[mask]
    aligned_rets = aligned_rets[mask]

    hits = (aligned_pos * aligned_rets > 0.0).astype(float)
    return float(hits.mean()) if hits.size > 0 else 0.0


