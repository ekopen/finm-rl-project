"""
Plotting utilities for visualizing equity curves and trading behavior.
"""

from __future__ import annotations

from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np


def plot_equity_curves(
    curves: dict[str, np.ndarray],
    dates: Iterable | None = None,
    out_path: str | None = None,
) -> None:
    """
    Plot one or more equity curves on the same axes.

    Parameters
    ----------
    curves : dict[str, np.ndarray]
        Mapping from label to equity curve array.
    dates : iterable or None
        Optional iterable of x-axis values (e.g., dates). If None, use
        integer steps.
    out_path : str or None
        If provided, save the figure to this path instead of showing it.
    """
    if not curves:
        return

    first_curve = next(iter(curves.values()))
    n = len(first_curve)

    if dates is not None:
        x = list(dates)
        if len(x) != n:
            x = list(range(n))
    else:
        x = list(range(n))

    plt.figure(figsize=(10, 6))
    for label, eq in curves.items():
        eq_arr = np.asarray(eq, dtype=float)
        # Normalize to start at 1.0 for fair comparison
        if len(eq_arr) > 0 and eq_arr[0] != 0:
            eq_arr = eq_arr / eq_arr[0]
        plt.plot(x[: len(eq_arr)], eq_arr, label=label)

    plt.xlabel("Time")
    plt.ylabel("Equity")
    plt.title("Equity Curves")
    plt.legend()
    plt.tight_layout()

    if out_path is not None:
        plt.savefig(out_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_price_with_positions(
    price: np.ndarray,
    positions: np.ndarray,
    out_path: str | None = None,
) -> None:
    """
    Plot price series with long/short/flat positions overlaid.

    Positions are visualized as colored markers:
    - long (pos > 0): green
    - short (pos < 0): red
    - flat (pos == 0): gray
    """
    price_arr = np.asarray(price, dtype=float)
    pos_arr = np.asarray(positions, dtype=float)

    if price_arr.shape[0] != pos_arr.shape[0]:
        raise ValueError("price and positions must have the same length.")

    x = np.arange(len(price_arr))

    plt.figure(figsize=(10, 6))
    plt.plot(x, price_arr, label="Price", color="black")

    long_mask = pos_arr > 0
    short_mask = pos_arr < 0
    flat_mask = pos_arr == 0

    plt.scatter(x[long_mask], price_arr[long_mask], color="green", s=10, label="Long")
    plt.scatter(x[short_mask], price_arr[short_mask], color="red", s=10, label="Short")
    plt.scatter(x[flat_mask], price_arr[flat_mask], color="gray", s=10, label="Flat")

    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.title("Price with Positions")
    plt.legend()
    plt.tight_layout()

    if out_path is not None:
        plt.savefig(out_path, dpi=150)
        plt.close()
    else:
        plt.show()


