"""
Regime-based PPO experiments (scaffold).

Currently trains PPO on SPY and evaluates on a test period overall.
Once regime labels are implemented in features.regimes, this script will
slice metrics by regime (bull/bear, high/low vol).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from eval.metrics import (
    compute_annualized_return,
    compute_max_drawdown,
    compute_sharpe,
    compute_total_return,
)
from eval.plotting import plot_equity_curves
from features.regimes import label_bull_bear, label_volatility
from ppo.eval_utils import run_policy_episode

from experiments.common import (
    load_spy_splits,
    make_base_config,
    make_results_dir,
    make_single_asset_env,
    train_env_with_config,
)


def _equity_from_returns(returns: pd.Series) -> list[float]:
    """Convert a return series into an equity curve starting at 1.0."""
    if returns.empty:
        return [1.0]
    equity = np.concatenate(([1.0], np.cumprod(1.0 + returns.values)))
    return equity.tolist()


def _metrics_from_returns(returns: pd.Series) -> dict[str, float]:
    equity = _equity_from_returns(returns)
    return {
        "total_return": compute_total_return(equity),
        "annualized_return": compute_annualized_return(equity, periods_per_year=252),
        "sharpe": compute_sharpe(equity, periods_per_year=252),
        "max_drawdown": compute_max_drawdown(equity),
    }


def _plot_equity_with_regimes(
    eq,
    dates,
    regimes,
    label_to_color=None,
    title="Equity Curve with Regimes",
    out_path=None,
):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.patches import Patch

    # Convert inputs
    eq = np.asarray(eq, dtype=float)
    x = pd.to_datetime(dates)  # Use actual dates on x-axis

    # Ensure regimes is a pandas Series aligned with eq
    regimes = pd.Series(regimes).reset_index(drop=True)

    # Get unique regimes in order of appearance
    unique_regimes = list(pd.unique(regimes))

    # If no color mapping is provided, assign colors automatically
    if label_to_color is None:
        default_colors = ["green", "red", "blue", "orange", "purple", "gray"]
        label_to_color = {}
        for i, reg in enumerate(unique_regimes):
            label_to_color[reg] = default_colors[i % len(default_colors)]

    plt.figure(figsize=(12, 6))

    # Plot equity curve
    plt.plot(x, eq, label="Equity", color="black")

    # Draw regime shading by contiguous segments
    current = regimes.iloc[0]
    start = 0

    for i in range(1, len(regimes)):
        if regimes.iloc[i] != current:
            color = label_to_color.get(current, "gray")
            plt.axvspan(x[start], x[i], color=color, alpha=0.15)
            current = regimes.iloc[i]
            start = i

    # Last segment
    color = label_to_color.get(current, "gray")
    plt.axvspan(x[start], x[len(regimes) - 1], color=color, alpha=0.15)

    # Legend for regimes + equity
    regime_handles = [
        Patch(facecolor=label_to_color[reg], alpha=0.15, label=str(reg))
        for reg in unique_regimes
    ]
    equity_handle = plt.Line2D([], [], color="black", label="Equity")

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.legend(handles=regime_handles + [equity_handle])
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=150)
        plt.close()
    else:
        plt.show()
        
        
def _compute_regime_metrics(
    returns: pd.Series, labels: pd.Series
) -> dict[str, dict[str, float]]:
    """Compute metrics for each label value given per-step returns."""
    metrics: dict[str, dict[str, float]] = {}
    valid = labels.dropna().unique()
    for regime in sorted(valid):
        mask = labels == regime
        regime_returns = returns[mask]
        if regime_returns.empty:
            continue
        metrics[regime] = {
            **_metrics_from_returns(regime_returns),
            "count": int(mask.sum()),
        }
    return metrics


def main() -> None:
    train_df, _, test_df = load_spy_splits()
    results_dir = make_results_dir("regimes")

    # Train on train period.
    train_env = make_single_asset_env(train_df)
    base_config = make_base_config()
    log_path = str(results_dir / "ppo_train_logs.json")
    agent = train_env_with_config(train_env, base_config, log_path=log_path)

    # Evaluate on test period overall.
    test_env = make_single_asset_env(test_df)
    eq_ppo = run_policy_episode(test_env, agent)
    min_len = len(eq_ppo)
    dates = test_df.index[:min_len]
    eq_ppo = eq_ppo[:min_len]
    eq_series = pd.Series(eq_ppo, index=dates, dtype=float)
    returns = eq_series.pct_change().iloc[1:]

    overall_metrics = {
        "total_return": compute_total_return(eq_ppo),
        "annualized_return": compute_annualized_return(eq_ppo, periods_per_year=252),
        "sharpe": compute_sharpe(eq_ppo, periods_per_year=252),
        "max_drawdown": compute_max_drawdown(eq_ppo),
    }

    out_json = results_dir / "overall_metrics.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(overall_metrics, f, indent=2)

    plot_path = results_dir / "overall_equity.png"
    plot_equity_curves({"PPO": eq_ppo}, dates=dates, out_path=str(plot_path))

    print("Overall metrics:", overall_metrics)
    print(f"Wrote overall metrics to {out_json}")
    print(f"Wrote equity plot to {plot_path}")

    # Regime splits (bull/bear and vol regimes).
    return_index = returns.index
    bull_bear = label_bull_bear(test_df).reindex(return_index).ffill().bfill()
    vol_regime = label_volatility(test_df).reindex(return_index).ffill().bfill()
        
    # Bull/Bear regimes
    plot_path = results_dir / "bull_bear_equity.png"
    _plot_equity_with_regimes(
        eq_ppo,
        dates,
        bull_bear,
        label_to_color={"bull": "green", "bear": "red"},
        title="Equity Curve with Bull/Bear Regimes",
        out_path=str(plot_path),
    )

    # Volatility regimes
    plot_path = results_dir / "vol_regime_equity.png"
    _plot_equity_with_regimes(
        eq_ppo,
        dates,
        vol_regime,
        label_to_color={"low_vol": "blue", "high_vol": "orange"},
        title="Equity Curve with Volatility Regimes",
        out_path=str(plot_path),
    )

    regime_metrics = {
        "bull_bear": _compute_regime_metrics(returns, bull_bear),
        "volatility": _compute_regime_metrics(returns, vol_regime),
    }

    regime_json = results_dir / "regime_metrics.json"
    with open(regime_json, "w", encoding="utf-8") as f:
        json.dump(regime_metrics, f, indent=2)

    print(f"Wrote regime metrics to {regime_json}")


if __name__ == "__main__":
    main()

