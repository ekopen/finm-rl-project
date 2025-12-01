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

