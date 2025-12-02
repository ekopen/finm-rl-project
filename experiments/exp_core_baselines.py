"""
Core benchmark: PPO vs simple baselines (buy-and-hold, MA crossover) on SPY.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from baselines.buy_and_hold import run_buy_and_hold
from baselines.ma_crossover import run_ma_crossover
from eval.metrics import (
    compute_annualized_return,
    compute_max_drawdown,
    compute_sharpe,
    compute_total_return,
)
from eval.plotting import plot_equity_curves
from ppo.eval_utils import run_policy_episode

from experiments.common import (
    load_spy_splits,
    make_base_config,
    make_results_dir,
    make_single_asset_env,
    train_env_with_config,
)


def main() -> None:
    train_df, _, test_df = load_spy_splits()
    results_dir = make_results_dir("core_baselines")

    # Train PPO on train period.
    train_env = make_single_asset_env(train_df)
    config = make_base_config()
    log_path = str(results_dir / "ppo_train_logs.json")
    agent = train_env_with_config(train_env, config, log_path=log_path)

    # Evaluate PPO policy on test period.
    test_env = make_single_asset_env(test_df)
    eq_ppo = run_policy_episode(test_env, agent, debug=True)

    # Baselines on test period.
    eq_bh = run_buy_and_hold(test_df)
    eq_ma, _ = run_ma_crossover(test_df)

    # Align curves by truncating to minimum length.
    min_len = min(len(eq_ppo), len(eq_bh), len(eq_ma))
    eq_ppo = eq_ppo[:min_len]
    eq_bh = eq_bh[:min_len]
    eq_ma = eq_ma[:min_len]
    dates = test_df.index[:min_len]

    metrics: dict[str, dict[str, float]] = {}
    for name, equity in [
        ("PPO", eq_ppo),
        ("BuyAndHold", eq_bh),
        ("MACrossover", eq_ma),
    ]:
        metrics[name] = {
            "total_return": compute_total_return(equity),
            "annualized_return": compute_annualized_return(equity, periods_per_year=252),
            "sharpe": compute_sharpe(equity, rf=0.0, periods_per_year=252),
            "max_drawdown": compute_max_drawdown(equity),
        }

    # Save metrics.
    metrics_path = results_dir / "core_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Save equity curve plot.
    curves = {
        "PPO": eq_ppo,
        "Buy & Hold": eq_bh,
        "MA Crossover": eq_ma,
    }
    plot_path = results_dir / "core_equity.png"
    plot_equity_curves(curves, dates=dates, out_path=str(plot_path))

    print(f"Wrote metrics to {metrics_path}")
    print(f"Wrote equity plot to {plot_path}")


if __name__ == "__main__":
    main()


