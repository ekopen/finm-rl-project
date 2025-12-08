"""
Core benchmark: PPO vs simple baselines (buy-and-hold, MA crossover) on SPY.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

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


def set_global_seed(seed: int) -> None:
    """Set global random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        # If torch is not available, just skip.
        pass


def main() -> None:
    # Set seed for reproducibility
    set_global_seed(42)
    
    train_df, _, test_df = load_spy_splits()
    results_dir = make_results_dir("core_baselines")

    # Train PPO on train period.
    # Explicitly set transaction_cost=0.0 for fair comparison with baselines
    train_env = make_single_asset_env(
        train_df, env_config={"transaction_cost": 0.0, "lambda_risk": 0.0, "lambda_drawdown": 0.0}
    )
    config = make_base_config()
    log_path = str(results_dir / "ppo_train_logs.json")
    agent = train_env_with_config(train_env, config, log_path=log_path)

    # Evaluate PPO policy on test period.
    # Explicitly set transaction_cost=0.0 for fair comparison with baselines
    test_env = make_single_asset_env(
        test_df, env_config={"transaction_cost": 0.0, "lambda_risk": 0.0, "lambda_drawdown": 0.0}
    )
    eq_ppo = run_policy_episode(test_env, agent, debug=True)

    # Baselines on test period.
    # MA crossover with shorting enabled for fair comparison with PPO
    eq_bh = run_buy_and_hold(test_df)
    eq_ma, dates_ma = run_ma_crossover(test_df, allow_short=True)

    # Get dates for PPO and Buy & Hold
    # PPO equity has length N (initial + N-1 steps), where N = len(test_df)
    # Dates correspond to test_df.index
    dates_ppo = test_df.index[:len(eq_ppo)]
    dates_bh = test_df.index[:len(eq_bh)]

    # Align all strategies by date intersection
    # Find common dates across all strategies
    common_dates = dates_ppo.intersection(dates_bh).intersection(dates_ma)
    
    if len(common_dates) == 0:
        error_msg = "No overlapping dates found between strategies!\n"
        error_msg += f"PPO date range: {dates_ppo[0]} to {dates_ppo[-1]} ({len(dates_ppo)} points)\n"
        error_msg += f"Buy & Hold date range: {dates_bh[0]} to {dates_bh[-1]} ({len(dates_bh)} points)\n"
        error_msg += f"MA Crossover date range: {dates_ma[0]} to {dates_ma[-1]} ({len(dates_ma)} points)"
        raise ValueError(error_msg)
    
    # Validate date ranges
    print(f"\n=== Date Range Validation ===")
    print(f"PPO dates: {dates_ppo[0]} to {dates_ppo[-1]} ({len(dates_ppo)} points)")
    print(f"Buy & Hold dates: {dates_bh[0]} to {dates_bh[-1]} ({len(dates_bh)} points)")
    print(f"MA Crossover dates: {dates_ma[0]} to {dates_ma[-1]} ({len(dates_ma)} points)")
    print(f"Common dates: {common_dates[0]} to {common_dates[-1]} ({len(common_dates)} points)")
    
    if len(common_dates) < len(dates_ppo) or len(common_dates) < len(dates_bh) or len(common_dates) < len(dates_ma):
        print(f"WARNING: Date ranges don't fully overlap. Using {len(common_dates)} common dates.")
    
    # Align equity curves to common dates
    # Create DataFrames with dates as index for easy alignment
    df_ppo = pd.Series(eq_ppo, index=dates_ppo)
    df_bh = pd.Series(eq_bh, index=dates_bh)
    df_ma = pd.Series(eq_ma, index=dates_ma)
    
    # Reindex to common dates
    eq_ppo_aligned = df_ppo.reindex(common_dates).values
    eq_bh_aligned = df_bh.reindex(common_dates).values
    eq_ma_aligned = df_ma.reindex(common_dates).values
    
    # Check for NaN values after reindexing (indicates date misalignment)
    nan_checks = {
        "PPO": np.isnan(eq_ppo_aligned).sum(),
        "Buy & Hold": np.isnan(eq_bh_aligned).sum(),
        "MA Crossover": np.isnan(eq_ma_aligned).sum(),
    }
    if any(count > 0 for count in nan_checks.values()):
        error_msg = "NaN values found after date alignment - dates may not align correctly!\n"
        error_msg += f"NaN counts: {nan_checks}\n"
        error_msg += f"PPO date range: {dates_ppo[0]} to {dates_ppo[-1]} ({len(dates_ppo)} points)\n"
        error_msg += f"Buy & Hold date range: {dates_bh[0]} to {dates_bh[-1]} ({len(dates_bh)} points)\n"
        error_msg += f"MA Crossover date range: {dates_ma[0]} to {dates_ma[-1]} ({len(dates_ma)} points)\n"
        error_msg += f"Common dates: {common_dates[0]} to {common_dates[-1]} ({len(common_dates)} points)"
        raise ValueError(error_msg)
    
    # Validate lengths match
    len_ppo = len(eq_ppo_aligned)
    len_bh = len(eq_bh_aligned)
    len_ma = len(eq_ma_aligned)
    len_common = len(common_dates)
    
    if not (len_ppo == len_bh == len_ma == len_common):
        error_msg = "Length mismatch after date alignment!\n"
        error_msg += f"PPO: {len_ppo}, Buy & Hold: {len_bh}, MA Crossover: {len_ma}, Common dates: {len_common}"
        raise ValueError(error_msg)
    
    # Use aligned values
    eq_ppo = eq_ppo_aligned
    eq_bh = eq_bh_aligned
    eq_ma = eq_ma_aligned
    dates = common_dates

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


