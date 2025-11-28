"""
Regime-based PPO experiments (scaffold).

Currently trains PPO on SPY and evaluates on a test period overall.
Once regime labels are implemented in features.regimes, this script will
slice metrics by regime (bull/bear, high/low vol).
"""

from __future__ import annotations

import json

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

    # TODO: once features.regimes.label_bull_bear / label_volatility are implemented:
    # - label each date in test_df by regime
    # - slice eq_ppo / returns series by regime and recompute metrics per regime
    # - save regime-split tables and, optionally, regime-colored equity plots


if __name__ == "__main__":
    main()

