"""
Reward-shaping experiment harness.

Note: SingleAssetEnv does not yet accept explicit reward-shaping knobs
(transaction costs, risk penalties). This script is structured so that once
those are added, you can just pass them through.
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


# Reward shaping configs passed directly to SingleAssetEnv.
SHAPING_CONFIGS = [
    {
        "name": "no_shaping",
        "transaction_cost": 0.0,
        "lambda_risk": 0.0,
        "lambda_drawdown": 0.0,
    },
    {
        "name": "high_cost",
        "transaction_cost": 0.002,
        "lambda_risk": 0.0,
        "lambda_drawdown": 0.0,
    },
    {
        "name": "risk_penalty",
        "transaction_cost": 0.0,
        "lambda_risk": 0.2,
        "lambda_drawdown": 0.0,
    },
    {
        "name": "drawdown_guard",
        "transaction_cost": 0.0,
        "lambda_risk": 0.0,
        "lambda_drawdown": 0.5,
    },
]


def main() -> None:
    train_df, _, test_df = load_spy_splits()
    results_dir = make_results_dir("reward_shaping")

    for cfg in SHAPING_CONFIGS:
        name = cfg["name"]
        print(f"\n=== Reward shaping config: {cfg} ===")

        env_config = {k: v for k, v in cfg.items() if k != "name"}
        train_env = make_single_asset_env(train_df, env_config=env_config)
        # Use epochs=30 for better convergence, aligned with baseline experiments
        base_config = make_base_config(epochs=30)
        log_path = str(results_dir / f"{name}_train_logs.json")
        agent = train_env_with_config(train_env, base_config, log_path=log_path)

        test_env = make_single_asset_env(test_df, env_config=env_config)
        eq_ppo = run_policy_episode(test_env, agent)
        min_len = len(eq_ppo)
        dates = test_df.index[:min_len]
        eq_ppo = eq_ppo[:min_len]

        metrics = {
            "total_return": compute_total_return(eq_ppo),
            "annualized_return": compute_annualized_return(eq_ppo, periods_per_year=252),
            "sharpe": compute_sharpe(eq_ppo, periods_per_year=252),
            "max_drawdown": compute_max_drawdown(eq_ppo),
            "shaping_config": cfg,
        }

        out_json = results_dir / f"{name}_metrics.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        plot_path = results_dir / f"{name}_equity.png"
        plot_equity_curves({"PPO": eq_ppo}, dates=dates, out_path=str(plot_path))

        print(f"[{name}] metrics: {metrics}")
        print(f"[{name}] wrote metrics to {out_json}")
        print(f"[{name}] wrote equity plot to {plot_path}")


if __name__ == "__main__":
    main()

