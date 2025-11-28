"""
Compare different state/env configurations for PPO.

Currently only:
- single_simple: SingleAssetEnv + make_simple_features

Placeholders:
- single_rich: SingleAssetEnv + rich features (TODO)
- pairs_simple: PairsEnv + pairs state (TODO)
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


CONFIGS = [
    {"name": "single_simple", "type": "single_simple"},
    {"name": "single_rich", "type": "single_rich"},   # TODO
    {"name": "pairs_simple", "type": "pairs_simple"}, # TODO
]


def main() -> None:
    train_df, _, test_df = load_spy_splits()
    results_dir = make_results_dir("states_envs")

    for cfg in CONFIGS:
        cfg_type = cfg["type"]
        name = cfg["name"]

        if cfg_type != "single_simple":
            print(f"Skipping {name} (type={cfg_type}) – not yet implemented.")
            continue

        print(f"\n=== Running state/env config: {name} ===")

        # single_simple: SingleAssetEnv + simple features.
        train_env = make_single_asset_env(train_df)
        base_config = make_base_config()
        log_path = str(results_dir / f"{name}_train_logs.json")
        agent = train_env_with_config(train_env, base_config, log_path=log_path)

        test_env = make_single_asset_env(test_df)
        eq_ppo = run_policy_episode(test_env, agent)
        min_len = len(eq_ppo)
        dates = test_df.index[:min_len]
        eq_ppo = eq_ppo[:min_len]

        metrics = {
            "total_return": compute_total_return(eq_ppo),
            "annualized_return": compute_annualized_return(eq_ppo, periods_per_year=252),
            "sharpe": compute_sharpe(eq_ppo, periods_per_year=252),
            "max_drawdown": compute_max_drawdown(eq_ppo),
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

