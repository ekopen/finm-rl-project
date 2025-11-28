"""
PPO hyperparameter sweep experiments.

Runs multiple short PPO trainings on SingleAssetEnv with different PPOConfig
settings and saves training + test metrics and configs.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import List

from eval.metrics import (
    compute_annualized_return,
    compute_max_drawdown,
    compute_sharpe,
    compute_total_return,
)
from eval.plotting import plot_equity_curves
from ppo.eval_utils import run_policy_episode
from ppo.ppo_agent import PPOAgent, PPOConfig
from ppo.trainer import train_ppo

from eval.summarize import summarize_runs
from experiments.common import (
    load_spy_splits,
    make_base_config,
    make_results_dir,
    make_single_asset_env,
)


def build_train_and_test_envs():
    train_df, _, test_df = load_spy_splits()
    train_env = make_single_asset_env(train_df)
    test_env = make_single_asset_env(test_df)
    return train_env, test_env, test_df.index


def main() -> None:
    base_config = make_base_config(
        steps_per_epoch=1024,
        epochs=8,
        log_interval=2,
    )

    sweep = [
        {
            "name": "clip0.1_ent0.00",
            "overrides": {"clip_eps": 0.1, "entropy_coef": 0.0},
        },
        {
            "name": "clip0.2_ent0.01",
            "overrides": {"clip_eps": 0.2, "entropy_coef": 0.01},
        },
        {
            "name": "clip0.3_ent0.01",
            "overrides": {"clip_eps": 0.3, "entropy_coef": 0.01},
        },
        {
            "name": "gamma0.95_lambda0.90",
            "overrides": {"gamma": 0.95, "gae_lambda": 0.90},
        },
    ]

    results_dir = make_results_dir("ppo_hyperparams")
    run_payloads: List[dict] = []

    train_env, test_env, test_index = build_train_and_test_envs()

    for setting in sweep:
        name = setting["name"]
        overrides = setting["overrides"]

        config = replace(base_config, **overrides)
        print(f"\n=== Running config: {name} ===")
        print("PPOConfig:", config)

        # Fresh agent for each run.
        state_dim = train_env.reset().shape[0]
        action_dim = train_env.action_space_n
        agent = PPOAgent(state_dim=state_dim, action_dim=action_dim, config=config)

        metrics_train = train_ppo(
            train_env,
            agent,
            config,
            log_path=str(results_dir / f"{name}_train_logs.json"),
        )

        # Evaluate on test env.
        eq_ppo = run_policy_episode(test_env, agent)
        min_len = len(eq_ppo)
        dates = test_index[:min_len]
        eq_ppo = eq_ppo[:min_len]

        test_metrics = {
            "total_return": compute_total_return(eq_ppo),
            "annualized_return": compute_annualized_return(eq_ppo, periods_per_year=252),
            "sharpe": compute_sharpe(eq_ppo, periods_per_year=252),
            "max_drawdown": compute_max_drawdown(eq_ppo),
        }

        print(f"[{name}] test_metrics: {test_metrics}")

        # Save equity plot for this run.
        plot_path = results_dir / f"{name}_equity.png"
        plot_equity_curves({"PPO": eq_ppo}, dates=dates, out_path=str(plot_path))

        run_payload = {
            "name": name,
            "config": asdict(config),
            "train_metrics": metrics_train,
            "test_metrics": test_metrics,
        }
        run_payloads.append(run_payload)

        out_path = results_dir / f"{name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(run_payload, f, indent=2)
        print(f"Saved run payload to {out_path}")

    # Simple summary artifact (JSON + CSV + printed table).
    summary = {p["name"]: p["test_metrics"] for p in run_payloads}
    summary_json = results_dir / "summary_test_metrics.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    df = summarize_runs(run_payloads, metric_key="test_metrics")
    summary_csv = results_dir / "summary_test_metrics.csv"
    df.to_csv(summary_csv, index=False)
    print("\nHyperparameter summary:")
    print(df.to_string(index=False))
    print(f"Wrote summary JSON to {summary_json}")
    print(f"Wrote summary CSV to {summary_csv}")


if __name__ == "__main__":
    # Ensure project root is importable when running directly.
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    main()


