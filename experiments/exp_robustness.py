"""
PPO seed robustness + pretraining comparison experiments.

Runs multiple short PPO trainings on SingleAssetEnv with the SAME PPOConfig
but different random seeds AND an additional switch:
    - without BC pretraining (randomly initialized PPO)
    - with BC pretraining (load weights from a checkpoint)

For each (seed, pretrained_flag) combination:
    - train on train_env
    - evaluate on test_env
    - save training + test metrics and configs

At the end:
    - write per-run summary (JSON + CSV)
    - compute mean and std of test metrics for:
        * pretrained = False
        * pretrained = True
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, replace, fields
from pathlib import Path
from typing import List

import random
import numpy as np
import pandas as pd  # used to save the group-wise mean/std as a CSV

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


def build_train_and_test_envs():
    """
    Build train and test environments using the same data split as before.
    """
    train_df, _, test_df = load_spy_splits()
    train_env = make_single_asset_env(train_df)
    test_env = make_single_asset_env(test_df)
    return train_env, test_env, test_df.index


def main() -> None:
    # Base PPO config: we will NOT sweep hyperparameters here.
    base_config: PPOConfig = make_base_config(
        steps_per_epoch=1024,
        epochs=8,
        log_interval=2,
    )

    # Seeds we want to test for robustness.
    seed_list = [0, 1, 2, 3, 4]

    # Build the sweep:
    # For each seed, we run:
    #   - one experiment WITHOUT pretraining (random init)
    #   - one experiment WITH pretraining (load BC checkpoint)
    sweep = []
    for seed in seed_list:
        sweep.append(
            {"name": f"seed_{seed}_nopre", "seed": seed, "pretrained": False}
        )
        sweep.append(
            {"name": f"seed_{seed}_pre", "seed": seed, "pretrained": True}
        )

    # Directory where all results will be stored.
    results_dir = make_results_dir("ppo_seed_pretrain_compare")
    run_payloads: List[dict] = []

    # Build envs once. If your env is stateful in a way that makes this problematic,
    # you could also rebuild envs inside the loop instead.
    train_env, test_env, test_index = build_train_and_test_envs()

    # Check if PPOConfig has a "seed" field (not strictly required, but nice if present).
    has_seed_field = any(f.name == "seed" for f in fields(PPOConfig))

    # Path to the BC-pretrained PPO checkpoint.
    # Make sure this path matches your actual pretraining output.
    ckpt = Path("results/pretrain_ma_bc/ppo_bc_pretrained.pt")

    for setting in sweep:
        name = setting["name"]
        seed = setting["seed"]
        pretrained = setting["pretrained"]

        print(f"\n=== Running config: {name} ===")
        print(f"Seed       : {seed}")
        print(f"Pretrained : {pretrained}")

        # 1. Set global random seed.
        set_global_seed(seed)

        # 2. If the env supports seed(), also set it.
        if hasattr(train_env, "seed"):
            train_env.seed(seed)
        if hasattr(test_env, "seed"):
            test_env.seed(seed)

        # 3. Build PPOConfig, optionally injecting the seed.
        if has_seed_field:
            config = replace(base_config, seed=seed)
        else:
            config = base_config

        print("PPOConfig:", config)

        # 4. Create a fresh agent for this run.
        state = train_env.reset()
        # Gym/Gymnasium compatibility: reset may return (obs, info).
        if isinstance(state, tuple) and len(state) >= 1:
            state = state[0]
        state_dim = state.shape[0]
        action_dim = train_env.action_space_n

        agent = PPOAgent(state_dim=state_dim, action_dim=action_dim, config=config)

        # 4.1 Optionally load BC-pretrained weights.
        if pretrained:
            if ckpt.exists():
                print(f"Loading BC-pretrained PPO weights from: {ckpt}")
                agent.load(str(ckpt))
            else:
                print(
                    f"[Warning] Pretraining checkpoint not found at {ckpt}. "
                    f"Continuing with randomly initialized weights."
                )

        # 5. Train PPO.
        log_path = results_dir / f"{name}_train_logs.json"
        metrics_train = train_ppo(
            train_env,
            agent,
            config,
            log_path=str(log_path),
        )

        # 6. Evaluate on the test environment.
        eq_ppo = run_policy_episode(test_env, agent)
        min_len = len(eq_ppo)
        dates = test_index[:min_len]
        eq_ppo = eq_ppo[:min_len]

        test_metrics = {
            "total_return": compute_total_return(eq_ppo),
            "annualized_return": compute_annualized_return(
                eq_ppo, periods_per_year=252
            ),
            "sharpe": compute_sharpe(eq_ppo, periods_per_year=252),
            "max_drawdown": compute_max_drawdown(eq_ppo),
        }

        print(f"[{name}] test_metrics: {test_metrics}")

        # 7. Save equity curve plot.
        plot_path = results_dir / f"{name}_equity.png"
        plot_equity_curves({"PPO": eq_ppo}, dates=dates, out_path=str(plot_path))

        # 8. Build run payload for later summarization.
        run_payload = {
            "name": name,
            "seed": seed,
            "pretrained": pretrained,
            "config": asdict(config),
            "train_metrics": metrics_train,
            "test_metrics": test_metrics,
        }
        run_payloads.append(run_payload)

        # Save one JSON per run.
        out_path = results_dir / f"{name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(run_payload, f, indent=2)
        print(f"Saved run payload to {out_path}")

    # -----------------------------------------------------------------------
    # 9. Per-run summary (same style as before).
    # -----------------------------------------------------------------------
    summary = {p["name"]: p["test_metrics"] for p in run_payloads}
    summary_json = results_dir / "summary_test_metrics.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    df = summarize_runs(run_payloads, metric_key="test_metrics")
    summary_csv = results_dir / "summary_test_metrics.csv"
    df.to_csv(summary_csv, index=False)
    print("\nPer-run seed + pretraining summary (test metrics):")
    print(df.to_string(index=False))
    print(f"Wrote summary JSON to {summary_json}")
    print(f"Wrote summary CSV  to {summary_csv}")

    # -----------------------------------------------------------------------
    # 10. Compute mean and std for each group: pretrained vs non-pretrained.
    # -----------------------------------------------------------------------
    metrics = ["total_return", "annualized_return", "sharpe", "max_drawdown"]
    rows = []

    for group_name, flag in [("nopre", False), ("pre", True)]:
        group_runs = [p for p in run_payloads if p["pretrained"] == flag]
        if not group_runs:
            continue

        print(f"\nGroup '{group_name}' (pretrained={flag}) has {len(group_runs)} runs.")

        for metric in metrics:
            values = [r["test_metrics"][metric] for r in group_runs]
            arr = np.array(values, dtype=float)
            mean = float(np.mean(arr))
            # Use sample standard deviation (ddof=1) if at least 2 runs exist.
            if len(arr) > 1:
                std = float(np.std(arr, ddof=1))
            else:
                std = 0.0

            rows.append(
                {
                    "group": group_name,
                    "pretrained": flag,
                    "metric": metric,
                    "mean": mean,
                    "std": std,
                }
            )

            print(
                f"[Group={group_name}] {metric:18s} : "
                f"mean={mean:.6f}, std={std:.6f}"
            )

    # Save group-wise statistics to a separate CSV.
    stats_df = pd.DataFrame(rows)
    group_stats_csv = results_dir / "summary_group_stats.csv"
    stats_df.to_csv(group_stats_csv, index=False)
    print(f"\nWrote group-wise mean/std CSV to {group_stats_csv}")


if __name__ == "__main__":
    # Ensure project root is importable when running directly.
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    main()

