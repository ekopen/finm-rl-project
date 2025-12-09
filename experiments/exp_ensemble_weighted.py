from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, replace, fields
from pathlib import Path
from typing import List

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

    # Set Python hash seed for deterministic dictionary iteration order
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.cuda.manual_seed(seed)

        # Enable deterministic algorithms (may warn if some ops can't be deterministic)
        torch.use_deterministic_algorithms(True, warn_only=True)

        # Set CUDNN flags for determinism
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        # If torch is not available, just skip.
        pass


def build_train_and_test_envs():
    """
    Build train and test environments using the same data split.
    """
    train_df, _, test_df = load_spy_splits()
    train_env = make_single_asset_env(train_df)
    test_env = make_single_asset_env(test_df)
    return train_env, test_env, test_df.index


def compute_run_weight(train_metrics: dict, test_metrics: dict) -> float:
    """
    Assign a weight to each run based on training / test performance.

    Priority:
      - If 'avg_ep_return' exists in train_metrics, use the mean of the last few values.
      - Else if 'episode_returns' exists, use the mean of the last few values.
      - Else fall back to test_metrics['sharpe'].

    The final weight is clipped to be non-negative (bad runs get near-zero weights).
    """
    base = 0.0

    # Try to use training-time signals to measure how well the run learned
    if isinstance(train_metrics, dict):
        if "avg_ep_return" in train_metrics:
            vals = np.asarray(train_metrics["avg_ep_return"], dtype=float)
            if vals.size > 0:
                base = float(np.mean(vals[-5:]))  # average over last few points
        elif "episode_returns" in train_metrics:
            vals = np.asarray(train_metrics["episode_returns"], dtype=float)
            if vals.size > 0:
                base = float(np.mean(vals[-5:]))

    # If no training signal is available, fall back to test Sharpe
    if base == 0.0:
        base = float(test_metrics.get("sharpe", 0.0))

    # Do not allow negative weights
    w = max(base, 0.0)
    return w


def main() -> None:
    # ----------------------------------------------------------------------
    # 0. Base PPO config: we do NOT sweep hyperparameters here.
    # ----------------------------------------------------------------------
    base_config: PPOConfig = make_base_config(
        steps_per_epoch=1024,
        epochs=8,
        log_interval=2,
    )

    # ----------------------------------------------------------------------
    # 1. Seeds we want to test for robustness / ensemble
    # ----------------------------------------------------------------------
    N_SEEDS = 100  # you can change this to 30, 50, etc.
    seed_list = list(range(N_SEEDS))

    # For each seed, only run the pretraining variant.
    sweep = []
    for seed in seed_list:
        # If later you want to add a no-pretraining ensemble, uncomment this:
        # sweep.append({"name": f"seed_{seed}_nopre", "seed": seed, "pretrained": False})
        sweep.append({"name": f"seed_{seed}_pre", "seed": seed, "pretrained": True})

    # Directory where all results will be stored.
    results_dir = make_results_dir("ppo_ensemble_weighted")
    run_payloads: List[dict] = []

    # Build envs once (if your env is very stateful, you may rebuild inside the loop)
    train_env, test_env, test_index = build_train_and_test_envs()

    # Check if PPOConfig has a "seed" field
    has_seed_field = any(f.name == "seed" for f in fields(PPOConfig))

    # Path to the BC-pretrained PPO checkpoint.
    ckpt = Path("results/pretrain_ma_bc/ppo_bc_pretrained.pt")

    # ----------------------------------------------------------------------
    # 2. Containers for ensemble equity curves (full paths) and weights
    # ----------------------------------------------------------------------
    ensemble_curves = {
        "pre": [],
        # "nopre": [],  # if you later add a non-pretrained ensemble
    }
    ensemble_weights = {
        "pre": [],
        # "nopre": [],
    }

    # ----------------------------------------------------------------------
    # 3. Main loop: iterate over seeds
    # ----------------------------------------------------------------------
    for setting in sweep:
        name = setting["name"]
        seed = setting["seed"]
        pretrained = setting["pretrained"]

        print(f"\n=== Running config: {name} ===")
        print(f"Seed       : {seed}")
        print(f"Pretrained : {pretrained}")

        # (1) Set global random seed.
        set_global_seed(seed)

        # (2) If the env supports seed(), also set it.
        if hasattr(train_env, "seed"):
            train_env.seed(seed)
        if hasattr(test_env, "seed"):
            test_env.seed(seed)

        # (3) Build PPOConfig, optionally injecting the seed.
        if has_seed_field:
            config = replace(base_config, seed=seed)
        else:
            config = base_config

        print("PPOConfig:", config)

        # (4) Create a fresh agent for this run.
        state = train_env.reset()
        # Gym/Gymnasium compatibility: reset may return (obs, info).
        if isinstance(state, tuple) and len(state) >= 1:
            state = state[0]
        state_dim = state.shape[0]
        action_dim = train_env.action_space_n

        agent = PPOAgent(state_dim=state_dim, action_dim=action_dim, config=config)

        # (4.1) Optionally load BC-pretrained weights.
        if pretrained:
            if ckpt.exists():
                print(f"Loading BC-pretrained PPO weights from: {ckpt}")
                agent.load(str(ckpt))
            else:
                print(
                    f"[Warning] Pretraining checkpoint not found at {ckpt}. "
                    f"Continuing with randomly initialized weights."
                )

        # (5) Train PPO.
        log_path = results_dir / f"{name}_train_logs.json"
        metrics_train = train_ppo(
            train_env,
            agent,
            config,
            log_path=str(log_path),
        )

        # (6) Evaluate on the test environment (get full equity curve).
        eq_ppo = run_policy_episode(test_env, agent)
        min_len = len(eq_ppo)
        dates = test_index[:min_len]
        eq_ppo = eq_ppo[:min_len]

        # (7) Compute scalar test metrics.
        test_metrics = {
            "total_return": compute_total_return(eq_ppo),
            "annualized_return": compute_annualized_return(
                eq_ppo, periods_per_year=252
            ),
            "sharpe": compute_sharpe(eq_ppo, periods_per_year=252),
            "max_drawdown": compute_max_drawdown(eq_ppo),
        }

        print(f"[{name}] test_metrics: {test_metrics}")

        # (7.1) Compute the weight for this run (prefer training signals)
        weight = compute_run_weight(metrics_train, test_metrics)

        # (6.1) Store full path and weight in the ensemble containers
        group_key = "pre" if pretrained else "nopre"
        if group_key not in ensemble_curves:
            ensemble_curves[group_key] = []
            ensemble_weights[group_key] = []
        ensemble_curves[group_key].append(eq_ppo.copy())
        ensemble_weights[group_key].append(weight)

        # (8) Save equity curve plot for this single run.
        plot_path = results_dir / f"{name}_equity.png"
        plot_equity_curves({"PPO": eq_ppo}, dates=dates, out_path=str(plot_path))

        # (9) Build run payload for later summarization.
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

    # ----------------------------------------------------------------------
    # 4. Per-run summary (scalar metrics)
    # ----------------------------------------------------------------------
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

    # ----------------------------------------------------------------------
    # 5. Group-wise mean and std for scalar metrics (kept from original logic)
    # ----------------------------------------------------------------------
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

    stats_df = pd.DataFrame(rows)
    group_stats_csv = results_dir / "summary_group_stats.csv"
    stats_df.to_csv(group_stats_csv, index=False)
    print(f"\nWrote group-wise mean/std CSV to {group_stats_csv}")

    # ----------------------------------------------------------------------
    # 6. Weighted ensemble over the full equity paths: mean & std at each step
    # ----------------------------------------------------------------------
    en
