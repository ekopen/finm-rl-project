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

    # Set Python hash seed for dictionary iteration order determinism
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
    Build train and test environments using the same data split as before.
    """
    train_df, _, test_df = load_spy_splits()
    train_env = make_single_asset_env(train_df)
    test_env = make_single_asset_env(test_df)
    return train_env, test_env, test_df.index


def main() -> None:
    # ----------------------------------------------------------------------
    # 0. Base PPO config: we will NOT sweep hyperparameters here.
    # ----------------------------------------------------------------------
    base_config: PPOConfig = make_base_config(
        steps_per_epoch=1024,
        epochs=8,
        log_interval=2,
    )

    # ----------------------------------------------------------------------
    # 1. Seeds we want to test for robustness / ensemble
    # ----------------------------------------------------------------------
    N_SEEDS = 100  # 你可以改成 30, 50 等
    seed_list = list(range(N_SEEDS))

    # For each seed, we run:
    #   - one experiment WITHOUT pretraining (random init)
    #   - one experiment WITH pretraining (load BC checkpoint)
    sweep = []
    for seed in seed_list:
        # sweep.append({"name": f"seed_{seed}_nopre", "seed": seed, "pretrained": False})
        sweep.append({"name": f"seed_{seed}_pre", "seed": seed, "pretrained": True})

    # Directory where all results will be stored.
    results_dir = make_results_dir("ppo_ensemble")
    run_payloads: List[dict] = []

    # Build envs once (if你的 env 很强 stateful，可以在 loop 里重建)
    train_env, test_env, test_index = build_train_and_test_envs()

    # Check if PPOConfig has a "seed" field
    has_seed_field = any(f.name == "seed" for f in fields(PPOConfig))

    # Path to the BC-pretrained PPO checkpoint.
    ckpt = Path("results/pretrain_ma_bc/ppo_bc_pretrained.pt")

    # ----------------------------------------------------------------------
    # 2. 用来存 ensemble equity curves（完整路径）
    # ----------------------------------------------------------------------
    ensemble_curves = {
        "pre": [],
    }

    # ----------------------------------------------------------------------
    # 3. 主循环：不同 seed × (pre / nopre)
    # ----------------------------------------------------------------------
    for setting in sweep:
        name = setting["name"]
        seed = setting["seed"]
        pretrained = setting["pretrained"]

        print(f"\n=== Running config: {name} ===")
        print(f"Seed       : {seed}")
        print(f"Pretrained : {pretrained}")

        # 1) Set global random seed.
        set_global_seed(seed)

        # 2) If the env supports seed(), also set it.
        if hasattr(train_env, "seed"):
            train_env.seed(seed)
        if hasattr(test_env, "seed"):
            test_env.seed(seed)

        # 3) Build PPOConfig, optionally injecting the seed.
        if has_seed_field:
            config = replace(base_config, seed=seed)
        else:
            config = base_config

        print("PPOConfig:", config)

        # 4) Create a fresh agent for this run.
        state = train_env.reset()
        # Gym/Gymnasium compatibility: reset may return (obs, info).
        if isinstance(state, tuple) and len(state) >= 1:
            state = state[0]
        state_dim = state.shape[0]
        action_dim = train_env.action_space_n

        agent = PPOAgent(state_dim=state_dim, action_dim=action_dim, config=config)

        # 4.1) Optionally load BC-pretrained weights.
        if pretrained:
            if ckpt.exists():
                print(f"Loading BC-pretrained PPO weights from: {ckpt}")
                agent.load(str(ckpt))
            else:
                print(
                    f"[Warning] Pretraining checkpoint not found at {ckpt}. "
                    f"Continuing with randomly initialized weights."
                )

        # 5) Train PPO.
        log_path = results_dir / f"{name}_train_logs.json"
        metrics_train = train_ppo(
            train_env,
            agent,
            config,
            log_path=str(log_path),
        )

        # 6) Evaluate on the test environment (get full equity curve).
        eq_ppo = run_policy_episode(test_env, agent)
        min_len = len(eq_ppo)
        dates = test_index[:min_len]
        eq_ppo = eq_ppo[:min_len]

        # 6.1) 把完整路径存进 ensemble 容器（按是否 pretrained 分开）
        group_key = "pre" if pretrained else "nopre"
        ensemble_curves[group_key].append(eq_ppo.copy())

        # 7) Compute scalar test metrics.
        test_metrics = {
            "total_return": compute_total_return(eq_ppo),
            "annualized_return": compute_annualized_return(
                eq_ppo, periods_per_year=252
            ),
            "sharpe": compute_sharpe(eq_ppo, periods_per_year=252),
            "max_drawdown": compute_max_drawdown(eq_ppo),
        }

        print(f"[{name}] test_metrics: {test_metrics}")

        # 8) Save equity curve plot for this single run.
        plot_path = results_dir / f"{name}_equity.png"
        plot_equity_curves({"PPO": eq_ppo}, dates=dates, out_path=str(plot_path))

        # 9) Build run payload for later summarization.
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
    # 4. Per-run summary (scalar metrics) – 和原来一样
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
    # 5. Compute mean and std for each group (scalar metrics) – 原有逻辑
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
    # 6. 重点：按路径做 ensemble，计算逐步均值 & 标准差
    # ----------------------------------------------------------------------
    ensemble_stats = {}

    for group_name, curves in ensemble_curves.items():
        if not curves:
            continue

        # 对齐长度（一般都一样，这里保险起见）
        min_len = min(len(c) for c in curves)
        aligned = np.stack([c[:min_len] for c in curves], axis=0)  # [n_runs, T]

        mean_path = aligned.mean(axis=0)
        std_path = aligned.std(axis=0, ddof=1)

        ensemble_stats[group_name] = {
            "mean_path": mean_path,
            "std_path": std_path,
        }

        # 保存到 CSV（带日期）
        dates = test_index[:min_len]
        df_path = pd.DataFrame(
            {
                "date": dates,
                "mean_equity": mean_path,
                "std_equity": std_path,
            }
        )
        out_csv = results_dir / f"ensemble_{group_name}_equity_path.csv"
        df_path.to_csv(out_csv, index=False)
        print(f"Saved ensemble path for group '{group_name}' to {out_csv}")

    # ----------------------------------------------------------------------
    # 7. 画 ensemble mean equity curve（带 std 阴影）
    # ----------------------------------------------------------------------
    if ensemble_stats:
        plt.figure(figsize=(10, 6))
        for group_name, stats in ensemble_stats.items():
            mean_path = stats["mean_path"]
            std_path = stats["std_path"]
            steps = np.arange(len(mean_path))

            if group_name == "pre":
                label = "Pre-training"
            else:
                label = "No Pre-training"

            plt.plot(steps, mean_path, label=label)
            plt.fill_between(
                steps,
                mean_path - std_path,
                mean_path + std_path,
                alpha=0.15,
            )

        plt.title("Ensemble Mean Equity Curve (Pre-training vs No Pre-training)")
        plt.xlabel("Time Step")
        plt.ylabel("Equity")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out_fig = results_dir / "ensemble_equity_curves.png"
        plt.savefig(out_fig, dpi=150)
        plt.show()
        print(f"Saved ensemble equity curve figure to {out_fig}")


if __name__ == "__main__":
    # Ensure project root is importable when running directly.
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    main()
