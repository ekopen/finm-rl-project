"""
PPO seed robustness experiments.

Runs multiple short PPO trainings on SingleAssetEnv with the SAME PPOConfig
but different random seeds, and saves training + test metrics and configs.

This checks how sensitive performance is to randomness (initialization, sampling, etc.).
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
        # torch 不存在也没关系，直接跳过
        pass


def build_train_and_test_envs():
    train_df, _, test_df = load_spy_splits()
    train_env = make_single_asset_env(train_df)
    test_env = make_single_asset_env(test_df)
    return train_env, test_env, test_df.index


def main() -> None:
    # 和原来一样的 base_config，只是现在不改里面的超参数了
    base_config: PPOConfig = make_base_config(
        steps_per_epoch=1024,
        epochs=8,
        log_interval=2,
    )

    # 要检验的随机种子列表（你可以根据需要增减）
    seed_list = [0, 1, 2, 3, 4]

    # 构造“搜索列表”，这里只是不同 seed
    sweep = [
        {
            "name": f"seed_{seed}",
            "seed": seed,
        }
        for seed in seed_list
    ]

    results_dir = make_results_dir("ppo_seed_robustness")
    run_payloads: List[dict] = []

    # 环境可以提前建好，然后在每个 seed 下重置 / reseed
    train_env, test_env, test_index = build_train_and_test_envs()

    # PPOConfig 是否有 seed 字段（有的话就一起写进去，没有就保持原样）
    has_seed_field = any(f.name == "seed" for f in fields(PPOConfig))

    for setting in sweep:
        name = setting["name"]
        seed = setting["seed"]

        print(f"\n=== Running config: {name} ===")
        print(f"Seed: {seed}")

        # 1. 设定全局随机种子
        set_global_seed(seed)

        # 2. 如果环境支持 seed() 方法，也设置一下
        if hasattr(train_env, "seed"):
            train_env.seed(seed)
        if hasattr(test_env, "seed"):
            test_env.seed(seed)

        # 3. 构造 PPOConfig：如果 dataclass 里有 seed 字段，就覆盖
        if has_seed_field:
            config = replace(base_config, seed=seed)
        else:
            config = base_config

        print("PPOConfig:", config)

        # 4. 为每个 run 构建新的 agent
        state = train_env.reset()
        # 兼容 gym 新版：reset 可能返回 (obs, info)
        if isinstance(state, tuple) and len(state) >= 1:
            state = state[0]
        state_dim = state.shape[0]
        action_dim = train_env.action_space_n

        agent = PPOAgent(state_dim=state_dim, action_dim=action_dim, config=config)
        
        from pathlib import Path
        ckpt = Path("results/pretrain_ma_bc/ppo_bc_pretrained.pt")
        if ckpt.exists():
            print("Loading BC-pretrained PPO weights...")
            agent.load(str(ckpt))

        # 5. 训练
        metrics_train = train_ppo(
            train_env,
            agent,
            config,
            log_path=str(results_dir / f"{name}_train_logs.json"),
        )

        # 6. 在测试集上评估
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

        # 7. 保存 equity 曲线图
        plot_path = results_dir / f"{name}_equity.png"
        plot_equity_curves({"PPO": eq_ppo}, dates=dates, out_path=str(plot_path))

        # 8. 组装 payload，便于之后做 summary
        run_payload = {
            "name": name,
            "seed": seed,
            "config": asdict(config),
            "train_metrics": metrics_train,
            "test_metrics": test_metrics,
        }
        run_payloads.append(run_payload)

        out_path = results_dir / f"{name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(run_payload, f, indent=2)
        print(f"Saved run payload to {out_path}")

    # 9. 生成整体 summary（JSON + CSV + 打印表）
    summary = {p["name"]: p["test_metrics"] for p in run_payloads}
    summary_json = results_dir / "summary_test_metrics.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    df = summarize_runs(run_payloads, metric_key="test_metrics")
    summary_csv = results_dir / "summary_test_metrics.csv"
    df.to_csv(summary_csv, index=False)
    print("\nSeed robustness summary (test metrics):")
    print(df.to_string(index=False))
    print(f"Wrote summary JSON to {summary_json}")
    print(f"Wrote summary CSV to {summary_csv}")


if __name__ == "__main__":
    # Ensure project root is importable when running directly.
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    main()
