"""
Compare different state/env configurations for PPO.

Currently only:
- single_simple: SingleAssetEnv + make_simple_features

Placeholders:
- single_rich: SingleAssetEnv + rich features (TODO)
- pairs_simple: PairsEnv + pairs state (solved)
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

from features.state_builders import make_rich_features, make_simple_pairs_features  # MODIFIED

from features.data_loader import load_price_data  # NEW
from envs.pairs_env import PairsEnv  # NEW

from experiments.common import (
    load_spy_splits,
    make_base_config,
    make_results_dir,
    make_single_asset_env,
    train_env_with_config,
)


CONFIGS = [
    {"name": "single_simple", "type": "single_simple"},
    {"name": "single_rich", "type": "single_rich"},
    {"name": "pairs_simple", "type": "pairs_simple"},  # PairsEnv 配置
]


# NEW: 专门封装一个 pairs_simple 运行函数
def run_pairs_simple(
    train_spy,
    test_spy,
    results_dir,
) -> None:
    """
    Run PPO on a simple 2-asset pairs environment (e.g. SPY / QQQ)
    using simple features on each leg.

    - Asset A: SPY (train_spy / test_spy)
    - Asset B: QQQ (loaded separately and aligned by date)
    """

    name = "pairs_simple"

    # 1) 加载第二个资产（例如 QQQ）的价格数据
    start_date = train_spy.index.min().strftime("%Y-%m-%d")
    end_date = test_spy.index.max().strftime("%Y-%m-%d")

    prices_b_full = load_price_data("QQQ", start=start_date, end=end_date)

    # 2) 对齐 index：保证两只资产在各自的 train/test 期有共同交易日
    train_idx = train_spy.index.intersection(prices_b_full.index)
    test_idx = test_spy.index.intersection(prices_b_full.index)

    train_a = train_spy.loc[train_idx].copy()
    test_a = test_spy.loc[test_idx].copy()
    train_b = prices_b_full.loc[train_idx].copy()
    test_b = prices_b_full.loc[test_idx].copy()

    # 3) 使用你在 state_builders 里写好的 pairs 特征函数  # ⭐ 这里是关键修改
    feats_train_a, feats_train_b = make_simple_pairs_features(train_a, train_b)
    feats_test_a, feats_test_b = make_simple_pairs_features(test_a, test_b)

    # 4) 构建 PairsEnv 并训练 PPO
    base_config = make_base_config()
    log_path = str(results_dir / f"{name}_train_logs.json")

    train_env = PairsEnv(
        prices_a=train_a,
        prices_b=train_b,
        features_a=feats_train_a,
        features_b=feats_train_b,
        initial_cash=1.0,
        config={},  # 如需交易成本/风险惩罚可在这里加
    )
    agent = train_env_with_config(train_env, base_config, log_path=log_path)

    # 5) 在测试集上评估
    test_env = PairsEnv(
        prices_a=test_a,
        prices_b=test_b,
        features_a=feats_test_a,
        features_b=feats_test_b,
        initial_cash=1.0,
        config={},
    )
    eq_ppo = run_policy_episode(test_env, agent)

    min_len = len(eq_ppo)
    dates = test_a.index[:min_len]
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

    # Save equity curve array for later analysis
    equity_json = results_dir / f"{name}_equity.json"
    equity_data = {
        "equity": eq_ppo.tolist(),
        "dates": dates.strftime("%Y-%m-%d").tolist() if hasattr(dates, 'strftime') else [str(d) for d in dates]
    }
    with open(equity_json, "w", encoding="utf-8") as f:
        json.dump(equity_data, f, indent=2)

    plot_path = results_dir / f"{name}_equity.png"
    plot_equity_curves({"PPO": eq_ppo}, dates=dates, out_path=str(plot_path))

    print(f"[{name}] metrics: {metrics}")
    print(f"[{name}] wrote metrics to {out_json}")
    print(f"[{name}] wrote equity data to {equity_json}")
    print(f"[{name}] wrote equity plot to {plot_path}")



def main() -> None:
    train_df, _, test_df = load_spy_splits()
    results_dir = make_results_dir("states_envs")

    for cfg in CONFIGS:
        cfg_type = cfg["type"]
        name = cfg["name"]


        if cfg_type == "single_simple":
            features_builder = None  # default simple features
        elif cfg_type == "single_rich":
            features_builder = make_rich_features
        # NEW: 
        elif cfg_type == "pairs_simple":  # NEW
            print(f"\n=== Running state/env config: {name} ===")  # NEW
            run_pairs_simple(train_df, test_df, results_dir)  # NEW
            continue  # NEW
        else:
            print(f"Skipping {name} (type={cfg_type}) – unknown type.")
            continue

        print(f"\n=== Running state/env config: {name} ===")

        builder_kwargs = {}
        if features_builder is not None:
            builder_kwargs["features_builder"] = features_builder

        train_env = make_single_asset_env(train_df, **builder_kwargs)
        base_config = make_base_config()
        log_path = str(results_dir / f"{name}_train_logs.json")
        agent = train_env_with_config(train_env, base_config, log_path=log_path)

        test_env = make_single_asset_env(test_df, **builder_kwargs)
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

        # Save equity curve array for later analysis
        equity_json = results_dir / f"{name}_equity.json"
        equity_data = {
            "equity": eq_ppo.tolist(),
            "dates": dates.strftime("%Y-%m-%d").tolist() if hasattr(dates, 'strftime') else [str(d) for d in dates]
        }
        with open(equity_json, "w", encoding="utf-8") as f:
            json.dump(equity_data, f, indent=2)

        plot_path = results_dir / f"{name}_equity.png"
        plot_equity_curves({"PPO": eq_ppo}, dates=dates, out_path=str(plot_path))

        print(f"[{name}] metrics: {metrics}")
        print(f"[{name}] wrote metrics to {out_json}")
        print(f"[{name}] wrote equity data to {equity_json}")
        print(f"[{name}] wrote equity plot to {plot_path}")


if __name__ == "__main__":
    main()
