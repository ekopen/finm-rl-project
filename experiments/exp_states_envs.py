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
import random

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
    # Use epochs=30 for better convergence, aligned with baseline experiments
    base_config = make_base_config(epochs=30)
    log_path = str(results_dir / f"{name}_train_logs.json")

    train_env = PairsEnv(
        prices_a=train_a,
        prices_b=train_b,
        features_a=feats_train_a,
        features_b=feats_train_b,
        initial_cash=1.0,
        config={"transaction_cost": 0.0},  # Explicit for fair comparison
    )
    agent = train_env_with_config(train_env, base_config, log_path=log_path)

    # 5) 在测试集上评估
    test_env = PairsEnv(
        prices_a=test_a,
        prices_b=test_b,
        features_a=feats_test_a,
        features_b=feats_test_b,
        initial_cash=1.0,
        config={"transaction_cost": 0.0},  # Explicit for fair comparison
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


def set_global_seed(seed: int) -> None:
    """Set global random seeds for reproducibility."""
    import os
    random.seed(seed)
    np.random.seed(seed)
    
    # Set Python hash seed for dictionary iteration order determinism
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.cuda.manual_seed(seed)
        
        # Enable deterministic algorithms (may warn if some ops can't be deterministic)
        torch.use_deterministic_algorithms(True, warn_only=True)
        
        # Set CUDNN flags for determinism (even for CPU, some ops may use CUDNN)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        # If torch is not available, just skip.
        pass


def create_state_comparison_plot(results_dir, test_df: pd.DataFrame) -> None:
    """
    Create comparison plot showing all state configs vs baselines.
    
    Args:
        results_dir: Directory containing state config equity JSON files
        test_df: Test period price data for running baselines
    """
    print("\n" + "=" * 70)
    print("Creating State Comparison Plot")
    print("=" * 70)
    
    # Load state config equity curves from saved JSON files
    state_configs = ["single_simple", "single_rich", "pairs_simple"]
    state_curves = {}
    state_dates = {}
    
    for config_name in state_configs:
        equity_json_path = results_dir / f"{config_name}_equity.json"
        if equity_json_path.exists():
            with open(equity_json_path, "r", encoding="utf-8") as f:
                equity_data = json.load(f)
            state_curves[config_name] = np.array(equity_data["equity"])
            state_dates[config_name] = pd.to_datetime(equity_data["dates"])
            print(f"  Loaded {config_name}: {len(state_curves[config_name])} points")
        else:
            print(f"  WARNING: Equity JSON not found for {config_name}")
    
    if not state_curves:
        print("ERROR: No state config equity curves found. Skipping comparison plot.")
        return
    
    # Run baselines on test data
    eq_bh = run_buy_and_hold(test_df)
    eq_ma, dates_ma = run_ma_crossover(test_df, allow_short=True)
    dates_bh = test_df.index[:len(eq_bh)]
    
    print(f"  Buy & Hold: {len(eq_bh)} points")
    print(f"  MA Crossover: {len(eq_ma)} points")
    
    # Find common dates across all strategies
    # Start with baseline dates
    common_dates = dates_bh.intersection(dates_ma)
    
    # Intersect with each state config's dates
    for config_name, config_dates in state_dates.items():
        common_dates = common_dates.intersection(config_dates)
    
    if len(common_dates) == 0:
        print("ERROR: No overlapping dates found between strategies!")
        return
    
    print(f"  Common dates: {common_dates[0]} to {common_dates[-1]} ({len(common_dates)} points)")
    
    # Align all equity curves to common dates
    df_bh = pd.Series(eq_bh, index=dates_bh)
    df_ma = pd.Series(eq_ma, index=dates_ma)
    
    aligned_curves = {
        "Buy & Hold": df_bh.reindex(common_dates).values,
        "MA Crossover": df_ma.reindex(common_dates).values,
    }
    
    # Add state config curves
    for config_name, equity in state_curves.items():
        df_config = pd.Series(equity, index=state_dates[config_name])
        aligned_curves[config_name] = df_config.reindex(common_dates).values
    
    # Validate no NaNs after alignment
    nan_checks = {name: np.isnan(eq).sum() for name, eq in aligned_curves.items()}
    if any(count > 0 for count in nan_checks.values()):
        error_msg = "NaN values found after date alignment!\n"
        error_msg += f"NaN counts: {nan_checks}"
        print(f"ERROR: {error_msg}")
        return
    
    # Compute metrics for all strategies
    metrics: dict[str, dict[str, float]] = {}
    for name, equity in aligned_curves.items():
        metrics[name] = {
            "total_return": compute_total_return(equity),
            "annualized_return": compute_annualized_return(equity, periods_per_year=252),
            "sharpe": compute_sharpe(equity, rf=0.0, periods_per_year=252),
            "max_drawdown": compute_max_drawdown(equity),
        }
    
    # Save metrics to JSON file
    metrics_path = results_dir / "states_envs_comparison_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to {metrics_path}")
    
    # Print metrics table
    print("\n" + "=" * 80)
    print("State Comparison Metrics")
    print("=" * 80)
    print(f"{'Strategy':<20} {'Total Return':>15} {'Ann. Return':>15} {'Sharpe':>10} {'Max DD':>10}")
    print("-" * 80)
    for name, m in metrics.items():
        print(f"{name:<20} {m['total_return']:>15.4f} {m['annualized_return']:>15.4f} "
              f"{m['sharpe']:>10.4f} {m['max_drawdown']:>10.4f}")
    print("=" * 80)
    
    # Create comparison plot
    plot_path = results_dir / "states_envs_comparison_equity.png"
    plot_equity_curves(aligned_curves, dates=common_dates, out_path=str(plot_path))
    
    print(f"\nSaved comparison plot to {plot_path}")
    print(f"Date range: {common_dates[0]} to {common_dates[-1]} ({len(common_dates)} points)")


def main() -> None:
    # Set seed for reproducibility BEFORE any random operations
    set_global_seed(42)
    
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

        # Explicitly set transaction_cost=0.0 for fair comparison with baselines
        train_env = make_single_asset_env(
            train_df, 
            env_config={"transaction_cost": 0.0, "lambda_risk": 0.0, "lambda_drawdown": 0.0},
            **builder_kwargs
        )
        # Use epochs=30 for better convergence, aligned with baseline experiments
        base_config = make_base_config(epochs=30)
        log_path = str(results_dir / f"{name}_train_logs.json")
        agent = train_env_with_config(train_env, base_config, log_path=log_path)

        test_env = make_single_asset_env(
            test_df,
            env_config={"transaction_cost": 0.0, "lambda_risk": 0.0, "lambda_drawdown": 0.0},
            **builder_kwargs
        )
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

    # Create combined comparison plot with all state configs + baselines
    create_state_comparison_plot(results_dir, test_df)


if __name__ == "__main__":
    main()
