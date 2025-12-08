"""
PPO hyperparameter sweep experiments.

Runs multiple short PPO trainings on SingleAssetEnv with different PPOConfig
settings and saves training + test metrics and configs.
"""

from __future__ import annotations

import itertools
import json
import os
import random
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import List

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
from ppo.ppo_agent import PPOAgent, PPOConfig
from ppo.trainer import train_ppo

from eval.summarize import summarize_runs
from experiments.common import (
    load_spy_splits,
    make_base_config,
    make_results_dir,
    make_single_asset_env,
    train_env_with_config,
)


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


def build_train_and_test_envs():
    train_df, _, test_df = load_spy_splits()
    # Explicitly set transaction_cost=0.0 for fair comparison across hyperparameter configurations
    env_config = {"transaction_cost": 0.0, "lambda_risk": 0.0, "lambda_drawdown": 0.0}
    train_env = make_single_asset_env(train_df, env_config=env_config)
    test_env = make_single_asset_env(test_df, env_config=env_config)
    return train_env, test_env, test_df.index


def find_best_config_by_sharpe(run_payloads: List[dict]) -> dict:
    """
    Find the best config from run_payloads by Sharpe ratio.
    In case of ties, prefer higher total_return.
    
    Args:
        run_payloads: List of run payload dictionaries with 'test_metrics' key
        
    Returns:
        The run_payload dict with the highest Sharpe ratio
    """
    best_payload = None
    best_sharpe = float("-inf")
    best_total_return = float("-inf")
    
    for payload in run_payloads:
        sharpe = payload["test_metrics"]["sharpe"]
        total_return = payload["test_metrics"]["total_return"]
        
        if sharpe > best_sharpe or (sharpe == best_sharpe and total_return > best_total_return):
            best_sharpe = sharpe
            best_total_return = total_return
            best_payload = payload
    
    if best_payload is None:
        raise ValueError("No run payloads provided to find_best_config_by_sharpe")
    
    return best_payload


def find_best_from_each_dimension(run_payloads: List[dict], base_config: PPOConfig) -> dict:
    """
    Find the best value from each hyperparameter dimension independently.
    
    Args:
        run_payloads: List of run payload dictionaries
        base_config: Base PPOConfig to use defaults for missing values
        
    Returns:
        Dict with keys: clip_eps, entropy_coef, gamma, gae_lambda, lr
    """
    best_dims = {
        "clip_eps": base_config.clip_eps,
        "entropy_coef": base_config.entropy_coef,
        "gamma": base_config.gamma,
        "gae_lambda": base_config.gae_lambda,
        "lr": base_config.lr,
    }
    
    # Find best clip_eps and entropy_coef from clip_eps × entropy_coef grid
    # Names are like "clip01_ent000", "clip02_ent001", etc.
    clip_ent_payloads = [
        p for p in run_payloads
        if p["name"].startswith("clip") and "_ent" in p["name"]
    ]
    if clip_ent_payloads:
        best_clip_ent = find_best_config_by_sharpe(clip_ent_payloads)
        best_dims["clip_eps"] = best_clip_ent["config"]["clip_eps"]
        best_dims["entropy_coef"] = best_clip_ent["config"]["entropy_coef"]
    
    # Find best gamma/gae_lambda from gamma/GAE grid
    # Names are like "gamma095_lambda090", "gamma095_lambda095", etc.
    gamma_gae_payloads = [
        p for p in run_payloads
        if p["name"].startswith("gamma") and "_lambda" in p["name"]
    ]
    if gamma_gae_payloads:
        best_gamma_gae = find_best_config_by_sharpe(gamma_gae_payloads)
        best_dims["gamma"] = best_gamma_gae["config"]["gamma"]
        best_dims["gae_lambda"] = best_gamma_gae["config"]["gae_lambda"]
    
    # Find best lr from initial LR sweep (exclude "best_lr*" configs from Phase 2)
    # Names are like "lr1e4", "lr3e4", "lr1e3"
    lr_payloads = [
        p for p in run_payloads
        if p["name"].startswith("lr") and not p["name"].startswith("best_")
    ]
    if lr_payloads:
        best_lr_payload = find_best_config_by_sharpe(lr_payloads)
        best_dims["lr"] = best_lr_payload["config"]["lr"]
    
    return best_dims


def run_single_config(
    name: str,
    config: PPOConfig,
    train_env,
    test_env,
    test_index,
    results_dir: Path,
) -> dict:
    """
    Run a single hyperparameter configuration and return the run payload.
    
    Args:
        name: Name for this configuration
        config: PPOConfig to use
        train_env: Training environment
        test_env: Test environment
        test_index: Test data index for date alignment
        results_dir: Directory to save results
        
    Returns:
        Run payload dictionary
    """
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

    # Align dates with equity curve
    # PPO equity has length N (initial + N-1 steps), where N = len(test_df)
    # Dates correspond to test_df.index
    dates = test_index[:len(eq_ppo)]
    
    # Validate equity curve
    if np.isnan(eq_ppo).any():
        nan_count = np.isnan(eq_ppo).sum()
        raise ValueError(
            f"[{name}] NaN values found in equity curve ({nan_count} NaNs). "
            f"Equity length: {len(eq_ppo)}, Dates length: {len(dates)}"
        )
    
    if len(eq_ppo) != len(dates):
        raise ValueError(
            f"[{name}] Length mismatch: equity ({len(eq_ppo)}) != dates ({len(dates)})"
        )
    
    if len(eq_ppo) == 0:
        raise ValueError(f"[{name}] Empty equity curve generated!")

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

    # Save equity curve array for later comparison
    equity_json = results_dir / f"{name}_equity.json"
    equity_data = {
        "equity": eq_ppo.tolist(),
        "dates": dates.strftime("%Y-%m-%d").tolist() if hasattr(dates, 'strftime') else [str(d) for d in dates]
    }
    with open(equity_json, "w", encoding="utf-8") as f:
        json.dump(equity_data, f, indent=2)

        run_payload = {
            "name": name,
            "config": asdict(config),
            "train_metrics": metrics_train,
            "test_metrics": test_metrics,
        }

        out_path = results_dir / f"{name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(run_payload, f, indent=2)
        print(f"Saved run payload to {out_path}")
    print(f"Saved equity data to {equity_json}")
    
    return run_payload


def load_baseline_equity_curves(test_df: pd.DataFrame) -> dict:
    """
    Load baseline equity curves by re-running baseline evaluations.
    
    Args:
        test_df: Test period price data
        
    Returns:
        Dict with keys "PPO Baseline", "Buy & Hold", "MA Crossover"
        Each value is a dict with "equity" (np.ndarray) and "dates" (pd.Index)
    """
    # Set seed to match core_baselines experiment for reproducibility
    # CRITICAL: Must set seed BEFORE model initialization to ensure identical weights
    # This ensures the baseline PPO is trained with the same random state as exp_core_baselines
    set_global_seed(42)
    
    # Train baseline PPO agent
    train_df, _, _ = load_spy_splits()
    train_env = make_single_asset_env(
        train_df, env_config={"transaction_cost": 0.0, "lambda_risk": 0.0, "lambda_drawdown": 0.0}
    )
    config = make_base_config()
    agent = train_env_with_config(train_env, config, log_path=None)
    
    # Evaluate baseline PPO on test period
    test_env = make_single_asset_env(
        test_df, env_config={"transaction_cost": 0.0, "lambda_risk": 0.0, "lambda_drawdown": 0.0}
    )
    eq_ppo_baseline = run_policy_episode(test_env, agent)
    dates_ppo = test_df.index[:len(eq_ppo_baseline)]
    
    # Run baselines
    eq_bh = run_buy_and_hold(test_df)
    eq_ma, dates_ma = run_ma_crossover(test_df, allow_short=True)
    dates_bh = test_df.index[:len(eq_bh)]
    
    return {
        "PPO Baseline": {"equity": eq_ppo_baseline, "dates": dates_ppo},
        "Buy & Hold": {"equity": eq_bh, "dates": dates_bh},
        "MA Crossover": {"equity": eq_ma, "dates": dates_ma},
    }


def create_baseline_comparison_plot(
    best_config_name: str,
    best_equity: np.ndarray,
    best_dates: pd.Index,
    results_dir: Path,
) -> None:
    """
    Create comparison plot showing best hyperparameter config vs all baselines.
    
    Args:
        best_config_name: Name of the best hyperparameter configuration
        best_equity: Equity curve array for best config
        best_dates: Date index for best config equity curve
        results_dir: Directory to save the plot
    """
    print("\n" + "=" * 70)
    print("Creating Baseline Comparison Plot")
    print("=" * 70)
    
    # Load baseline equity curves
    _, _, test_df = load_spy_splits()
    baseline_curves = load_baseline_equity_curves(test_df)
    
    # Extract dates and equity arrays
    dates_ppo = baseline_curves["PPO Baseline"]["dates"]
    dates_bh = baseline_curves["Buy & Hold"]["dates"]
    dates_ma = baseline_curves["MA Crossover"]["dates"]
    
    # Find common dates across all strategies
    common_dates = dates_ppo.intersection(dates_bh).intersection(dates_ma).intersection(best_dates)
    
    if len(common_dates) == 0:
        error_msg = "No overlapping dates found between strategies!\n"
        error_msg += f"PPO Baseline date range: {dates_ppo[0]} to {dates_ppo[-1]} ({len(dates_ppo)} points)\n"
        error_msg += f"Buy & Hold date range: {dates_bh[0]} to {dates_bh[-1]} ({len(dates_bh)} points)\n"
        error_msg += f"MA Crossover date range: {dates_ma[0]} to {dates_ma[-1]} ({len(dates_ma)} points)\n"
        error_msg += f"Best Config date range: {best_dates[0]} to {best_dates[-1]} ({len(best_dates)} points)"
        raise ValueError(error_msg)
    
    # Align all equity curves to common dates
    df_ppo = pd.Series(baseline_curves["PPO Baseline"]["equity"], index=dates_ppo)
    df_bh = pd.Series(baseline_curves["Buy & Hold"]["equity"], index=dates_bh)
    df_ma = pd.Series(baseline_curves["MA Crossover"]["equity"], index=dates_ma)
    df_best = pd.Series(best_equity, index=best_dates)
    
    # Reindex to common dates
    eq_ppo_aligned = df_ppo.reindex(common_dates).values
    eq_bh_aligned = df_bh.reindex(common_dates).values
    eq_ma_aligned = df_ma.reindex(common_dates).values
    eq_best_aligned = df_best.reindex(common_dates).values
    
    # Validate no NaNs after alignment
    nan_checks = {
        "PPO Baseline": np.isnan(eq_ppo_aligned).sum(),
        "Buy & Hold": np.isnan(eq_bh_aligned).sum(),
        "MA Crossover": np.isnan(eq_ma_aligned).sum(),
        "Best Config": np.isnan(eq_best_aligned).sum(),
    }
    if any(count > 0 for count in nan_checks.values()):
        error_msg = "NaN values found after date alignment!\n"
        error_msg += f"NaN counts: {nan_checks}\n"
        error_msg += f"Common dates: {common_dates[0]} to {common_dates[-1]} ({len(common_dates)} points)"
        raise ValueError(error_msg)
    
    # Validate lengths match
    len_ppo = len(eq_ppo_aligned)
    len_bh = len(eq_bh_aligned)
    len_ma = len(eq_ma_aligned)
    len_best = len(eq_best_aligned)
    len_common = len(common_dates)
    
    if not (len_ppo == len_bh == len_ma == len_best == len_common):
        error_msg = "Length mismatch after date alignment!\n"
        error_msg += f"PPO Baseline: {len_ppo}, Buy & Hold: {len_bh}, "
        error_msg += f"MA Crossover: {len_ma}, Best Config: {len_best}, Common dates: {len_common}"
        raise ValueError(error_msg)
    
    # Compute metrics for all strategies using aligned equity curves
    metrics: dict[str, dict[str, float]] = {}
    strategy_names = {
        "PPO Baseline": eq_ppo_aligned,
        "Buy & Hold": eq_bh_aligned,
        "MA Crossover": eq_ma_aligned,
        f"Best Config ({best_config_name})": eq_best_aligned,
    }
    
    for name, equity in strategy_names.items():
        metrics[name] = {
            "total_return": compute_total_return(equity),
            "annualized_return": compute_annualized_return(equity, periods_per_year=252),
            "sharpe": compute_sharpe(equity, rf=0.0, periods_per_year=252),
            "max_drawdown": compute_max_drawdown(equity),
        }
    
    # Save metrics to JSON file
    metrics_path = results_dir / "best_vs_baselines_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")
    
    # Print metrics table
    print("\n" + "=" * 70)
    print("Baseline Comparison Metrics")
    print("=" * 70)
    print(f"{'Strategy':<30} {'Total Return':>15} {'Ann. Return':>15} {'Sharpe':>10} {'Max DD':>10}")
    print("-" * 70)
    for name, m in metrics.items():
        print(f"{name:<30} {m['total_return']:>15.6f} {m['annualized_return']:>15.6f} "
              f"{m['sharpe']:>10.4f} {m['max_drawdown']:>10.4f}")
    print("=" * 70)
    
    # Create comparison plot
    curves = {
        "PPO Baseline": eq_ppo_aligned,
        "Buy & Hold": eq_bh_aligned,
        "MA Crossover": eq_ma_aligned,
        f"Best Config ({best_config_name})": eq_best_aligned,
    }
    
    plot_path = results_dir / "best_vs_baselines_equity.png"
    plot_equity_curves(curves, dates=common_dates, out_path=str(plot_path))
    
    print(f"\nSaved baseline comparison plot to {plot_path}")
    print(f"Date range: {common_dates[0]} to {common_dates[-1]} ({len(common_dates)} points)")


def main() -> None:
    # Set seed for reproducibility BEFORE any random operations (model initialization, etc.)
    # This ensures model weights and all random operations are deterministic
    set_global_seed(42)
    
    base_config = make_base_config(
        steps_per_epoch=1024,
        epochs=15,  # Increased from 8 for better convergence
        log_interval=2,
    )

    # Build sweep using grid search and additional configs
    sweep: List[dict] = []
    
    # Grid search: clip_eps × entropy_coef
    clip_eps_values = [0.1, 0.2, 0.3]
    entropy_coef_values = [0.0, 0.01, 0.02]
    
    for clip_eps, entropy_coef in itertools.product(clip_eps_values, entropy_coef_values):
        # Format: clip0.1_ent0.00, clip0.2_ent0.01, etc.
        clip_str = f"{clip_eps:.1f}".replace(".", "")
        ent_str = f"{entropy_coef:.2f}".replace(".", "")
        name = f"clip{clip_str}_ent{ent_str}"
        sweep.append({
            "name": name,
            "overrides": {"clip_eps": clip_eps, "entropy_coef": entropy_coef},
        })
    
    # Learning rate sweep (using baseline clip_eps=0.2, entropy_coef=0.01)
    lr_values = [1e-4, 3e-4, 1e-3]
    for lr in lr_values:
        # Format: lr1e4, lr3e4, lr1e3
        if lr == 1e-4:
            name = "lr1e4"
        elif lr == 3e-4:
            name = "lr3e4"
        elif lr == 1e-3:
            name = "lr1e3"
        else:
            # Fallback for other values
            name = f"lr{lr:.0e}".replace("e-0", "e").replace("e+0", "e").replace(".", "")
        sweep.append({
            "name": name,
            "overrides": {"lr": lr},
        })
    
    # Gamma/GAE lambda grid (expanded from single config)
    gamma_gae_combos = [
        (0.95, 0.90),  # existing
        (0.95, 0.95),  # new
        (0.99, 0.90),  # new
        (0.97, 0.92),  # new, intermediate
    ]
    for gamma, gae_lambda in gamma_gae_combos:
        name = f"gamma{gamma:.2f}_lambda{gae_lambda:.2f}".replace(".", "")
        sweep.append({
            "name": name,
            "overrides": {"gamma": gamma, "gae_lambda": gae_lambda},
        })

    results_dir = make_results_dir("ppo_hyperparams")
    run_payloads: List[dict] = []

    train_env, test_env, test_index = build_train_and_test_envs()

    # ============================================================================
    # Phase 1: Initial Sweep
    # ============================================================================
    print("\n" + "=" * 70)
    print("PHASE 1: Initial Hyperparameter Sweep")
    print("=" * 70)
    
    for setting in sweep:
        name = setting["name"]
        overrides = setting["overrides"]
        config = replace(base_config, **overrides)
        run_payload = run_single_config(
            name, config, train_env, test_env, test_index, results_dir
        )
        run_payloads.append(run_payload)

    # ============================================================================
    # Phase 2: Best Config Learning Rate Sweep
    # ============================================================================
    print("\n" + "=" * 70)
    print("PHASE 2: Learning Rate Sweep on Best Config")
    print("=" * 70)
    
    best_config_payload = find_best_config_by_sharpe(run_payloads)
    best_config_dict = best_config_payload["config"]
    print(f"\nBest config from initial sweep: {best_config_payload['name']}")
    print(f"Sharpe: {best_config_payload['test_metrics']['sharpe']:.4f}")
    print(f"Total Return: {best_config_payload['test_metrics']['total_return']:.4f}")
    print(f"Config: clip_eps={best_config_dict['clip_eps']}, "
          f"entropy_coef={best_config_dict['entropy_coef']}, "
          f"gamma={best_config_dict['gamma']}, "
          f"gae_lambda={best_config_dict['gae_lambda']}, "
          f"lr={best_config_dict['lr']}")
    
    # Run LR sweep on best config
    for lr in lr_values:
        # Create config with best config's params + new LR
        overrides = {
            "clip_eps": best_config_dict["clip_eps"],
            "entropy_coef": best_config_dict["entropy_coef"],
            "gamma": best_config_dict["gamma"],
            "gae_lambda": best_config_dict["gae_lambda"],
            "lr": lr,
        }
        config = replace(base_config, **overrides)
        
        # Format name
        if lr == 1e-4:
            lr_name = "lr1e4"
        elif lr == 3e-4:
            lr_name = "lr3e4"
        elif lr == 1e-3:
            lr_name = "lr1e3"
        else:
            lr_name = f"lr{lr:.0e}".replace("e-0", "e").replace("e+0", "e").replace(".", "")
        
        name = f"best_{lr_name}"
        run_payload = run_single_config(
            name, config, train_env, test_env, test_index, results_dir
        )
        run_payloads.append(run_payload)

    # ============================================================================
    # Phase 3: Best Combo (Combining Best from Each Dimension)
    # ============================================================================
    print("\n" + "=" * 70)
    print("PHASE 3: Best Combo Configuration")
    print("=" * 70)
    
    best_dims = find_best_from_each_dimension(run_payloads, base_config)
    print(f"\nBest from each dimension:")
    print(f"  clip_eps: {best_dims['clip_eps']}")
    print(f"  entropy_coef: {best_dims['entropy_coef']}")
    print(f"  gamma: {best_dims['gamma']}")
    print(f"  gae_lambda: {best_dims['gae_lambda']}")
    print(f"  lr: {best_dims['lr']}")
    
    # Create config combining all best dimensions
    config = replace(base_config, **best_dims)
    name = "best_combo"
    run_payload = run_single_config(
        name, config, train_env, test_env, test_index, results_dir
    )
    run_payloads.append(run_payload)

    # ============================================================================
    # Final Summary Generation
    # ============================================================================
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

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
    print(f"\nWrote summary JSON to {summary_json}")
    print(f"Wrote summary CSV to {summary_csv}")
    
    # Print best configs for easy reference
    final_best = find_best_config_by_sharpe(run_payloads)
    print(f"\nOverall best config: {final_best['name']}")
    print(f"  Sharpe: {final_best['test_metrics']['sharpe']:.4f}")
    print(f"  Total Return: {final_best['test_metrics']['total_return']:.4f}")
    
    # Create baseline comparison plot
    best_config_name = final_best['name']
    equity_json_path = results_dir / f"{best_config_name}_equity.json"
    
    if equity_json_path.exists():
        with open(equity_json_path, "r", encoding="utf-8") as f:
            equity_data = json.load(f)
        
        best_equity = np.array(equity_data["equity"])
        best_dates = pd.to_datetime(equity_data["dates"])
        
        create_baseline_comparison_plot(
            best_config_name, best_equity, best_dates, results_dir
        )
    else:
        print(f"\nWARNING: Equity JSON not found for best config '{best_config_name}'")
        print(f"  Expected path: {equity_json_path}")
        print("  Skipping baseline comparison plot generation.")


if __name__ == "__main__":
    # Ensure project root is importable when running directly.
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    main()


