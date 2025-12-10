from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Callable, Tuple

import pandas as pd

from envs.single_asset_env import SingleAssetEnv
from features.data_loader import load_price_data
from features.state_builders import make_simple_features
from ppo.ppo_agent import PPOAgent, PPOConfig
from ppo.trainer import train_ppo


RESULTS_ROOT = Path("results")


def load_spy_splits() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load SPY daily data and split into train / val / test.

    - Train: 2010-01-01 to 2017-12-31 (inclusive)
    - Val:   2018-01-01 to 2019-12-31
    - Test:  2020-01-01 onwards
    """
    df = load_price_data("SPY", start="2010-01-01", end="2023-12-31")
    df = df.sort_index()

    train = df[df.index <= "2017-12-31"].copy()
    val = df[(df.index >= "2018-01-01") & (df.index <= "2019-12-31")].copy()
    test = df[df.index >= "2020-01-01"].copy()
    return train, val, test


FeatureBuilder = Callable[[pd.DataFrame], pd.DataFrame]


def make_single_asset_env(
    prices_df: pd.DataFrame,
    features_builder: FeatureBuilder = make_simple_features,
    env_config: dict | None = None,
) -> SingleAssetEnv:
    """Build SingleAssetEnv from a price dataframe using the provided builder."""
    features_df = features_builder(prices_df)
    return SingleAssetEnv(
        prices_df=prices_df,
        features_df=features_df,
        initial_cash=1.0,
        config=env_config,
    )


def make_base_config(**overrides) -> PPOConfig:
    """
    Base PPOConfig used across experiments, with optional field overrides.
    """
    base = PPOConfig(
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        lr=3e-4,
        max_grad_norm=0.5,
        update_epochs=4,  # Reduced from 10 for more frequent, stable updates
        minibatch_size=64,
        steps_per_epoch=2048,  # Increased from 1024 for better sample diversity
        epochs=10,
        log_interval=2,
        device="cpu",
    )
    return replace(base, **overrides)


def make_results_dir(exp_name: str) -> Path:
    """Create (if needed) and return results/<exp_name>/ directory."""
    d = RESULTS_ROOT / exp_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def train_env_with_config(
    env: SingleAssetEnv,
    config: PPOConfig,
    log_path: str | None = None,
) -> PPOAgent:
    """
    Convenience: create an agent for the env, run train_ppo, and return the agent.
    """
    state_dim = env.reset().shape[0]
    action_dim = env.action_space_n
    agent = PPOAgent(state_dim=state_dim, action_dim=action_dim, config=config)
    train_ppo(env, agent, config, log_path=log_path)
    return agent



