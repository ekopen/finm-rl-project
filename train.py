"""
Basic training entrypoint: run PPO on the SingleAssetEnv using SPY data.

This script is intentionally simple; more complex experiments live under
`experiments/`.
"""

from __future__ import annotations

from pathlib import Path

from envs.single_asset_env import SingleAssetEnv
from features.data_loader import load_price_data, train_test_split_by_date
from features.state_builders import make_simple_features
from ppo.ppo_agent import PPOAgent, PPOConfig
from ppo.trainer import train_ppo


def main() -> None:
    try:
        # Load data and build simple features.
        prices = load_price_data("SPY", start="2010-01-01", end="2020-01-01")
        train_prices, _ = train_test_split_by_date(prices, "2017-12-31")
        train_features = make_simple_features(train_prices)

        env = SingleAssetEnv(
            prices_df=train_prices,
            features_df=train_features,
            initial_cash=1.0,
        )

        state = env.reset()
        state_dim = state.shape[0]
        action_dim = env.action_space_n

        # Default PPO configuration for basic experiments.
        config = PPOConfig(
            gamma=0.99,
            gae_lambda=0.95,
            clip_eps=0.2,
            entropy_coef=0.01,
            value_coef=0.5,
            lr=3e-4,
            max_grad_norm=0.5,
            update_epochs=10,
            minibatch_size=64,
            steps_per_epoch=1024,
            epochs=10,
            log_interval=1,
        )
        print("Using PPOConfig:", config)
        agent = PPOAgent(state_dim=state_dim, action_dim=action_dim, config=config)

        # Optional: save logs next to this script.
        log_path = Path("ppo_train_logs.json")
        metrics = train_ppo(env, agent, config, log_path=str(log_path))

        final_metrics = {k: v[-1] for k, v in metrics.items() if v}
        print("Final epoch metrics:", final_metrics)
        print(f"Saved training logs to {log_path.resolve()}")
    except Exception as exc:  # pragma: no cover - debug aid
        print("[train.py] Encountered an error during training:")
        raise


if __name__ == "__main__":
    main()

