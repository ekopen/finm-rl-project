"""
Behavior Cloning pretraining for PPO using a moving-average crossover expert.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from experiments.common import (
    load_spy_splits,
    make_base_config,
    make_single_asset_env,
    make_results_dir,
)
from ppo.ppo_agent import PPOAgent


# ----- 1. Define a simple MA-crossover expert policy -----

SHORT_WINDOW = 20
LONG_WINDOW = 50
# action encoding consistent with run_policy_episode debug:
# 0 = short, 1 = flat, 2 = long


def ma_crossover_expert(short_ma: float, long_ma: float) -> int:
    if short_ma > long_ma:
        return 2  # long
    elif short_ma < long_ma:
        return 0  # short
    else:
        return 1  # flat


# ----- 2. Collect (state, expert_action) dataset from the training env -----

def collect_expert_dataset(env, train_df) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Roll through the training data with an MA expert to build (state, action) pairs.

    We assume env steps through train_df in order; we compute MAs on train_df,
    and at each step where both MAs are defined, we record (state, expert_action).
    """

    df = train_df.copy()
    df["ma_short"] = df["close"].rolling(SHORT_WINDOW).mean()
    df["ma_long"] = df["close"].rolling(LONG_WINDOW).mean()

    short_arr = df["ma_short"].to_numpy()
    long_arr = df["ma_long"].to_numpy()

    states: list[np.ndarray] = []
    actions: list[int] = []

    state = env.reset()
    idx = 0

    while True:
        # Skip steps where long MA is not defined yet
        if not np.isnan(long_arr[idx]):
            short_ma = short_arr[idx]
            long_ma = long_arr[idx]
            a = ma_crossover_expert(short_ma, long_ma)
            states.append(state.copy())
            actions.append(a)
        else:
            # Before we have enough history, we can stay flat
            a = 1  # flat

        next_state, reward, done, _info = env.step(a)
        state = next_state

        idx += 1
        if done or idx >= len(df):
            break

    if len(states) == 0:
        raise RuntimeError("No valid (state, action) pairs collected; check MA windows and data length.")

    states_t = torch.as_tensor(np.asarray(states), dtype=torch.float32)
    actions_t = torch.as_tensor(actions, dtype=torch.long)
    return states_t, actions_t


# ----- 3. Main: build env, agent, do BC pretraining, save checkpoint -----

def main() -> None:
    # Use the same split utilities as other experiments
    train_df, _, _ = load_spy_splits()

    # Build training env
    # Explicitly set transaction_cost=0.0 for consistency with other experiments
    train_env = make_single_asset_env(
        train_df,
        env_config={"transaction_cost": 0.0, "lambda_risk": 0.0, "lambda_drawdown": 0.0}
    )

    # Build PPO agent with base config
    base_config = make_base_config(epochs=30)  # Use epochs=30 for consistency
    dummy_state = train_env.reset()
    state_dim = dummy_state.shape[0]
    action_dim = train_env.action_space_n

    agent = PPOAgent(state_dim=state_dim, action_dim=action_dim, config=base_config)

    # Collect expert dataset
    print("Collecting expert dataset using MA-crossover expert...")
    expert_states, expert_actions = collect_expert_dataset(train_env, train_df)
    print(f"Collected {expert_states.shape[0]} expert (state, action) pairs.")

    # Behavior cloning pretraining
    print("Starting behavior cloning pretraining...")
    bc_loss = agent.behavior_cloning_update(
        expert_states,
        expert_actions,
        bc_epochs=10,
        batch_size=256,
        lr=1e-3,  # can be a bit larger than PPO lr
    )
    print(f"Finished BC pretraining, final loss = {bc_loss:.6f}")

    # Save pretrained weights
    results_dir = make_results_dir("pretrain_ma_bc")
    ckpt_path = results_dir / "ppo_bc_pretrained.pt"
    agent.save(str(ckpt_path))
    print(f"Saved BC-pretrained PPO weights to {ckpt_path.resolve()}")


if __name__ == "__main__":
    main()
