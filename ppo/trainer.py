"""
High-level PPO training loop utilities.

Provides:
- `collect_rollout` to build a `RolloutBatch` from an environment and agent
- `train_ppo` to run multi-epoch PPO training and return simple metrics
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import torch

from .ppo_agent import PPOAgent, PPOConfig, RolloutBatch


def collect_rollout(env: Any, agent: PPOAgent, num_steps: int) -> RolloutBatch:
    """
    Collect a fixed-length rollout using the given environment and agent.

    The environment is reset at the beginning and whenever it signals `done`,
    but collection continues until `num_steps` transitions have been recorded.
    """
    obs_list: list[np.ndarray] = []
    actions_list: list[int] = []
    log_probs_list: list[float] = []
    rewards_list: list[float] = []
    dones_list: list[float] = []
    values_list: list[float] = []

    state = env.reset()

    for _ in range(num_steps):
        obs_list.append(state.copy())

        action, log_prob, value = agent.act(state)
        next_state, reward, done, _info = env.step(action)

        actions_list.append(action)
        log_probs_list.append(log_prob)
        rewards_list.append(float(reward))
        dones_list.append(float(done))
        values_list.append(value)

        state = next_state
        if done:
            state = env.reset()

    # Bootstrap value for the final state.
    last_value = agent.value(state)

    return RolloutBatch(
        obs=torch.as_tensor(np.asarray(obs_list), dtype=torch.float32),
        actions=torch.as_tensor(actions_list, dtype=torch.long),
        log_probs=torch.as_tensor(log_probs_list, dtype=torch.float32),
        rewards=torch.as_tensor(rewards_list, dtype=torch.float32),
        dones=torch.as_tensor(dones_list, dtype=torch.float32),
        values=torch.as_tensor(values_list, dtype=torch.float32),
        last_value=torch.as_tensor(last_value, dtype=torch.float32),
    )


def train_ppo(
    env: Any,
    agent: PPOAgent,
    config: PPOConfig,
    log_path: str | None = None,
) -> dict[str, list[float]]:
    """
    Run PPO training on the provided environment.

    Parameters
    ----------
    env
        Environment exposing `reset()` and `step(action)` methods.
    agent
        PPO agent with `act` / `update` methods.
    config
        PPO configuration including `steps_per_epoch`, `epochs`, and
        `log_interval`.
    log_path
        Optional path to a JSON file where metrics will be saved.
    """
    metrics: dict[str, list[float]] = {
        "epoch": [],
        "mean_reward": [],
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        "approx_kl": [],
    }

    for epoch in range(config.epochs):
        batch = collect_rollout(env, agent, num_steps=config.steps_per_epoch)
        stats = agent.update(batch)

        mean_reward = float(batch.rewards.mean().item())

        metrics["epoch"].append(epoch)
        metrics["mean_reward"].append(mean_reward)
        metrics["policy_loss"].append(stats.get("policy_loss", float("nan")))
        metrics["value_loss"].append(stats.get("value_loss", float("nan")))
        metrics["entropy"].append(stats.get("entropy", float("nan")))
        metrics["approx_kl"].append(stats.get("approx_kl", float("nan")))

        if (
            (epoch + 1) % config.log_interval == 0
            or epoch == 0
            or epoch == config.epochs - 1
        ):
            print(
                f"[epoch {epoch+1}/{config.epochs}] "
                f"mean_reward={mean_reward:.6f}, "
                f"policy_loss={metrics['policy_loss'][-1]:.4f}, "
                f"value_loss={metrics['value_loss'][-1]:.4f}, "
                f"entropy={metrics['entropy'][-1]:.4f}, "
                f"approx_kl={metrics['approx_kl'][-1]:.6f}",
                flush=True,
            )

    if log_path is not None:
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    return metrics


