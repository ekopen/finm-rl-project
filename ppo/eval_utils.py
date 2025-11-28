"""
Helpers for evaluating PPO agents, such as generating equity curves from
trained policies.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .ppo_agent import PPOAgent


def run_policy_episode(
    env: Any,
    agent: PPOAgent,
    num_steps: int | None = None,
) -> np.ndarray:
    """
    Run a single episode (or a fixed number of steps) with the given policy.

    Parameters
    ----------
    env
        Environment exposing `reset()` and `step(action)` and tracking
        `portfolio_value` in the step `info` dict (as in `SingleAssetEnv`).
    agent
        Trained PPOAgent with an `act(state)` method.
    num_steps
        Optional maximum number of steps. If None, the episode ends when
        the environment signals `done`.

    Returns
    -------
    equity : np.ndarray
        Equity curve over the episode, starting from the environment's
        `initial_cash` attribute if present, otherwise 1.0.
    """
    state = env.reset()
    initial_cash = float(getattr(env, "initial_cash", 1.0))

    equity: list[float] = [initial_cash]
    steps = 0

    done = False
    while True:
        action, log_prob, value = agent.act(state)
        next_state, reward, done, info = env.step(action)

        # Prefer portfolio_value from the environment; otherwise, approximate.
        if "portfolio_value" in info:
            pv = float(info["portfolio_value"])
        else:
            pv = equity[-1] * (1.0 + float(reward))

        equity.append(pv)

        state = next_state
        steps += 1

        if done or (num_steps is not None and steps >= num_steps):
            break

    return np.asarray(equity, dtype=float)



