"""
Helpers for evaluating PPO agents, such as generating equity curves from
trained policies.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from .ppo_agent import PPOAgent


def run_policy_episode(
    env: Any,
    agent: PPOAgent,
    num_steps: int | None = None,
    debug: bool = False,
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
    debug
        If True, print action / position distributions for inspection.

    Returns
    -------
    equity : np.ndarray
        Equity curve over the episode, starting from the environment's
        `initial_cash` attribute if present, otherwise 1.0.
    """
    state = env.reset()

    # 兼容有些 env 可能返回 (obs, info) 的情况
    if isinstance(state, tuple) and len(state) == 2:
        state, _ = state

    initial_cash = float(getattr(env, "initial_cash", 1.0))

    equity: list[float] = [initial_cash]
    actions: list[int] = []
    positions: list[float] = []

    steps = 0
    done = False

    while True:
        # 让 agent 给一个动作
        # Use deterministic=False to sample from learned policy and make actual trading decisions
        # This avoids collapsing to a single action during evaluation
        action, log_prob, value = agent.act(state, deterministic=False)
        actions.append(int(action))

        next_state, reward, done, info = env.step(action)

        # 优先使用 env 里的 portfolio_value
        if "portfolio_value" in info:
            pv = float(info["portfolio_value"])
        else:
            pv = equity[-1] * (1.0 + float(reward))

        equity.append(pv)

        # 记录 position（如果 env 没提供就从属性里拿）
        if "position" in info:
            positions.append(float(info["position"]))
        else:
            positions.append(float(getattr(env, "position", 0.0)))

        state = next_state
        steps += 1

        if done or (num_steps is not None and steps >= num_steps):
            break

    if debug:
        action_counts = Counter(actions)
        position_counts = Counter(positions)
        print("=== PPO EVAL DEBUG ===")
        print("Action counts (0=short, 1=flat, 2=long):", action_counts)
        print("Position counts (-1=short, 0=flat, 1=long):", position_counts)
        print("First 5 equity values:", equity[:5])
        print("Last 5 equity values:", equity[-5:])

    return np.asarray(equity, dtype=float)




