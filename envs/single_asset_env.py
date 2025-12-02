"""
Single-asset trading environment with a very simple daily PnL reward.

This v1 environment is intentionally minimal: it trades a single asset with
discrete actions {short, flat, long} and uses daily returns as reward.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class SingleAssetEnv:
    """
    Simple single-asset environment compatible with PPO-style agents.

    State consists of:
    - features: [ret, ma, vol] for the current time step
    - current position: scalar in {-1, 0, +1}
    """

    def __init__(
        self,
        prices_df: pd.DataFrame,
        features_df: pd.DataFrame,
        initial_cash: float = 1.0,
        config: dict[str, Any] | None = None,
    ) -> None:
        if not prices_df.index.equals(features_df.index):
            raise ValueError(
                "prices_df and features_df must have the same index for alignment."
            )

        self.prices = prices_df["close"].to_numpy()
        self.features = features_df[["ret", "ma", "vol"]].to_numpy().astype(
            np.float32
        )
        self.initial_cash = float(initial_cash)
        self.config: dict[str, Any] = config or {}
        self.transaction_cost = float(self.config.get("transaction_cost", 0.0))
        self.lambda_risk = float(self.config.get("lambda_risk", 0.0))
        self.lambda_drawdown = float(self.config.get("lambda_drawdown", 0.0))

        self.num_steps = len(self.prices)
        if self.num_steps != self.features.shape[0]:
            raise ValueError("prices and features must have the same number of rows.")

        # Discrete action space: 0 -> short, 1 -> flat, 2 -> long.
        self.action_space_n = 3

        # These will be set in reset().
        self.t: int = 0
        self.position: int = 0
        self.portfolio_value: float = self.initial_cash
        self.peak_value: float = self.initial_cash

    def reset(self) -> np.ndarray:
        """
        Reset the environment to the beginning of the episode.

        Returns
        -------
        state : np.ndarray
            Initial observation of the environment.
        """
        self.t = 0
        self.position = 0  # -1 short, 0 flat, +1 long
        self.portfolio_value = self.initial_cash
        self.peak_value = self.initial_cash
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        feat = self.features[self.t]
        pos = np.array([self.position], dtype=np.float32)
        return np.concatenate([feat, pos], axis=0).astype(np.float32)

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, dict[str, float]]:
        """
        Take one step in the environment given an action.

        Parameters
        ----------
        action : int
            Discrete action index: 0=short, 1=flat, 2=long.

        Returns
        -------
        next_state, reward, done, info : tuple
            Standard gym-style step output.
        """
        if self.t >= self.num_steps - 1:
            raise RuntimeError("Episode has already terminated. Call reset().")

        # Map discrete action to position.
        if action == 0:
            new_pos = -1
        elif action == 1:
            new_pos = 0
        else:
            new_pos = 1

        prev_price = self.prices[self.t]
        self.t += 1
        done = self.t == self.num_steps - 1
        curr_price = self.prices[self.t]

        price_return = (curr_price - prev_price) / prev_price
        raw_reward = float(new_pos * price_return)
        cost_penalty = self.transaction_cost * abs(new_pos - self.position)
        risk_penalty = self.lambda_risk * (price_return**2)

        reward = raw_reward - cost_penalty - risk_penalty

        proposed_value = self.portfolio_value * (1.0 + reward)
        peak_candidate = max(self.peak_value, proposed_value)
        drawdown_ratio = 0.0
        if peak_candidate > 0.0:
            drawdown_ratio = max(0.0, (peak_candidate - proposed_value) / peak_candidate)
        drawdown_penalty = self.lambda_drawdown * drawdown_ratio
        reward -= drawdown_penalty

        self.position = new_pos
        self.portfolio_value *= 1.0 + reward
        self.peak_value = max(self.peak_value, self.portfolio_value)

        next_state = self._get_state()
        info: dict[str, float] = {
            "portfolio_value": float(self.portfolio_value),
            "raw_reward": raw_reward,
            "transaction_cost": cost_penalty,
            "risk_penalty": risk_penalty,
            "drawdown_penalty": drawdown_penalty,
            "position": float(self.position),  # <<< 新增这一行
        }
        return next_state, reward, done, info


