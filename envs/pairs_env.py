# envs/pairs_env.py
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class PairsEnv:
    """
    Simple 2-asset (pairs) trading environment.

    - Two underlyings: A and B
    - Discrete actions:
        0 -> flat both
        1 -> long A, short B
        2 -> short A, long B
    - Reward = combined PnL (plus optional shaping).
    """

    def __init__(
        self,
        prices_a: pd.DataFrame,
        prices_b: pd.DataFrame,
        features_a: pd.DataFrame,
        features_b: pd.DataFrame,
        initial_cash: float = 1.0,
        config: dict[str, Any] | None = None,
    ) -> None:
        # 对齐 index
        aligned_index = prices_a.index.intersection(prices_b.index)
        prices_a = prices_a.loc[aligned_index]
        prices_b = prices_b.loc[aligned_index]
        features_a = features_a.loc[aligned_index]
        features_b = features_b.loc[aligned_index]

        self.prices_a = prices_a["close"].to_numpy()
        self.prices_b = prices_b["close"].to_numpy()

        # 简单特征: [ret, ma, vol] 各三列
        self.features_a = features_a[["ret", "ma", "vol"]].to_numpy().astype(np.float32)
        self.features_b = features_b[["ret", "ma", "vol"]].to_numpy().astype(np.float32)

        # spread 可以用 log-price 差或者单纯价差，这里简单价差:
        spread = (self.prices_a - self.prices_b).reshape(-1, 1).astype(np.float32)
        self.spread = spread

        self.initial_cash = float(initial_cash)
        self.config: dict[str, Any] = config or {}
        self.transaction_cost = float(self.config.get("transaction_cost", 0.0))
        self.lambda_risk = float(self.config.get("lambda_risk", 0.0))
        self.lambda_drawdown = float(self.config.get("lambda_drawdown", 0.0))

        self.num_steps = len(self.prices_a)
        assert self.num_steps == len(self.prices_b) == self.features_a.shape[0] == self.features_b.shape[0]

        # 动作空间: 0 flat, 1 long A short B, 2 short A long B
        self.action_space_n = 3

        # 状态:
        # [feat_a(3), feat_b(3), spread(1), pos_a(1), pos_b(1)] => 3+3+1+1+1 = 9
        self.t: int = 0
        self.pos_a: int = 0
        self.pos_b: int = 0
        self.portfolio_value: float = self.initial_cash
        self.peak_value: float = self.initial_cash

    def reset(self) -> np.ndarray:
        self.t = 0
        self.pos_a = 0
        self.pos_b = 0
        self.portfolio_value = self.initial_cash
        self.peak_value = self.initial_cash
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        feat_a = self.features_a[self.t]  # (3,)
        feat_b = self.features_b[self.t]  # (3,)
        spr = self.spread[self.t]         # (1,)
        pos = np.array([self.pos_a, self.pos_b], dtype=np.float32)  # (2,)
        state = np.concatenate([feat_a, feat_b, spr, pos], axis=0).astype(np.float32)
        return state

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, dict[str, float]]:
        if self.t >= self.num_steps - 1:
            raise RuntimeError("Episode has already terminated. Call reset().")

        # 动作 -> 新仓位
        if action == 0:
            new_pos_a, new_pos_b = 0, 0
        elif action == 1:
            # long A, short B
            new_pos_a, new_pos_b = +1, -1
        else:
            # short A, long B
            new_pos_a, new_pos_b = -1, +1

        prev_price_a = self.prices_a[self.t]
        prev_price_b = self.prices_b[self.t]

        self.t += 1
        done = self.t == self.num_steps - 1

        curr_price_a = self.prices_a[self.t]
        curr_price_b = self.prices_b[self.t]

        ret_a = (curr_price_a - prev_price_a) / prev_price_a
        ret_b = (curr_price_b - prev_price_b) / prev_price_b

        raw_pnl = new_pos_a * ret_a + new_pos_b * ret_b
        cost_penalty = self.transaction_cost * (
            abs(new_pos_a - self.pos_a) + abs(new_pos_b - self.pos_b)
        )
        risk_penalty = self.lambda_risk * (ret_a**2 + ret_b**2)

        reward = raw_pnl - cost_penalty - risk_penalty

        proposed_value = self.portfolio_value * (1.0 + reward)
        peak_candidate = max(self.peak_value, proposed_value)
        drawdown_ratio = 0.0
        if peak_candidate > 0.0:
            drawdown_ratio = max(0.0, (peak_candidate - proposed_value) / peak_candidate)
        drawdown_penalty = self.lambda_drawdown * drawdown_ratio
        reward -= drawdown_penalty

        self.pos_a = new_pos_a
        self.pos_b = new_pos_b
        self.portfolio_value *= 1.0 + reward
        self.peak_value = max(self.peak_value, self.portfolio_value)

        next_state = self._get_state()
        info: dict[str, float] = {
            "portfolio_value": float(self.portfolio_value),
            "raw_pnl": float(raw_pnl),
            "transaction_cost": float(cost_penalty),
            "risk_penalty": float(risk_penalty),
            "drawdown_penalty": float(drawdown_penalty),
            "pos_a": float(self.pos_a),
            "pos_b": float(self.pos_b),
        }
        return next_state, reward, done, info
