from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .models import ActorCritic


@dataclass
class PPOConfig:
    """
    Hyperparameters for PPO training and experiments.

    These fields are intended to be swept in experiments; see the notes
    alongside each attribute for typical effects.
    """

    # Discount factor for future rewards (higher = more farsighted).
    gamma: float = 0.99
    # GAE lambda parameter (higher = smoother but more biased advantages).
    gae_lambda: float = 0.95
    # Clipping range for policy ratio in PPO objective.
    clip_eps: float = 0.2
    # Entropy bonus coefficient; higher values encourage more exploration.
    entropy_coef: float = 0.01
    # Weight on value loss relative to policy loss.
    value_coef: float = 0.5
    # Learning rate for Adam optimizer.
    lr: float = 3e-4
    # Maximum gradient norm for global gradient clipping.
    max_grad_norm: float = 0.5
    # Number of passes over the rollout per update.
    update_epochs: int = 10
    # Minibatch size for PPO updates.
    minibatch_size: int = 64
    # How many environment steps to collect before each update phase.
    steps_per_epoch: int = 2048
    # Number of training epochs (update cycles).
    epochs: int = 50
    # How often (in epochs) to print training metrics.
    log_interval: int = 10
    # Device string understood by torch.device, e.g. "cpu" or "cuda".
    device: str = "cpu"


@dataclass
class RolloutBatch:
    """
    Simple container for a PPO rollout.

    All tensors are expected to be 1D over time (T, ...) and will be moved
    to the agent's device inside `PPOAgent.update`.
    """

    obs: torch.Tensor          # (T, obs_dim)
    actions: torch.Tensor      # (T,)
    log_probs: torch.Tensor    # (T,)
    rewards: torch.Tensor      # (T,)
    dones: torch.Tensor        # (T,)
    values: torch.Tensor       # (T,)
    last_value: torch.Tensor   # ()


class PPOAgent:
    """
    PPO agent built around an `ActorCritic` model for discrete actions.

    Exposes:
      - `act(state)` for interaction
      - `update(rollout)` for applying PPO updates
    """

    def __init__(self, state_dim: int, action_dim: int, config: PPOConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)

        self.model = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)

    def act(self, state: np.ndarray) -> tuple[int, float, float]:
        """
        Sample an action for a single state.

        Returns
        -------
        action : int
        log_prob : float
        value : float
        """
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            probs, value = self.model(state_t)
            dist = Categorical(probs=probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return (
            int(action.item()),
            float(log_prob.item()),
            float(value.squeeze(0).item()),
        )

    def value(self, state: np.ndarray) -> float:
        """Estimate V(s) for a single state without sampling an action."""
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            _, value = self.model(state_t)
        return float(value.squeeze(0).item())

    def evaluate(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate log-probs, entropy, and values for given states/actions.

        Parameters
        ----------
        states : (T, obs_dim)
        actions : (T,)
        """
        probs, values = self.model(states)
        dist = Categorical(probs=probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy, values

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        last_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generalized Advantage Estimation (GAE-Lambda).

        All inputs are 1D tensors of length T except `last_value` (scalar).
        """
        gamma = self.config.gamma
        lam = self.config.gae_lambda

        T = rewards.shape[0]
        advantages = torch.zeros(T, device=self.device)
        last_gae = 0.0

        for t in reversed(range(T)):
            next_non_terminal = 1.0 - dones[t]
            next_value = last_value if t == T - 1 else values[t + 1]
            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            last_gae = delta + gamma * lam * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        return advantages, returns

    def update(self, rollout: RolloutBatch) -> dict[str, float]:
        """
        Apply PPO updates using a rollout batch.
        """
        obs = rollout.obs.to(self.device)
        actions = rollout.actions.to(self.device).long()
        old_log_probs = rollout.log_probs.to(self.device)
        rewards = rollout.rewards.to(self.device)
        dones = rollout.dones.to(self.device)
        values = rollout.values.to(self.device)
        last_value = rollout.last_value.to(self.device)

        advantages, returns = self.compute_gae(
            rewards=rewards,
            values=values,
            dones=dones,
            last_value=last_value,
        )

        # Normalize advantages for stability.
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        clip_eps = self.config.clip_eps
        batch_size = obs.shape[0]
        minibatch_size = min(self.config.minibatch_size, batch_size)

        stats: dict[str, float] = {}

        for _ in range(self.config.update_epochs):
            permutation = torch.randperm(batch_size, device=self.device)

            for start in range(0, batch_size, minibatch_size):
                idx = permutation[start : start + minibatch_size]

                mb_obs = obs[idx]
                mb_actions = actions[idx]
                mb_old_log_probs = old_log_probs[idx]
                mb_advantages = advantages[idx]
                mb_returns = returns[idx]

                new_log_probs, entropy, value_pred = self.evaluate(mb_obs, mb_actions)
                ratio = (new_log_probs - mb_old_log_probs).exp()

                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(value_pred, mb_returns)
                entropy_loss = -entropy.mean()

                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    + self.config.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = (mb_old_log_probs - new_log_probs).mean().clamp(min=0).item()

                # Track last minibatch stats from the last epoch; good enough for a smoke test.
                stats.update(
                    {
                        "policy_loss": float(policy_loss.item()),
                        "value_loss": float(value_loss.item()),
                        "entropy": float(entropy.mean().item()),
                        "approx_kl": float(approx_kl),
                    }
                )

        return stats
