"""
Shared model definitions for PPO (actor-critic).

This is a thin wrapper around the existing `Actor` and `Critic` networks so
that experiments can depend on a single `ActorCritic` interface.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .actor import Actor
from .critic import Critic


class ActorCritic(nn.Module):
    """
    Minimal actor-critic container with a unified forward interface.

    The underlying policy and value networks are defined in `actor.py` and
    `critic.py`; this class simply ties them together.

    Expected shapes
    ---------------
    obs : (batch_size, obs_dim)
    returns
      - policy_out : (batch_size, act_dim) -- action probabilities
      - values     : (batch_size,)        -- state values
    """

    def __init__(self, obs_dim: int, act_dim: int) -> None:
        super().__init__()
        self.actor = Actor(obs_dim, act_dim)
        self.critic = Critic(obs_dim)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        policy_out = self.actor(obs)
        values = self.critic(obs).squeeze(-1)
        return policy_out, values


