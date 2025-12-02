# import torch
# import torch.nn as nn

# class Actor(nn.Module):
#     """
#     Minimal policy network template.
#     """

#     def __init__(self, obs_dim, act_dim):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(obs_dim, 32),
#             nn.ReLU(),
#             nn.Linear(32, act_dim),
#             nn.Softmax(dim=-1)
#         )

#     def forward(self, obs):
#         return self.net(obs)

# ppo/actor.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    """
    Simple MLP policy network for discrete actions.

    Input:  obs_dim
    Output: action probabilities of shape (batch_size, act_dim)
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, act_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: (batch_size, obs_dim)
        logits = self.net(obs)              # (batch_size, act_dim)
        probs = F.softmax(logits, dim=-1)   # softmax over action dimension
        return probs
