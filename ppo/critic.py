# import torch
# import torch.nn as nn

# class Critic(nn.Module):
#     """
#     Minimal value network template.
#     """

#     def __init__(self, obs_dim):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(obs_dim, 32),
#             nn.ReLU(),
#             nn.Linear(32, 1)
#         )

#     def forward(self, obs):
#         return self.net(obs)

# ppo/critic.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    """
    Simple MLP value network V(s).
    """

    def __init__(self, obs_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: (batch_size, obs_dim)
        v = self.net(obs)   # (batch_size, 1)
        return v

