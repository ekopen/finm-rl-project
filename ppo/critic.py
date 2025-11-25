import torch
import torch.nn as nn

class Critic(nn.Module):
    """
    Minimal value network template.
    """

    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, obs):
        return self.net(obs)
