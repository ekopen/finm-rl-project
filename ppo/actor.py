import torch
import torch.nn as nn

class Actor(nn.Module):
    """
    Minimal policy network template.
    """

    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 32),
            nn.ReLU(),
            nn.Linear(32, act_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, obs):
        return self.net(obs)
