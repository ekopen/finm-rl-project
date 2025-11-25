import torch

class PPOAgent:
    """
    Minimal PPO agent template.
    No training logic yet — just the structure.
    """

    def __init__(self, actor, critic, config):
        self.actor = actor
        self.critic = critic
        self.config = config

    def select_action(self, obs):
        """
        Returns: action, log_prob
        """
        with torch.no_grad():
            probs = self.actor(torch.tensor(obs, dtype=torch.float32))
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return int(action.item()), log_prob.item()

    def update(self, buffer):
        """
        Placeholder — training logic goes here.
        """
        pass
