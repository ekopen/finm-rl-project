class RolloutBuffer:
    """
    Minimal rollout buffer for PPO.
    """

    def __init__(self):
        self.obs = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []

    def reset(self):
        self.__init__()
