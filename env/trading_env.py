import numpy as np

class TradingEnv:
    """
    Minimal trading environment template.
    """

    def __init__(self, prices):
        self.prices = prices
        self.t = 0

    def reset(self):
        self.t = 0
        return np.array([self.prices[self.t]])

    def step(self, action):
        """
        action: placeholder (0 = hold, 1 = buy, 2 = sell)
        """
        reward = 0.0  # fill in later
        self.t += 1
        done = self.t >= len(self.prices) - 1
        obs = np.array([self.prices[self.t]])
        return obs, reward, done, {}
