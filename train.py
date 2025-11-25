import numpy as np
from env.trading_env import TradingEnv
from ppo.actor import Actor
from ppo.critic import Critic
from ppo.ppo_agent import PPOAgent
from ppo.utils import RolloutBuffer

def main():
    prices = np.random.randn(100).cumsum() + 100
    env = TradingEnv(prices)

    actor = Actor(obs_dim=1, act_dim=3)
    critic = Critic(obs_dim=1)
    agent = PPOAgent(actor, critic, config={"dummy": True})

    buffer = RolloutBuffer()

    obs = env.reset()
    for _ in range(10):
        action, logp = agent.select_action(obs)
        next_obs, reward, done, _ = env.step(action)

        buffer.obs.append(obs)
        buffer.actions.append(action)
        buffer.log_probs.append(logp)
        buffer.rewards.append(reward)
        buffer.dones.append(done)

        obs = next_obs
        if done:
            break

    # Placeholder training call
    agent.update(buffer)

if __name__ == "__main__":
    main()
