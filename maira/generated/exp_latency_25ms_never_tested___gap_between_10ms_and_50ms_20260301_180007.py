# This experiment fills the gap between 10ms and 50ms latency with a 25ms setting in the aerial combat environment
from environment.base_env import AerialCombatEnv
from stable_baselines3 import PPO
import pandas as pd
import os

env = AerialCombatEnv(latency=0.025)
model = PPO('MlpPolicy', env, verbose=0)

timesteps = 0
rewards = []
while timesteps < 100000:
    model.learn(10000)
    timesteps += 10000
    rewards.extend([env.reset()['reward'] for _ in range(100)])
    mean_reward = sum(rewards) / len(rewards)
    std_reward = (sum((x - mean_reward) ** 2 for x in rewards) / len(rewards)) ** 0.5
    print(f"{timesteps}, {mean_reward}, {std_reward}")
    pd.DataFrame({'timestep': [timesteps], 'mean_reward': [mean_reward], 'std_reward': [std_reward]}).to_csv('logs/metrics/latency_25ms.csv', mode='a', header=False, index=False)
model.save('models/ppo_latency_25ms')