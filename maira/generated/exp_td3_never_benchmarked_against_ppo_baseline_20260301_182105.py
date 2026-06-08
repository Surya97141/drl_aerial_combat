# This experiment fills the gap of benchmarking TD3 against PPO in AerialCombatEnv, providing a comparison of two different reinforcement learning algorithms in a complex environment.

from environment.base_env import AerialCombatEnv
from stable_baselines3 import TD3
import numpy as np
import pandas as pd
import os

env = AerialCombatEnv()
model = TD3('MlpPolicy', env, verbose=0)
logs = []

for i in range(100000):
    obs = env.reset()
    done = False
    rewards = []
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        rewards.append(reward)
    if i % 10000 == 0:
        print(f'Step {i}')
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        logs.append([i, mean_reward, std_reward])
model.save('models/td3_aerial_combat')
pd.DataFrame(logs, columns=['timestep', 'mean_reward', 'std_reward']).to_csv('logs/metrics/td3_metrics.csv', index=False)