# This experiment fills the gap of training an agent in AerialCombatEnv with a latency of 200ms to find the upper bound of its performance.
from environment.base_env import AerialCombatEnv
from stable_baselines3 import PPO
import os
import pandas as pd

env = AerialCombatEnv(latency=200)
model = PPO('MlpPolicy', env, verbose=0)
timesteps = 100000
results = []

for i in range(0, timesteps, 10000):
    model.learn(10000)
    rewards = [sum(model.collect_episodes(10)[0].rewards) for _ in range(10)]
    results.append([i, sum(rewards)/len(rewards), (sum((x - sum(rewards)/len(rewards)) ** 2 for x in rewards) / len(rewards)) ** 0.5])
    print(f'Training at {i} timesteps')
    
df = pd.DataFrame(results, columns=['timestep', 'mean_reward', 'std_reward'])
df.to_csv('logs/metrics/latency_200ms.csv', index=False)
model.save('models/ppo_latency_200ms')