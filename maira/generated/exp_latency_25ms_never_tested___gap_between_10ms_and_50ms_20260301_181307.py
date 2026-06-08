# This experiment fills the gap in latency testing between 10ms and 50ms, specifically targeting a 25ms latency.

from environment.base_env import AerialCombatEnv
from stable_baselines3 import PPO
import pandas as pd
import numpy as np
import os

env = AerialCombatEnv(latency=0.025)
model = PPO('MlpPolicy', env, verbose=0)

timesteps = 100000
results = []
for i in range(0, timesteps, 10000):
    model.learn(10000)
    rewards = [env.reset(); sum([env.step(model.predict(env.get_state()))[1] for _ in range(100)]) for _ in range(100)]
    results.append([i, np.mean(rewards), np.std(rewards)])
    print(f'Progress: {i/10000}%')
    env.reset()

pd.DataFrame(results, columns=['timestep', 'mean_reward', 'std_reward']).to_csv('logs/metrics/latency_25ms.csv', index=False)
model.save('models/ppo_latency_25ms')