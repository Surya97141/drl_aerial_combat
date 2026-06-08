# This experiment fills the gap between 10ms and 50ms latency in the AerialCombatEnv, specifically testing a 25ms latency scenario.

from environment.base_env import AerialCombatEnv
from stable_baselines3 import PPO
import pandas as pd
import os

env = AerialCombatEnv(latency=0.025)
model = PPO('MlpPolicy', env, verbose=0)

log_dir = 'logs/metrics/'
os.makedirs(log_dir, exist_ok=True)
timesteps = 100000
for i in range(0, timesteps, 10000):
    model.learn(10000)
    rewards = [env.reset()[1] for _ in range(10)]
    df = pd.DataFrame({'timestep': [i], 'mean_reward': [sum(rewards)/len(rewards)], 'std_reward': [pd.DataFrame(rewards).std().values[0]]})
    df.to_csv(log_dir + 'ppo_latency_25ms.csv', mode='a', header=False if i else True, index=False)
    print(f'Trained {i} timesteps')
model.save('models/ppo_latency_25ms')