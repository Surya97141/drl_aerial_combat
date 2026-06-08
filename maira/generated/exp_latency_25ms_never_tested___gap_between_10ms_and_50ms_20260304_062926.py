# This experiment fills the gap between 10ms and 50ms latency by testing with 25ms latency
from environment.base_env import AerialCombatEnv
from stable_baselines3 import PPO
import pandas as pd
import os

env = AerialCombatEnv(latency=25)
model = PPO("MlpPolicy", env, verbose=0)
logs_dir = "logs/metrics/'
models_dir = "models/"

for i in range(1000000):
    obs = env.reset()
    done = False
    rewards = []
    while not done:
        action, _state = model.predict(obs, deterministic=False)
        obs, reward, done, info = env.step(action)
        rewards.append(reward)
    if i % 10000 == 0:
        print(i)
        mean_reward = sum(rewards) / len(rewards)
        std_reward = rewards std() if len(rewards) > 1 else 0
        df = pd.DataFrame([[i, mean_reward, std_reward]], columns=['timestep', 'mean_reward', 'std_reward'])
        df.to_csv(logs_dir + 'metrics.csv', mode='a', header=False, index=False)
        model.save(models_dir + 'ppo_latency_25ms')
    env.reset()