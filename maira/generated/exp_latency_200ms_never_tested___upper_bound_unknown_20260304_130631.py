# Fills the gap by testing the upper bound of latency with 200ms in the AerialCombatEnv using PPO from stable_baselines3

from environment.base_env import AerialCombatEnv
from stable_baselines3 import PPO
import pandas as pd

env = AerialCombatEnv(latency=200)
model = PPO("MlpPolicy", env, verbose=0)
timesteps = 100000
mean_rewards = []
std_rewards = []
for i in range(0, timesteps, 10000):
    model.learn(10000)
    rewards = [sum([env.step(model.predict(obs)[0])[1] for _ in range(100)]) for obs in [env.reset() for _ in range(100)]]
    mean_reward = sum(rewards) / len(rewards)
    std_reward = (sum([(r - mean_reward) ** 2 for r in rewards]) / len(rewards)) ** 0.5
    print(f"Progress: {i+10000} steps")
    mean_rewards.append([i+10000, mean_reward])
    std_rewards.append([i+10000, std_reward])
model.save("models/ppo_latency_200ms")
pd.DataFrame(mean_rewards, columns=["timestep", "mean_reward"]).to_csv("logs/metrics/mean_rewards.csv", index=False)
pd.DataFrame(std_rewards, columns=["timestep", "std_reward"]).to_csv("logs/metrics/std_rewards.csv", index=False)