# This script fills the gap of benchmarking TD3 against PPO baseline in the AerialCombatEnv environment

from environment.base_env import AerialCombatEnv
from stable_baselines3 import TD3
import os
import csv

env = AerialCombatEnv()
model = TD3("MlpPolicy", env, verbose=0)

timestep = 0
mean_rewards = []
std_rewards = []

while timestep < 100000:
    model.learn(10000)
    rewards = []
    for _ in range(10):
        obs = env.reset()
        done = False
        reward = 0
        while not done:
            action, _ = model.predict(obs)
            obs, r, done, _ = env.step(action)
            reward += r
        rewards.append(reward)
    mean_reward = sum(rewards) / len(rewards)
    std_reward = (sum((x - mean_reward) ** 2 for x in rewards) / len(rewards)) ** 0.5
    mean_rewards.append(mean_reward)
    std_rewards.append(std_reward)
    print(f"Timestep: {timestep}, Mean Reward: {mean_reward}, Std Reward: {std_reward}")
    timestep += 10000

model.save("models/td3_aerial_combat")
with open("logs/metrics/td3_metrics.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["timestep", "mean_reward", "std_reward"])
    for i in range(len(mean_rewards)):
        writer.writerow([i * 10000, mean_rewards[i], std_rewards[i]])