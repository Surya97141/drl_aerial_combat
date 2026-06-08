# This experiment fills the gap of evaluating the performance of reinforcement learning models 
# with multiple seeds to assess their robustness and reliability. 
# Expect to see the mean reward of the model at each timestep, saved to a CSV file and 
# model checkpoints saved to the models directory.

from stable_baselines3 import PPO
from custom_environment import CustomEnvironment
import numpy as np
import csv
import os

env = CustomEnvironment()
model = PPO("MlpPolicy", env, verbose=1)

seeds = [123, 456, 789]
for seed in seeds:
    env.seed(seed)
    model.seed(seed)
    model.learn(total_timesteps=1000)
    rewards = []
    for _ in range(100):
        obs = env.reset()
        done = False
        reward = 0
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, _ = env.step(action)
        rewards.append(reward)
    mean_reward = np.mean(rewards)
    with open("logs/metrics/metrics.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([seed, "mean_reward", mean_reward])
    model.save(f"models/ppo_{seed}")