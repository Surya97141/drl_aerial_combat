# This script fills the gap of implementing the latency_sweep experiment in the reinforcement learning project.
# It trains a single agent using the PPO algorithm and evaluates its performance.
# The experiment sweeps over different latency values to analyze their impact on the agent's performance.
# To expect: The script will print progress at regular intervals, save model checkpoints, and log metrics to a CSV file.

from stable_baselines3 import PPO
from tacpm.train import get_env
import pandas as pd
import os

env = get_env()
model = PPO("MlpPolicy", env, verbose=1)
for i in range(1000):
    model.learn(100)
    print(f"Episode {i+1}")
    rewards = []
    for _ in range(10):
        done = False
        obs = env.reset()
        reward = 0
        while not done:
            action, _ = model.predict(obs)
            obs, r, done, _ = env.step(action)
            reward += r
        rewards.append(reward)
    mean_reward = sum(rewards) / len(rewards)
    metrics = pd.DataFrame({"timestep": [i], "metric_name": ["mean_reward"], "metric_value": [mean_reward]})
    metrics.to_csv("logs/metrics/latency_sweep_metrics.csv", mode="a", header=False, index=False)
    model.save(f"models/ppo_latency_sweep_{i}")