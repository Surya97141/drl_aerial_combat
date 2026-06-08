# This experiment fills the gap between 10ms and 50ms latency, specifically testing 25ms latency in the AerialCombatEnv

from environment.base_env import AerialCombatEnv
from stable_baselines3 import PPO
import os
import csv

env = AerialCombatEnv(latency=25)
model = PPO("MlpPolicy", env, verbose=0)

timestep = 0
rewards = []
while timestep < 100000:
    obs = env.reset()
    done = False
    episode_reward = 0
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        episode_reward += reward
    rewards.append(episode_reward)
    timestep += 1000
    if timestep % 10000 == 0:
        print(f"Timestep: {timestep}, Mean Reward: {sum(rewards)/len(rewards)}, Std Reward: {rewards and (sum((x - sum(rewards)/len(rewards)) ** 2 for x in rewards) / len(rewards)) ** 0.5 or 0}")
        with open('logs/metrics/latency_25ms.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestep, sum(rewards)/len(rewards), rewards and (sum((x - sum(rewards)/len(rewards)) ** 2 for x in rewards) / len(rewards)) ** 0.5 or 0])
    model.learn(total_timesteps=1000)
model.save("models/ppo_latency_25ms")