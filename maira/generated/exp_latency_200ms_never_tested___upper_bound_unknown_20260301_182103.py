# Filling the gap of training AerialCombatEnv with PPO and 200ms latency, exploring its upper bound
from environment.base_env import AerialCombatEnv
from stable_baselines3 import PPO
import csv
import os

env = AerialCombatEnv(latency=200)
model = PPO('MlpPolicy', env, verbose=0)

timestep = 0
rewards = []
while timestep < 1000000:
    obs = env.reset()
    done = False
    episode_reward = 0
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        episode_reward += reward
        timestep += 1
        if timestep % 10000 == 0:
            print(f"Timestep: {timestep}")
    rewards.append(episode_reward)
    if len(rewards) > 100:
        rewards.pop(0)
    mean_reward = sum(rewards) / len(rewards)
    std_reward = (sum((x - mean_reward) ** 2 for x in rewards) / len(rewards)) ** 0.5
    with open('logs/metrics/latency_200ms.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(['timestep', 'mean_reward', 'std_reward'])
        writer.writerow([timestep, mean_reward, std_reward])
model.save('models/ppo_latency_200ms')