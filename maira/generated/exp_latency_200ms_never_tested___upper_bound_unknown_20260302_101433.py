# This script fills the gap of testing the AerialCombatEnv with PPO and 200ms latency, where the upper bound of the results is unknown.

from environment.base_env import AerialCombatEnv
from stable_baselines3 import PPO
import csv

env = AerialCombatEnv(latency=200)
model = PPO("MlpPolicy", env, verbose=0)

timestep = 0
mean_rewards = []
std_rewards = []

while timestep < 100000:
    model.learn(10000)
    timestep += 10000
    rewards = []
    for _ in range(100):
        obs = env.reset()
        done = False
        reward = 0
        while not done:
            action, _ = model.predict(obs)
            obs, r, done, _ = env.step(action)
            reward += r
        rewards.append(reward)
    mean_reward = sum(rewards) / len(rewards)
    std_reward = (sum((r - mean_reward) ** 2 for r in rewards) / len(rewards)) ** 0.5
    mean_rewards.append(mean_reward)
    std_rewards.append(std_reward)
    print(f"{timestep}: {mean_reward}")
    with open("logs/metrics/latency_200ms.csv", "a", newline="") as f:
        writer = csv.writer(f)
        if timestep == 10000:
            writer.writerow(["timestep", "mean_reward", "std_reward"])
        writer.writerow([timestep, mean_reward, std_reward])
model.save("models/ppo_latency_200ms")