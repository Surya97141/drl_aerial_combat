# This script fills the gap of never benchmarking TD3 against PPO baseline in the existing reinforcement learning project
from environment.base_env import AerialCombatEnv
from stable_baselines3 import TD3
import csv
import os

env = AerialCombatEnv()
model = TD3("MlpPolicy", env)
log_dir = "logs/metrics/"
os.makedirs(log_dir, exist_ok=True)
csv_file = open(log_dir + "td3_metrics.csv", "w", newline="")
writer = csv.DictWriter(csv_file, fieldnames=["timestep", "mean_reward", "std_reward"])
writer.writeheader()

for i in range(100000):
    model.learn(100)
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
    writer.writerow({"timestep": i * 100, "mean_reward": mean_reward, "std_reward": std_reward})
    if i % 100 == 0:
        print(f"Step {i * 100}")
model.save("models/td3_model")
csv_file.close()