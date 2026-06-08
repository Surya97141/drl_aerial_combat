# This script fills the gap of training a PPO model with 200ms latency in the AerialCombatEnv environment.

from environment.base_env import AerialCombatEnv
from stable_baselines3 import PPO
import csv
import os

env = AerialCombatEnv(latency=200)
model = PPO("MlpPolicy", env, verbose=0)

log_dir = "logs/metrics/"
os.makedirs(log_dir, exist_ok=True)
csv_file = open(os.path.join(log_dir, "ppo_latency_200ms.csv"), "w", newline="")
csv_writer = csv.writer(csv_file)

for i in range(100000):
    model.learn(10000)
    rewards = [env.reset(); sum([env.step(model.predict(env.get_observation()))[1] for _ in range(100)]) for _ in range(10)]
    mean_reward, std_reward = sum(rewards) / len(rewards), (sum((x - sum(rewards) / len(rewards)) ** 2 for x in rewards) / len(rewards)) ** 0.5
    csv_writer.writerow([i * 10000 + 10000, mean_reward, std_reward])
    print(f"Timestep: {i * 10000 + 10000}")
    csv_file.flush()

model.save("models/ppo_latency_200ms") 
csv_file.close()