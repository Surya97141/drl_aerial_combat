# This experiment fills the gap between 10ms and 50ms latency, specifically testing 25ms latency in the AerialCombatEnv environment
from environment.base_env import AerialCombatEnv
from stable_baselines3 import PPO
import csv

env = AerialCombatEnv(latency=0.025)
model = PPO("MlpPolicy", env, verbose=0)

timestep = 0
mean_rewards = []
std_rewards = []

while timestep < 100000:
    model.learn(10000)
    timestep += 10000
    rewards = [sum(env.run_episode(model)) for _ in range(100)]
    mean_rewards.append([timestep, sum(rewards)/len(rewards)])
    std_rewards.append([timestep, (sum((x - sum(rewards)/len(rewards)) ** 2 for x in rewards) / len(rewards)) ** 0.5])
    print(f"Timestep {timestep}")

model.save("models/ppo_latency_25ms")
with open('logs/metrics/ppo_latency_25ms.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["timestep", "mean_reward", "std_reward"])
    for i in range(len(mean_rewards)):
        writer.writerow([mean_rewards[i][0], mean_rewards[i][1], std_rewards[i][1]])