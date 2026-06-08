# This experiment fills the gap between 10ms and 50ms latency by testing a 25ms latency environment

from environment.base_env import AerialCombatEnv
from stable_baselines3 import PPO
import csv

env = AerialCombatEnv(latency=25)
model = PPO("MlpPolicy", env, verbose=0)
results = []

for i in range(100000):
    obs = env.reset()
    done = False
    rewards = []
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        rewards.append(reward)
    rewards = sum(rewards)
    results.append((i+1, rewards))
    if i % 10000 == 0:
        print(f"Step {i+1}, Reward: {sum([r[1] for r in results[-10:]]) / 10}")
        mean_reward = sum([r[1] for r in results[-10:]]) / 10
        std_reward = (sum([(r[1] - mean_reward) ** 2 for r in results[-10:]]) / 10) ** 0.5
        with open("logs/metrics/latency_25ms.csv", "a", newline="") as f:
            csv.writer(f).writerow([i+1, mean_reward, std_reward])
model.save("models/ppo_latency_25ms")