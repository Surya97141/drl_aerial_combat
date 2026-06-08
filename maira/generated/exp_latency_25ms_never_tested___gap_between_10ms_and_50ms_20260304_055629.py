# This experiment fills the gap between 10ms and 50ms latency by testing with 25ms latency in the AerialCombatEnv

from environment.base_env import AerialCombatEnv
from stable_baselines3 import PPO
import pandas as pd
import os

env = AerialCombatEnv(latency=25)
model = PPO("MlpPolicy", env, verbose=0)

results = {"timestep": [], "mean_reward": [], "std_reward": []}
timestep = 0
while timestep < 100000:
    model.learn(10000)
    timestep += 10000
    rewards = [sum(env.run_episode(model)) for _ in range(100)]
    results["timestep"].append(timestep)
    results["mean_reward"].append(sum(rewards) / len(rewards))
    results["std_reward"].append((sum((x - sum(rewards) / len(rewards)) ** 2 for x in rewards) / len(rewards)) ** 0.5)
    print(f"Trained for {timestep} steps")
    pd.DataFrame(results).to_csv("logs/metrics/latency_25ms.csv", index=False)
    model.save("models/ppo_latency_25ms")