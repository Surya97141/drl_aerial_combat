# This script fills the gap of comparing TD3 with PPO baseline in the AerialCombatEnv, providing insight into their relative performance.

from environment.base_env import AerialCombatEnv
from stable_baselines3 import TD3
import csv

env = AerialCombatEnv()
model = TD3("MlpPolicy", env, verbose=0)

timestep = 0
mean_rewards = []
while timestep < 100000:
    model.learn(10000)
    timestep += 10000
    rewards = []
    for _ in range(10):
        obs = env.reset()
        reward = 0
        for _ in range(100):
            action, _ = model.predict(obs)
            obs, r, _, _ = env.step(action)
            reward += r
        rewards.append(reward)
    mean_reward = sum(rewards) / len(rewards)
    std_reward = (sum((x - mean_reward) ** 2 for x in rewards) / len(rewards)) ** 0.5
    mean_rewards.append((timestep, mean_reward, std_reward))
    print(f"Timestep {timestep}, Mean Reward {mean_reward}, Std Reward {std_reward}")
    with open('logs/metrics/td3.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["timestep", "mean_reward", "std_reward"])
        writer.writerows(mean_rewards)
model.save("models/td3_model")