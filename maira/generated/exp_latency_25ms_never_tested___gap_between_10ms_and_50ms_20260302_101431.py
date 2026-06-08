# This experiment fills the gap in latency testing between 10ms and 50ms, specifically testing a latency of 25ms
from environment.base_env import AerialCombatEnv
from stable_baselines3 import PPO
import csv

env = AerialCombatEnv(latency=25)
model = PPO("MlpPolicy", env)
timesteps = 100000
for i in range(0, timesteps, 10000):
    model.learn(10000)
    rewards = []
    for _ in range(100):
        done = False
        obs = env.reset()
        episode_reward = 0
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
        rewards.append(episode_reward)
    mean_reward = sum(rewards) / len(rewards)
    std_reward = (sum((x - mean_reward) ** 2 for x in rewards) / len(rewards)) ** 0.5
    print(f"Steps: {i}, Mean Reward: {mean_reward}, Std Reward: {std_reward}")
    with open("logs/metrics/latency_25ms.csv", "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if i == 0:
            writer.writerow(["timestep", "mean_reward", "std_reward"])
        writer.writerow([i, mean_reward, std_reward])
model.save("models/ppo_latency_25ms")