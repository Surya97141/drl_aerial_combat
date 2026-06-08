# This experiment fills the gap in latency testing between 10ms and 50ms with a 25ms latency setting
from environment.base_env import AerialCombatEnv
from stable_baselines3 import PPO
import csv

env = AerialCombatEnv(latency=25)
model = PPO("MlpPolicy", env, verbose=0)
timestep = 0
mean_rewards = []
while timestep < 100000:
    model.learn(10000)
    timestep += 10000
    rewards = []
    for _ in range(10):
        done = False
        reward = 0
        obs = env.reset()
        while not done:
            action, _ = model.predict(obs)
            obs, r, done, _ = env.step(action)
            reward += r
        rewards.append(reward)
    mean_reward = sum(rewards) / len(rewards)
    std_reward = (sum((x - mean_reward) ** 2 for x in rewards) / len(rewards)) ** 0.5
    mean_rewards.append((timestep, mean_reward, std_reward))
    print(f"Timestep: {timestep}, Mean Reward: {mean_reward}, Std Reward: {std_reward}")
    with open('logs/metrics/latency_25ms.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["timestep", "mean_reward", "std_reward"])
        writer.writerows(mean_rewards)
model.save("models/ppo_latency_25ms")