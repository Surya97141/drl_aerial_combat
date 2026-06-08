# This experiment fills the gap between the 10ms and 50ms latency tests, evaluating the model's performance at 25ms latency
from environment.base_env import AerialCombatEnv
from stable_baselines3 import PPO
import pandas as pd

env = AerialCombatEnv(latency=25)
model = PPO("MlpPolicy", env, verbose=0)
rewards = []
for i in range(100000):
    obs = env.reset()
    done = False
    episode_reward = 0
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        episode_reward += reward
    rewards.append(episode_reward)
    if (i+1) % 10000 == 0:
        print(f"Step: {i+1}")
        mean_reward = sum(rewards[-100:]) / 100
        std_reward = (sum((x - mean_reward) ** 2 for x in rewards[-100:]) / 100) ** 0.5
        pd.DataFrame([[i+1, mean_reward, std_reward]], columns=['timestep', 'mean_reward', 'std_reward']).to_csv('logs/metrics/latency_25ms.csv', mode='a', header=False, index=False)
model.save("models/ppo_latency_25ms")