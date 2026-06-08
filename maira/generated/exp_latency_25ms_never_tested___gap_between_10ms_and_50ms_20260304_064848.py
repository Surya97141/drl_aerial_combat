# This experiment fills the gap in latency testing between 10ms and 50ms by evaluating the model at 25ms latency
from environment.base_env import AerialCombatEnv
from stable_baselines3 import PPO
import pandas as pd
import numpy as np

env = AerialCombatEnv(latency=0.025)
model = PPO("MlpPolicy", env, verbose=0)
rewards = []
timesteps = []

for i in range(100000):
    obs = env.reset()
    done = False
    reward = 0.0
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        if i % 10000 == 0:
            print(f"Step {i}, Reward: {reward}")
    rewards.append(reward)
    timesteps.append(i+1)
    if i % 10000 == 0:
        df = pd.DataFrame({"timestep": timesteps, "mean_reward": np.array(rewards).cumsum()/(np.arange(len(rewards))+1), "std_reward": np.array(rewards).cumsum()/(np.arange(len(rewards))+1)})
        df.to_csv("logs/metrics/ppo_latency_25ms.csv", index=False)
model.save("models/ppo_latency_25ms")