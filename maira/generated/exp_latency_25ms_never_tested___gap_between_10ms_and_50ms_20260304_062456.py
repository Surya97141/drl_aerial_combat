# Fills the gap between 10ms and 50ms latency by testing 25ms latency in AerialCombatEnv
from environment.base_env import AerialCombatEnv
from stable_baselines3 import PPO
import pandas as pd

env = AerialCombatEnv(latency=25)
model = PPO("MlpPolicy", env, verbose=0)
rewards = []
timesteps = []
for i in range(100000):
    obs = env.reset()
    done = False
    episode_reward = 0
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        episode_reward += reward
    rewards.append(episode_reward)
    timesteps.append((i+1)*1000)
    if (i+1) % 10 == 0:
        print(f"Step {timesteps[-1]}")
    if (i+1) % 100 == 0:
        model.save(f"models/ppo_latency_25ms_{i+1}")
        df = pd.DataFrame({"timestep": timesteps, "mean_reward": [sum(rewards)/len(rewards)], "std_reward": [0]})
        df.to_csv(f"logs/metrics/{i+1}.csv", index=False)