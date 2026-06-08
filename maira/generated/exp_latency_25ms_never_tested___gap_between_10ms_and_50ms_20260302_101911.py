# This experiment fills the gap between 10ms and 50ms latency by testing a 25ms latency environment in the aerial combat scenario

from environment.base_env import AerialCombatEnv
from stable_baselines3 import PPO
import pandas as pd
from datetime import datetime

env = AerialCombatEnv(latency=25)
model = PPO("MlpPolicy", env, verbose=0)

log_data = []
timestep = 0
while True:
    model.learn(10000)
    timestep += 10000
    rewards = [env.reset(); sum([env.step(model.predict(obs)[0])[1] for _ in range(100)]) for _ in range(10)]
    log_data.append((timestep, sum(rewards)/len(rewards), (sum([x**2 for x in rewards])/len(rewards))**0.5))
    print(f"Timestep: {timestep}")
    if timestep > 500000:
        break

pd.DataFrame(log_data, columns=["timestep", "mean_reward", "std_reward"]).to_csv(f"logs/metrics/{datetime.now()}.csv", index=False)
model.save(f"models/ppo_latency_25ms_{datetime.now()}")