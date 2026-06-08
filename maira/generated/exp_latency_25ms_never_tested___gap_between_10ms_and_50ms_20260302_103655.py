# This experiment fills the gap between 10ms and 50ms latency, with a 25ms latency
from environment.base_env import AerialCombatEnv
from stable_baselines3 import PPO
import csv

env = AerialCombatEnv(latency=25)
model = PPO("MlpPolicy", env, verbose=0)
timestep = 0

while timestep < 100000:
    model.learn(10000)
    timestep += 10000
    rewards = [env.reset(); sum([env.step(model.predict(obs)[0])[1] for _ in range(100)]) for _ in range(10)]
    with open("logs/metrics/latency_25ms.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([timestep, sum(rewards)/10, (sum((x - sum(rewards)/10)**2 for x in rewards)/10)**0.5])
    print(f"Timestep: {timestep}")
model.save("models/ppo_latency_25ms")