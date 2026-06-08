# This experiment fills the gap of comparing the performance of SAC and PPO algorithms in the aerial combat environment.
# It trains SAC and PPO agents and saves their metrics, allowing us to expect a comparison of their mean rewards.

from environment.base_env import BaseEnv
from stable_baselines3 import PPO, SAC

env = BaseEnv()
models = []
for algo in ['PPO', 'SAC']:
    if algo == 'PPO':
        model = PPO('MlpPolicy', env)
    else:
        model = SAC('MlpPolicy', env)
    models.append(model)

for i, model in enumerate(models):
    model.learn(total_timesteps=1000)
    model.save(f"models/{model.__class__.__name__}_sac_vs_ppo")
    with open(f"logs/metrics/{model.__class__.__name__}_metrics.csv", 'w') as f:
        f.write("timestep,metric_name,metric_value\n")
        f.write(f"1000,mean_reward,{model.rewards[-1]}\n")
    print(f"Trained {model.__class__.__name__}")