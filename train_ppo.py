"""Train PPO agent on aerial combat environment."""

import gymnasium as gym
import stable_baselines3 as sb3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.logger import configure
import numpy as np
import sys
from pathlib import Path

# FIRST: Fix paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "1_environment"))

# NOW import envs
from enhanced_env import EnhancedAerialCombatEnv as AerialCombatEnv
# from enhanced_env import EnhancedAerialCombatEnv  # Uncomment later

def main():
    print(" Training PPO on Aerial Combat Environment...")
    
    # Training envs (4 parallel for speed)
    train_env_fn = lambda: AerialCombatEnv(max_steps=1000)
    train_env = DummyVecEnv([train_env_fn for _ in range(4)])
    
    # Eval env
    eval_env = AerialCombatEnv(max_steps=1000)
    
    # Callbacks
    eval_callback = EvalCallback(eval_env, best_model_save_path="./models/",
                                 log_path="./ppo_logs/", eval_freq=5000,
                                 deterministic=True, render=False)
    
    checkpoint_callback = CheckpointCallback(save_freq=50_000,
                                             save_path="./models/",
                                             name_prefix="ppo_checkpoint")
    
    # PPO with good defaults
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        tensorboard_log="./ppo_logs/",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0
    )
    
    print("Starting training (500k steps)...")
    model.learn(
        total_timesteps=500_000,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )
    
    # Save
    model.save("models/ppo_aerial_combat_final")
    
    # Test 20 episodes
    print("\n Testing final agent (20 episodes)...")
    episode_rewards = []
    
    for ep in range(20):
        ep_reward = 0
        obs, _ = eval_env.reset()
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            ep_reward += reward
            done = terminated or truncated
        
        episode_rewards.append(ep_reward)
        print(f"Ep {ep+1}: {ep_reward:.1f}")
    
    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    print(f"\n Final: {avg_reward:.1f} ± {std_reward:.1f}")
    print(" Done! tensorboard --logdir=./ppo_logs/")
    
    return avg_reward

if __name__ == "__main__":
    main()