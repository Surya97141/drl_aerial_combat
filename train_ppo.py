"""Train PPO agent on aerial combat environment."""

import sys
import numpy as np
from pathlib import Path
from collections import Counter

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "environment"))

from enhanced_env import EnhancedAerialCombatEnv


def make_env(noise_std: float = 0.0, latency_steps: int = 0):
    return lambda: EnhancedAerialCombatEnv(
        noise_std=noise_std,
        latency_steps=latency_steps,
        max_steps=1000,
    )


def train(
    noise_std: float = 0.0,
    latency_steps: int = 0,
    total_timesteps: int = 500_000,
    seed: int = 42,
    run_name: str = "baseline",
):
    print(f"\nTraining PPO — run='{run_name}' seed={seed} "
          f"noise={noise_std} latency={latency_steps}")

    train_env = DummyVecEnv([make_env(noise_std, latency_steps) for _ in range(4)])
    eval_env = EnhancedAerialCombatEnv(noise_std=noise_std, latency_steps=latency_steps)

    model_dir = f"./models/{run_name}"
    log_dir = f"./ppo_logs/{run_name}"

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=5_000,
        deterministic=True,
        render=False,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=model_dir,
        name_prefix="ppo_checkpoint",
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        seed=seed,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # small entropy bonus encourages exploration
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
    )

    model.save(f"{model_dir}/ppo_final")
    print(f"Model saved to {model_dir}/ppo_final")

    # --- Evaluation: 20 episodes, track outcomes ---
    print(f"\nEvaluating {run_name} (20 episodes)...")
    episode_rewards = []
    outcomes = []
    kills = 0
    damage_dealt_list = []

    for ep in range(20):
        ep_reward = 0.0
        obs, _ = eval_env.reset(seed=ep)   # different seed per episode for varied starts
        done = False
        ep_info = {}

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            ep_reward += reward
            done = terminated or truncated
            if done:
                ep_info = info

        episode_rewards.append(ep_reward)
        outcome = ep_info.get("outcome", "unknown")
        outcomes.append(outcome)
        if ep_info.get("kill", False):
            kills += 1
        damage_dealt_list.append(ep_info.get("damage_dealt", 0.0))

        print(
            f"  Ep {ep+1:02d}: reward={ep_reward:7.1f}  outcome={outcome:<8}  "
            f"dmg_dealt={ep_info.get('damage_dealt', 0):.0f}  "
            f"min_dist={ep_info.get('min_distance', -1):.0f}m"
        )

    avg_r = np.mean(episode_rewards)
    std_r = np.std(episode_rewards)
    kill_rate = kills / 20
    outcome_counts = Counter(outcomes)

    print(f"\n  avg_reward : {avg_r:.1f} ± {std_r:.1f}")
    print(f"  kill_rate  : {kill_rate:.0%}  ({kills}/20)")
    print(f"  outcomes   : {dict(outcome_counts)}")
    print(f"  avg_dmg    : {np.mean(damage_dealt_list):.1f}")
    print(f"\ntensorboard --logdir={log_dir}")

    return {
        "run_name": run_name,
        "seed": seed,
        "avg_reward": float(avg_r),
        "std_reward": float(std_r),
        "kill_rate": kill_rate,
        "outcomes": dict(outcome_counts),
        "avg_damage_dealt": float(np.mean(damage_dealt_list)),
    }


if __name__ == "__main__":
    train(
        noise_std=0.0,
        latency_steps=0,
        total_timesteps=500_000,
        seed=42,
        run_name="baseline",
    )
