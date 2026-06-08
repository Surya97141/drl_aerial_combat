"""
TacPM — Replay Generator
Runs trained-model episodes and saves timestep-level data for post-mortem analysis.
"""

import json
import numpy as np
import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "environment"))

from enhanced_env import EnhancedAerialCombatEnv


def run_episode(env: EnhancedAerialCombatEnv, model, episode_id: int) -> dict:
    """Run one episode with the given policy and record every timestep."""
    obs, _ = env.reset()
    timesteps = []
    ep_info = {}

    while True:
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()

        next_obs, reward, terminated, truncated, info = env.step(action)

        # obs indices: pos[0:3] vel[3:6] opp_pos[6:9] opp_vel[9:12] health[12:14]
        timesteps.append({
            "t": env.current_step,
            "agent_pos":      obs[0:3].tolist(),
            "agent_vel":      obs[3:6].tolist(),
            "opp_pos":        obs[6:9].tolist(),
            "opp_vel":        obs[9:12].tolist(),
            "agent_health":   float(obs[12]) * 100.0,
            "opp_health":     float(obs[13]) * 100.0,
            "action":         action.tolist(),
            "reward":         float(reward),
            "distance":       float(np.linalg.norm(obs[0:3] - obs[6:9])),
        })

        obs = next_obs
        if terminated or truncated:
            ep_info = info
            break

    outcome = ep_info.get("outcome", "timeout" if not terminated else "terminated")
    total_reward = sum(t["reward"] for t in timesteps)

    return {
        "episode_id":               episode_id,
        "outcome":                  outcome,
        "kill":                     ep_info.get("kill", False),
        "total_reward":             round(total_reward, 4),
        "length":                   len(timesteps),
        "damage_dealt":             ep_info.get("damage_dealt", 0.0),
        "damage_received":          ep_info.get("damage_received", 0.0),
        "net_damage":               ep_info.get("net_damage", 0.0),
        "min_distance":             ep_info.get("min_distance", -1.0),
        "time_to_first_engagement": ep_info.get("time_to_first_engagement"),
        "agent_health_final":       ep_info.get("agent_health_final", 0.0),
        "opp_health_final":         ep_info.get("opp_health_final", 100.0),
        "timesteps":                timesteps,
    }


def generate_replays(
    n_episodes: int = 10,
    save_path: str = None,
    model_path: str = None,
    noise_std: float = 0.0,
    latency_steps: int = 0,
):
    """
    Generate n_episodes and save to JSON.

    Args:
        model_path: Path to a saved SB3 model (.zip). If None, uses random policy.
        noise_std / latency_steps: Environment conditions to match the trained variant.
    """
    if save_path is None:
        save_path = str(project_root / "data" / "replays.json")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    model = None
    if model_path is not None:
        from stable_baselines3 import PPO
        model = PPO.load(model_path)
        print(f"Loaded model from {model_path}")
    else:
        print("No model path given — using random policy")

    env = EnhancedAerialCombatEnv(
        noise_std=noise_std,
        latency_steps=latency_steps,
    )
    replays = []
    kills = 0

    for i in range(n_episodes):
        ep = run_episode(env, model, episode_id=i)
        replays.append(ep)
        if ep["kill"]:
            kills += 1
        print(
            f"  Episode {i+1:02d}/{n_episodes} | outcome={ep['outcome']:<8} "
            f"reward={ep['total_reward']:7.1f}  dmg_dealt={ep['damage_dealt']:.0f}  "
            f"min_dist={ep['min_distance']:.0f}m"
        )

    env.close()

    print(f"\nKill rate: {kills}/{n_episodes} ({kills/n_episodes:.0%})")

    with open(save_path, "w") as f:
        json.dump(replays, f, indent=2)

    print(f"Saved {n_episodes} replays → {save_path}")
    return save_path


if __name__ == "__main__":
    # Example: run with trained baseline model
    default_model = str(project_root / "models" / "baseline" / "ppo_final.zip")
    model_path = default_model if Path(default_model).exists() else None

    generate_replays(
        n_episodes=10,
        model_path=model_path,
        noise_std=0.0,
        latency_steps=0,
    )
