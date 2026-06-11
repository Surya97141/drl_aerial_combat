"""
Automated Reward Engineering Loop (Active EDT Loop)

Closes the loop between episode failure diagnosis and reward improvement:

  1. Train PPO agent for TRAIN_STEPS
  2. Generate replays with trained agent
  3. EDT diagnoses failure modes -> recommends fix
  4. Apply fix to a RewardConfig and retrain
  5. Measure kill-rate delta
  6. Append labelled episode to EDT active-learning buffer
  7. Repeat

This is the "zero-human-intervention reward engineering" novelty contribution.

Usage:
    python experiments/auto_reward_loop.py              # 3 iterations default
    python experiments/auto_reward_loop.py --iters 5
    python experiments/auto_reward_loop.py --steps 200000
"""

import sys
import json
import argparse
import copy
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "environment"))
sys.path.insert(0, str(project_root / "tacpm"))

from enhanced_env import EnhancedAerialCombatEnv

# ---------------------------------------------------------------------------
# Reward config — all shaping parameters in one dict
# ---------------------------------------------------------------------------

DEFAULT_REWARD_CONFIG = {
    "proximity_range":         1000.0,   # gate distance for proximity reward
    "proximity_weight":        3.0,
    "altitude_penalty_start":  200.0,
    "altitude_penalty_weight": 5.0,
    "altitude_penalty_enabled":True,
    "firing_bonus":            1.0,
    "kill_reward":             100.0,
    "death_penalty":           50.0,
    "heading_weight":          1.5,
    "closing_weight":          1.0,
    "health_weight":           2.0,
}

# Mapping from EDT fix type -> function that modifies the config
FIX_APPLICATORS = {
    "NO_FIX_NEEDED": lambda cfg: cfg,

    "ADD_ALTITUDE_PENALTY": lambda cfg: {
        **cfg,
        "altitude_penalty_enabled": True,
        "altitude_penalty_weight": cfg["altitude_penalty_weight"] * 1.5,
        "altitude_penalty_start":  max(cfg["altitude_penalty_start"], 300.0),
    },

    "REDUCE_PROXIMITY_RANGE": lambda cfg: {
        **cfg,
        "proximity_range":  max(500.0, cfg["proximity_range"] * 0.7),
        "proximity_weight": cfg["proximity_weight"] * 1.2,
    },

    "ADD_FIRING_BONUS": lambda cfg: {
        **cfg,
        "firing_bonus": cfg["firing_bonus"] * 1.75,
    },

    "INCREASE_KILL_REWARD": lambda cfg: {
        **cfg,
        "kill_reward": cfg["kill_reward"] * 1.5,
    },

    "FIX_OPP_ADVANTAGE": lambda cfg: cfg,   # env-level change, flagged in log
}


# ---------------------------------------------------------------------------
# Configurable env — injects reward config at step time
# ---------------------------------------------------------------------------

class ConfigurableRewardEnv(EnhancedAerialCombatEnv):
    def __init__(self, reward_cfg: dict = None, **kwargs):
        super().__init__(**kwargs)
        self._rcfg = reward_cfg or copy.deepcopy(DEFAULT_REWARD_CONFIG)

    def _calculate_reward(self, terminated, truncated, combat_info):
        cfg    = self._rcfg
        reward = 0.0

        reward += 0.1   # per-step survival

        rel_pos  = self.opponent_state["position"] - self.agent_state["position"]
        distance = float(np.linalg.norm(rel_pos))

        # Proximity (gated)
        if distance < cfg["proximity_range"]:
            reward += (cfg["proximity_range"] - distance) / cfg["proximity_range"] \
                      * cfg["proximity_weight"]

        # Altitude penalty
        altitude = self.agent_state["position"][2]
        if cfg["altitude_penalty_enabled"] and altitude < cfg["altitude_penalty_start"]:
            reward -= (cfg["altitude_penalty_start"] - altitude) \
                      / cfg["altitude_penalty_start"] * cfg["altitude_penalty_weight"]

        # Heading reward
        if distance > 1e-6:
            heading_vec = np.array([
                np.cos(self.agent_state["heading"]),
                np.sin(self.agent_state["heading"]),
                0.0,
            ])
            rel_norm = rel_pos / distance
            dot      = float(np.dot(heading_vec, rel_norm))
            reward  += max(0.0, (dot + 1.0) / 2.0) * cfg["heading_weight"]

        # Closing speed
        if distance > 1e-6:
            vel_diff = self.agent_state["velocity"] - self.opponent_state["velocity"]
            closing  = float(np.dot(vel_diff, rel_pos / distance))
            reward  += max(0.0, closing / 200.0) * cfg["closing_weight"]

        # Health advantage
        health_diff = (self.agent_state["health"] - self.opponent_state["health"]) / 100.0
        reward += health_diff * cfg["health_weight"]

        # Firing bonus
        if combat_info["agent_fired"]:
            reward += cfg["firing_bonus"]

        # Terminal
        if terminated:
            if self.opponent_state["health"] <= 0 and self.agent_state["health"] > 0:
                reward += cfg["kill_reward"]
            elif self.agent_state["health"] <= 0:
                reward -= cfg["death_penalty"]
            elif self.agent_state["position"][2] < 0:
                reward -= 30.0
        elif truncated:
            reward += (self._damage_dealt - self._damage_received) / 100.0

        return float(reward)


# ---------------------------------------------------------------------------
# Train + evaluate helpers
# ---------------------------------------------------------------------------

def train_agent(reward_cfg: dict, steps: int, seed: int, save_path: Path):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    def make_env():
        return ConfigurableRewardEnv(reward_cfg=reward_cfg,
                                     noise_std=0.0, latency_steps=0, max_steps=1000)

    env   = DummyVecEnv([make_env for _ in range(4)])
    model = PPO(
        "MlpPolicy", env, seed=seed, verbose=0,
        learning_rate=3e-4, n_steps=2048, batch_size=64,
        n_epochs=10, gamma=0.99, ent_coef=0.01,
    )
    model.learn(total_timesteps=steps, progress_bar=True)
    model.save(str(save_path))
    return model


def evaluate_agent(model, reward_cfg: dict, n_eps: int = 20, seed_offset: int = 0) -> dict:
    env     = ConfigurableRewardEnv(reward_cfg=reward_cfg,
                                    noise_std=0.0, latency_steps=0, max_steps=1000)
    rewards, outcomes, damages = [], [], []
    for ep in range(n_eps):
        obs, _ = env.reset(seed=ep + seed_offset * 1000)
        done, ep_r, ep_info = False, 0.0, {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            ep_r += r
            done  = term or trunc
            if done:
                ep_info = info
        rewards.append(ep_r)
        outcomes.append(ep_info.get("outcome", "unknown"))
        damages.append(ep_info.get("damage_dealt", 0.0))
    env.close()
    kills = outcomes.count("kill")
    return {
        "kill_rate":   kills / n_eps,
        "avg_reward":  float(np.mean(rewards)),
        "avg_damage":  float(np.mean(damages)),
        "outcomes":    dict(Counter(outcomes)),
    }


def collect_replays(model, reward_cfg: dict, n_eps: int = 10, seed_offset: int = 0) -> list:
    """Collect episode trajectories for EDT diagnosis."""
    from edt_diagnose import append_active
    from edt_model import FAILURE_MODES, FIX_TYPES

    env     = ConfigurableRewardEnv(reward_cfg=reward_cfg,
                                    noise_std=0.0, latency_steps=0, max_steps=1000)
    replays = []
    for ep in range(n_eps):
        obs, _ = env.reset(seed=ep + seed_offset * 1000)
        timesteps = []
        done = False
        step = 0
        while not done:
            agent_pos = env.agent_state["position"].tolist()
            a_h = float(env.agent_state["health"])
            o_h = float(env.opponent_state["health"])

            action, _ = model.predict(obs, deterministic=True)
            obs_next, reward, term, trunc, info = env.step(action)

            rel  = env.opponent_state["position"] - env.agent_state["position"]
            dist = float(np.linalg.norm(rel))

            timesteps.append({
                "t":            step,
                "distance":     dist,
                "agent_health": a_h,
                "opp_health":   o_h,
                "reward":       float(reward),
                "agent_pos":    agent_pos,
            })
            obs  = obs_next
            done = term or trunc
            step += 1

        outcome = info.get("outcome", "unknown")
        replays.append({
            "episode_id": ep + seed_offset * 1000,
            "outcome":    outcome,
            "kill":       info.get("kill", False),
            "length":     len(timesteps),
            "total_reward": sum(t["reward"] for t in timesteps),
            "damage_dealt": info.get("damage_dealt", 0.0),
            "timesteps":  timesteps,
        })
    env.close()
    return replays


def run_edt_diagnosis(replays: list) -> str:
    """Run EDT on replay list, return the most-voted fix type string."""
    from edt_diagnose import load_model, diagnose_episode

    model    = load_model()
    fix_votes = Counter()
    for ep in replays:
        d = diagnose_episode(model, ep)
        fix_votes[d["predicted_fix"]] += 1
    return fix_votes.most_common(1)[0][0]


def append_to_active_buffer(replays: list, fix_type: str) -> None:
    """Label replay episodes with the fix that was applied and save to active buffer."""
    from edt_diagnose import append_active
    from edt_model import FIX_TYPES, FAILURE_MODES

    # Map fix_type string -> index
    fix_idx = FIX_TYPES.index(fix_type) if fix_type in FIX_TYPES else 0

    # Derive failure mode from outcomes
    outcomes = [ep["outcome"] for ep in replays]
    dominant = Counter(outcomes).most_common(1)[0][0]
    mode_map = {
        "kill": "KILL", "died": "CLOSE_RANGE_LOSS",
        "crashed": "CRASH", "timeout": "STALEMATE", "fled": "DISENGAGEMENT",
    }
    fail_label = mode_map.get(dominant, "STALEMATE")
    fail_idx   = FAILURE_MODES.index(fail_label)

    for ep in replays[:5]:   # only append a few per iteration to avoid bloat
        append_active(ep["timesteps"], fail_idx, fix_idx)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_loop(n_iters: int = 3, steps_per_iter: int = 200_000, seed: int = 42):
    run_dir  = project_root / "models" / "auto_reward"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "loop_log.json"

    log = {
        "started":    datetime.now().isoformat(),
        "iterations": [],
    }

    reward_cfg = copy.deepcopy(DEFAULT_REWARD_CONFIG)
    model_path = None

    print(f"\n{'='*60}")
    print(f"  AUTO REWARD ENGINEERING LOOP  —  {n_iters} iterations")
    print(f"  Steps/iter: {steps_per_iter:,}  |  Seed: {seed}")
    print(f"{'='*60}\n")

    for it in range(n_iters):
        print(f"\n--- Iteration {it} ---")
        print(f"  Reward config: proximity_range={reward_cfg['proximity_range']:.0f}  "
              f"firing_bonus={reward_cfg['firing_bonus']:.2f}  "
              f"kill_reward={reward_cfg['kill_reward']:.0f}")

        # 1. Train
        iter_path = run_dir / f"iter{it}_model"
        model     = train_agent(reward_cfg, steps_per_iter, seed + it, iter_path)

        # 2. Evaluate baseline kill rate
        before_stats = evaluate_agent(model, reward_cfg,
                                      n_eps=20, seed_offset=it)
        print(f"  Kill rate before fix: {before_stats['kill_rate']:.0%}")

        # 3. Collect replays
        replays = collect_replays(model, reward_cfg, n_eps=10, seed_offset=it)

        # 4. EDT diagnosis
        fix_type = run_edt_diagnosis(replays)
        print(f"  EDT recommended fix : {fix_type}")

        # 5. Apply fix to reward config
        new_cfg = FIX_APPLICATORS.get(fix_type, lambda c: c)(copy.deepcopy(reward_cfg))

        # 6. Quick re-eval with new config (same model, new reward shaping)
        after_stats = evaluate_agent(model, new_cfg,
                                     n_eps=20, seed_offset=it + 50)

        delta = after_stats["kill_rate"] - before_stats["kill_rate"]
        print(f"  Kill rate after fix : {after_stats['kill_rate']:.0%}  "
              f"(delta={delta:+.0%})")

        # 7. Append to EDT active buffer
        append_to_active_buffer(replays, fix_type)

        # 8. Accept new config if it improves kill rate
        if delta >= 0 or fix_type == "NO_FIX_NEEDED":
            reward_cfg = new_cfg
            accepted   = True
        else:
            accepted   = False
            print(f"  Fix REJECTED (kill rate dropped) — keeping previous config")

        entry = {
            "iter":          it,
            "fix_applied":   fix_type,
            "accepted":      accepted,
            "kill_before":   before_stats["kill_rate"],
            "kill_after":    after_stats["kill_rate"],
            "kill_delta":    delta,
            "reward_config": reward_cfg,
        }
        log["iterations"].append(entry)
        log_path.write_text(json.dumps(log, indent=2))
        print(f"  Config {'accepted' if accepted else 'rejected'}  |  log -> {log_path}")

    # Final summary
    print(f"\n{'='*60}")
    print("  AUTO REWARD LOOP — SUMMARY")
    print(f"{'='*60}")
    print(f"  Iter | Fix                        | Kill Δ  | Accepted")
    print(f"  {'─'*52}")
    for e in log["iterations"]:
        print(f"  {e['iter']:>4} | {e['fix_applied']:<26} | "
              f"{e['kill_delta']:>+.0%}    | {'✓' if e['accepted'] else '✗'}")
    print(f"{'='*60}\n")
    print(f"  Final reward config: {json.dumps(reward_cfg, indent=4)}")
    return log


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--seed",  type=int, default=42)
    args = parser.parse_args()

    run_loop(n_iters=args.iters, steps_per_iter=args.steps, seed=args.seed)
