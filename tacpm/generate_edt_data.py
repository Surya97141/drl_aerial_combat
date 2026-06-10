"""
EDT Synthetic Data Generator

Creates training data for the Episode Diagnostic Transformer by running
deliberately misconfigured reward functions and labelling outcomes.

Six broken configurations × N_EPISODES each = labelled trajectory dataset.
Saves: data/edt_train.npz

Usage:
    python tacpm/generate_edt_data.py              # 50 eps per config
    python tacpm/generate_edt_data.py --episodes 20  # faster, for smoke test
"""

import sys
import argparse
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "environment"))

from enhanced_env import EnhancedAerialCombatEnv
from edt_model import (
    FAILURE_MODES, FIX_TYPES, N_FEATURES, MAX_LEN,
)

# ---------------------------------------------------------------------------
# Broken-env subclasses — override _calculate_reward only
# ---------------------------------------------------------------------------

class _NoAltitudePenaltyEnv(EnhancedAerialCombatEnv):
    """Altitude crash penalty removed -> agent will crash frequently."""
    def _calculate_reward(self, terminated, truncated, combat_info):
        r = super()._calculate_reward(terminated, truncated, combat_info)
        # Cancel the altitude penalty the parent added
        altitude = self.agent_state["position"][2]
        if altitude < 200.0:
            r += (200.0 - altitude) / 200.0 * 5.0
        return r


class _LargeProximityRangeEnv(EnhancedAerialCombatEnv):
    """Proximity reward active at 5 km instead of 1 km -> agent orbits far away."""
    def _calculate_reward(self, terminated, truncated, combat_info):
        r = super()._calculate_reward(terminated, truncated, combat_info)
        # Remove the parent's 1-km gated proximity reward
        rel = self.opponent_state["position"] - self.agent_state["position"]
        dist = float(np.linalg.norm(rel))
        r -= max(0.0, (1000.0 - dist) / 1000.0) * 3.0
        # Replace with 5-km gated proximity reward (much weaker pull)
        r += max(0.0, (5000.0 - dist) / 5000.0) * 1.0
        return r


class _NoFiringBonusEnv(EnhancedAerialCombatEnv):
    """Firing bonus removed -> agent doesn't learn to aim and fire."""
    def _calculate_reward(self, terminated, truncated, combat_info):
        r = super()._calculate_reward(terminated, truncated, combat_info)
        if combat_info["agent_fired"]:
            r -= 1.0   # remove the +1.0 firing bonus from parent
        return r


class _NoKillRewardEnv(EnhancedAerialCombatEnv):
    """Kill terminal reward removed -> agent has no incentive to finish the fight."""
    def _calculate_reward(self, terminated, truncated, combat_info):
        r = super()._calculate_reward(terminated, truncated, combat_info)
        if (terminated
                and self.opponent_state["health"] <= 0
                and self.agent_state["health"] > 0):
            r -= 100.0   # cancel the parent's +100 kill reward
        return r


class _OppAdvantageEnv(EnhancedAerialCombatEnv):
    """Opponent fires from 900 m, agent only from 300 m -> agent always dies first."""
    def __init__(self, **kwargs):
        super().__init__(
            engagement_range=300.0,
            opp_engagement_range=900.0,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Config registry
# ---------------------------------------------------------------------------

CONFIGS = [
    {
        "name":         "working",
        "EnvClass":     EnhancedAerialCombatEnv,
        "failure_mode": FAILURE_MODES.index("KILL"),
        "fix_type":     FIX_TYPES.index("NO_FIX_NEEDED"),
        "description":  "Normal config — agent kills",
    },
    {
        "name":         "no_altitude_penalty",
        "EnvClass":     _NoAltitudePenaltyEnv,
        "failure_mode": FAILURE_MODES.index("CRASH"),
        "fix_type":     FIX_TYPES.index("ADD_ALTITUDE_PENALTY"),
        "description":  "No altitude penalty — agent crashes",
    },
    {
        "name":         "large_proximity_range",
        "EnvClass":     _LargeProximityRangeEnv,
        "failure_mode": FAILURE_MODES.index("STALEMATE"),
        "fix_type":     FIX_TYPES.index("REDUCE_PROXIMITY_RANGE"),
        "description":  "5-km proximity range — agent orbits far",
    },
    {
        "name":         "no_firing_bonus",
        "EnvClass":     _NoFiringBonusEnv,
        "failure_mode": FAILURE_MODES.index("TIMEOUT_DRIFT"),
        "fix_type":     FIX_TYPES.index("ADD_FIRING_BONUS"),
        "description":  "No firing bonus — agent approaches but won't engage",
    },
    {
        "name":         "no_kill_reward",
        "EnvClass":     _NoKillRewardEnv,
        "failure_mode": FAILURE_MODES.index("STALEMATE"),
        "fix_type":     FIX_TYPES.index("INCREASE_KILL_REWARD"),
        "description":  "No kill reward — agent fights but never finishes",
    },
    {
        "name":         "opp_advantage",
        "EnvClass":     _OppAdvantageEnv,
        "failure_mode": FAILURE_MODES.index("CLOSE_RANGE_LOSS"),
        "fix_type":     FIX_TYPES.index("FIX_OPP_ADVANTAGE"),
        "description":  "Opponent range 900m vs agent 300m — agent always loses",
    },
]

# ---------------------------------------------------------------------------
# Policy: mix of pursuit heuristic and random for realistic trajectories
# ---------------------------------------------------------------------------

def _heuristic_action(obs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Simple pursuit policy: fly toward opponent, maintain altitude.
    50% chance to inject random noise so trajectories aren't perfectly smooth.
    """
    agent_pos = obs[0:3]
    opp_pos   = obs[6:9]

    rel  = opp_pos - agent_pos
    dist = np.linalg.norm(rel) + 1e-8
    rel_norm = rel / dist

    throttle = 0.8 + rng.uniform(-0.2, 0.2)

    # Pitch to maintain ~1000 m altitude
    alt   = agent_pos[2]
    pitch = np.clip((1000.0 - alt) / 500.0, -1.0, 1.0) * 0.4

    # Yaw: align with horizontal direction to opponent (crude approximation)
    yaw = np.clip(rel_norm[1] * 2.0, -1.0, 1.0) + rng.uniform(-0.3, 0.3)

    action = np.array([throttle, pitch, 0.0, yaw], dtype=np.float32)

    # 40% random perturbation
    if rng.random() < 0.4:
        action += rng.uniform(-0.5, 0.5, size=4).astype(np.float32)

    return np.clip(action, -1.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Trajectory extraction
# ---------------------------------------------------------------------------

def _extract_features(timesteps: list) -> np.ndarray:
    """Convert timestep list to (MAX_LEN, N_FEATURES) padded array."""
    arr = np.zeros((MAX_LEN, N_FEATURES), dtype=np.float32)
    for i, t in enumerate(timesteps[:MAX_LEN]):
        dist  = t["distance"]
        alt   = t["agent_pos"][2]
        a_h   = t["agent_health"]
        o_h   = t["opp_health"]
        r     = t["reward"]
        step  = t["t"]
        fired = 1.0 if dist <= 600.0 else 0.0
        arr[i] = [
            dist / 15000.0,
            a_h  / 100.0,
            o_h  / 100.0,
            float(np.clip(r / 30.0, -1.0, 1.0)),
            alt  / 2000.0,
            fired,
            step / 1000.0,
        ]
    return arr


def _load_ppo_model():
    """Try to load the best trained PPO baseline for the working config."""
    from stable_baselines3 import PPO
    candidates = [
        project_root / "models" / "robustness" / "baseline_s42"  / "best_model.zip",
        project_root / "models" / "robustness" / "baseline_s42"  / "ppo_final.zip",
        project_root / "models" / "ppo_agent.zip",
    ]
    for p in candidates:
        if p.exists():
            return PPO.load(str(p))
    return None


def _run_episodes(
    EnvClass,
    n_episodes:   int,
    seed_offset:  int  = 0,
    ppo_model           = None,
) -> list:
    """Run n_episodes, return list of (MAX_LEN, N_FEATURES) arrays.
    Uses ppo_model if provided, otherwise falls back to heuristic policy.
    """
    rng  = np.random.default_rng(42 + seed_offset)
    env  = EnvClass(noise_std=0.0, latency_steps=0, max_steps=1000)
    episodes = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep + seed_offset * 1000)
        timesteps = []
        done = False
        step = 0

        while not done:
            if ppo_model is not None:
                action, _ = ppo_model.predict(obs, deterministic=False)
            else:
                action = _heuristic_action(obs, rng)

            agent_pos = env.agent_state["position"].tolist()
            a_h = float(env.agent_state["health"])
            o_h = float(env.opponent_state["health"])

            obs_next, reward, term, trunc, info = env.step(action)
            done = term or trunc

            rel  = (env.opponent_state["position"] - env.agent_state["position"])
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
            step += 1

        episodes.append(_extract_features(timesteps))

    env.close()
    return episodes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate(n_episodes: int = 50) -> None:
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)
    out_path = data_dir / "edt_train.npz"

    all_X, all_failure, all_fix, all_lengths = [], [], [], []

    # Load trained PPO model once — used for the "working" config only
    ppo_model = _load_ppo_model()
    if ppo_model is not None:
        print(f"  PPO model loaded for 'working' config trajectories.\n")
    else:
        print(f"  No PPO model found — using heuristic policy for all configs.\n")

    for i, cfg in enumerate(CONFIGS):
        print(f"  [{i+1}/{len(CONFIGS)}] {cfg['name']}  ({cfg['description']})")
        use_ppo = ppo_model if cfg["name"] == "working" else None
        episodes = _run_episodes(cfg["EnvClass"], n_episodes,
                                 seed_offset=i, ppo_model=use_ppo)

        for arr in episodes:
            all_X.append(arr)
            all_failure.append(cfg["failure_mode"])
            all_fix.append(cfg["fix_type"])
            # Length = last non-zero row index + 1
            nonzero = np.any(arr != 0, axis=1)
            length  = int(nonzero.sum()) if nonzero.any() else 1
            all_lengths.append(length)

    X        = np.stack(all_X)            # (N, MAX_LEN, N_FEATURES)
    failures = np.array(all_failure, dtype=np.int64)
    fixes    = np.array(all_fix,     dtype=np.int64)
    lengths  = np.array(all_lengths, dtype=np.int64)

    np.savez(out_path, X=X, failures=failures, fixes=fixes, lengths=lengths)

    total = len(X)
    print(f"\n  Saved {total} trajectories -> {out_path}")
    print(f"  Breakdown:")
    from collections import Counter
    for mode_idx, count in Counter(failures.tolist()).items():
        print(f"    {FAILURE_MODES[mode_idx]:<22}  {count} eps")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50,
                        help="Episodes per broken config (default 50)")
    args = parser.parse_args()

    print(f"\nGenerating EDT training data ({args.episodes} eps × {len(CONFIGS)} configs)...\n")
    generate(args.episodes)
