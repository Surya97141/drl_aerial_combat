"""
Generational Self-Play Training

Trains the agent across G generations.  At the end of each generation the
current policy is frozen and added to the opponent pool.  The next generation
trains against a random sample from that pool (league-style mixing), producing
an agent that has faced an ever-stronger, diverse set of opponents.

Architecture (for the paper):
  - Generation 0 : trains against heuristic baseline (same as PPO baseline)
  - Generation 1+ : trains against pool of frozen past checkpoints
  - Opponent pool  : keeps the last POOL_SIZE checkpoints (oldest dropped)
  - Steps/gen      : STEPS_PER_GEN  (default 300k — affordable on CPU)
  - Evaluation     : after every generation, tested on both heuristic AND
                     the current best self-play opponent

Outputs (all under models/selfplay/):
  gen{N}/agent.zip          — trained agent
  gen{N}/opponent.zip       — frozen copy used as next-gen opponent
  selfplay_log.json         — per-generation kill rates vs heuristic + self

Usage:
  python experiments/train_selfplay.py               # full run (~5 generations)
  python experiments/train_selfplay.py --gens 2       # quick smoke test
  python experiments/train_selfplay.py --steps 100000 # fewer steps/gen
"""

import sys
import json
import argparse
import shutil
import numpy as np
from pathlib import Path
from collections import Counter
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "environment"))

from enhanced_env import EnhancedAerialCombatEnv
from self_play_env import SelfPlayEnv

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------

GENERATIONS    = 5
STEPS_PER_GEN  = 300_000
N_EVAL_EPS     = 20
POOL_SIZE      = 3    # keep last 3 frozen checkpoints as possible opponents
SEEDS          = [42, 123]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_ppo(env, seed: int, log_name: str):
    from stable_baselines3 import PPO
    return PPO(
        "MlpPolicy", env,
        seed           = seed,
        verbose        = 0,
        learning_rate  = 3e-4,
        n_steps        = 2048,
        batch_size     = 64,
        n_epochs       = 10,
        gamma          = 0.99,
        gae_lambda     = 0.95,
        clip_range     = 0.2,
        ent_coef       = 0.01,
        tensorboard_log= str(project_root / "ppo_logs" / "selfplay" / log_name),
    )


def evaluate_agent(model, env_fn, n_eps: int = N_EVAL_EPS, seed_offset: int = 0) -> dict:
    """Run n_eps episodes and return outcome stats."""
    env      = env_fn()
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
        "kill_count":  kills,
        "avg_reward":  float(np.mean(rewards)),
        "avg_damage":  float(np.mean(damages)),
        "outcomes":    dict(Counter(outcomes)),
    }


# ---------------------------------------------------------------------------
# One generation of training
# ---------------------------------------------------------------------------

def train_one_generation(
    gen:        int,
    pool:       list,         # list of .zip path strings for opponents
    seed:       int,
    model_dir:  Path,
    prev_model: Path = None,  # warm-start from previous gen if provided
) -> Path:
    """Train for one generation and return path to saved model."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    out_dir = model_dir / f"gen{gen}_s{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / "agent.zip"

    if final_path.exists():
        print(f"    [skip] gen{gen}_s{seed} already trained")
        return final_path

    print(f"\n    Training gen{gen} seed={seed}"
          f"  pool_size={len(pool)}"
          f"  {'(heuristic opp)' if not pool else '(selfplay opp)'}")

    def make_env():
        env = SelfPlayEnv(
            opp_model_paths=list(pool),
            noise_std=0.0,
            latency_steps=0,
            max_steps=1000,
        )
        return env

    train_env = DummyVecEnv([make_env for _ in range(4)])
    eval_env  = make_env()

    if prev_model and prev_model.exists():
        # Warm-start: load weights, replace env
        model = PPO.load(str(prev_model), env=train_env)
        model.set_random_seed(seed)
    else:
        model = make_ppo(train_env, seed, f"gen{gen}_s{seed}")

    from stable_baselines3.common.callbacks import EvalCallback
    model.learn(
        total_timesteps = STEPS_PER_GEN,
        callback        = EvalCallback(
            eval_env,
            best_model_save_path = str(out_dir),
            log_path             = str(out_dir),
            eval_freq            = 10_000,
            deterministic        = True,
            render               = False,
            verbose              = 0,
        ),
        progress_bar = True,
        reset_num_timesteps = (prev_model is None),
    )
    model.save(str(final_path.with_suffix("")))
    print(f"    Saved -> {final_path}")
    return final_path


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_selfplay(
    n_gens:     int = GENERATIONS,
    steps:      int = STEPS_PER_GEN,
    seeds:      list = None,
    warm_start: bool = True,
) -> dict:
    global STEPS_PER_GEN
    STEPS_PER_GEN = steps
    if seeds is None:
        seeds = SEEDS

    model_dir  = project_root / "models" / "selfplay"
    model_dir.mkdir(parents=True, exist_ok=True)
    log_path   = model_dir / "selfplay_log.json"

    # Load existing log (for resuming)
    if log_path.exists():
        log = json.loads(log_path.read_text())
    else:
        log = {"generations": [], "started": datetime.now().isoformat()}

    # Opponent pool — paths of frozen checkpoints
    pool: list = []
    # Seed → last trained model path (for warm-starting)
    prev_models = {s: None for s in seeds}

    print(f"\n{'='*60}")
    print(f"  SELF-PLAY TRAINING  —  {n_gens} generations × {steps:,} steps")
    print(f"  Seeds: {seeds}  |  Pool size: {POOL_SIZE}")
    print(f"{'='*60}\n")

    for gen in range(n_gens):
        gen_entry = {
            "gen":              gen,
            "pool_size":        len(pool),
            "opp_type":         "heuristic" if not pool else "selfplay",
            "seeds":            {},
        }

        # ---- Train all seeds in this generation ----
        gen_models = []
        for seed in seeds:
            prev = prev_models[seed]
            model_path = train_one_generation(
                gen, pool, seed, model_dir,
                prev_model = prev if warm_start else None,
            )
            gen_models.append(model_path)
            prev_models[seed] = model_path

        # ---- Evaluate best model from this generation ----
        from stable_baselines3 import PPO
        # Use first seed's best model for evaluation
        best_path = model_dir / f"gen{gen}_s{seeds[0]}" / "best_model.zip"
        if not best_path.exists():
            best_path = gen_models[0]

        eval_model = PPO.load(str(best_path))

        # Vs heuristic opponent
        def heuristic_env():
            return EnhancedAerialCombatEnv(noise_std=0.0, latency_steps=0, max_steps=1000)

        heur_stats = evaluate_agent(eval_model, heuristic_env, seed_offset=gen)

        print(f"\n  Gen {gen} vs heuristic : kill_rate={heur_stats['kill_rate']:.0%}  "
              f"avg_damage={heur_stats['avg_damage']:.0f}")

        gen_entry["seeds"][str(seeds[0])] = {
            "vs_heuristic": heur_stats,
        }

        # Vs best self-play opponent (if pool non-empty)
        if pool:
            def sp_eval_env():
                return SelfPlayEnv(
                    opp_model_paths=[pool[-1]],   # vs latest frozen opp
                    opp_deterministic=True,
                    noise_std=0.0, latency_steps=0, max_steps=1000,
                )
            sp_stats = evaluate_agent(eval_model, sp_eval_env, seed_offset=gen + 100)
            print(f"  Gen {gen} vs selfplay  : kill_rate={sp_stats['kill_rate']:.0%}  "
                  f"avg_damage={sp_stats['avg_damage']:.0f}")
            gen_entry["seeds"][str(seeds[0])]["vs_selfplay"] = sp_stats

        log["generations"].append(gen_entry)
        log_path.write_text(json.dumps(log, indent=2))

        # ---- Freeze best model and add to pool ----
        frozen_path = model_dir / f"opp_gen{gen}.zip"
        shutil.copy(str(best_path), str(frozen_path))
        pool.append(str(frozen_path))
        if len(pool) > POOL_SIZE:
            pool.pop(0)   # drop oldest

        print(f"  Pool updated: {len(pool)} opponents  (gen{max(0,gen-POOL_SIZE+1)}–gen{gen})")

    # ---- Final summary ----
    print(f"\n{'='*60}")
    print("  SELF-PLAY RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  Gen  | Opp Type   | Kill Rate (vs heuristic)")
    print(f"  {'─'*40}")
    for g in log["generations"]:
        seed_key = str(seeds[0])
        kr = g["seeds"].get(seed_key, {}).get("vs_heuristic", {}).get("kill_rate", 0)
        print(f"  {g['gen']:>3}  | {g['opp_type']:<10} | {kr:.0%}")
    print(f"{'='*60}\n")
    print(f"  Log -> {log_path}")

    return log


# ---------------------------------------------------------------------------
# Robustness evaluation of final self-play agent
# ---------------------------------------------------------------------------

def evaluate_selfplay_robustness(log: dict = None) -> dict:
    """
    Run the final self-play agent through the 4×4 robustness grid and compare
    with the baseline.  Saves to results/selfplay_robustness.json.
    """
    from stable_baselines3 import PPO

    # Find latest gen model
    model_dir = project_root / "models" / "selfplay"
    gens = sorted(model_dir.glob("gen*_s*/best_model.zip"))
    if not gens:
        gens = sorted(model_dir.glob("gen*_s*/agent.zip"))
    if not gens:
        print("No selfplay models found — run train_selfplay.py first.")
        return {}

    final_model_path = gens[-1]
    print(f"\n  Evaluating final self-play model: {final_model_path}\n")
    model = PPO.load(str(final_model_path))

    TEST_CONDITIONS = {
        "clean":    {"noise_std": 0.0, "latency_steps": 0},
        "latency":  {"noise_std": 0.0, "latency_steps": 2},
        "noise":    {"noise_std": 8.0, "latency_steps": 0},
        "degraded": {"noise_std": 8.0, "latency_steps": 2},
    }

    results = {}
    for test_name, cfg in TEST_CONDITIONS.items():
        def env_fn(c=cfg):
            return EnhancedAerialCombatEnv(**c, max_steps=1000)
        stats = evaluate_agent(model, env_fn, n_eps=20, seed_offset=999)
        results[test_name] = stats
        print(f"  selfplay vs {test_name:<10}: kill_rate={stats['kill_rate']:.0%}  "
              f"avg_damage={stats['avg_damage']:.0f}")

    out_path = project_root / "results" / "selfplay_robustness.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n  Saved -> {out_path}")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gens",       type=int,   default=GENERATIONS,
                        help="Number of self-play generations")
    parser.add_argument("--steps",      type=int,   default=STEPS_PER_GEN,
                        help="Training steps per generation")
    parser.add_argument("--no-warmstart", action="store_true",
                        help="Train from scratch each generation (no weight carry-over)")
    parser.add_argument("--eval-only",  action="store_true",
                        help="Skip training, run robustness eval on saved models")
    args = parser.parse_args()

    if args.eval_only:
        evaluate_selfplay_robustness()
    else:
        log = run_selfplay(
            n_gens     = args.gens,
            steps      = args.steps,
            warm_start = not args.no_warmstart,
        )
        print("\n  Running robustness evaluation of final self-play agent...\n")
        evaluate_selfplay_robustness(log)
