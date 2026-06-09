"""
Robustness Sweep -- Option 2 core experiment.

Runs a 4x4 train/test grid:
  - 4 training conditions x 3 seeds  =>  12 training runs
  - each trained model evaluated on all 4 test conditions
  => 48 (train_variant, test_condition, seed) evaluation cells

Results are written to results/robustness_grid.json.
From that file, collect_results.py produces Table 1 for the paper.

Usage:
  python experiments/robustness_sweep.py               # full run (~6h on CPU)
  python experiments/robustness_sweep.py --dry-run      # print plan, no training
  python experiments/robustness_sweep.py --eval-only    # skip training, re-eval saved models
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from collections import Counter
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "environment"))

from enhanced_env import EnhancedAerialCombatEnv

# ---------------------------------------------------------------------------
# Experiment grid definition
# ---------------------------------------------------------------------------

TRAIN_VARIANTS = {
    "baseline":      {"noise_std": 0.0, "latency_steps": 0},
    "latency":       {"noise_std": 0.0, "latency_steps": 2},
    "noise":         {"noise_std": 8.0, "latency_steps": 0},
    "degraded":      {"noise_std": 8.0, "latency_steps": 2},
}

TEST_CONDITIONS = {
    "clean":         {"noise_std": 0.0, "latency_steps": 0},
    "latency":       {"noise_std": 0.0, "latency_steps": 2},
    "noise":         {"noise_std": 8.0, "latency_steps": 0},
    "degraded":      {"noise_std": 8.0, "latency_steps": 2},
}

SEEDS           = [42, 123, 456]
TRAIN_STEPS     = 500_000
N_EVAL_EPISODES = 20

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_variant(variant_name: str, config: dict, seed: int) -> Path:
    """Train one (variant, seed) and return the saved model path."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

    run_name  = f"{variant_name}_s{seed}"
    model_dir = project_root / "models" / "robustness" / run_name
    model_dir.mkdir(parents=True, exist_ok=True)
    final_path = model_dir / "ppo_final.zip"

    if final_path.exists():
        print(f"  [skip] {run_name} already trained -> {final_path}")
        return final_path

    print(f"\n  Training {run_name}  noise={config['noise_std']}  latency={config['latency_steps']}  seed={seed}")

    def make_env():
        return EnhancedAerialCombatEnv(**config, max_steps=1000)

    train_env = DummyVecEnv([make_env for _ in range(4)])
    eval_env  = make_env()

    model = PPO(
        "MlpPolicy", train_env, seed=seed, verbose=0,
        tensorboard_log=str(project_root / "ppo_logs" / "robustness" / run_name),
        learning_rate=3e-4, n_steps=2048, batch_size=64,
        n_epochs=10, gamma=0.99, gae_lambda=0.95,
        clip_range=0.2, ent_coef=0.01,
    )
    model.learn(
        total_timesteps=TRAIN_STEPS,
        callback=[
            EvalCallback(eval_env, best_model_save_path=str(model_dir),
                         log_path=str(model_dir), eval_freq=10_000,
                         deterministic=True, render=False, verbose=0),
            CheckpointCallback(save_freq=100_000, save_path=str(model_dir),
                               name_prefix="ckpt", verbose=0),
        ],
        progress_bar=True,
    )
    model.save(str(final_path.with_suffix("")))
    print(f"  Saved -> {final_path}")
    return final_path


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model_path: Path, test_config: dict, seed_offset: int = 0) -> dict:
    """Evaluate a saved model under a given test condition."""
    from stable_baselines3 import PPO

    model   = PPO.load(str(model_path))
    env     = EnhancedAerialCombatEnv(**test_config, max_steps=1000)

    rewards, outcomes, damages, min_dists = [], [], [], []

    for ep in range(N_EVAL_EPISODES):
        obs, _ = env.reset(seed=ep + seed_offset * 1000)
        ep_reward, done, ep_info = 0.0, False, {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            ep_reward += r
            done = term or trunc
            if done:
                ep_info = info
        rewards.append(ep_reward)
        outcomes.append(ep_info.get("outcome", "unknown"))
        damages.append(ep_info.get("damage_dealt", 0.0))
        min_dists.append(ep_info.get("min_distance", -1.0))

    env.close()
    kills = outcomes.count("kill")

    return {
        "kill_rate":      kills / N_EVAL_EPISODES,
        "kill_count":     kills,
        "n_episodes":     N_EVAL_EPISODES,
        "avg_reward":     float(np.mean(rewards)),
        "std_reward":     float(np.std(rewards)),
        "avg_damage":     float(np.mean(damages)),
        "avg_min_dist":   float(np.mean([d for d in min_dists if d >= 0])),
        "outcome_counts": dict(Counter(outcomes)),
    }


# ---------------------------------------------------------------------------
# Main sweep loop
# ---------------------------------------------------------------------------

def run_sweep(dry_run: bool = False, eval_only: bool = False):
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)
    grid_path = results_dir / "robustness_grid.json"

    # Load existing results so we can resume
    if grid_path.exists():
        with open(grid_path) as f:
            grid = json.load(f)
    else:
        grid = {}

    total_cells = len(TRAIN_VARIANTS) * len(SEEDS) * len(TEST_CONDITIONS)
    done_cells  = 0

    print(f"\n{'='*60}")
    print(f"  ROBUSTNESS SWEEP -- {len(TRAIN_VARIANTS)} variants x {len(SEEDS)} seeds")
    print(f"  Train steps: {TRAIN_STEPS:,}  |  Eval episodes: {N_EVAL_EPISODES}")
    print(f"  Total evaluation cells: {total_cells}")
    print(f"{'='*60}\n")

    for variant_name, train_config in TRAIN_VARIANTS.items():
        for seed in SEEDS:
            run_key = f"{variant_name}_s{seed}"

            # --- Training ---
            if not dry_run and not eval_only:
                model_path = train_variant(variant_name, train_config, seed)
            else:
                model_path = project_root / "models" / "robustness" / run_key / "ppo_final.zip"
                if not model_path.exists() and not dry_run:
                    print(f"  [warn] No model found for {run_key}, skipping eval.")
                    continue

            if dry_run:
                print(f"  [dry] Would train {run_key}")
                continue

            # --- Evaluation on all test conditions ---
            if run_key not in grid:
                grid[run_key] = {
                    "variant": variant_name,
                    "seed": seed,
                    "train_config": train_config,
                    "eval": {},
                }

            for test_name, test_config in TEST_CONDITIONS.items():
                cell_key = f"test_{test_name}"
                if cell_key in grid[run_key]["eval"]:
                    done_cells += 1
                    continue

                print(f"  Eval {run_key} on {test_name}... ", end="", flush=True)
                cell = evaluate_model(model_path, test_config, seed_offset=seed)
                cell["test_condition"] = test_name
                grid[run_key]["eval"][cell_key] = cell
                done_cells += 1

                print(f"kill_rate={cell['kill_rate']:.0%}  ({done_cells}/{total_cells})")

                # Save after every cell so we can resume
                with open(grid_path, "w") as f:
                    json.dump(grid, f, indent=2)

    if not dry_run:
        print(f"\nGrid complete -> {grid_path}")
        _print_summary_table(grid)

    return grid


def _print_summary_table(grid: dict):
    """Print a quick kill-rate table to stdout (averaged across seeds)."""
    from collections import defaultdict

    # Accumulate kill rates: cell[variant][test_cond] = [kill_rates...]
    acc = defaultdict(lambda: defaultdict(list))
    for run_key, run in grid.items():
        v = run["variant"]
        for cell_key, cell in run["eval"].items():
            test = cell["test_condition"]
            acc[v][test].append(cell["kill_rate"])

    test_names = list(TEST_CONDITIONS.keys())
    col_w = 12

    print(f"\n{'='*60}")
    print("  KILL RATE TABLE  (mean across seeds)")
    print(f"  Train \\ Test  |" + "|".join(f"{t:^{col_w}}" for t in test_names))
    print("  " + "-" * (14 + col_w * len(test_names) + len(test_names)))
    for v in TRAIN_VARIANTS:
        row = f"  {v:<13} |"
        for t in test_names:
            if acc[v][t]:
                mean = np.mean(acc[v][t])
                row += f"{mean:^{col_w}.0%}"
            else:
                row += f"{'---':^{col_w}}"
            row += "|"
        print(row)
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run",   action="store_true", help="Print plan only")
    parser.add_argument("--eval-only", action="store_true", help="Skip training, re-eval saved models")
    args = parser.parse_args()

    run_sweep(dry_run=args.dry_run, eval_only=args.eval_only)
