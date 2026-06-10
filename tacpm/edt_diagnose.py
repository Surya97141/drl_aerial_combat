"""
EDT Inference — Episode Diagnosis

Loads a trained EDT and runs it over a replays.json file produced by
tacpm/replay_generator.py.  Outputs:
  - Console diagnosis with attention-weighted critical timestep
  - data/edt_diagnosis.json — structured output compatible with reward_shaper

Active-learning hook:
  If --label <failure_idx> --fix <fix_idx> is passed, the episode trajectory
  is appended to data/edt_active.npz for the next training round.

Usage:
    python tacpm/edt_diagnose.py
    python tacpm/edt_diagnose.py --replays data/replays.json
    python tacpm/edt_diagnose.py --episode 3   # diagnose one episode
    python tacpm/edt_diagnose.py --list-labels  # print failure/fix label tables
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "tacpm"))

from edt_model import (
    EpisodeDiagnosticTransformer, FAILURE_MODES, FIX_TYPES,
    trajectory_to_tensor, make_pad_mask,
    MAX_LEN, N_FEATURES,
)

DEVICE = torch.device("cpu")

# ---------------------------------------------------------------------------
# Concrete reward-shaping advice per fix type
# ---------------------------------------------------------------------------

FIX_ADVICE = {
    "NO_FIX_NEEDED": (
        "Agent is performing well. No reward changes recommended.",
        None,
    ),
    "ADD_ALTITUDE_PENALTY": (
        "Agent crashes frequently. Add or strengthen the altitude penalty.",
        "if altitude < 200:\n    reward -= (200 - altitude) / 200 * 5.0",
    ),
    "REDUCE_PROXIMITY_RANGE": (
        "Agent orbits too far. Gate proximity reward to <=1000 m.",
        "reward += max(0, (1000 - distance) / 1000) * 3.0  # was wider",
    ),
    "ADD_FIRING_BONUS": (
        "Agent closes but doesn't engage. Add a per-step firing bonus.",
        "if agent_fired:\n    reward += 1.5",
    ),
    "INCREASE_KILL_REWARD": (
        "Agent fights but doesn't finish. Raise the kill terminal reward.",
        "if kill:\n    reward += 150.0  # was 100",
    ),
    "FIX_OPP_ADVANTAGE": (
        "Opponent out-ranges agent. Increase agent engagement_range or reduce opp_engagement_range.",
        "env = EnhancedAerialCombatEnv(engagement_range=700, opp_engagement_range=400)",
    ),
}

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model() -> EpisodeDiagnosticTransformer:
    model_path = project_root / "models" / "edt" / "edt_model.pth"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No EDT model at {model_path}.\n"
            "Run: python tacpm/generate_edt_data.py && python tacpm/train_edt.py"
        )
    model = EpisodeDiagnosticTransformer().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Diagnosis of a single episode
# ---------------------------------------------------------------------------

def diagnose_episode(model, episode: dict) -> dict:
    timesteps = episode.get("timesteps", [])
    if not timesteps:
        return {"error": "no timesteps"}

    x, length = trajectory_to_tensor(timesteps, MAX_LEN)
    x         = x.to(DEVICE)
    pad_mask  = make_pad_mask(length.to(DEVICE), x.size(1))

    with torch.no_grad():
        fail_logits, fix_logits, attn = model(x, pad_mask)

    fail_probs  = torch.softmax(fail_logits, dim=-1)[0].cpu().numpy()
    fix_probs   = torch.softmax(fix_logits,  dim=-1)[0].cpu().numpy()
    attn_seq    = attn[0].cpu().numpy()                  # (MAX_LEN,)

    fail_idx = int(fail_probs.argmax())
    fix_idx  = int(fix_probs.argmax())
    T        = min(len(timesteps), MAX_LEN)

    # Top-3 critical timesteps (highest attention within valid range)
    valid_attn = attn_seq[:T].copy()
    top3_steps = valid_attn.argsort()[::-1][:3].tolist()

    fix_label, fix_code = FIX_ADVICE[FIX_TYPES[fix_idx]]

    result = {
        "episode_id":         episode.get("episode_id", -1),
        "outcome":            episode.get("outcome", "unknown"),
        "kill":               episode.get("kill", False),
        "predicted_failure":  FAILURE_MODES[fail_idx],
        "predicted_fix":      FIX_TYPES[fix_idx],
        "failure_probs":      {m: float(f"{p:.4f}") for m, p in zip(FAILURE_MODES, fail_probs)},
        "fix_probs":          {f: float(f"{p:.4f}") for f, p in zip(FIX_TYPES, fix_probs)},
        "fix_confidence":     float(f"{fix_probs[fix_idx]:.4f}"),
        "critical_timesteps": top3_steps,
        "fix_advice":         fix_label,
        "fix_code_snippet":   fix_code,
        "attention_weights":  attn_seq[:T].tolist(),
    }
    return result


# ---------------------------------------------------------------------------
# Active-learning: append a labelled real episode to edt_active.npz
# ---------------------------------------------------------------------------

def append_active(timesteps: list, failure_idx: int, fix_idx: int) -> None:
    active_path = project_root / "data" / "edt_active.npz"

    arr = np.zeros((MAX_LEN, N_FEATURES), dtype=np.float32)
    for i, t in enumerate(timesteps[:MAX_LEN]):
        dist  = t.get("distance", 0.0)
        alt   = (t.get("agent_pos") or [0, 0, 1000])[2]
        a_h   = t.get("agent_health", 100.0)
        o_h   = t.get("opp_health",   100.0)
        r     = t.get("reward", 0.0)
        step  = t.get("t", i)
        fired = 1.0 if dist <= 600.0 else 0.0
        arr[i] = [dist/15000, a_h/100, o_h/100,
                  float(np.clip(r/30, -1, 1)), alt/2000, fired, step/1000]

    length = min(len(timesteps), MAX_LEN)
    new_X   = arr[np.newaxis]                            # (1, MAX_LEN, N_FEATURES)
    new_f   = np.array([failure_idx], dtype=np.int64)
    new_fx  = np.array([fix_idx],     dtype=np.int64)
    new_l   = np.array([length],      dtype=np.int64)

    if active_path.exists():
        old = np.load(active_path)
        X       = np.concatenate([old["X"],        new_X],  axis=0)
        fails   = np.concatenate([old["failures"],  new_f],  axis=0)
        fixes   = np.concatenate([old["fixes"],     new_fx], axis=0)
        lengths = np.concatenate([old["lengths"],   new_l],  axis=0)
    else:
        X, fails, fixes, lengths = new_X, new_f, new_fx, new_l

    np.savez(active_path, X=X, failures=fails, fixes=fixes, lengths=lengths)
    print(f"  Active-learning buffer: {len(X)} examples -> {active_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_diagnosis(
    replays_path: str = None,
    episode_id: int   = None,
) -> list:
    if replays_path is None:
        replays_path = str(project_root / "data" / "replays.json")

    with open(replays_path, "r") as f:
        replays = json.load(f)

    if episode_id is not None:
        replays = [ep for ep in replays if ep.get("episode_id") == episode_id]
        if not replays:
            print(f"Episode {episode_id} not found in {replays_path}")
            return []

    model   = load_model()
    results = []

    print(f"\n{'='*60}")
    print(f"  EDT DIAGNOSIS — {len(replays)} episode(s)")
    print(f"{'='*60}\n")

    for ep in replays:
        r = diagnose_episode(model, ep)
        results.append(r)

        conf_str = f"({r['fix_confidence']:.0%} confidence)"
        print(f"  Episode {r['episode_id']:02d}  |  outcome={r['outcome']}")
        print(f"  Failure : {r['predicted_failure']}")
        print(f"  Fix     : {r['predicted_fix']}  {conf_str}")
        print(f"  Advice  : {r['fix_advice']}")
        if r["fix_code_snippet"]:
            print(f"  Code    :\n    {r['fix_code_snippet'].replace(chr(10), chr(10)+'    ')}")
        print(f"  Critical timesteps: {r['critical_timesteps']}")
        print()

    # Summary
    from collections import Counter
    fix_counter = Counter(r["predicted_fix"] for r in results)
    print(f"{'='*60}")
    print("  FIX RECOMMENDATION SUMMARY")
    for fix, count in fix_counter.most_common():
        print(f"  {count}x  {fix}")
    print(f"{'='*60}\n")

    # Save
    out_path = Path(replays_path).parent / "edt_diagnosis.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Diagnosis saved -> {out_path}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--replays",     default=None, help="Path to replays.json")
    parser.add_argument("--episode",     type=int,     default=None,
                        help="Diagnose a single episode by id")
    parser.add_argument("--list-labels", action="store_true",
                        help="Print failure/fix label tables and exit")
    parser.add_argument("--add-active",  action="store_true",
                        help="Append this episode to active-learning buffer")
    parser.add_argument("--label",       type=int,     default=None,
                        help="Failure mode label index (for --add-active)")
    parser.add_argument("--fix",         type=int,     default=None,
                        help="Fix type label index (for --add-active)")
    args = parser.parse_args()

    if args.list_labels:
        print("\nFailure modes:")
        for i, m in enumerate(FAILURE_MODES):
            print(f"  {i}  {m}")
        print("\nFix types:")
        for i, f in enumerate(FIX_TYPES):
            print(f"  {i}  {f}")
        sys.exit(0)

    results = run_diagnosis(args.replays, args.episode)

    if args.add_active and results and args.label is not None and args.fix is not None:
        replays_path = args.replays or str(project_root / "data" / "replays.json")
        with open(replays_path) as f:
            replays = json.load(f)
        ep_id  = results[0]["episode_id"]
        ep     = next((e for e in replays if e["episode_id"] == ep_id), None)
        if ep:
            append_active(ep["timesteps"], args.label, args.fix)
