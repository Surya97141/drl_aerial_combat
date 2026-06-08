"""
TacPM — Episode Analyzer
Reads replay JSON and extracts tactical failure patterns.
No LLM yet — pure data analysis first.
"""

import json
import numpy as np
import os


def load_replays(path: str) -> list:
    with open(path, "r") as f:
        return json.load(f)


def analyze_episode(ep: dict) -> dict:
    """Extract tactical insights from one episode."""
    timesteps = ep["timesteps"]
    rewards   = [t["reward"] for t in timesteps]
    distances = [t["distance"] for t in timesteps]

    # Find worst moment (lowest reward timestep)
    worst_t   = int(np.argmin(rewards))
    worst_ts  = timesteps[worst_t]

    # Find closest approach
    closest_t = int(np.argmin(distances))
    min_dist  = round(distances[closest_t], 1)
    max_dist  = round(max(distances), 1)

    # Reward trend — is agent improving or degrading?
    first_half = np.mean(rewards[:len(rewards)//2])
    second_half= np.mean(rewards[len(rewards)//2:])
    trend      = "improving" if second_half > first_half else "degrading"

    # Detect reward collapse — 20 consecutive steps below -0.5
    collapse_start = None
    streak = 0
    for i, r in enumerate(rewards):
        if r < -0.5:
            streak += 1
            if streak >= 20 and collapse_start is None:
                collapse_start = i - 19
        else:
            streak = 0

    # Classify failure mode
    if ep["outcome"] == "timeout" and trend == "degrading":
        failure_mode = "GRADUAL_DRIFT — agent loses ground over time"
    elif ep["outcome"] == "timeout" and trend == "improving":
        failure_mode = "STALEMATE — agent holds but never closes for kill"
    elif ep["outcome"] == "terminated" and worst_ts["distance"] < 100:
        failure_mode = "CLOSE_RANGE_LOSS — agent lost at close quarters"
    elif ep["outcome"] == "terminated" and worst_ts["distance"] > 5000:
        failure_mode = "DISENGAGEMENT — agent flew too far from opponent"
    else:
        failure_mode = "UNKNOWN"

    return {
        "episode_id":     ep["episode_id"],
        "outcome":        ep["outcome"],
        "length":         ep["length"],
        "total_reward":   ep["total_reward"],
        "reward_trend":   trend,
        "failure_mode":   failure_mode,
        "worst_timestep": worst_t,
        "worst_reward":   round(rewards[worst_t], 4),
        "min_distance":   min_dist,
        "max_distance":   max_dist,
        "collapse_start": collapse_start,
        "avg_reward":     round(np.mean(rewards), 4),
    }


def analyze_all(replays_path: str) -> list:
    replays = load_replays(replays_path)
    results = []

    print(f"\n{'='*60}")
    print(f"  TacPM EPISODE ANALYSIS — {len(replays)} episodes")
    print(f"{'='*60}\n")

    for ep in replays:
        analysis = analyze_episode(ep)
        results.append(analysis)

        print(f"Episode {analysis['episode_id']:02d} | {analysis['failure_mode']}")
        print(f"  Outcome: {analysis['outcome']} | Length: {analysis['length']}")
        print(f"  Reward: avg={analysis['avg_reward']} worst={analysis['worst_reward']} trend={analysis['reward_trend']}")
        print(f"  Distance: min={analysis['min_distance']}m max={analysis['max_distance']}m")
        if analysis["collapse_start"]:
            print(f"  ⚠ Reward collapse detected at step {analysis['collapse_start']}")
        print()

    # Summary
    from collections import Counter
    modes = Counter(a["failure_mode"] for a in results)
    print(f"{'='*60}")
    print("  FAILURE MODE SUMMARY")
    print(f"{'='*60}")
    for mode, count in modes.most_common():
        print(f"  {count}x  {mode}")
    print()

    # Save analysis
    out_path = replays_path.replace("replays.json", "analysis.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Analysis saved to {out_path}")

    return results


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base, "data", "replays.json")
    analyze_all(path)
    