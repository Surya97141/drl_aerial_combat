"""
TacPM — Episode Analyzer
Reads replay JSON and extracts tactical failure (and success) patterns.
"""

import json
import numpy as np
import os
from collections import Counter


def analyze_episode(ep: dict) -> dict:
    timesteps = ep["timesteps"]
    rewards   = [t["reward"] for t in timesteps]
    distances = [t["distance"] for t in timesteps]

    worst_t  = int(np.argmin(rewards))
    min_dist = round(min(distances), 1)
    max_dist = round(max(distances), 1)

    first_half  = np.mean(rewards[:max(len(rewards) // 2, 1)])
    second_half = np.mean(rewards[max(len(rewards) // 2, 1):]) if len(rewards) > 1 else first_half
    trend = "improving" if second_half > first_half else "degrading"

    # Reward collapse detection
    collapse_start = None
    streak = 0
    for i, r in enumerate(rewards):
        if r < -0.5:
            streak += 1
            if streak >= 20 and collapse_start is None:
                collapse_start = i - 19
        else:
            streak = 0

    outcome = ep.get("outcome", "unknown")
    kill = ep.get("kill", False)

    # Classify outcome
    if kill:
        failure_mode = "KILL — agent successfully eliminated opponent"
    elif outcome == "timeout" and trend == "degrading":
        failure_mode = "GRADUAL_DRIFT — agent loses ground over time"
    elif outcome == "timeout":
        failure_mode = "STALEMATE — agent survives but never closes for kill"
    elif outcome == "died":
        if min_dist < 100:
            failure_mode = "CLOSE_RANGE_LOSS — agent lost at close quarters"
        else:
            failure_mode = "ATTRITION_LOSS — agent bled out under sustained fire"
    elif outcome == "crashed":
        failure_mode = "GROUND_COLLISION — agent flew into ground"
    elif outcome == "fled":
        failure_mode = "DISENGAGEMENT — agent flew too far from opponent"
    else:
        failure_mode = "UNKNOWN"

    return {
        "episode_id":               ep["episode_id"],
        "outcome":                  outcome,
        "kill":                     kill,
        "length":                   ep["length"],
        "total_reward":             ep["total_reward"],
        "reward_trend":             trend,
        "failure_mode":             failure_mode,
        "worst_timestep":           worst_t,
        "worst_reward":             round(rewards[worst_t], 4),
        "min_distance":             min_dist,
        "max_distance":             max_dist,
        "collapse_start":           collapse_start,
        "avg_reward":               round(float(np.mean(rewards)), 4),
        "damage_dealt":             ep.get("damage_dealt", 0.0),
        "damage_received":          ep.get("damage_received", 0.0),
        "net_damage":               ep.get("net_damage", 0.0),
        "time_to_first_engagement": ep.get("time_to_first_engagement"),
    }


def analyze_all(replays_path: str, save: bool = True) -> list:
    with open(replays_path, "r") as f:
        replays = json.load(f)

    print(f"\n{'='*60}")
    print(f"  TacPM EPISODE ANALYSIS — {len(replays)} episodes")
    print(f"{'='*60}\n")

    results = []
    for ep in replays:
        a = analyze_episode(ep)
        results.append(a)

        print(f"Episode {a['episode_id']:02d} | {a['failure_mode']}")
        print(f"  outcome={a['outcome']:<8}  length={a['length']}")
        print(f"  reward: avg={a['avg_reward']}  worst={a['worst_reward']}  trend={a['reward_trend']}")
        print(f"  distance: min={a['min_distance']}m  max={a['max_distance']}m")
        print(f"  damage: dealt={a['damage_dealt']:.0f}  received={a['damage_received']:.0f}  net={a['net_damage']:.0f}")
        if a["time_to_first_engagement"]:
            print(f"  first engagement: step {a['time_to_first_engagement']}")
        if a["collapse_start"]:
            print(f"  WARNING reward collapse at step {a['collapse_start']}")
        print()

    # Summary
    modes    = Counter(a["failure_mode"] for a in results)
    outcomes = Counter(a["outcome"] for a in results)
    kills    = sum(1 for a in results if a["kill"])
    kill_rate = kills / len(results)
    avg_dmg  = np.mean([a["damage_dealt"] for a in results])

    print(f"{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"  kill_rate  : {kill_rate:.0%}  ({kills}/{len(results)})")
    print(f"  avg_damage : {avg_dmg:.1f} HP dealt per episode")
    print(f"\n  Outcomes:")
    for o, c in outcomes.most_common():
        print(f"    {c}x  {o}")
    print(f"\n  Failure modes:")
    for m, c in modes.most_common():
        print(f"    {c}x  {m}")
    print()

    if save:
        out_path = replays_path.replace("replays.json", "analysis.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Analysis saved -> {out_path}")

    return results


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base, "data", "replays.json")
    analyze_all(path)
