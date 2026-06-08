"""
TacPM — LLM Analyst (Gemini)
Takes episode analysis JSON → sends to Gemini → gets tactical diagnosis.
"""

import json
import os
import google.generativeai as genai


def build_prompt(analyses: list) -> str:
    lines = []
    for a in analyses:
        lines.append(
            f"Episode {a['episode_id']}: outcome={a['outcome']}, "
            f"failure={a['failure_mode']}, avg_reward={a['avg_reward']:.3f}, "
            f"worst_reward={a['worst_reward']:.3f}, trend={a['reward_trend']}, "
            f"min_dist={a['min_distance']}m, "
            f"dmg_dealt={a.get('damage_dealt', 'N/A')}, "
            f"dmg_received={a.get('damage_received', 'N/A')}, "
            f"time_to_engagement={a.get('time_to_first_engagement', 'N/A')}"
        )

    episodes_text = "\n".join(lines)

    return f"""You are a deep reinforcement learning expert analyzing a DRL aerial combat agent.

ENVIRONMENT:
- Observation space (14-dim): agent_pos(3) + agent_vel(3) + opp_pos(3) + opp_vel(3) + agent_health(1) + opp_health(1)
- Action space: [throttle, pitch, roll, yaw] — continuous -1 to 1
- Combat: agent fires when within 500m AND pointing within 30° of opponent → 5 HP/step damage
- Opponent fires when within 600m (always faces agent) → 3 HP/step damage
- Win condition: opponent health ≤ 0. Loss: agent health ≤ 0 or ground collision or fled > 15km.
- Reward: survival(0.1/step) + proximity(max 3.0) + heading(max 1.5) + closing_vel(max 1.0) + health_advantage(±2.0) + firing_bonus(1.0/step) + terminal(kill=+100, death=-50)

EPISODE RESULTS:
{episodes_text}

FAILURE MODE DEFINITIONS:
- STALEMATE: agent times out without closing for a kill (improving trend but no kill)
- GRADUAL_DRIFT: agent times out while degrading (getting further from opponent)
- CLOSE_RANGE_LOSS: agent died at close range (< 100m)
- DISENGAGEMENT: agent flew too far away (> 5000m at worst moment)

Please provide:
1. ROOT CAUSE — What is the fundamental tactical reason the agent is failing?
2. REWARD PROBLEM — What specific flaw in the reward function is causing this behavior?
3. OBSERVATION PROBLEM — What critical information is missing or misleading in the observations?
4. TOP 3 FIXES — Specific, actionable code-level changes ranked by expected impact.
5. NEXT EXPERIMENT — What single experiment should be run next to confirm your diagnosis?

Be specific and technical. Reference the actual numbers from the data."""


def run_llm_analysis(analysis_path: str, model_name: str = "gemini-1.5-flash") -> dict:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY environment variable not set.")

    genai.configure(api_key=api_key)
    llm = genai.GenerativeModel(model_name)

    with open(analysis_path, "r") as f:
        analyses = json.load(f)

    prompt = build_prompt(analyses)

    print(f"Sending {len(analyses)} episodes to Gemini for tactical diagnosis...\n")
    response = llm.generate_content(prompt)
    diagnosis = response.text

    # Aggregate stats
    outcomes = [a["outcome"] for a in analyses]
    kill_count = outcomes.count("kill")
    failure_modes = list(set(a["failure_mode"] for a in analyses))
    avg_damage = sum(a.get("damage_dealt", 0) for a in analyses) / len(analyses)

    report = {
        "episodes_analyzed": len(analyses),
        "kill_rate": kill_count / len(analyses),
        "failure_mode_summary": failure_modes,
        "avg_damage_dealt": avg_damage,
        "outcome_counts": {o: outcomes.count(o) for o in set(outcomes)},
        "diagnosis": diagnosis,
    }

    out_path = analysis_path.replace("analysis.json", "llm_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print("=" * 60)
    print("  TacPM TACTICAL DIAGNOSIS")
    print("=" * 60)
    print(diagnosis)
    print(f"\nKill rate: {kill_count}/{len(analyses)} ({report['kill_rate']:.0%})")
    print(f"Avg damage dealt: {avg_damage:.1f}")
    print(f"Full report saved → {out_path}")

    return report


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base, "data", "analysis.json")
    run_llm_analysis(path)
