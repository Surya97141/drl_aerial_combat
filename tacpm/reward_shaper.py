"""
TacPM -- Reward Shaper
Automated reward function improvement loop:
  analysis.json  -->  EDT or Gemini  -->  suggested Python reward diff  -->  saved patch

Two backends (pass use_edt=True/False or --edt / --gemini CLI flag):
  EDT    (default) : local Episode Diagnostic Transformer, zero API cost, <1 s
  Gemini           : Google Gemini 1.5 Flash, requires GEMINI_API_KEY env var
"""

import json
import os
import re
import sys
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _load_reward_source() -> str:
    """Extract the _calculate_reward method from enhanced_env.py as context."""
    env_path = project_root / "environment" / "enhanced_env.py"
    source = env_path.read_text(encoding="utf-8")
    # Pull out just the reward method so the prompt stays focused
    match = re.search(
        r"(    def _calculate_reward\(.*?)(\n    def )",
        source,
        re.DOTALL,
    )
    return match.group(1).rstrip() if match else source


def build_shaping_prompt(analyses: list, reward_source: str) -> str:
    # Summarise failure modes
    from collections import Counter
    modes    = Counter(a["failure_mode"] for a in analyses)
    outcomes = Counter(a["outcome"] for a in analyses)
    kills    = sum(1 for a in analyses if a.get("kill", False))
    avg_dmg  = sum(a.get("damage_dealt", 0) for a in analyses) / len(analyses)
    avg_dist = sum(a.get("min_distance", 0) for a in analyses) / len(analyses)

    mode_lines = "\n".join(f"  {c}x  {m}" for m, c in modes.most_common())
    outcome_lines = "\n".join(f"  {c}x  {o}" for o, c in outcomes.most_common())

    episode_lines = []
    for a in analyses:
        episode_lines.append(
            f"  ep{a['episode_id']:02d}: outcome={a['outcome']:<8} "
            f"failure={a['failure_mode'][:30]:<30}  "
            f"avg_reward={a['avg_reward']:6.2f}  "
            f"min_dist={a['min_distance']}m  "
            f"dmg_dealt={a.get('damage_dealt',0):.0f}"
        )

    return f"""You are an expert in deep reinforcement learning reward engineering for aerial combat agents.

TASK
----
Analyse the episode failure data below and produce a SPECIFIC, MINIMAL Python code change
to the reward function that will fix the dominant failure mode. Output only what is needed.

ENVIRONMENT CONTEXT
-------------------
- Agent fires when within 600m AND facing within 30 degrees of opponent (5 HP/step).
- Opponent fires when within 500m (always faces agent, 3 HP/step).
- Kill = opponent health <= 0.  Death = agent health <= 0.  Episode cap = 1000 steps.
- Starting altitude: ~1000m.  Ground = altitude 0 (terminates episode).
- Observation: [agent_pos(3), agent_vel(3), opp_pos(3), opp_vel(3), agent_health(1), opp_health(1)]

EPISODE SUMMARY  ({len(analyses)} episodes)
---------------
Kill rate  : {kills}/{len(analyses)} ({kills/len(analyses):.0%})
Avg damage : {avg_dmg:.1f} HP dealt per episode
Avg min dist: {avg_dist:.0f} m

Outcome breakdown:
{outcome_lines}

Failure mode breakdown:
{mode_lines}

Per-episode detail:
{chr(10).join(episode_lines)}

CURRENT _calculate_reward METHOD
---------------------------------
```python
{reward_source}
```

INSTRUCTIONS
------------
1. Identify the SINGLE most impactful reward problem causing the dominant failure mode.
2. Output your diagnosis in 2-3 sentences.
3. Output the MINIMAL code change needed — a unified diff or a clearly labelled
   "ADD AFTER LINE X" / "REPLACE LINES X-Y" block.
4. Explain in one sentence WHY this change will fix the failure mode.
5. Predict the expected change in kill rate after retraining.

Format your response EXACTLY as:

DIAGNOSIS:
<your 2-3 sentence diagnosis>

CODE CHANGE:
```python
<the exact Python lines to add or change, with a comment marking where they go>
```

RATIONALE:
<one sentence>

PREDICTED KILL RATE CHANGE:
<e.g. "0% -> 40-60%">
"""


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def call_gemini(prompt: str, model_name: str = "gemini-1.5-flash") -> str:
    import google.generativeai as genai
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY not set.")
    genai.configure(api_key=api_key)
    response = genai.GenerativeModel(model_name).generate_content(prompt)
    return response.text


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def parse_response(text: str) -> dict:
    """Extract structured fields from the LLM response."""

    def _between(label_start: str, label_end: str) -> str:
        pattern = rf"{re.escape(label_start)}\s*(.*?)\s*{re.escape(label_end)}"
        m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else ""

    diagnosis = _between("DIAGNOSIS:", "CODE CHANGE:")
    rationale = _between("RATIONALE:", "PREDICTED KILL RATE CHANGE:")
    prediction = text.split("PREDICTED KILL RATE CHANGE:")[-1].strip() if "PREDICTED KILL RATE CHANGE:" in text else ""

    # Extract code block
    code_match = re.search(r"CODE CHANGE:.*?```python\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    code = code_match.group(1).strip() if code_match else ""

    return {
        "diagnosis":   diagnosis,
        "code":        code,
        "rationale":   rationale,
        "prediction":  prediction,
        "raw":         text,
    }


# ---------------------------------------------------------------------------
# EDT backend — local, zero-cost
# ---------------------------------------------------------------------------

def run_edt_shaping(analyses: list) -> dict:
    """Run EDT over replays that match the analysis, return parsed patch dict."""
    sys.path.insert(0, str(project_root / "tacpm"))
    from edt_diagnose import load_model, diagnose_episode, FIX_ADVICE
    from edt_model import FIX_TYPES

    replays_path = project_root / "data" / "replays.json"
    if not replays_path.exists():
        raise FileNotFoundError(
            f"replays.json not found at {replays_path}. "
            "Run tacpm/replay_generator.py first."
        )
    with open(replays_path) as f:
        replays = json.load(f)

    model   = load_model()
    ep_diags = []
    for ep in replays:
        d = diagnose_episode(model, ep)
        ep_diags.append(d)

    # Aggregate: most-voted fix
    from collections import Counter
    fix_votes = Counter(d["predicted_fix"] for d in ep_diags)
    top_fix   = fix_votes.most_common(1)[0][0]
    top_fail  = Counter(d["predicted_failure"] for d in ep_diags).most_common(1)[0][0]
    advice, code = FIX_ADVICE[top_fix]

    print("=" * 60)
    print("  TacPM REWARD SHAPING (EDT backend)")
    print("=" * 60)
    print(f"\nDOMINANT FAILURE : {top_fail}")
    print(f"RECOMMENDED FIX  : {top_fix}")
    print(f"\nDIAGNOSIS:\n  {advice}")
    if code:
        print(f"\nCODE SNIPPET:\n  {code}")
    print(f"\nFix vote breakdown:")
    for fix, cnt in fix_votes.most_common():
        print(f"  {cnt}x  {fix}")

    return {
        "backend":          "EDT",
        "dominant_failure": top_fail,
        "diagnosis":        advice,
        "code":             code or "",
        "rationale":        f"EDT voted {top_fix} ({fix_votes[top_fix]}/{len(ep_diags)} episodes)",
        "prediction":       "Retrain to observe impact",
        "edt_votes":        dict(fix_votes),
        "raw":              json.dumps(ep_diags, indent=2),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_reward_shaping(
    analysis_path: str = None,
    dry_run:       bool = False,
    use_edt:       bool = True,
) -> dict:
    """
    Full pipeline: read analysis -> EDT or Gemini diagnosis -> save patch.

    Args:
        analysis_path : Path to analysis.json. Defaults to data/analysis.json.
        dry_run       : If True, skip the backend call (Gemini path only).
        use_edt       : True = EDT backend (default), False = Gemini backend.
    """
    if analysis_path is None:
        analysis_path = str(project_root / "data" / "analysis.json")

    with open(analysis_path, "r") as f:
        analyses = json.load(f)

    # --- EDT path ---
    if use_edt:
        parsed = run_edt_shaping(analyses)

    # --- Gemini path ---
    else:
        reward_source = _load_reward_source()
        prompt        = build_shaping_prompt(analyses, reward_source)

        if dry_run:
            print("=== DRY RUN: prompt only ===")
            print(prompt)
            return {"prompt": prompt}

        print("Sending failure analysis to Gemini for reward shaping suggestions...\n")
        raw_response = call_gemini(prompt)
        parsed = parse_response(raw_response)
        parsed["backend"] = "Gemini"

        print("=" * 60)
        print("  TacPM REWARD SHAPING SUGGESTION (Gemini backend)")
        print("=" * 60)
        print(f"\nDIAGNOSIS:\n{parsed['diagnosis']}")
        print(f"\nCODE CHANGE:\n{parsed['code']}")
        print(f"\nRATIONALE:\n{parsed['rationale']}")
        print(f"\nPREDICTED KILL RATE CHANGE:\n{parsed['prediction']}")

    # Save patch
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir    = Path(analysis_path).parent
    patch_path = out_dir / f"reward_patch_{timestamp}.json"

    result = {
        "timestamp":        timestamp,
        "analysis_path":    analysis_path,
        "kill_rate_before": sum(1 for a in analyses if a.get("kill", False)) / len(analyses),
        **parsed,
    }

    with open(patch_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nPatch saved -> {patch_path}")
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    dry    = "--dry-run" in sys.argv
    gemini = "--gemini"  in sys.argv   # default is EDT
    run_reward_shaping(dry_run=dry, use_edt=not gemini)
