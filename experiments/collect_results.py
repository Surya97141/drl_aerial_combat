"""
Results Aggregator -- produces paper-ready tables from robustness_grid.json.

Outputs:
  results/table1_kill_rates.csv   -- the main 4x4 kill-rate grid (Table 1)
  results/table2_outcomes.csv     -- outcome breakdown per cell
  results/robustness_gap.json     -- degradation deltas for inline text
  (prints LaTeX snippet for Table 1)

Usage:
  python experiments/collect_results.py
"""

import json
import csv
import numpy as np
from pathlib import Path
from collections import defaultdict

project_root = Path(__file__).parent.parent
results_dir  = project_root / "results"

VARIANTS   = ["baseline", "latency", "noise", "degraded"]
TEST_CONDS = ["clean", "latency", "noise", "degraded"]

VARIANT_LABELS = {
    "baseline": "Baseline (clean)",
    "latency":  "Latency-aware",
    "noise":    "Noise-aware",
    "degraded": "Degraded (both)",
}
TEST_LABELS = {
    "clean":    "Clean",
    "latency":  "Latency",
    "noise":    "Noise",
    "degraded": "Both",
}


def load_grid() -> dict:
    path = results_dir / "robustness_grid.json"
    if not path.exists():
        raise FileNotFoundError(f"No results found at {path}.\nRun robustness_sweep.py first.")
    with open(path) as f:
        return json.load(f)


def aggregate(grid: dict) -> dict:
    """
    Returns: agg[variant][test_cond] = {
        "kill_rates": [...],   # one per seed
        "mean_kill":  float,
        "std_kill":   float,
        "mean_reward": float,
        "mean_damage": float,
        "outcomes":   {outcome: mean_count},
    }
    """
    agg = defaultdict(lambda: defaultdict(lambda: {
        "kill_rates": [], "rewards": [], "damages": [], "outcomes": defaultdict(list)
    }))

    for run_key, run in grid.items():
        v = run["variant"]
        for cell_key, cell in run["eval"].items():
            t = cell["test_condition"]
            agg[v][t]["kill_rates"].append(cell["kill_rate"])
            agg[v][t]["rewards"].append(cell["avg_reward"])
            agg[v][t]["damages"].append(cell["avg_damage"])
            for outcome, cnt in cell.get("outcome_counts", {}).items():
                agg[v][t]["outcomes"][outcome].append(cnt)

    # Finalise
    result = {}
    for v in agg:
        result[v] = {}
        for t in agg[v]:
            d = agg[v][t]
            result[v][t] = {
                "kill_rates":   d["kill_rates"],
                "mean_kill":    float(np.mean(d["kill_rates"])),
                "std_kill":     float(np.std(d["kill_rates"])),
                "mean_reward":  float(np.mean(d["rewards"])),
                "mean_damage":  float(np.mean(d["damages"])),
                "n_seeds":      len(d["kill_rates"]),
                "outcomes":     {o: float(np.mean(v2)) for o, v2 in d["outcomes"].items()},
            }
    return result


def write_table1(agg: dict):
    """Kill-rate grid as CSV."""
    path = results_dir / "table1_kill_rates.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["Train \\ Test"] + [TEST_LABELS[t] for t in TEST_CONDS]
        writer.writerow(header)
        for v in VARIANTS:
            row = [VARIANT_LABELS.get(v, v)]
            for t in TEST_CONDS:
                if v in agg and t in agg[v]:
                    cell = agg[v][t]
                    row.append(f"{cell['mean_kill']:.1%} +/- {cell['std_kill']:.1%}")
                else:
                    row.append("---")
            writer.writerow(row)
    print(f"Table 1 (kill rates) -> {path}")


def write_table2(agg: dict):
    """Outcome breakdown CSV."""
    path = results_dir / "table2_outcomes.csv"
    all_outcomes = sorted({o for v in agg for t in agg[v] for o in agg[v][t]["outcomes"]})
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Variant", "Test"] + all_outcomes)
        for v in VARIANTS:
            for t in TEST_CONDS:
                if v not in agg or t not in agg[v]:
                    continue
                row = [VARIANT_LABELS.get(v, v), TEST_LABELS.get(t, t)]
                for o in all_outcomes:
                    row.append(f"{agg[v][t]['outcomes'].get(o, 0):.1f}")
                writer.writerow(row)
    print(f"Table 2 (outcomes)   -> {path}")


def compute_robustness_gaps(agg: dict) -> dict:
    """
    Key metrics for the paper's inline text:
      - Robustness gap:   baseline clean - baseline degraded
      - Training benefit: degraded_clean - baseline_degraded
      - Cross-transfer:   latency_aware tested on noise vs. noise_aware tested on noise
    """
    gaps = {}

    def get(v, t):
        return agg.get(v, {}).get(t, {}).get("mean_kill", None)

    baseline_clean    = get("baseline", "clean")
    baseline_degraded = get("baseline", "degraded")
    degraded_clean    = get("degraded", "clean")
    degraded_degraded = get("degraded", "degraded")
    latency_on_noise  = get("latency",  "noise")
    noise_on_latency  = get("noise",    "latency")

    if baseline_clean is not None and baseline_degraded is not None:
        gaps["robustness_gap"] = {
            "desc": "Kill-rate drop: baseline (clean) -> baseline (degraded)",
            "value": baseline_clean - baseline_degraded,
            "baseline_clean": baseline_clean,
            "baseline_degraded": baseline_degraded,
        }

    if degraded_clean is not None and baseline_clean is not None:
        gaps["robustness_overhead"] = {
            "desc": "Cost of robustness training (clean test): degraded vs. baseline",
            "value": baseline_clean - degraded_clean,
        }

    if degraded_degraded is not None and baseline_degraded is not None:
        gaps["robustness_benefit"] = {
            "desc": "Benefit of robustness training (degraded test): degraded vs. baseline",
            "value": degraded_degraded - baseline_degraded,
        }

    if latency_on_noise is not None and noise_on_latency is not None:
        gaps["cross_transfer"] = {
            "desc": "Cross-transfer: latency-trained vs. noise-trained on each other's condition",
            "latency_trained_on_noise": latency_on_noise,
            "noise_trained_on_latency": noise_on_latency,
        }

    path = results_dir / "robustness_gap.json"
    with open(path, "w") as f:
        json.dump(gaps, f, indent=2)
    print(f"Robustness gaps      -> {path}")
    return gaps


def print_latex_table(agg: dict):
    """Print a LaTeX table snippet for Table 1."""
    print("\n--- LaTeX Table 1 (copy into paper) ---")
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\caption{Kill rate (\%) across training and test conditions (mean $\pm$ std, 3 seeds, 20 eval episodes each).}")
    print(r"\label{tab:robustness}")
    print(r"\begin{tabular}{l|cccc}")
    print(r"\hline")
    header = "Train $\\backslash$ Test & " + " & ".join(TEST_LABELS[t] for t in TEST_CONDS) + r" \\"
    print(header)
    print(r"\hline")
    for v in VARIANTS:
        cells = []
        for t in TEST_CONDS:
            if v in agg and t in agg[v]:
                c = agg[v][t]
                pct  = c["mean_kill"] * 100
                std  = c["std_kill"]  * 100
                # Bold the diagonal (same train/test condition)
                val = f"{pct:.0f}$\\pm${std:.0f}"
                if v.replace("_aware","").replace("_","") == t.replace("_",""):
                    val = r"\textbf{" + val + "}"
                cells.append(val)
            else:
                cells.append("---")
        label = VARIANT_LABELS.get(v, v)
        print(f"{label} & " + " & ".join(cells) + r" \\")
    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\end{table}")
    print("--- end LaTeX ---\n")


def print_inline_numbers(gaps: dict):
    """Print the key numbers for inline paper text."""
    print("\n--- Key numbers for paper inline text ---")
    for key, g in gaps.items():
        print(f"\n{key}:")
        print(f"  {g['desc']}")
        if "value" in g:
            direction = "improvement" if g["value"] > 0 else "drop"
            print(f"  Delta: {abs(g['value']):.1%} {direction}")
        for k, v in g.items():
            if k not in ("desc", "value"):
                print(f"  {k}: {v:.1%}" if isinstance(v, float) else f"  {k}: {v}")
    print("--- end ---\n")


def main():
    results_dir.mkdir(exist_ok=True)
    grid = load_grid()
    agg  = aggregate(grid)

    write_table1(agg)
    write_table2(agg)
    gaps = compute_robustness_gaps(agg)

    print_latex_table(agg)
    print_inline_numbers(gaps)


if __name__ == "__main__":
    main()
