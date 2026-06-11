"""
DRL Aerial Combat Research — Interactive Dashboard
Run: streamlit run dashboard/app.py
"""

import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from collections import defaultdict, Counter

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent.parent

st.set_page_config(
    page_title="Aerial Combat Research",
    page_icon="✈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Data loaders (cached so hot reloads are fast)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=30)
def load_grid():
    p = ROOT / "results" / "robustness_grid.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())

@st.cache_data(ttl=30)
def load_replays():
    p = ROOT / "data" / "replays.json"
    if not p.exists():
        return []
    return json.loads(p.read_text())

@st.cache_data(ttl=30)
def load_diagnosis():
    p = ROOT / "data" / "edt_diagnosis.json"
    if not p.exists():
        return []
    return json.loads(p.read_text())

@st.cache_data(ttl=30)
def load_edt_history():
    p = ROOT / "models" / "edt" / "edt_history.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())

@st.cache_data(ttl=30)
def load_selfplay_log():
    p = ROOT / "models" / "selfplay" / "selfplay_log.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())

@st.cache_data(ttl=30)
def load_auto_reward_log():
    p = ROOT / "models" / "auto_reward" / "loop_log.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())

@st.cache_data(ttl=30)
def load_analysis():
    p = ROOT / "data" / "analysis.json"
    if not p.exists():
        return []
    return json.loads(p.read_text())

# ---------------------------------------------------------------------------
# Helper: aggregate grid
# ---------------------------------------------------------------------------

VARIANTS   = ["baseline", "latency", "noise", "degraded"]
TEST_CONDS = ["clean", "latency", "noise", "degraded"]

VARIANT_LABELS = {
    "baseline": "Baseline",
    "latency":  "Latency-aware",
    "noise":    "Noise-aware",
    "degraded": "Degraded",
}

OUTCOME_COLORS = {
    "kill":    "#2ecc71",
    "died":    "#e74c3c",
    "crashed": "#e67e22",
    "timeout": "#3498db",
    "fled":    "#9b59b6",
    "unknown": "#95a5a6",
}

def aggregate_grid(grid: dict):
    acc = defaultdict(lambda: defaultdict(list))
    for run_key, run in grid.items():
        v = run["variant"]
        for ck, cell in run["eval"].items():
            t = cell["test_condition"]
            acc[v][t].append(cell["kill_rate"])
    result = {}
    for v in acc:
        result[v] = {}
        for t in acc[v]:
            vals = acc[v][t]
            result[v][t] = {
                "mean": float(np.mean(vals)),
                "std":  float(np.std(vals)),
                "n":    len(vals),
            }
    return result


# ---------------------------------------------------------------------------
# Tab 1 — Overview
# ---------------------------------------------------------------------------

def tab_overview(grid, replays, diagnosis):
    st.header("Project Overview")

    # KPI row
    c1, c2, c3, c4 = st.columns(4)

    best_kill = 0.0
    total_cells = 0
    if grid:
        for run in grid.values():
            for cell in run["eval"].values():
                best_kill = max(best_kill, cell["kill_rate"])
                total_cells += 1

    with c1:
        st.metric("Best Kill Rate", f"{best_kill:.0%}")
    with c2:
        st.metric("Sweep Cells Done", f"{total_cells} / 48")
    with c3:
        kills = sum(1 for ep in replays if ep.get("kill"))
        st.metric("Episodes Analysed", f"{len(replays)} ({kills} kills)")
    with c4:
        st.metric("EDT Model", "Trained  ✓" if (ROOT / "models" / "edt" / "edt_model.pth").exists() else "Not trained")

    st.divider()

    # Pipeline diagram
    st.subheader("TacPM Pipeline")
    stages = ["PPO Training", "Replay Generator", "Episode Analyser", "EDT Diagnosis", "Reward Shaper", "Retrain"]
    fig = go.Figure()
    for i, s in enumerate(stages):
        fig.add_trace(go.Scatter(
            x=[i], y=[0], mode="markers+text",
            marker=dict(size=40, color="#3498db", line=dict(width=2, color="white")),
            text=[s], textposition="bottom center",
            textfont=dict(size=11),
            name=s, showlegend=False,
        ))
        if i < len(stages) - 1:
            fig.add_annotation(
                x=i + 0.5, y=0, ax=i + 0.1, ay=0,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=2, arrowsize=1.5,
                arrowwidth=2, arrowcolor="#2ecc71",
            )
    fig.update_layout(
        height=160, margin=dict(l=20, r=20, t=10, b=60),
        xaxis=dict(visible=False, range=[-0.5, len(stages) - 0.5]),
        yaxis=dict(visible=False, range=[-0.5, 0.5]),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Outcome summary from replays
    if replays:
        st.subheader("Episode Outcome Distribution")
        outcomes = Counter(ep["outcome"] for ep in replays)
        fig2 = px.pie(
            names=list(outcomes.keys()),
            values=list(outcomes.values()),
            color=list(outcomes.keys()),
            color_discrete_map=OUTCOME_COLORS,
            hole=0.45,
        )
        fig2.update_traces(textposition="inside", textinfo="percent+label")
        fig2.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=20, b=20),
            showlegend=False,
        )
        col1, col2 = st.columns([1, 2])
        with col1:
            st.plotly_chart(fig2, use_container_width=True)
        with col2:
            df = pd.DataFrame([
                {
                    "Episode": ep["episode_id"],
                    "Outcome": ep["outcome"],
                    "Kill": "✓" if ep["kill"] else "",
                    "Length": ep["length"],
                    "Reward": f"{ep['total_reward']:.1f}",
                    "Dmg Dealt": f"{ep.get('damage_dealt', 0):.0f}",
                    "Min Dist": f"{ep.get('min_distance', -1):.0f}m",
                }
                for ep in replays
            ])
            st.dataframe(df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Tab 2 — Robustness Grid
# ---------------------------------------------------------------------------

def tab_robustness(grid):
    st.header("Robustness Grid — Kill Rate Heatmap")

    if not grid:
        st.warning("No sweep results yet. Run `experiments/robustness_sweep.py` first.")
        return

    agg    = aggregate_grid(grid)
    metric = st.radio("Metric", ["Kill Rate", "Std Dev", "Samples"],
                      horizontal=True)

    # Build matrix
    z, text_arr, hover = [], [], []
    for v in VARIANTS:
        row_z, row_t, row_h = [], [], []
        for t in TEST_CONDS:
            cell = agg.get(v, {}).get(t)
            if cell:
                val  = cell["mean"] if metric == "Kill Rate" else (cell["std"] if metric == "Std Dev" else cell["n"])
                disp = f"{cell['mean']:.0%}<br>±{cell['std']:.0%}" if metric == "Kill Rate" else f"{val:.2f}"
                row_z.append(val)
                row_t.append(f"{cell['mean']:.0%}" if metric in ("Kill Rate", "Std Dev") else str(int(val)))
                row_h.append(f"Train: {VARIANT_LABELS[v]}<br>Test: {t}<br>Mean: {cell['mean']:.1%}<br>Std: {cell['std']:.1%}<br>Seeds: {cell['n']}")
            else:
                row_z.append(None)
                row_t.append("—")
                row_h.append("No data yet")
        z.append(row_z)
        text_arr.append(row_t)
        hover.append(row_h)

    fig = go.Figure(go.Heatmap(
        z=z,
        x=[t.capitalize() for t in TEST_CONDS],
        y=[VARIANT_LABELS[v] for v in VARIANTS],
        text=text_arr,
        customdata=hover,
        texttemplate="%{text}",
        hovertemplate="%{customdata}<extra></extra>",
        colorscale="RdYlGn",
        zmin=0, zmax=1,
        colorbar=dict(title="Kill Rate", tickformat=".0%"),
    ))

    # Bold diagonal (same train/test condition)
    for i, v in enumerate(VARIANTS):
        for j, t in enumerate(TEST_CONDS):
            if v == t or (v == "degraded" and t == "degraded"):
                fig.add_shape(
                    type="rect",
                    x0=j - 0.5, x1=j + 0.5,
                    y0=i - 0.5, y1=i + 0.5,
                    line=dict(color="white", width=3),
                )

    fig.update_layout(
        height=380,
        xaxis_title="Test Condition",
        yaxis_title="Train Variant",
        font=dict(size=13),
        margin=dict(l=20, r=20, t=30, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("White borders = matched train/test condition (diagonal). Green = high kill rate.")

    # Key numbers
    st.subheader("Key Numbers")
    cols = st.columns(3)
    baseline_clean = agg.get("baseline", {}).get("clean", {}).get("mean")
    baseline_deg   = agg.get("baseline", {}).get("degraded", {}).get("mean")
    latency_mean   = np.mean([agg.get("latency", {}).get(t, {}).get("mean", 0) for t in TEST_CONDS if agg.get("latency", {}).get(t)])
    with cols[0]:
        if baseline_clean and baseline_deg:
            st.metric("Robustness Gap (Baseline)", f"{abs(baseline_clean - baseline_deg):.1%}",
                      delta=f"clean→degraded", delta_color="off")
    with cols[1]:
        if latency_mean:
            st.metric("Latency-trained avg kill rate", f"{latency_mean:.0%}", delta="across all test conditions")
    with cols[2]:
        completed = sum(len(r["eval"]) for r in grid.values())
        st.metric("Sweep progress", f"{completed}/48 cells")


# ---------------------------------------------------------------------------
# Tab 3 — Episode Replay
# ---------------------------------------------------------------------------

def tab_replay(replays):
    st.header("Episode Replay — 3D Combat Visualiser")

    if not replays:
        st.warning("No replay data. Run `python tacpm/replay_generator.py` first.")
        return

    ep_options = {
        f"Ep {ep['episode_id']:02d}  [{ep['outcome']}]  reward={ep['total_reward']:.0f}": ep
        for ep in replays
    }
    chosen_label = st.selectbox("Select episode", list(ep_options.keys()))
    ep = ep_options[chosen_label]
    ts = ep["timesteps"]

    # ---- 3D trajectory ----
    agent_x = [t["agent_pos"][0] for t in ts]
    agent_y = [t["agent_pos"][1] for t in ts]
    agent_z = [t["agent_pos"][2] for t in ts]
    opp_x   = [t["opp_pos"][0] for t in ts]
    opp_y   = [t["opp_pos"][1] for t in ts]
    opp_z   = [t["opp_pos"][2] for t in ts]
    steps   = list(range(len(ts)))

    fig = go.Figure()

    # Agent trajectory — coloured by step (light→dark blue)
    fig.add_trace(go.Scatter3d(
        x=agent_x, y=agent_y, z=agent_z,
        mode="lines+markers",
        line=dict(color=steps, colorscale="Blues", width=4),
        marker=dict(size=2, color=steps, colorscale="Blues"),
        name="Agent",
    ))
    # Opponent trajectory — red
    fig.add_trace(go.Scatter3d(
        x=opp_x, y=opp_y, z=opp_z,
        mode="lines",
        line=dict(color="red", width=3),
        name="Opponent",
        opacity=0.6,
    ))
    # Start / End markers
    fig.add_trace(go.Scatter3d(
        x=[agent_x[0]], y=[agent_y[0]], z=[agent_z[0]],
        mode="markers", marker=dict(size=10, color="lime", symbol="circle"),
        name="Agent start",
    ))
    fig.add_trace(go.Scatter3d(
        x=[agent_x[-1]], y=[agent_y[-1]], z=[agent_z[-1]],
        mode="markers",
        marker=dict(size=10, symbol="x",
                    color="#2ecc71" if ep["kill"] else "#e74c3c"),
        name=f"Agent end ({ep['outcome']})",
    ))
    # Ground plane
    gx = np.linspace(min(agent_x + opp_x) - 200, max(agent_x + opp_x) + 200, 4)
    gy = np.linspace(min(agent_y + opp_y) - 200, max(agent_y + opp_y) + 200, 4)
    gX, gY = np.meshgrid(gx, gy)
    fig.add_trace(go.Surface(
        x=gX, y=gY, z=np.zeros_like(gX),
        colorscale=[[0, "#1a1a2e"], [1, "#16213e"]],
        opacity=0.3, showscale=False, name="Ground",
    ))

    fig.update_layout(
        height=520,
        scene=dict(
            xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Altitude (m)",
            bgcolor="rgb(10,10,30)",
            xaxis=dict(gridcolor="#334"),
            yaxis=dict(gridcolor="#334"),
            zaxis=dict(gridcolor="#334", range=[0, max(agent_z + opp_z) + 100]),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=20, b=0),
        legend=dict(bgcolor="rgba(30,30,60,0.8)", font=dict(color="white")),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ---- Health timeline ----
    st.subheader("Health & Reward Timeline")
    col1, col2 = st.columns(2)
    times = [t["t"] for t in ts]

    with col1:
        fig_h = go.Figure()
        fig_h.add_trace(go.Scatter(
            x=times, y=[t["agent_health"] for t in ts],
            fill="tozeroy", name="Agent HP",
            line=dict(color="#3498db", width=2),
            fillcolor="rgba(52,152,219,0.25)",
        ))
        fig_h.add_trace(go.Scatter(
            x=times, y=[t["opp_health"] for t in ts],
            fill="tozeroy", name="Opp HP",
            line=dict(color="#e74c3c", width=2),
            fillcolor="rgba(231,76,60,0.25)",
        ))
        fig_h.update_layout(
            height=220, title="Health over time",
            yaxis=dict(range=[0, 105], title="HP"),
            xaxis_title="Step",
            margin=dict(l=10, r=10, t=30, b=30),
        )
        st.plotly_chart(fig_h, use_container_width=True)

    with col2:
        rewards = [t["reward"] for t in ts]
        cumulative = np.cumsum(rewards).tolist()
        fig_r = go.Figure()
        fig_r.add_trace(go.Scatter(
            x=times, y=rewards,
            name="Per-step reward",
            line=dict(color="#f39c12", width=1.5),
            opacity=0.7,
        ))
        fig_r.add_trace(go.Scatter(
            x=times, y=cumulative,
            name="Cumulative",
            line=dict(color="#2ecc71", width=2),
        ))
        fig_r.update_layout(
            height=220, title="Reward over time",
            xaxis_title="Step",
            margin=dict(l=10, r=10, t=30, b=30),
        )
        st.plotly_chart(fig_r, use_container_width=True)

    # ---- Distance ----
    dists = [t["distance"] for t in ts]
    fig_d = go.Figure()
    fig_d.add_trace(go.Scatter(
        x=times, y=dists,
        fill="tozeroy", name="Distance",
        line=dict(color="#9b59b6", width=2),
        fillcolor="rgba(155,89,182,0.2)",
    ))
    fig_d.add_hline(y=600, line_dash="dash", line_color="#3498db",
                    annotation_text="Agent fire range 600m")
    fig_d.add_hline(y=500, line_dash="dash", line_color="#e74c3c",
                    annotation_text="Opp fire range 500m")
    fig_d.update_layout(
        height=200, title="Distance to opponent",
        xaxis_title="Step", yaxis_title="m",
        margin=dict(l=10, r=10, t=30, b=30),
    )
    st.plotly_chart(fig_d, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 4 — EDT Diagnosis
# ---------------------------------------------------------------------------

def tab_edt(diagnosis, replays):
    st.header("EDT Episode Diagnosis")

    if not diagnosis:
        st.warning("No diagnosis yet. Run `python tacpm/edt_diagnose.py` first.")
        return

    # Summary bar chart
    st.subheader("Fix Recommendation Summary")
    fix_counts = Counter(d["predicted_fix"] for d in diagnosis)
    fail_counts = Counter(d["predicted_failure"] for d in diagnosis)

    col1, col2 = st.columns(2)
    with col1:
        fig_f = px.bar(
            x=list(fix_counts.values()),
            y=list(fix_counts.keys()),
            orientation="h",
            color=list(fix_counts.values()),
            color_continuous_scale="teal",
            labels={"x": "Episodes", "y": "Fix type"},
            title="Fix Recommendations",
        )
        fig_f.update_layout(height=280, showlegend=False,
                            coloraxis_showscale=False,
                            margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_f, use_container_width=True)

    with col2:
        fig_fm = px.bar(
            x=list(fail_counts.values()),
            y=list(fail_counts.keys()),
            orientation="h",
            color=list(fail_counts.values()),
            color_continuous_scale="reds",
            labels={"x": "Episodes", "y": "Failure mode"},
            title="Predicted Failure Modes",
        )
        fig_fm.update_layout(height=280, showlegend=False,
                             coloraxis_showscale=False,
                             margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_fm, use_container_width=True)

    # Per-episode detail with attention timeline
    st.subheader("Per-episode Detail")
    ep_labels = [
        f"Ep {d['episode_id']:02d} | {d['outcome']} | {d['predicted_fix']} ({d['fix_confidence']:.0%})"
        for d in diagnosis
    ]
    sel = st.selectbox("Select episode", ep_labels)
    diag = diagnosis[ep_labels.index(sel)]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Predicted Failure", diag["predicted_failure"])
    with c2:
        st.metric("Recommended Fix", diag["predicted_fix"])
    with c3:
        st.metric("Confidence", f"{diag['fix_confidence']:.0%}")

    # Fix confidence breakdown
    fig_conf = go.Figure(go.Bar(
        x=list(diag["fix_probs"].values()),
        y=list(diag["fix_probs"].keys()),
        orientation="h",
        marker_color=[
            "#2ecc71" if k == diag["predicted_fix"] else "#3498db"
            for k in diag["fix_probs"]
        ],
    ))
    fig_conf.update_layout(
        height=220, title="Fix type confidence",
        xaxis=dict(range=[0, 1], tickformat=".0%"),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig_conf, use_container_width=True)

    # Attention timeline
    attn = diag.get("attention_weights", [])
    if attn:
        fig_attn = go.Figure(go.Bar(
            x=list(range(len(attn))),
            y=attn,
            marker=dict(
                color=attn,
                colorscale="Oranges",
                showscale=False,
            ),
        ))
        # Mark critical timesteps
        for cts in diag.get("critical_timesteps", []):
            if cts < len(attn):
                fig_attn.add_vline(x=cts, line_dash="dot",
                                   line_color="#e74c3c", opacity=0.8)
        fig_attn.update_layout(
            height=200,
            title="EDT Attention weights (red lines = critical timesteps)",
            xaxis_title="Timestep", yaxis_title="Attention",
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig_attn, use_container_width=True)

    # Advice box
    st.info(f"**Advice:** {diag['fix_advice']}")
    if diag.get("fix_code_snippet"):
        st.code(diag["fix_code_snippet"], language="python")


# ---------------------------------------------------------------------------
# Tab 5 — EDT Training History
# ---------------------------------------------------------------------------

def tab_edt_training(history):
    st.header("EDT Model Training History")

    if not history:
        st.warning("No training history. Run `python tacpm/train_edt.py` first.")
        return

    epochs = list(range(1, len(history["train_loss"]) + 1))

    col1, col2 = st.columns(2)
    with col1:
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(
            x=epochs, y=history["train_loss"],
            mode="lines", name="Train loss",
            line=dict(color="#e74c3c", width=2),
        ))
        fig_loss.update_layout(
            height=280, title="Training Loss",
            xaxis_title="Epoch", yaxis_title="Loss",
            margin=dict(l=10, r=10, t=40, b=30),
        )
        st.plotly_chart(fig_loss, use_container_width=True)

    with col2:
        fig_acc = go.Figure()
        fig_acc.add_trace(go.Scatter(
            x=epochs, y=[v * 100 for v in history["val_failure_acc"]],
            mode="lines", name="Failure mode acc",
            line=dict(color="#3498db", width=2),
        ))
        fig_acc.add_trace(go.Scatter(
            x=epochs, y=[v * 100 for v in history["val_fix_acc"]],
            mode="lines", name="Fix type acc",
            line=dict(color="#2ecc71", width=2),
        ))
        best = max(
            (a + b) / 2
            for a, b in zip(history["val_failure_acc"], history["val_fix_acc"])
        ) * 100
        fig_acc.add_hline(y=best, line_dash="dash", line_color="#f39c12",
                          annotation_text=f"Best: {best:.1f}%")
        fig_acc.update_layout(
            height=280, title="Validation Accuracy",
            xaxis_title="Epoch", yaxis_title="Accuracy (%)",
            yaxis=dict(range=[0, 105]),
            margin=dict(l=10, r=10, t=40, b=30),
        )
        st.plotly_chart(fig_acc, use_container_width=True)

    # Final per-class accuracy from last checkpoint
    st.subheader("Model Accuracy at Best Checkpoint")
    best_idx = int(np.argmax([
        (a + b) / 2
        for a, b in zip(history["val_failure_acc"], history["val_fix_acc"])
    ]))
    col3, col4 = st.columns(2)
    with col3:
        fa = history["val_failure_acc"][best_idx] * 100
        st.metric("Failure mode accuracy", f"{fa:.1f}%", f"epoch {best_idx + 1}")
    with col4:
        fxa = history["val_fix_acc"][best_idx] * 100
        st.metric("Fix type accuracy", f"{fxa:.1f}%", f"epoch {best_idx + 1}")


# ---------------------------------------------------------------------------
# Tab 6 — Self-play progress
# ---------------------------------------------------------------------------

def tab_selfplay(sp_log, auto_log):
    st.header("Self-Play & Auto Reward Loop")

    col_sp, col_ar = st.columns(2)

    # ---- Self-play ----
    with col_sp:
        st.subheader("Self-Play Generations")
        if not sp_log:
            st.info("Not started yet.\n\n```\npython experiments/train_selfplay.py --gens 5\n```")
        else:
            gens = sp_log.get("generations", [])
            if gens:
                seed_key = str(list(gens[0]["seeds"].keys())[0])
                gen_nums  = [g["gen"] for g in gens]
                kr_heur   = [g["seeds"].get(seed_key, {}).get("vs_heuristic", {}).get("kill_rate", 0) for g in gens]
                kr_sp     = [g["seeds"].get(seed_key, {}).get("vs_selfplay",  {}).get("kill_rate", None) for g in gens]

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=gen_nums, y=[v * 100 for v in kr_heur],
                    mode="lines+markers", name="vs Heuristic",
                    line=dict(color="#3498db", width=2),
                    marker=dict(size=8),
                ))
                if any(v is not None for v in kr_sp):
                    fig.add_trace(go.Scatter(
                        x=[g for g, v in zip(gen_nums, kr_sp) if v is not None],
                        y=[v * 100 for v in kr_sp if v is not None],
                        mode="lines+markers", name="vs Self-play opp",
                        line=dict(color="#e74c3c", width=2, dash="dot"),
                        marker=dict(size=8),
                    ))
                fig.update_layout(
                    height=280,
                    xaxis_title="Generation", yaxis_title="Kill Rate (%)",
                    yaxis=dict(range=[0, 105]),
                    margin=dict(l=10, r=10, t=20, b=30),
                )
                st.plotly_chart(fig, use_container_width=True)

                df = pd.DataFrame([{
                    "Gen": g["gen"],
                    "Opp Type": g["opp_type"],
                    "Kill % vs Heuristic": f"{g['seeds'].get(seed_key,{}).get('vs_heuristic',{}).get('kill_rate',0):.0%}",
                    "Kill % vs Self": f"{g['seeds'].get(seed_key,{}).get('vs_selfplay',{}).get('kill_rate','—'):.0%}"
                        if isinstance(g['seeds'].get(seed_key,{}).get('vs_selfplay',{}).get('kill_rate'), float) else "—",
                } for g in gens])
                st.dataframe(df, hide_index=True, use_container_width=True)

    # ---- Auto reward loop ----
    with col_ar:
        st.subheader("Auto Reward Engineering Loop")
        if not auto_log:
            st.info("Not started yet.\n\n```\npython experiments/auto_reward_loop.py --iters 3\n```")
        else:
            iters = auto_log.get("iterations", [])
            if iters:
                fig2 = go.Figure()
                fig2.add_trace(go.Bar(
                    x=[f"Iter {i['iter']}" for i in iters],
                    y=[i["kill_before"] * 100 for i in iters],
                    name="Before fix",
                    marker_color="#e74c3c",
                ))
                fig2.add_trace(go.Bar(
                    x=[f"Iter {i['iter']}" for i in iters],
                    y=[i["kill_after"] * 100 for i in iters],
                    name="After fix",
                    marker_color="#2ecc71",
                ))
                fig2.update_layout(
                    barmode="group", height=280,
                    yaxis=dict(range=[0, 105], title="Kill Rate (%)"),
                    margin=dict(l=10, r=10, t=20, b=30),
                )
                st.plotly_chart(fig2, use_container_width=True)

                df2 = pd.DataFrame([{
                    "Iter": i["iter"],
                    "Fix Applied": i["fix_applied"],
                    "Kill Before": f"{i['kill_before']:.0%}",
                    "Kill After":  f"{i['kill_after']:.0%}",
                    "Delta":       f"{i['kill_delta']:+.0%}",
                    "Accepted":    "✓" if i["accepted"] else "✗",
                } for i in iters])
                st.dataframe(df2, hide_index=True, use_container_width=True)


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------

def main():
    st.title("DRL Aerial Combat Research — Live Dashboard")
    st.caption("Robustness study + EDT + Self-play + Auto Reward Loop  |  Auto-refreshes on file change")

    grid      = load_grid()
    replays   = load_replays()
    diagnosis = load_diagnosis()
    history   = load_edt_history()
    sp_log    = load_selfplay_log()
    auto_log  = load_auto_reward_log()

    tabs = st.tabs([
        "Overview",
        "Robustness Grid",
        "Episode Replay",
        "EDT Diagnosis",
        "EDT Training",
        "Self-Play & Auto Loop",
    ])

    with tabs[0]:
        tab_overview(grid, replays, diagnosis)
    with tabs[1]:
        tab_robustness(grid)
    with tabs[2]:
        tab_replay(replays)
    with tabs[3]:
        tab_edt(diagnosis, replays)
    with tabs[4]:
        tab_edt_training(history)
    with tabs[5]:
        tab_selfplay(sp_log, auto_log)


if __name__ == "__main__":
    main()
