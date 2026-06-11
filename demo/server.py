"""
Aerial Combat Demo — FastAPI + WebSocket Game Server

Runs the EnhancedAerialCombatEnv at 30 fps and streams state to the
Babylon.js frontend over a WebSocket connection.

Supports three control modes (set by client message):
  manual   — player controls the agent via keyboard input
  ai       — trained PPO model controls the agent
  degraded — PPO model + sensor noise + latency (robustness demo)

Usage (local):
    python demo/server.py

Usage (cloud, swap python path):
    uvicorn demo.server:app --host 0.0.0.0 --port 8000
"""

import sys
import json
import asyncio
import numpy as np
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Path bootstrap — works both as local subfolder AND standalone repo
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent.parent          # repo root when inside monorepo
DEMO = Path(__file__).parent

# Try monorepo paths first, fall back to local copies (for cloud deployment)
for candidate in [ROOT / "environment", DEMO / "environment"]:
    if candidate.exists():
        sys.path.insert(0, str(candidate))
        break

for candidate in [ROOT / "tacpm", DEMO / "tacpm"]:
    if candidate.exists():
        sys.path.insert(0, str(candidate))
        break

from enhanced_env import EnhancedAerialCombatEnv

# Model paths — prefer robustness sweep best model, fall back to baseline
MODEL_CANDIDATES = [
    ROOT / "models" / "robustness" / "latency_s123" / "best_model.zip",
    ROOT / "models" / "robustness" / "baseline_s42"  / "best_model.zip",
    ROOT / "models" / "ppo_agent.zip",
    DEMO  / "models" / "ppo_agent.zip",
]

EDT_CANDIDATES = [
    ROOT / "models" / "edt" / "edt_model.pth",
    DEMO  / "models" / "edt_model.pth",
]

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI(title="Aerial Combat Demo")

# Serve static files (HTML / JS / assets)
static_dir = DEMO / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/")
async def index():
    return FileResponse(str(static_dir / "index.html"))


# ---------------------------------------------------------------------------
# Model loader (lazy, cached)
# ---------------------------------------------------------------------------

_ppo_model = None
_edt_model = None

def get_ppo_model():
    global _ppo_model
    if _ppo_model is None:
        from stable_baselines3 import PPO
        for p in MODEL_CANDIDATES:
            if p.exists():
                _ppo_model = PPO.load(str(p))
                print(f"  PPO loaded: {p}")
                return _ppo_model
        print("  WARNING: No PPO model found — AI mode will use random policy")
    return _ppo_model

def get_edt_model():
    global _edt_model
    if _edt_model is None:
        try:
            from edt_model import EpisodeDiagnosticTransformer
            import torch
            for p in EDT_CANDIDATES:
                if p.exists():
                    m = EpisodeDiagnosticTransformer()
                    m.load_state_dict(torch.load(str(p), map_location="cpu"))
                    m.eval()
                    _edt_model = m
                    print(f"  EDT loaded: {p}")
                    return _edt_model
        except Exception as e:
            print(f"  EDT not available: {e}")
    return _edt_model


# ---------------------------------------------------------------------------
# Game session — one per WebSocket connection
# ---------------------------------------------------------------------------

TICK_RATE   = 30          # fps
TICK_MS     = 1.0 / TICK_RATE
MAX_SESSIONS = 4          # prevent overload

active_sessions = 0

class GameSession:
    def __init__(self, ws: WebSocket):
        self.ws        = ws
        self.mode      = "manual"       # manual | ai | degraded
        self.env: Optional[EnhancedAerialCombatEnv] = None
        self.obs       = None
        self.done      = False
        self.step_n    = 0
        self.episode   = 0
        self.trajectory = []            # for EDT diagnosis

        # Player input state
        self.keys = {
            "throttle": 0.0,   # W/S
            "pitch":    0.0,   # up/down arrows
            "roll":     0.0,   # Q/E
            "yaw":      0.0,   # A/D
        }

        # Stats
        self.kills        = 0
        self.deaths       = 0
        self.edt_diagnosis = None

    # ------------------------------------------------------------------

    def _make_env(self) -> EnhancedAerialCombatEnv:
        if self.mode == "degraded":
            return EnhancedAerialCombatEnv(
                noise_std=8.0, latency_steps=2, max_steps=1000
            )
        return EnhancedAerialCombatEnv(
            noise_std=0.0, latency_steps=0, max_steps=1000
        )

    def _reset(self):
        if self.env:
            self.env.close()
        self.env      = self._make_env()
        self.obs, _   = self.env.reset(seed=self.episode)
        self.done     = False
        self.step_n   = 0
        self.trajectory = []
        self.edt_diagnosis = None

    def _get_action(self) -> np.ndarray:
        if self.mode == "manual":
            return np.array([
                self.keys["throttle"],
                self.keys["pitch"],
                self.keys["roll"],
                self.keys["yaw"],
            ], dtype=np.float32)
        else:
            model = get_ppo_model()
            if model:
                action, _ = model.predict(self.obs, deterministic=True)
                return action
            return self.env.action_space.sample()

    def _run_edt(self):
        """Run EDT on the completed trajectory and return diagnosis."""
        edt = get_edt_model()
        if edt is None or len(self.trajectory) < 5:
            return None
        try:
            from edt_model import trajectory_to_tensor, make_pad_mask, FAILURE_MODES, FIX_TYPES
            import torch
            x, length = trajectory_to_tensor(self.trajectory)
            pad_mask  = make_pad_mask(length, x.size(1))
            with torch.no_grad():
                fl, fxl, attn = edt(x, pad_mask)
            fail_idx = int(fl.argmax(1)[0])
            fix_idx  = int(fxl.argmax(1)[0])
            fail_conf = float(torch.softmax(fl, 1)[0, fail_idx])
            fix_conf  = float(torch.softmax(fxl, 1)[0, fix_idx])
            T = min(len(self.trajectory), x.size(1))
            attn_vals = attn[0, :T].tolist()
            top3 = sorted(range(T), key=lambda i: attn_vals[i], reverse=True)[:3]
            return {
                "failure_mode": FAILURE_MODES[fail_idx],
                "fix_type":     FIX_TYPES[fix_idx],
                "fail_conf":    round(fail_conf, 3),
                "fix_conf":     round(fix_conf, 3),
                "critical_steps": top3,
            }
        except Exception as e:
            return {"error": str(e)}

    def _build_state(self, reward: float, info: dict) -> dict:
        env = self.env
        ap  = env.agent_state["position"].tolist()
        av  = env.agent_state["velocity"].tolist()
        op  = env.opponent_state["position"].tolist()
        ov  = env.opponent_state["velocity"].tolist()
        ah  = float(env.agent_state["heading"])
        oh  = float(env.opponent_state["heading"])

        rel  = np.array(op) - np.array(ap)
        dist = float(np.linalg.norm(rel))

        return {
            "type":          "state",
            "step":          self.step_n,
            "episode":       self.episode,
            "mode":          self.mode,
            "agent": {
                "pos":     ap,
                "vel":     av,
                "heading": ah,
                "health":  float(env.agent_state["health"]),
            },
            "opponent": {
                "pos":     op,
                "vel":     ov,
                "heading": oh,
                "health":  float(env.opponent_state["health"]),
            },
            "distance":      round(dist, 1),
            "in_range":      dist <= env.engagement_range,
            "opp_in_range":  dist <= env.opp_engagement_range,
            "reward":        round(reward, 3),
            "done":          self.done,
            "outcome":       info.get("outcome", ""),
            "kills":         self.kills,
            "deaths":        self.deaths,
            "edt":           self.edt_diagnosis,
        }

    # ------------------------------------------------------------------

    async def handle_message(self, msg: dict):
        t = msg.get("type")

        if t == "set_mode":
            self.mode = msg.get("mode", "manual")
            self._reset()
            await self.ws.send_text(json.dumps({
                "type": "mode_changed", "mode": self.mode
            }))

        elif t == "input":
            self.keys["throttle"] = float(msg.get("throttle", 0))
            self.keys["pitch"]    = float(msg.get("pitch",    0))
            self.keys["roll"]     = float(msg.get("roll",     0))
            self.keys["yaw"]      = float(msg.get("yaw",      0))

        elif t == "reset":
            self.episode += 1
            self._reset()

    async def run(self):
        self._reset()
        last_tick = asyncio.get_event_loop().time()

        while True:
            now  = asyncio.get_event_loop().time()
            wait = TICK_MS - (now - last_tick)
            if wait > 0:
                await asyncio.sleep(wait)
            last_tick = asyncio.get_event_loop().time()

            if self.done:
                await asyncio.sleep(0.1)
                continue

            # Step environment
            action = self._get_action()
            self.obs, reward, term, trunc, info = self.env.step(action)
            self.step_n += 1

            # Record trajectory for EDT
            env = self.env
            rel = env.opponent_state["position"] - env.agent_state["position"]
            self.trajectory.append({
                "t":            self.step_n,
                "distance":     float(np.linalg.norm(rel)),
                "agent_health": float(env.agent_state["health"]),
                "opp_health":   float(env.opponent_state["health"]),
                "reward":       float(reward),
                "agent_pos":    env.agent_state["position"].tolist(),
            })

            self.done = term or trunc

            if self.done:
                outcome = info.get("outcome", "")
                if outcome == "kill":
                    self.kills += 1
                elif outcome in ("died", "crashed"):
                    self.deaths += 1
                # Run EDT diagnosis
                self.edt_diagnosis = self._run_edt()
                # Auto-restart after 4 seconds
                asyncio.get_event_loop().call_later(
                    4.0, lambda: setattr(self, "_restart_pending", True)
                )

            # Check pending restart
            if getattr(self, "_restart_pending", False):
                self._restart_pending = False
                self.episode += 1
                self._reset()

            state = self._build_state(reward, info)
            try:
                await self.ws.send_text(json.dumps(state))
            except Exception:
                break


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    global active_sessions
    if active_sessions >= MAX_SESSIONS:
        await ws.close(code=1008, reason="Server full")
        return

    await ws.accept()
    active_sessions += 1
    session = GameSession(ws)
    print(f"  Client connected  (active: {active_sessions})")

    game_task = asyncio.create_task(session.run())

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            await session.handle_message(msg)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"  Session error: {e}")
    finally:
        game_task.cancel()
        if session.env:
            session.env.close()
        active_sessions -= 1
        print(f"  Client disconnected (active: {active_sessions})")


# ---------------------------------------------------------------------------
# Dev entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*50)
    print("  Aerial Combat Demo Server")
    print("  Open: http://localhost:8000")
    print("="*50 + "\n")
    uvicorn.run("server:app", host="0.0.0.0", port=8000,
                reload=False, app_dir=str(DEMO))
