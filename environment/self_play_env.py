"""
Self-play aerial combat environment.

The opponent is driven by a loaded PPO policy (or falls back to the
heuristic if no model is provided).  Both combatants follow identical
physics so kill/death conditions are symmetric.

Key design decisions:
  - Opponent observation is the *mirrored* 14-dim vector (its own state
    as "agent", the actual agent as "opponent").
  - Opponent uses the same heading / throttle / pitch physics as the agent,
    so a learned opponent can genuinely out-manoeuvre the agent.
  - Fire-cone checks already use opponent_state["heading"], so the PPO
    opponent must learn to face the target — it cannot free-ride on the
    heuristic's perfect aim.
  - Opponent pool: __init__ accepts a list of model paths; at each
    episode reset a random one is sampled (league-style mixing).
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "environment"))

from enhanced_env import EnhancedAerialCombatEnv


class SelfPlayEnv(EnhancedAerialCombatEnv):
    """
    Extends EnhancedAerialCombatEnv so the opponent can run a PPO policy.

    Args:
        opp_model_paths : list of .zip paths to sample opponents from.
                          Pass [] or None to use the heuristic opponent.
        opp_deterministic: If True, opponent acts deterministically.
    """

    def __init__(
        self,
        opp_model_paths: Optional[List[str]] = None,
        opp_deterministic: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._opp_paths      = opp_model_paths or []
        self._opp_det        = opp_deterministic
        self._opp_model      = None          # loaded at reset
        self._opp_model_path = None          # currently active path
        self._opp_rng        = np.random.default_rng(0)

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------

    def _load_opponent(self, path: str):
        from stable_baselines3 import PPO
        self._opp_model      = PPO.load(path)
        self._opp_model_path = path

    def update_opponent_pool(self, paths: List[str]) -> None:
        """Hot-swap the pool of opponent checkpoints (call between generations)."""
        self._opp_paths = list(paths)
        self._opp_model = None   # force reload on next reset

    # ------------------------------------------------------------------
    # Reset — sample a new opponent from the pool
    # ------------------------------------------------------------------

    def reset(self, seed=None, **kwargs):
        obs, info = super().reset(seed=seed, **kwargs)
        if self._opp_paths:
            path = self._opp_paths[
                self._opp_rng.integers(len(self._opp_paths))
            ]
            if path != self._opp_model_path:
                self._load_opponent(path)
        else:
            self._opp_model = None
        return obs, info

    # ------------------------------------------------------------------
    # Opponent observation (mirrored 14-dim)
    # ------------------------------------------------------------------

    def _get_opponent_observation(self) -> np.ndarray:
        """Build the 14-dim obs from the opponent's perspective."""
        obs = np.concatenate([
            self.opponent_state["position"],
            self.opponent_state["velocity"],
            self.agent_state["position"],
            self.agent_state["velocity"],
            [np.clip(self.opponent_state["health"] / 100.0, 0.0, 1.0)],
            [np.clip(self.agent_state["health"]    / 100.0, 0.0, 1.0)],
        ]).astype(np.float32)

        # Apply same noise to opponent obs if noise mode active
        if self.noise_std > 0.0:
            noise = np.zeros(14, dtype=np.float32)
            noise[:12] = self.np_random.normal(
                0.0, self.noise_std, size=12
            ).astype(np.float32)
            obs = obs + noise

        return obs

    # ------------------------------------------------------------------
    # Opponent physics — identical to _update_agent
    # ------------------------------------------------------------------

    def _apply_opponent_action(self, action: np.ndarray) -> None:
        """Update opponent state using the same physics as the agent."""
        throttle, pitch, roll, yaw = np.clip(action, -1.0, 1.0)

        self.opponent_state["heading"] += yaw * 0.05

        speed = np.linalg.norm(self.opponent_state["velocity"])
        target_speed = 50.0 + throttle * 120.0
        if speed > 1e-6:
            self.opponent_state["velocity"] *= target_speed / speed

        self.opponent_state["velocity"][2] += pitch * 5.0
        self.opponent_state["velocity"][2]  = np.clip(
            self.opponent_state["velocity"][2], -50.0, 50.0
        )
        self.opponent_state["position"] += self.opponent_state["velocity"] * 0.1

    # ------------------------------------------------------------------
    # Override _update_opponent
    # ------------------------------------------------------------------

    def _update_opponent(self) -> None:
        if self._opp_model is None:
            # Fall back to heuristic (always faces + closes on agent)
            super()._update_opponent()
            return

        opp_obs    = self._get_opponent_observation()
        action, _  = self._opp_model.predict(opp_obs, deterministic=self._opp_det)

        # Apply latency to opponent too if configured (optional — we leave it off
        # for the opponent to keep training tractable)
        self._apply_opponent_action(action)

    # ------------------------------------------------------------------
    # Terminate opponent if it hits the ground too
    # ------------------------------------------------------------------

    def _check_termination(self) -> bool:
        if super()._check_termination():
            return True
        # Learned opponent can also crash
        if (self._opp_model is not None
                and self.opponent_state["position"][2] < 0):
            # Mark as agent win (opponent crashed — count as kill)
            self.opponent_state["health"] = 0.0
            return True
        return False
