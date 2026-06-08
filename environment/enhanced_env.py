"""
Enhanced aerial combat environment with realistic operational constraints.
Features: sensor noise, action latency, fire control system, health tracking.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple, Optional
from collections import deque


class EnhancedAerialCombatEnv(gym.Env):
    """
    Aerial combat environment with realistic degradations for robustness research.

    Observation space (14-dim):
        agent_pos(3) + agent_vel(3) + opp_pos(3) + opp_vel(3) + agent_health(1) + opp_health(1)

    Action space (4-dim continuous [-1, 1]):
        [throttle, pitch, roll, yaw]

    Combat mechanic:
        Agent fires when within engagement_range AND pointing within fire_cone_deg of opponent.
        Opponent fires back when within opp_engagement_range (always faces agent).
        Kill = opponent health <= 0. Death = agent health <= 0.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        noise_std: float = 0.0,
        latency_steps: int = 0,
        max_steps: int = 1000,
        engagement_range: float = 600.0,
        fire_cone_deg: float = 30.0,
        agent_damage_per_step: float = 5.0,
        opp_engagement_range: float = 500.0,
        opp_damage_per_step: float = 3.0,
    ):
        """
        Args:
            noise_std: Gaussian noise std added to position/velocity observations.
            latency_steps: Action buffer size (0 = no latency, N = N-1 step delay).
            max_steps: Episode length cap.
            engagement_range: Agent fires when closer than this (metres).
            fire_cone_deg: Half-angle of agent firing cone (degrees).
            agent_damage_per_step: HP dealt per step when firing conditions met.
            opp_engagement_range: Opponent fires when closer than this.
            opp_damage_per_step: HP dealt by opponent per step when firing.
        """
        super().__init__()

        self.noise_std = noise_std
        self.latency_steps = latency_steps
        self.max_steps = max_steps

        self.engagement_range = engagement_range
        self.fire_cone_rad = np.deg2rad(fire_cone_deg)
        self.agent_damage_per_step = agent_damage_per_step
        self.opp_engagement_range = opp_engagement_range
        self.opp_fire_cone_rad = np.deg2rad(45.0)
        self.opp_damage_per_step = opp_damage_per_step

        self.action_buffer = deque(maxlen=max(latency_steps, 1))
        self.agent_state = None
        self.opponent_state = None
        self.current_step = 0

        # Episode metric accumulators (reset each episode)
        self._min_distance = np.inf
        self._damage_dealt = 0.0
        self._damage_received = 0.0
        self._time_to_first_engagement = None

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )
        # 12 kinematic dims + 2 normalised health dims
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32
        )

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        self.current_step = 0
        self.action_buffer.clear()
        self._min_distance = np.inf
        self._damage_dealt = 0.0
        self._damage_received = 0.0
        self._time_to_first_engagement = None

        # Randomise starting positions so the policy must generalise
        rng = self.np_random
        pos_jitter = rng.uniform(-200.0, 200.0, size=2)   # x, y offset
        alt_jitter  = rng.uniform(-100.0, 100.0)
        hdg_jitter  = rng.uniform(-0.3, 0.3)

        self.agent_state = {
            "position": np.array([0.0 + pos_jitter[0],
                                  0.0 + pos_jitter[1],
                                  1000.0 + alt_jitter], dtype=np.float64),
            "velocity": np.array([100.0, 0.0, 0.0], dtype=np.float64),
            "heading": 0.0 + hdg_jitter,
            "health": 100.0,
        }

        opp_jitter = rng.uniform(-200.0, 200.0, size=2)
        self.opponent_state = {
            "position": np.array([1000.0 + opp_jitter[0],
                                  0.0 + opp_jitter[1],
                                  1000.0 + alt_jitter], dtype=np.float64),
            "velocity": np.array([-80.0, 0.0, 0.0], dtype=np.float64),
            "heading": np.pi + hdg_jitter,
            "health": 100.0,
        }

        return self._get_observation(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self.current_step += 1

        delayed = self._get_delayed_action(action)
        self._update_agent(delayed)
        self._update_opponent()
        combat_info = self._apply_combat()

        terminated = self._check_termination()
        truncated = self.current_step >= self.max_steps

        reward = self._calculate_reward(terminated, truncated, combat_info)
        obs = self._get_observation()

        info = {}
        if terminated or truncated:
            info = self._build_episode_info(terminated, truncated)

        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Physics
    # ------------------------------------------------------------------

    def _update_agent(self, action: np.ndarray) -> None:
        throttle, pitch, roll, yaw = np.clip(action, -1.0, 1.0)

        # Heading control
        self.agent_state["heading"] += yaw * 0.05

        # Speed control (50–170 m/s)
        speed = np.linalg.norm(self.agent_state["velocity"])
        target_speed = 50.0 + throttle * 120.0
        if speed > 1e-6:
            self.agent_state["velocity"] *= target_speed / speed

        # Pitch affects vertical velocity component
        self.agent_state["velocity"][2] += pitch * 5.0
        self.agent_state["velocity"][2] = np.clip(
            self.agent_state["velocity"][2], -50.0, 50.0
        )

        self.agent_state["position"] += self.agent_state["velocity"] * 0.1

    def _update_opponent(self) -> None:
        direction = self.agent_state["position"] - self.opponent_state["position"]
        distance = np.linalg.norm(direction)

        if distance > 1e-6:
            direction /= distance
            self.opponent_state["velocity"] = direction * 80.0
            self.opponent_state["heading"] = np.arctan2(direction[1], direction[0])

        self.opponent_state["position"] += self.opponent_state["velocity"] * 0.1

    # ------------------------------------------------------------------
    # Combat
    # ------------------------------------------------------------------

    def _apply_combat(self) -> Dict:
        """Resolve weapon fire for this timestep."""
        rel_pos = self.opponent_state["position"] - self.agent_state["position"]
        distance = float(np.linalg.norm(rel_pos))
        self._min_distance = min(self._min_distance, distance)

        agent_fired = False
        opp_fired = False

        # Agent fires at opponent
        if distance <= self.engagement_range:
            heading_vec = np.array([
                np.cos(self.agent_state["heading"]),
                np.sin(self.agent_state["heading"]),
                0.0,
            ])
            rel_norm = rel_pos / (distance + 1e-8)
            angle = np.arccos(np.clip(np.dot(heading_vec, rel_norm), -1.0, 1.0))

            if angle <= self.fire_cone_rad:
                self.opponent_state["health"] -= self.agent_damage_per_step
                self._damage_dealt += self.agent_damage_per_step
                agent_fired = True
                if self._time_to_first_engagement is None:
                    self._time_to_first_engagement = self.current_step

        # Opponent fires at agent (heuristic always faces agent, so angle ≈ 0)
        if distance <= self.opp_engagement_range:
            opp_heading_vec = np.array([
                np.cos(self.opponent_state["heading"]),
                np.sin(self.opponent_state["heading"]),
                0.0,
            ])
            agent_dir = -rel_pos / (distance + 1e-8)
            opp_angle = np.arccos(np.clip(np.dot(opp_heading_vec, agent_dir), -1.0, 1.0))

            if opp_angle <= self.opp_fire_cone_rad:
                self.agent_state["health"] -= self.opp_damage_per_step
                self._damage_received += self.opp_damage_per_step
                opp_fired = True

        return {"agent_fired": agent_fired, "opp_fired": opp_fired}

    def _check_termination(self) -> bool:
        if self.agent_state["health"] <= 0 or self.opponent_state["health"] <= 0:
            return True
        if self.agent_state["position"][2] < 0:
            return True
        distance = np.linalg.norm(
            self.agent_state["position"] - self.opponent_state["position"]
        )
        if distance > 15_000.0:
            return True
        return False

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _calculate_reward(
        self, terminated: bool, truncated: bool, combat_info: Dict
    ) -> float:
        reward = 0.0

        # Per-step survival
        reward += 0.1

        rel_pos = self.opponent_state["position"] - self.agent_state["position"]
        distance = float(np.linalg.norm(rel_pos))

        # Proximity reward (encourage closing within 2 km)
        reward += max(0.0, (2000.0 - distance) / 2000.0) * 3.0

        # Heading reward (agent facing opponent)
        if distance > 1e-6:
            heading_vec = np.array([
                np.cos(self.agent_state["heading"]),
                np.sin(self.agent_state["heading"]),
                0.0,
            ])
            rel_norm = rel_pos / distance
            dot = float(np.dot(heading_vec, rel_norm))
            reward += max(0.0, (dot + 1.0) / 2.0) * 1.5

        # Closing velocity reward
        if distance > 1e-6:
            vel_diff = self.agent_state["velocity"] - self.opponent_state["velocity"]
            closing_speed = float(np.dot(vel_diff, rel_pos / distance))
            reward += max(0.0, closing_speed / 200.0)

        # Health advantage (proportional, not binary)
        health_diff = (
            self.agent_state["health"] - self.opponent_state["health"]
        ) / 100.0
        reward += health_diff * 2.0

        # Firing reward (incentivise being in firing position)
        if combat_info["agent_fired"]:
            reward += 1.0

        # Terminal rewards
        if terminated:
            if self.opponent_state["health"] <= 0 and self.agent_state["health"] > 0:
                reward += 100.0   # kill
            elif self.agent_state["health"] <= 0:
                reward -= 50.0    # death
            elif self.agent_state["position"][2] < 0:
                reward -= 30.0    # ground collision
        elif truncated:
            # Partial reward for net damage at timeout
            reward += (self._damage_dealt - self._damage_received) / 100.0

        return float(reward)

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_observation(self) -> np.ndarray:
        obs = np.concatenate([
            self.agent_state["position"],
            self.agent_state["velocity"],
            self.opponent_state["position"],
            self.opponent_state["velocity"],
            [np.clip(self.agent_state["health"] / 100.0, 0.0, 1.0)],
            [np.clip(self.opponent_state["health"] / 100.0, 0.0, 1.0)],
        ]).astype(np.float32)

        if self.noise_std > 0.0:
            noise = np.zeros(14, dtype=np.float32)
            noise[:12] = self.np_random.normal(0.0, self.noise_std, size=12).astype(np.float32)
            obs = obs + noise

        return obs

    def _get_delayed_action(self, action: np.ndarray) -> np.ndarray:
        if self.latency_steps <= 1:
            return np.clip(action, -1.0, 1.0).astype(np.float32)
        self.action_buffer.append(action.copy())
        if len(self.action_buffer) < self.latency_steps:
            return np.zeros(4, dtype=np.float32)
        return np.array(list(self.action_buffer)[0], dtype=np.float32)

    # ------------------------------------------------------------------
    # Episode summary (returned in info on episode end)
    # ------------------------------------------------------------------

    def _build_episode_info(self, terminated: bool, truncated: bool) -> Dict:
        if self.opponent_state["health"] <= 0 and self.agent_state["health"] > 0:
            outcome = "kill"
        elif self.agent_state["health"] <= 0:
            outcome = "died"
        elif self.agent_state["position"][2] < 0:
            outcome = "crashed"
        elif truncated:
            outcome = "timeout"
        else:
            outcome = "fled"

        return {
            "outcome": outcome,
            "kill": outcome == "kill",
            "survived": outcome in ("kill", "timeout"),
            "episode_length": self.current_step,
            "agent_health_final": float(self.agent_state["health"]),
            "opp_health_final": float(self.opponent_state["health"]),
            "damage_dealt": float(self._damage_dealt),
            "damage_received": float(self._damage_received),
            "net_damage": float(self._damage_dealt - self._damage_received),
            "min_distance": float(self._min_distance) if self._min_distance != np.inf else -1.0,
            "time_to_first_engagement": self._time_to_first_engagement,
        }


if __name__ == "__main__":
    env = EnhancedAerialCombatEnv(noise_std=0.0, latency_steps=0)
    obs, _ = env.reset()
    print(f"Obs shape: {obs.shape}  (expected 14)")
    print(f"Action space: {env.action_space}")

    total_r = 0.0
    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_r += reward
        if terminated or truncated:
            print(f"Episode done — {info}")
            break

    env.close()
    print(f"Total reward: {total_r:.2f}")
    print("Enhanced env OK.")
