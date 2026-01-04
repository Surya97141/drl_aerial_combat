"""
Week 3: Research-grade aerial combat environment.
Features: sensor noise, latency, advanced multi-objective rewards.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple, Optional
from collections import deque


class EnhancedAerialCombatEnv(gym.Env):
    """
    Enhanced aerial combat environment with realistic constraints.
    
    Additions:
    - Sensor noise (observation degradation)
    - Action latency (communication delay)
    - Multi-objective reward function
    - Advanced opponent behavior
    """
    
    metadata = {"render_modes": []}
    
    def __init__(self, noise_std: float = 8.0, latency_steps: int = 2, max_steps: int = 1000):
        """
        Initialize enhanced environment.
        
        Args:
            noise_std: Standard deviation of sensor noise
            latency_steps: Number of steps to delay actions (1-3)
            max_steps: Maximum steps per episode
        """
        super().__init__()
        
        self.noise_std = noise_std
        self.latency_steps = latency_steps
        self.max_steps = max_steps
        self.current_step = 0
        
        # Latency buffer
        self.action_buffer = deque(maxlen=latency_steps)
        
        # States
        self.agent_state = None
        self.opponent_state = None
        
        # Action/observation spaces
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(4,), 
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(12,),
            dtype=np.float32
        )
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.action_buffer.clear()
        
        # Initialize agent at origin
        self.agent_state = {
            "position": np.array([0.0, 0.0, 1000.0]),
            "velocity": np.array([100.0, 0.0, 0.0]),
            "heading": 0.0,
            "health": 100.0
        }
        
        # Initialize opponent at distance
        self.opponent_state = {
            "position": np.array([1000.0, 0.0, 1000.0]),
            "velocity": np.array([-80.0, 0.0, 0.0]),
            "heading": np.pi,
            "health": 100.0
        }
        
        obs = self._get_noisy_observation()
        return obs, {}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one environment step."""
        self.current_step += 1
        
        # Apply latency
        delayed_action = self._get_delayed_action(action)
        
        # Update physics
        self._update_agent(delayed_action)
        self._update_opponent()
        
        # Calculate reward
        reward = self._calculate_advanced_reward()
        
        # Check termination
        terminated = self._check_termination()
        truncated = self.current_step >= self.max_steps
        
        # Get noisy observation
        obs = self._get_noisy_observation()
        
        return obs, reward, terminated, truncated, {}
    
    def _get_delayed_action(self, action: np.ndarray) -> np.ndarray:
        """Simulate communication latency."""
        self.action_buffer.append(action)
        
        if len(self.action_buffer) < self.latency_steps:
            return np.zeros(4)
        
        return list(self.action_buffer)[0]
    
    def _get_noisy_observation(self) -> np.ndarray:
        """Add realistic sensor noise to observations."""
        clean_obs = np.concatenate([
            self.agent_state["position"],
            self.agent_state["velocity"],
            self.opponent_state["position"],
            self.opponent_state["velocity"]
        ]).astype(np.float32)
        
        # Gaussian noise
        noise = np.random.normal(0, self.noise_std, size=clean_obs.shape).astype(np.float32)
        noisy_obs = (clean_obs + noise).astype(np.float32)
        
        return noisy_obs
    
    def _calculate_advanced_reward(self) -> float:
        """
        Multi-objective reward function.
        
        Components:
        1. Survival bonus
        2. Distance to opponent
        3. Angle/heading to opponent
        4. Closing velocity
        5. Health advantage
        """
        reward = 0.0
        
        # 1. Survival bonus (per step)
        reward += 0.1
        
        # 2. Distance reward (closer = better, normalized to 2km range)
        distance = np.linalg.norm(
            self.agent_state["position"] - self.opponent_state["position"]
        )
        distance_reward = max(0, (2000.0 - distance) / 2000.0) * 3.0
        reward += distance_reward
        
        # 3. Angle to opponent (facing = better)
        rel_pos = self.opponent_state["position"] - self.agent_state["position"]
        if np.linalg.norm(rel_pos) > 1e-6:
            rel_heading = np.arctan2(rel_pos[1], rel_pos[0]) - self.agent_state["heading"]
            rel_heading = (rel_heading + np.pi) % (2 * np.pi) - np.pi  # Normalize to [-π, π]
            angle_reward = max(0, (np.pi - abs(rel_heading)) / np.pi) * 1.5
            reward += angle_reward
        
        # 4. Closing velocity (moving towards opponent = good)
        vel_diff = self.agent_state["velocity"] - self.opponent_state["velocity"]
        closing_speed = np.dot(vel_diff, rel_pos / (np.linalg.norm(rel_pos) + 1e-6))
        closing_reward = max(0, closing_speed / 200.0) * 1.0
        reward += closing_reward
        
        # 5. Health advantage
        if self.agent_state["health"] > self.opponent_state["health"]:
            reward += 0.5
        
        return float(reward)
    
    def _update_agent(self, action: np.ndarray) -> None:
        """Update agent state based on control input."""
        # Unpack action
        throttle, pitch, roll, yaw = np.clip(action, -1.0, 1.0)
        
        # Update heading
        self.agent_state["heading"] += yaw * 0.05
        
        # Update velocity magnitude (throttle control)
        speed = np.linalg.norm(self.agent_state["velocity"])
        target_speed = 50.0 + throttle * 120.0  # 50-170 m/s
        
        if speed > 1e-6:
            self.agent_state["velocity"] *= (target_speed / speed)
        
        # Update position
        self.agent_state["position"] += self.agent_state["velocity"] * 0.1
    
    def _update_opponent(self) -> None:
        """Update opponent with simple heuristic AI."""
        # Vector to agent
        direction = self.agent_state["position"] - self.opponent_state["position"]
        distance = np.linalg.norm(direction)
        
        # Pursue agent
        if distance > 1e-6:
            direction = direction / distance
            self.opponent_state["velocity"] = direction * 80.0
            self.opponent_state["heading"] = np.arctan2(direction[1], direction[0])
        
        # Update position
        self.opponent_state["position"] += self.opponent_state["velocity"] * 0.1
    
    def _check_termination(self) -> bool:
        """Check if episode should terminate."""
        # Aircraft below ground
        if self.agent_state["position"][2] < 0:
            return True
        
        # Aircraft too far apart
        distance = np.linalg.norm(
            self.agent_state["position"] - self.opponent_state["position"]
        )
        if distance > 15000.0:
            return True
        
        # One of them dead (optional for future)
        if self.agent_state["health"] <= 0 or self.opponent_state["health"] <= 0:
            return True
        
        return False
    
    def close(self) -> None:
        """Clean up resources."""
        pass


if __name__ == "__main__":
    # Quick test
    env = EnhancedAerialCombatEnv(noise_std=8.0, latency_steps=2)
    obs, info = env.reset()
    print(f" Enhanced env created!")
    print(f"Obs shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    
    # Run 50 steps
    for _ in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    
    print(f" Enhanced env working! (50 steps ran successfully)")
    env.close()