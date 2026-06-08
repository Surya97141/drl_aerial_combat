"""
Core aerial combat environment for DRL training.
Gym-compatible interface for fighter aircraft control.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple, Optional


class AerialCombatEnv(gym.Env):
    """
    Aerial combat environment.
    
    Agent controls a fighter aircraft in combat against an opponent.
    State: Aircraft state (position, velocity, heading, health)
    Action: Control inputs (throttle, pitch, roll, yaw)
    Reward: Combat success metrics
    """
    
    metadata = {"render_modes": []}
    
    def __init__(self, 
                 max_steps: int = 1000,
                 opponent_type: str = "heuristic"):
        """
        Initialize the environment.
        
        Args:
            max_steps: Maximum steps per episode
            opponent_type: Type of opponent AI
        """
        super().__init__()
        
        self.max_steps = max_steps
        self.opponent_type = opponent_type
        
        # Step counter
        self.current_step = 0
        
        # Agent state (position, velocity, heading, health, etc.)
        self.agent_state = None
        self.opponent_state = None
        
        # Define action space
        # 4 continuous actions: [throttle, pitch, roll, yaw]
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(4,), 
            dtype=np.float32
        )
        
        # Define observation space
        # 12 observations: agent_pos(3) + agent_vel(3) + opponent_pos(3) + opponent_vel(3)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(12,),
            dtype=np.float32
        )
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment to initial state.
        
        Returns:
            observation: Initial state
            info: Additional info
        """
        super().reset(seed=seed)
        
        self.current_step = 0
        
        # Initialize agent at origin
        self.agent_state = {
            "position": np.array([0.0, 0.0, 1000.0]),  # x, y, altitude
            "velocity": np.array([100.0, 0.0, 0.0]),   # vx, vy, vz
            "heading": 0.0,
            "health": 100.0
        }
        
        # Initialize opponent at distance
        self.opponent_state = {
            "position": np.array([1000.0, 0.0, 1000.0]),
            "velocity": np.array([-80.0, 0.0, 0.0]),
            "heading": np.pi,  # Facing agent
            "health": 100.0
        }
        
        obs = self._get_observation()
        return obs, {}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one environment step.
        
        Args:
            action: Control input [throttle, pitch, roll, yaw]
        
        Returns:
            observation: New state
            reward: Reward signal
            terminated: Episode end flag
            truncated: Max steps reached flag
            info: Additional info
        """
        self.current_step += 1
        
        # Clip action to valid range
        action = np.clip(action, -1.0, 1.0)
        
        # Update agent state based on action
        self._update_agent(action)
        
        # Update opponent (AI)
        self._update_opponent()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check termination conditions
        terminated = self._check_termination()
        truncated = self.current_step >= self.max_steps
        
        # Get observation
        obs = self._get_observation()
        
        return obs, reward, terminated, truncated, {}
    
    def _update_agent(self, action: np.ndarray) -> None:
        """Update agent state based on control input."""
        # Simple physics: just update velocity based on throttle
        throttle, pitch, roll, yaw = action
        
        # Update heading
        self.agent_state["heading"] += yaw * 0.01
        
        # Update velocity magnitude based on throttle
        speed = np.linalg.norm(self.agent_state["velocity"])
        target_speed = 50.0 + throttle * 100.0  # 50-150 m/s
        self.agent_state["velocity"] *= (target_speed / (speed + 1e-6))
        
        # Update position
        self.agent_state["position"] += self.agent_state["velocity"] * 0.1
    
    def _update_opponent(self) -> None:
        """Update opponent state (simple heuristic)."""
        # Opponent tries to maintain distance and face agent
        direction = self.agent_state["position"] - self.opponent_state["position"]
        distance = np.linalg.norm(direction)
        
        if distance > 0:
            direction = direction / distance
        
        # Simple pursuit: move towards agent
        self.opponent_state["velocity"] = direction * 80.0
        self.opponent_state["position"] += self.opponent_state["velocity"] * 0.1
        
        # Face agent
        if distance > 0:
            self.opponent_state["heading"] = np.arctan2(direction[1], direction[0])
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        obs = np.concatenate([
            self.agent_state["position"],
            self.agent_state["velocity"],
            self.opponent_state["position"],
            self.opponent_state["velocity"]
        ]).astype(np.float32)
        return obs
    
    def _calculate_reward(self) -> float:
        """Calculate reward signal."""
        reward = 0.0
        
        # Distance to opponent (closer is better)
        distance = np.linalg.norm(
            self.agent_state["position"] - self.opponent_state["position"]
        )
        reward += (1000.0 - distance) / 1000.0  # -1 to +1
        
        # Survival bonus
        reward += 0.01
        
        # Damage bonus
        if self.agent_state["health"] > self.opponent_state["health"]:
            reward += 0.1
        
        return float(reward)
    
    def _check_termination(self) -> bool:
        """Check if episode should terminate."""
        # Either aircraft dies
        if self.agent_state["health"] <= 0 or self.opponent_state["health"] <= 0:
            return True
        
        # Aircraft leaves bounds (altitude or distance)
        if self.agent_state["position"][2] < 0:  # Below ground
            return True
        
        distance = np.linalg.norm(
            self.agent_state["position"] - self.opponent_state["position"]
        )
        if distance > 10000:  # Too far apart
            return True
        
        return False
    
    def close(self) -> None:
        """Clean up resources."""
        pass


if __name__ == "__main__":
    # Quick test
    env = AerialCombatEnv()
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Run 10 steps
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Reward: {reward:.3f}, Distance: {np.linalg.norm(obs[:3] - obs[6:9]):.1f}m")
        if terminated or truncated:
            break
    
    env.close()
    print("yes Basic environment working!")