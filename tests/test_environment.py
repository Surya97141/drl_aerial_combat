"""Unit tests for aerial combat environment."""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add 1_environment folder to path
sys.path.insert(0, str(Path(__file__).parent.parent / "1_environment"))

# Now import from base_env
from base_env import AerialCombatEnv


class TestAerialCombatEnv:
    """Test suite for AerialCombatEnv."""
    
    def setup_method(self):
        """Set up test environment."""
        self.env = AerialCombatEnv(max_steps=100)
    
    def test_env_creation(self):
        """Test that environment can be created."""
        assert self.env is not None
        assert self.env.action_space is not None
        assert self.env.observation_space is not None
    
    def test_reset(self):
        """Test reset functionality."""
        obs, info = self.env.reset()
        assert obs.shape == (12,)
        assert isinstance(info, dict)
        assert self.env.current_step == 0
    
    def test_action_space(self):
        """Test action space properties."""
        assert self.env.action_space.shape == (4,)
        assert self.env.action_space.dtype == np.float32
        
        # Test sampling
        action = self.env.action_space.sample()
        assert action.shape == (4,)
        assert np.all(action >= -1.0) and np.all(action <= 1.0)
    
    def test_observation_space(self):
        """Test observation space properties."""
        assert self.env.observation_space.shape == (12,)
        assert self.env.observation_space.dtype == np.float32
    
    def test_step(self):
        """Test step functionality."""
        obs, _ = self.env.reset()
        action = np.array([0.0, 0.0, 0.0, 0.0])
        
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        assert obs.shape == (12,)
        assert isinstance(reward, (float, np.floating))
        assert isinstance(terminated, (bool, np.bool_))
        assert isinstance(truncated, (bool, np.bool_))
        assert isinstance(info, dict)
        assert self.env.current_step == 1
    
    def test_multiple_steps(self):
        """Test multiple steps."""
        obs, _ = self.env.reset()
        
        for _ in range(50):
            action = self.env.action_space.sample()
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            if terminated or truncated:
                break
        
        assert self.env.current_step <= 50
    
    def test_max_steps_truncation(self):
        """Test that episode truncates at max_steps."""
        env = AerialCombatEnv(max_steps=10)
        obs, _ = env.reset()
        
        for _ in range(15):
            action = np.array([0.0, 0.0, 0.0, 0.0])
            obs, reward, terminated, truncated, info = env.step(action)
            
            if truncated:
                break
        
        assert env.current_step == 10
        assert truncated is True
    
    def test_observation_bounds(self):
        """Test that observations are reasonable."""
        obs, _ = self.env.reset()
        
        # Run several steps
        for _ in range(50):
            action = self.env.action_space.sample()
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            # Check no NaN or Inf
            assert not np.any(np.isnan(obs))
            assert not np.any(np.isinf(obs))
            
            if terminated or truncated:
                break
    
    def test_reproducibility(self):
        """Test that environment is reproducible with seed."""
        env1 = AerialCombatEnv()
        env2 = AerialCombatEnv()
        
        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)
        
        np.testing.assert_array_equal(obs1, obs2)
    
    def test_close(self):
        """Test that environment can be closed."""
        self.env.close()  # Should not raise
    
    def test_reward_is_scalar(self):
        """Test that reward is a scalar."""
        obs, _ = self.env.reset()
        action = self.env.action_space.sample()
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        assert isinstance(reward, (float, np.floating, np.ndarray))
        if isinstance(reward, np.ndarray):
            assert reward.shape == ()
    
    def test_long_episode(self):
        """Test that environment runs for full episode."""
        env = AerialCombatEnv(max_steps=100)
        obs, _ = env.reset()
        
        steps = 0
        while True:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            
            if terminated or truncated:
                break
        
        assert steps > 0
        assert steps <= 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])