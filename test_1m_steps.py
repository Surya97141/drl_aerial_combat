"""Test that environment can run 1M steps efficiently."""

import numpy as np
import time
import sys
from pathlib import Path

# Add 1_environment folder to path
sys.path.insert(0, str(Path(__file__).parent / "1_environment"))

from base_env import AerialCombatEnv


def test_1m_steps():
    """Run 1M steps and measure performance."""
    print("\n" + "="*60)
    print("Testing 1M Steps Performance")
    print("="*60 + "\n")
    
    env = AerialCombatEnv()
    
    obs, _ = env.reset()
    total_steps = 0
    total_reward = 0.0
    episodes = 0
    
    start_time = time.time()
    
    # Run episodes until 1M steps
    while total_steps < 1_000_000:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_steps += 1
        total_reward += reward
        
        if terminated or truncated:
            episodes += 1
            obs, _ = env.reset()
        
        # Print progress every 100k steps
        if total_steps % 100_000 == 0:
            elapsed = time.time() - start_time
            steps_per_sec = total_steps / elapsed
            avg_reward = total_reward / total_steps
            print(f"Steps: {total_steps:>9,} | "
                  f"Episodes: {episodes:>4,} | "
                  f"Avg Reward: {avg_reward:>7.4f} | "
                  f"Speed: {steps_per_sec:>8.0f} steps/sec")
    
    elapsed = time.time() - start_time
    avg_reward = total_reward / total_steps
    
    print("\n" + "="*60)
    print("yess SUCCESS: 1,000,000 Steps Completed!")
    print("="*60)
    print(f"Total time:        {elapsed:>10.2f} seconds")
    print(f"Total steps:       {total_steps:>10,}")
    print(f"Total episodes:    {episodes:>10,}")
    print(f"Speed:             {1_000_000/elapsed:>10.0f} steps/sec")
    print(f"Avg reward:        {avg_reward:>10.4f}")
    print(f"Avg episode len:   {total_steps/episodes:>10.1f} steps")
    print("="*60 + "\n")
    
    return elapsed, total_steps, avg_reward


if __name__ == "__main__":
    try:
        elapsed, steps, reward = test_1m_steps()
        print("yess Environment can handle production training!")
    except Exception as e:
        print(f"noo Error: {e}")
        import traceback
        traceback.print_exc()