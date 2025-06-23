"""
Simple test script to verify PPO implementation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import gymnasium as gym
import torch
import numpy as np

from ppo.agent import PPOAgent
from ppo.trainer import PPOTrainer
from ppo.memory import PPOMemory
from ppo.utils import compute_gae, normalize


def test_agent():
    """Test PPO agent creation and action selection."""
    print("Testing PPO Agent...")
    
    # Create agent
    agent = PPOAgent(state_dim=4, action_dim=2, hidden_dim=64)
    
    # Test action selection
    state = np.random.randn(4)
    action, log_prob, value = agent.get_action(state)
    
    print(f"State shape: {state.shape}")
    print(f"Action: {action}")
    print(f"Log probability: {log_prob:.4f}")
    print(f"Value: {value:.4f}")
    
    assert isinstance(action, int)
    assert isinstance(log_prob, float)
    assert isinstance(value, float)
    print("✓ Agent test passed!")


def test_memory():
    """Test memory buffer functionality."""
    print("\nTesting Memory Buffer...")
    
    memory = PPOMemory(buffer_size=100)
    
    # Store some experiences
    for i in range(10):
        memory.store(
            state=np.random.randn(4),
            action=np.random.randint(0, 2),
            reward=np.random.randn(),
            value=np.random.randn(),
            log_prob=np.random.randn(),
            done=(i == 9)
        )
    
    # Compute advantages
    memory.compute_advantages()
    
    # Get batch
    batch = memory.get_batch(batch_size=5)
    
    print(f"Memory size: {len(memory)}")
    print(f"Batch keys: {list(batch.keys())}")
    print(f"States shape: {batch['states'].shape}")
    print(f"Actions shape: {batch['actions'].shape}")
    
    assert len(memory) == 10
    assert batch['states'].shape[0] == 5
    print("✓ Memory test passed!")


def test_trainer():
    """Test trainer initialization."""
    print("\nTesting PPO Trainer...")
    
    # Create environment
    env = gym.make("CartPole-v1")
    
    # Create agent
    agent = PPOAgent(state_dim=4, action_dim=2, hidden_dim=64)
    
    # Create trainer
    trainer = PPOTrainer(agent=agent, env=env)
    
    print(f"Environment: {env}")
    print(f"Agent: {type(agent)}")
    print(f"Trainer: {type(trainer)}")
    
    # Test episode collection
    episode_data = trainer.collect_episode()
    print(f"Episode reward: {episode_data['reward']}")
    print(f"Episode length: {episode_data['length']}")
    
    assert episode_data['reward'] >= 0
    assert episode_data['length'] > 0
    print("✓ Trainer test passed!")
    
    env.close()


def test_utils():
    """Test utility functions."""
    print("\nTesting Utility Functions...")
    
    # Test GAE computation
    rewards = np.array([1.0, 0.5, -0.5, 1.0])
    values = np.array([0.1, 0.2, 0.3, 0.4])
    dones = np.array([False, False, False, True])
    
    advantages, returns = compute_gae(rewards, values, dones)
    
    print(f"Advantages shape: {advantages.shape}")
    print(f"Returns shape: {returns.shape}")
    
    assert advantages.shape == rewards.shape
    assert returns.shape == rewards.shape
    print("✓ Utils test passed!")


def main():
    """Run all tests."""
    print("Running PPO Implementation Tests...\n")
    
    try:
        test_agent()
        test_memory()
        test_trainer()
        test_utils()
        
        print("\n🎉 All tests passed! PPO implementation is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        raise


if __name__ == "__main__":
    main() 