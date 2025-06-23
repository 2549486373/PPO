"""
Test script for PPO implementation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from ppo.agent import PPOAgent
from ppo.trainer import PPOTrainer
from configs.default_config import PPOConfig


def test_agent_creation():
    """Test agent creation and basic functionality."""
    print("Testing agent creation...")
    
    agent = PPOAgent(
        state_dim=4, 
        action_dim=2, 
        hidden_dim=PPOConfig.HIDDEN_DIM,  # Use config value
        num_layers=PPOConfig.NUM_LAYERS,
        learning_rate=PPOConfig.LEARNING_RATE,
        device=PPOConfig.DEVICE
    )
    
    print(f"✅ Agent created successfully")
    print(f"   Device: {agent.device}")
    print(f"   Hidden dim: {PPOConfig.HIDDEN_DIM}")
    print(f"   Num layers: {PPOConfig.NUM_LAYERS}")
    
    return agent


def test_action_selection(agent):
    """Test action selection functionality."""
    print("\nTesting action selection...")
    
    # Test state
    state = np.array([0.1, 0.2, 0.3, 0.4])
    
    # Test deterministic action
    action, log_prob, value = agent.get_action(state, deterministic=True)
    print(f"✅ Deterministic action: {action}, log_prob: {log_prob:.4f}, value: {value:.4f}")
    
    # Test stochastic action
    action, log_prob, value = agent.get_action(state, deterministic=False)
    print(f"✅ Stochastic action: {action}, log_prob: {log_prob:.4f}, value: {value:.4f}")
    
    # Test batch of states
    states = torch.randn(5, 4)  # Batch of 5 states
    actions = torch.randint(0, 2, (5,))  # Random actions
    
    log_probs, values, entropy = agent.evaluate_actions(states, actions)
    print(f"✅ Batch evaluation: log_probs shape: {log_probs.shape}, values shape: {values.shape}")
    print(f"   Mean entropy: {entropy.mean().item():.4f}")


def test_training_step(agent):
    """Test a single training step."""
    print("\nTesting training step...")
    
    # Create dummy batch
    batch_size = 32
    batch = {
        'states': torch.randn(batch_size, 4),
        'actions': torch.randint(0, 2, (batch_size,)),
        'old_log_probs': torch.randn(batch_size),
        'advantages': torch.randn(batch_size),
        'returns': torch.randn(batch_size)
    }
    
    # Test update
    loss_info = agent.update(batch)
    print(f"✅ Training step completed")
    print(f"   Total loss: {loss_info['total_loss']:.4f}")
    print(f"   Policy loss: {loss_info['policy_loss']:.4f}")
    print(f"   Value loss: {loss_info['value_loss']:.4f}")
    print(f"   Current LR: {loss_info['learning_rate']:.2e}")


def test_trainer():
    """Test trainer functionality."""
    print("\nTesting trainer...")
    
    # Create agent
    agent = PPOAgent(
        state_dim=4, 
        action_dim=2, 
        hidden_dim=PPOConfig.HIDDEN_DIM,  # Use config value
        num_layers=PPOConfig.NUM_LAYERS,
        learning_rate=PPOConfig.LEARNING_RATE,
        device=PPOConfig.DEVICE
    )
    
    # Create trainer
    trainer = PPOTrainer(
        agent=agent,
        env_name="CartPole-v1",
        config={
            'gamma': PPOConfig.GAMMA,
            'gae_lambda': PPOConfig.GAE_LAMBDA,
            'clip_epsilon': PPOConfig.CLIP_EPSILON,
            'value_coef': PPOConfig.VALUE_COEF,
            'entropy_coef': PPOConfig.ENTROPY_COEF,
            'max_grad_norm': PPOConfig.MAX_GRAD_NORM,
            'batch_size': 32,  # Smaller for testing
            'num_epochs': 2,   # Smaller for testing
            'buffer_size': 256, # Smaller for testing
            'target_kl': PPOConfig.TARGET_KL,
            'max_episode_length': 500,
            'log_interval': 5,
            'save_interval': 10,
            'eval_interval': 5,
            'print_interval': 5,
            'early_stopping_patience': 0,  # Disable for testing
            'early_stopping_threshold': 0.0
        }
    )
    
    print(f"✅ Trainer created successfully")
    
    # Test episode collection
    print("\nTesting episode collection...")
    episode_data = trainer.collect_episode()
    print(f"✅ Episode collected: reward={episode_data['reward']:.2f}, length={episode_data['length']}")
    
    # Test batch collection
    print("\nTesting batch collection...")
    num_episodes = trainer.collect_batch()
    print(f"✅ Batch collected: {num_episodes} episodes")
    
    # Test policy update
    print("\nTesting policy update...")
    loss_info = trainer.update_policy()
    print(f"✅ Policy updated: total_loss={loss_info['total_loss']:.4f}")
    
    # Test evaluation
    print("\nTesting evaluation...")
    eval_info = trainer.evaluate(num_episodes=3)
    print(f"✅ Evaluation completed: mean_reward={eval_info['mean_reward']:.2f}")


def test_save_load():
    """Test model saving and loading."""
    print("\nTesting save/load functionality...")
    
    # Create agent
    agent = PPOAgent(
        state_dim=4, 
        action_dim=2, 
        hidden_dim=PPOConfig.HIDDEN_DIM,  # Use config value
        num_layers=PPOConfig.NUM_LAYERS,
        learning_rate=PPOConfig.LEARNING_RATE,
        device=PPOConfig.DEVICE
    )
    
    # Save model
    save_path = "test_model.pth"
    agent.save(save_path)
    print(f"✅ Model saved to {save_path}")
    
    # Create new agent and load
    new_agent = PPOAgent(
        state_dim=4, 
        action_dim=2, 
        hidden_dim=PPOConfig.HIDDEN_DIM,  # Use config value
        num_layers=PPOConfig.NUM_LAYERS,
        learning_rate=PPOConfig.LEARNING_RATE,
        device=PPOConfig.DEVICE
    )
    
    new_agent.load(save_path)
    print(f"✅ Model loaded from {save_path}")
    
    # Clean up
    if os.path.exists(save_path):
        os.remove(save_path)
        print(f"✅ Test file cleaned up")


def main():
    """Run all tests."""
    print("🧪 Running PPO tests...")
    print("=" * 50)
    
    try:
        # Test agent creation
        agent = test_agent_creation()
        
        # Test action selection
        test_action_selection(agent)
        
        # Test training step
        test_training_step(agent)
        
        # Test trainer
        test_trainer()
        
        # Test save/load
        test_save_load()
        
        print("\n🎉 All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 