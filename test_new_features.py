"""
Test script to demonstrate the new PPO features:
- Print rewards every 100 epochs
- Save model every 100 epochs  
- Early stopping functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from ppo.agent import PPOAgent
from ppo.trainer import PPOTrainer
from configs.default_config import PPOConfig


def test_new_features():
    """Test the new features with a simple CartPole environment."""
    
    print("🧪 Testing new PPO features...")
    print("Features to test:")
    print("  ✅ Print rewards every 100 epochs")
    print("  ✅ Save model every 100 epochs")
    print("  ✅ Early stopping disabled")
    
    # Environment parameters
    env_name = "CartPole-v1"
    state_dim = 4
    action_dim = 2
    
    # Create agent
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=PPOConfig.HIDDEN_DIM,  # Use config value
        num_layers=PPOConfig.NUM_LAYERS,
        learning_rate=PPOConfig.LEARNING_RATE,
        device="cpu"  # Use CPU for consistent testing
    )
    
    # Create trainer with custom config for testing
    trainer = PPOTrainer(
        agent=agent,
        env_name=env_name,
        config={
            'gamma': PPOConfig.GAMMA,
            'gae_lambda': PPOConfig.GAE_LAMBDA,
            'clip_epsilon': PPOConfig.CLIP_EPSILON,
            'value_coef': PPOConfig.VALUE_COEF,
            'entropy_coef': PPOConfig.ENTROPY_COEF,
            'max_grad_norm': PPOConfig.MAX_GRAD_NORM,
            'batch_size': 32,  # Smaller batch for faster testing
            'num_epochs': 5,
            'buffer_size': 512,  # Smaller buffer for faster testing
            'target_kl': PPOConfig.TARGET_KL,
            'max_episode_length': 500,
            'log_interval': 5,
            'save_interval': 50,  # Save every 50 episodes for testing
            'eval_interval': 15,  # Reduced from 25 for more frequent evaluation
            'print_interval': 25,  # Print every 25 episodes for testing
            'early_stopping_patience': 0,  # Disabled early stopping for testing
            'early_stopping_threshold': 0.0  # Not used when early stopping is disabled
        }
    )
    
    # Setup logging
    trainer.setup_logging("runs/test_new_features")
    
    # Create directories for saving
    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    print(f"\n🚀 Starting training...")
    print(f"   - Will print rewards every 25 episodes")
    print(f"   - Will save model every 50 episodes")
    print(f"   - Early stopping disabled")
    
    # Train the agent
    trainer.train(episodes=500, save_path="models/test_cartpole.pth")
    
    # Plot training curves
    trainer.plot_training_curves("plots/test_training.png")
    
    # Final evaluation
    print("\n📊 Final evaluation:")
    eval_info = trainer.evaluate(num_episodes=10)
    for key, value in eval_info.items():
        print(f"   {key}: {value:.2f}")
    
    print(f"\n✅ Test completed!")
    print(f"   Check the following files:")
    print(f"   - models/test_cartpole.pth (best model)")
    print(f"   - models/test_cartpole_episode_*.pth (checkpoint models)")
    print(f"   - plots/test_training.png (training curves)")
    print(f"   - runs/test_new_features/ (tensorboard logs)")


if __name__ == "__main__":
    test_new_features() 