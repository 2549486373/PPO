"""
Test script to demonstrate epoch-based plotting.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from ppo.agent import PPOAgent
from ppo.trainer import PPOTrainer
from configs.default_config import PPOConfig


def test_epoch_plotting():
    """Test the epoch-based plotting with a simple CartPole environment."""
    
    print("🧪 Testing Epoch-Based Plotting")
    print("=" * 50)
    
    # Environment parameters
    env_name = "CartPole-v1"
    state_dim = 4
    action_dim = 2
    
    # Create agent
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=PPOConfig.HIDDEN_DIM,
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
            'num_epochs': 5,   # 5 epochs per episode for testing
            'buffer_size': 512,  # Smaller buffer for faster testing
            'target_kl': PPOConfig.TARGET_KL,
            'max_episode_length': 500,
            'log_interval': 5,
            'save_interval': 20,  # Save every 20 episodes
            'eval_interval': 10,  # Evaluate every 10 episodes
            'print_interval': 10,  # Print every 10 episodes
            'early_stopping_patience': 0,  # Disabled early stopping
            'early_stopping_threshold': 0.0
        }
    )
    
    # Setup logging
    trainer.setup_logging("runs/test_epoch_plotting")
    
    # Create directories for saving
    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    print(f"\n🚀 Starting training...")
    print(f"   - Will train for 50 episodes")
    print(f"   - 5 epochs per episode = 250 total epochs")
    print(f"   - Will print every 10 episodes (epochs 50, 100, 150, 200, 250)")
    print(f"   - X-axis in plots will show epochs (0, 5, 10, 15, ..., 250)")
    
    # Train the agent
    trainer.train(episodes=50, save_path="models/test_epoch_plotting.pth")
    
    # Get epoch information
    epoch_info = trainer.get_epoch_info()
    print(f"\n📊 Epoch Information:")
    print(f"   Total epochs completed: {epoch_info['total_epochs']}")
    print(f"   Total episodes completed: {epoch_info['total_episodes']}")
    print(f"   Epochs per episode: {epoch_info['epochs_per_episode']}")
    
    # Plot training curves with epoch-based x-axis
    trainer.plot_training_curves("plots/test_epoch_plotting.png")
    
    # Final evaluation
    print("\n📊 Final evaluation:")
    eval_info = trainer.evaluate(num_episodes=10)
    for key, value in eval_info.items():
        print(f"   {key}: {value:.2f}")
    
    print(f"\n✅ Test completed!")
    print(f"Check plots/test_epoch_plotting.png for epoch-based training curves")


if __name__ == "__main__":
    test_epoch_plotting() 