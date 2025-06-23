"""
Quick Start: Basic PPO training example.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import gymnasium as gym
import torch
from ppo.agent import PPOAgent
from ppo.trainer import PPOTrainer


def quick_start():
    """Quick start example for PPO training."""
    
    print("🚀 PPO Quick Start Example")
    print("=" * 40)
    
    # Auto-detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
    
    # Create environment
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    print(f"Environment: CartPole-v1")
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    
    # Create PPO agent
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=64,
        learning_rate=3e-4,
        device=device
    )
    
    # Create trainer
    trainer = PPOTrainer(
        agent=agent,
        env=env,
        config={
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_epsilon': 0.2,
            'batch_size': 64,
            'num_epochs': 10,
            'buffer_size': 2048
        }
    )
    
    # Setup logging
    trainer.setup_logging("runs/quick_start")
    
    print("\n🎯 Starting training...")
    
    # Train for a short time
    trainer.train(episodes=100, save_path="models/quick_start_best.pth")
    
    # Evaluate the trained agent
    print("\n📊 Evaluating trained agent...")
    eval_info = trainer.evaluate(num_episodes=10)
    
    print("\n📈 Final Results:")
    for key, value in eval_info.items():
        print(f"  {key}: {value:.2f}")
    
    # Plot training curves
    trainer.plot_training_curves("plots/quick_start_training.png")
    
    # Visualize the trained agent
    print("\n🎬 Visualizing trained agent...")
    viz_results = trainer.visualize_agent(num_episodes=3, render=True)
    
    print("\n✅ Training completed!")
    print("Check the 'runs/quick_start' folder for TensorBoard logs")
    print("Check the 'plots/quick_start_training.png' for training curves")
    print("You just watched the trained agent balance the pole!")
    
    env.close()


if __name__ == "__main__":
    quick_start() 