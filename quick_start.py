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
from configs.default_config import PPOConfig


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
        hidden_dim=PPOConfig.HIDDEN_DIM,
        num_layers=PPOConfig.NUM_LAYERS,
        learning_rate=PPOConfig.LEARNING_RATE,
        lr_schedule=PPOConfig.LR_SCHEDULE,
        device=PPOConfig.DEVICE
    )
    
    # Create trainer
    trainer = PPOTrainer(
        agent=agent,
        env=env,
        config={
            'gamma': PPOConfig.GAMMA,
            'gae_lambda': PPOConfig.GAE_LAMBDA,
            'clip_epsilon': PPOConfig.CLIP_EPSILON,
            'value_coef': PPOConfig.VALUE_COEF,
            'entropy_coef': PPOConfig.ENTROPY_COEF,
            'max_grad_norm': PPOConfig.MAX_GRAD_NORM,
            'batch_size': PPOConfig.BATCH_SIZE,
            'num_epochs': PPOConfig.NUM_EPOCHS,
            'buffer_size': PPOConfig.BUFFER_SIZE,
            'target_kl': PPOConfig.TARGET_KL,
            'max_episode_length': PPOConfig.MAX_EPISODE_LENGTH,
            'log_interval': PPOConfig.LOG_INTERVAL,
            'save_interval': PPOConfig.SAVE_INTERVAL,
            'eval_interval': PPOConfig.EVAL_INTERVAL,
            'print_interval': PPOConfig.PRINT_INTERVAL,
            'early_stopping_patience': PPOConfig.EARLY_STOPPING_PATIENCE,
            'early_stopping_threshold': PPOConfig.EARLY_STOPPING_THRESHOLD
        }
    )
    
    # Setup logging
    trainer.setup_logging("runs/quick_start")
    
    print("\n🎯 Starting training...")
    
    # Train for a short time
    trainer.train(episodes=500, save_path="models/quick_start_best.pth")
    
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