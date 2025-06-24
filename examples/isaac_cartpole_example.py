"""
Example: Training PPO on Isaac Sim CartPole environment.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from ppo.agent import PPOAgent
from ppo.trainer import PPOTrainer
from ppo.isaac_env import create_isaac_env, ISAAC_AVAILABLE
from configs.default_config import PPOConfig


def main():
    """Main training function for Isaac Sim CartPole."""
    
    if not ISAAC_AVAILABLE:
        print("Isaac Sim is not available. Please install omni-isaac-gym first.")
        print("You can install it using: pip install omni-isaac-gym")
        return
    
    # Environment parameters
    env_name = "cartpole"
    num_envs = 4  # Number of parallel environments
    state_dim = 4
    action_dim = 2  # Discrete actions for CartPole
    
    # Show device information
    device = PPOConfig.DEVICE
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
    
    # Create Isaac Sim environment
    print("Creating Isaac Sim CartPole environment...")
    env = create_isaac_env(
        env_name=env_name,
        num_envs=num_envs,
        device=device,
        headless=True  # Set to False for visualization
    )
    
    print(f"Environment created successfully!")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Create agent with learning rate scheduling
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=PPOConfig.HIDDEN_DIM,
        num_layers=PPOConfig.NUM_LAYERS,
        learning_rate=PPOConfig.LEARNING_RATE,
        device=device,
        lr_schedule=PPOConfig.LR_SCHEDULE
    )
    
    # Create trainer with Isaac Sim environment
    trainer = PPOTrainer(
        agent=agent,
        env=env,  # Pass the Isaac Sim environment directly
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
    trainer.setup_logging("runs/ppo_isaac_cartpole")
    
    # Train the agent
    print("Starting PPO training on Isaac Sim CartPole...")
    trainer.train(episodes=500, save_path="models/isaac_cartpole_best.pth")
    
    # Plot training curves
    trainer.plot_training_curves("plots/isaac_cartpole_training.png")
    
    # Final evaluation
    print("\nFinal evaluation:")
    eval_info = trainer.evaluate(num_episodes=20)
    for key, value in eval_info.items():
        print(f"{key}: {value:.2f}")
    
    # Close environment
    env.close()


if __name__ == "__main__":
    main() 