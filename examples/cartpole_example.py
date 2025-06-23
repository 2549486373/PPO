"""
Example: Training PPO on CartPole-v1 environment.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from ppo.agent import PPOAgent
from ppo.trainer import PPOTrainer
from configs.default_config import PPOConfig


def main():
    """Main training function for CartPole."""
    
    # Environment parameters
    env_name = "CartPole-v1"
    state_dim = 4
    action_dim = 2
    
    # Show device information
    device = PPOConfig.DEVICE
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
    
    # Create agent with learning rate scheduling
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=PPOConfig.HIDDEN_DIM,
        num_layers=PPOConfig.NUM_LAYERS,
        learning_rate=PPOConfig.LEARNING_RATE,
        device=device,
        lr_schedule=PPOConfig.LR_SCHEDULE,
        lr_schedule_kwargs=PPOConfig.LR_SCHEDULE_KWARGS
    )
    
    # Create trainer
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
    trainer.setup_logging("runs/ppo_cartpole")
    
    # Train the agent
    print("Starting PPO training on CartPole-v1...")
    trainer.train(episodes=500, save_path="models/cartpole_best.pth")
    
    # Plot training curves
    trainer.plot_training_curves("plots/cartpole_training.png")
    
    # Final evaluation
    print("\nFinal evaluation:")
    eval_info = trainer.evaluate(num_episodes=20)
    for key, value in eval_info.items():
        print(f"{key}: {value:.2f}")


if __name__ == "__main__":
    main() 