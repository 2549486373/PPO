"""
Example: Training PPO on LunarLander-v2 environment.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from ppo.agent import PPOAgent
from ppo.trainer import PPOTrainer
from configs.default_config import PPOConfig


def main():
    """Main training function for LunarLander."""
    
    # Environment parameters
    env_name = "LunarLander-v2"
    state_dim = 8
    action_dim = 4
    
    # Show device information
    device = PPOConfig.DEVICE
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
    
    # Create agent
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=PPOConfig.HIDDEN_DIM,
        num_layers=PPOConfig.NUM_LAYERS,
        learning_rate=PPOConfig.LEARNING_RATE,
        device=device
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
            'eval_interval': PPOConfig.EVAL_INTERVAL
        }
    )
    
    # Setup logging
    trainer.setup_logging("runs/ppo_lunar_lander")
    
    # Train the agent
    print("Starting PPO training on LunarLander-v2...")
    trainer.train(episodes=1000, save_path="models/lunar_lander_best.pth")
    
    # Plot training curves
    trainer.plot_training_curves("plots/lunar_lander_training.png")
    
    # Final evaluation
    print("\nFinal evaluation:")
    eval_info = trainer.evaluate(num_episodes=20)
    for key, value in eval_info.items():
        print(f"{key}: {value:.2f}")


if __name__ == "__main__":
    main() 