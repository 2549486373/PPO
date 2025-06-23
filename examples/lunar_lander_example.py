"""
Example: Training PPO on LunarLander-v2 environment.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
from ppo.agent import PPOAgent
from ppo.trainer import PPOTrainer
from configs.default_config import PPOConfig


def main():
    """Main training function for LunarLander."""
    
    # Environment parameters
    env_name = "LunarLander-v2"
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Create agent with larger network for LunarLander
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=128,  # Larger network for more complex environment
        num_layers=3,    # More layers
        learning_rate=PPOConfig.LEARNING_RATE,
        device=PPOConfig.DEVICE
    )
    
    # Create trainer with adjusted config for LunarLander
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
            'batch_size': 128,  # Larger batch size
            'num_epochs': PPOConfig.NUM_EPOCHS,
            'buffer_size': 4096,  # Larger buffer
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
    trainer.train(episodes=2000, save_path="models/lunar_lander_best.pth")
    
    # Plot training curves
    trainer.plot_training_curves("plots/lunar_lander_training.png")
    
    # Final evaluation
    print("\nFinal evaluation:")
    eval_info = trainer.evaluate(num_episodes=20)
    for key, value in eval_info.items():
        print(f"{key}: {value:.2f}")
    
    # Close environment
    env.close()


if __name__ == "__main__":
    main() 