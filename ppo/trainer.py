"""
PPO Trainer for handling training loops and environment interaction.
"""

import gymnasium as gym
import numpy as np
import torch
import time
from tqdm import tqdm
from typing import Optional, Dict, Any
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from .agent import PPOAgent
from .memory import PPOMemory, EpisodeBuffer
from .utils import normalize


class PPOTrainer:
    """
    PPO Trainer for managing training loops and environment interaction.
    """
    
    def __init__(self, agent: PPOAgent, env_name: str = None, env: gym.Env = None,
                 config: Dict[str, Any] = None):
        """
        Initialize PPO Trainer.
        
        Args:
            agent: PPO agent instance
            env_name: Name of the environment (if env is not provided)
            env: Environment instance (if env_name is not provided)
            config: Configuration dictionary
        """
        self.agent = agent
        
        # Create or use environment
        if env is not None:
            self.env = env
        elif env_name is not None:
            self.env = gym.make(env_name)
        else:
            raise ValueError("Either env_name or env must be provided")
        
        # Configuration
        self.config = config or {}
        self.gamma = self.config.get('gamma', 0.99)
        self.gae_lambda = self.config.get('gae_lambda', 0.95)
        self.clip_epsilon = self.config.get('clip_epsilon', 0.2)
        self.value_coef = self.config.get('value_coef', 0.5)
        self.entropy_coef = self.config.get('entropy_coef', 0.01)
        self.max_grad_norm = self.config.get('max_grad_norm', 0.5)
        self.batch_size = self.config.get('batch_size', 64)
        self.num_epochs = self.config.get('num_epochs', 10)
        self.buffer_size = self.config.get('buffer_size', 2048)
        self.target_kl = self.config.get('target_kl', 0.01)
        self.max_episode_length = self.config.get('max_episode_length', 1000)
        
        # Memory and buffers
        self.memory = PPOMemory(buffer_size=self.buffer_size)
        self.episode_buffer = EpisodeBuffer(max_episodes=100)
        
        # Logging
        self.writer = None
        self.log_interval = self.config.get('log_interval', 10)
        self.save_interval = self.config.get('save_interval', 100)
        self.eval_interval = self.config.get('eval_interval', 50)
        
        # Training stats
        self.episode_count = 0
        self.total_steps = 0
        self.best_reward = float('-inf')
        
    def setup_logging(self, log_dir: str = "runs/ppo"):
        """Setup TensorBoard logging."""
        self.writer = SummaryWriter(log_dir)
    
    def collect_episode(self, deterministic: bool = False) -> Dict[str, Any]:
        """
        Collect a single episode.
        
        Args:
            deterministic: Whether to use deterministic action selection
            
        Returns:
            episode_data: Dictionary containing episode information
        """
        state, _ = self.env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        truncated = False
        
        while not (done or truncated) and episode_length < self.max_episode_length:
            # Get action from agent
            action, log_prob, value = self.agent.get_action(state, deterministic)
            
            # Take action in environment
            next_state, reward, done, truncated, _ = self.env.step(action)
            
            # Store experience
            self.memory.store(state, action, reward, value, log_prob, done)
            
            # Update state and counters
            state = next_state
            episode_reward += reward
            episode_length += 1
            self.total_steps += 1
        
        # Store episode data
        episode_data = {
            'reward': episode_reward,
            'length': episode_length,
            'total_reward': episode_reward
        }
        self.episode_buffer.store_episode(episode_data)
        
        return episode_data
    
    def collect_batch(self) -> int:
        """
        Collect a batch of experiences.
        
        Returns:
            num_episodes: Number of episodes collected
        """
        self.memory.reset()
        num_episodes = 0
        
        while len(self.memory) < self.buffer_size:
            episode_data = self.collect_episode()
            num_episodes += 1
            self.episode_count += 1
            
            # Log episode info
            if self.writer and self.episode_count % self.log_interval == 0:
                self.writer.add_scalar('Episode/Reward', episode_data['reward'], self.episode_count)
                self.writer.add_scalar('Episode/Length', episode_data['length'], self.episode_count)
        
        # Compute advantages
        last_state = self.env.reset()[0]
        _, _, last_value = self.agent.get_action(last_state, deterministic=True)
        self.memory.compute_advantages(
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            last_value=last_value
        )
        
        return num_episodes
    
    def update_policy(self) -> Dict[str, float]:
        """
        Update the policy using collected experiences.
        
        Returns:
            loss_info: Dictionary containing loss information
        """
        total_loss_info = {}
        
        for epoch in range(self.num_epochs):
            # Get batch
            batch = self.memory.get_batch(batch_size=self.batch_size)
            
            # Update agent
            loss_info = self.agent.update(
                batch=batch,
                clip_epsilon=self.clip_epsilon,
                value_coef=self.value_coef,
                entropy_coef=self.entropy_coef,
                max_grad_norm=self.max_grad_norm
            )
            
            # Accumulate loss info
            for key, value in loss_info.items():
                if key not in total_loss_info:
                    total_loss_info[key] = []
                total_loss_info[key].append(value)
            
            # Early stopping if KL divergence is too high
            if loss_info['kl_div'] > self.target_kl:
                break
        
        # Average loss info
        avg_loss_info = {key: np.mean(values) for key, values in total_loss_info.items()}
        
        return avg_loss_info
    
    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate the current policy.
        
        Args:
            num_episodes: Number of episodes to evaluate
            
        Returns:
            eval_info: Dictionary containing evaluation metrics
        """
        rewards = []
        lengths = []
        
        for _ in range(num_episodes):
            episode_data = self.collect_episode(deterministic=True)
            rewards.append(episode_data['reward'])
            lengths.append(episode_data['length'])
        
        eval_info = {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards),
            'mean_length': np.mean(lengths)
        }
        
        return eval_info
    
    def train(self, episodes: int = 1000, save_path: str = None):
        """
        Train the agent.
        
        Args:
            episodes: Number of episodes to train for
            save_path: Path to save the best model
        """
        print(f"Starting PPO training for {episodes} episodes...")
        
        # Setup logging if not already done
        if self.writer is None:
            self.setup_logging()
        
        # Training loop
        for episode in tqdm(range(episodes), desc="Training"):
            # Collect batch
            num_episodes = self.collect_batch()
            
            # Update policy
            loss_info = self.update_policy()
            
            # Log training info
            if self.writer:
                for key, value in loss_info.items():
                    self.writer.add_scalar(f'Loss/{key}', value, self.episode_count)
                
                # Log episode stats
                episode_stats = self.episode_buffer.get_episode_stats()
                for key, value in episode_stats.items():
                    self.writer.add_scalar(f'Stats/{key}', value, self.episode_count)
            
            # Evaluation
            if episode % self.eval_interval == 0:
                eval_info = self.evaluate()
                
                if self.writer:
                    for key, value in eval_info.items():
                        self.writer.add_scalar(f'Eval/{key}', value, self.episode_count)
                
                # Save best model
                if save_path and eval_info['mean_reward'] > self.best_reward:
                    self.best_reward = eval_info['mean_reward']
                    self.agent.save(save_path)
                    print(f"New best model saved! Reward: {self.best_reward:.2f}")
            
            # Save checkpoint
            if save_path and episode % self.save_interval == 0:
                checkpoint_path = save_path.replace('.pth', f'_episode_{episode}.pth')
                self.agent.save(checkpoint_path)
        
        print("Training completed!")
        if self.writer:
            self.writer.close()
    
    def plot_training_curves(self, save_path: str = None):
        """
        Plot training curves.
        
        Args:
            save_path: Path to save the plot
        """
        episode_stats = self.episode_buffer.get_episode_stats()
        
        if not episode_stats:
            print("No training data available for plotting.")
            return
        
        # Get recent episodes
        recent_episodes = self.episode_buffer.get_recent_episodes(100)
        rewards = [ep['reward'] for ep in recent_episodes]
        episodes = list(range(len(rewards)))
        
        plt.figure(figsize=(12, 8))
        
        # Plot reward curve
        plt.subplot(2, 2, 1)
        plt.plot(episodes, rewards)
        plt.title('Training Reward')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.grid(True)
        
        # Plot moving average
        plt.subplot(2, 2, 2)
        window = min(20, len(rewards))
        if window > 0:
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            plt.plot(episodes[window-1:], moving_avg)
            plt.title(f'Moving Average Reward (window={window})')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.grid(True)
        
        # Plot reward distribution
        plt.subplot(2, 2, 3)
        plt.hist(rewards, bins=20, alpha=0.7)
        plt.title('Reward Distribution')
        plt.xlabel('Reward')
        plt.ylabel('Frequency')
        plt.grid(True)
        
        # Plot stats
        plt.subplot(2, 2, 4)
        stats_names = ['mean_reward', 'std_reward', 'min_reward', 'max_reward']
        stats_values = [episode_stats.get(name, 0) for name in stats_names]
        plt.bar(stats_names, stats_values)
        plt.title('Training Statistics')
        plt.ylabel('Value')
        plt.xticks(rotation=45)
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        
        plt.close() 