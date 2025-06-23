"""
Experience replay buffer for PPO training.
"""

import numpy as np
import torch
from collections import deque


class PPOMemory:
    """
    Memory buffer for storing PPO training experiences.
    """
    
    def __init__(self, buffer_size=2048):
        """
        Initialize PPO memory buffer.
        
        Args:
            buffer_size: Maximum size of the buffer
        """
        self.buffer_size = buffer_size
        self.reset()
    
    def reset(self):
        """Reset the memory buffer."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.advantages = []
        self.returns = []
    
    def store(self, state, action, reward, value, log_prob, done):
        """
        Store a single experience in the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            value: Value estimate
            log_prob: Log probability of the action
            done: Whether episode is done
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def compute_advantages(self, gamma=0.99, gae_lambda=0.95, last_value=0):
        """
        Compute advantages using GAE.
        
        Args:
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            last_value: Value estimate for the last state
        """
        rewards = np.array(self.rewards)
        values = np.array(self.values + [last_value])
        dones = np.array(self.dones + [True])
        
        advantages = np.zeros_like(rewards)
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]
                
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = delta + gamma * gae_lambda * (1 - dones[t]) * last_advantage
            last_advantage = advantages[t]
        
        returns = advantages + values[:-1]
        
        self.advantages = advantages
        self.returns = returns
    
    def get_batch(self, batch_size=None):
        """
        Get a batch of experiences for training.
        
        Args:
            batch_size: Size of the batch (if None, returns all data)
            
        Returns:
            Dictionary containing batched data
        """
        if batch_size is None:
            batch_size = len(self.states)
        
        # Sample random indices
        indices = np.random.choice(len(self.states), batch_size, replace=False)
        
        batch = {
            'states': torch.FloatTensor(np.array(self.states)[indices]),
            'actions': torch.LongTensor(np.array(self.actions)[indices]),
            'old_log_probs': torch.FloatTensor(np.array(self.log_probs)[indices]),
            'advantages': torch.FloatTensor(self.advantages[indices]),
            'returns': torch.FloatTensor(self.returns[indices])
        }
        
        return batch
    
    def __len__(self):
        """Return the number of experiences in the buffer."""
        return len(self.states)


class EpisodeBuffer:
    """
    Buffer for storing complete episodes.
    """
    
    def __init__(self, max_episodes=100):
        """
        Initialize episode buffer.
        
        Args:
            max_episodes: Maximum number of episodes to store
        """
        self.max_episodes = max_episodes
        self.episodes = deque(maxlen=max_episodes)
    
    def store_episode(self, episode_data):
        """
        Store a complete episode.
        
        Args:
            episode_data: Dictionary containing episode data
        """
        self.episodes.append(episode_data)
    
    def get_recent_episodes(self, num_episodes):
        """
        Get the most recent episodes.
        
        Args:
            num_episodes: Number of episodes to retrieve
            
        Returns:
            List of episode data
        """
        return list(self.episodes)[-num_episodes:]
    
    def get_episode_stats(self):
        """
        Get statistics about stored episodes.
        
        Returns:
            Dictionary containing episode statistics
        """
        if not self.episodes:
            return {}
        
        rewards = [ep['total_reward'] for ep in self.episodes]
        lengths = [ep['length'] for ep in self.episodes]
        
        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards),
            'mean_length': np.mean(lengths),
            'num_episodes': len(self.episodes)
        } 