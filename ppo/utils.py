"""
Utility functions for PPO implementation.
"""

import numpy as np
import torch


def compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95):
    """
    Compute Generalized Advantage Estimation (GAE).
    
    Args:
        rewards: Array of rewards [batch_size]
        values: Array of value estimates [batch_size]
        dones: Array of done flags [batch_size]
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        
    Returns:
        advantages: Computed advantages [batch_size]
        returns: Computed returns [batch_size]
    """
    advantages = np.zeros_like(rewards)
    last_advantage = 0
    last_value = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = last_value
        else:
            next_value = values[t + 1]
            
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        advantages[t] = delta + gamma * gae_lambda * (1 - dones[t]) * last_advantage
        last_advantage = advantages[t]
    
    returns = advantages + values
    return advantages, returns


def normalize(x, eps=1e-8):
    """
    Normalize array to zero mean and unit variance.
    
    Args:
        x: Input array
        eps: Small constant to avoid division by zero
        
    Returns:
        Normalized array
    """
    return (x - x.mean()) / (x.std() + eps)


def compute_entropy(probs):
    """
    Compute entropy of probability distribution.
    
    Args:
        probs: Probability distribution [batch_size, num_actions]
        
    Returns:
        entropy: Entropy values [batch_size]
    """
    return -(probs * torch.log(probs + 1e-8)).sum(dim=-1)


def compute_kl_divergence(old_probs, new_probs):
    """
    Compute KL divergence between two probability distributions.
    
    Args:
        old_probs: Old probability distribution [batch_size, num_actions]
        new_probs: New probability distribution [batch_size, num_actions]
        
    Returns:
        kl_div: KL divergence values [batch_size]
    """
    return (old_probs * torch.log(old_probs / (new_probs + 1e-8) + 1e-8)).sum(dim=-1)


def discount_cumsum(x, discount):
    """
    Compute discounted cumulative sum.
    
    Args:
        x: Input array
        discount: Discount factor
        
    Returns:
        Discounted cumulative sum
    """
    return np.array([sum([x[i] * (discount ** i) for i in range(len(x))])])


def create_log_gaussian(mean, log_std, t):
    """
    Create log probability for Gaussian distribution.
    
    Args:
        mean: Mean of the distribution
        log_std: Log standard deviation
        t: Target value
        
    Returns:
        Log probability
    """
    return -0.5 * (((t - mean) / (log_std.exp() + 1e-8)) ** 2 + 2 * log_std + np.log(2 * np.pi)) 