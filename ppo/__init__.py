"""
PPO (Proximal Policy Optimization) Implementation

A clean, modular implementation of the PPO algorithm for reinforcement learning.
"""

from .agent import PPOAgent
from .trainer import PPOTrainer
from .memory import PPOMemory
from .utils import compute_gae, normalize

__version__ = "1.0.0"
__all__ = ["PPOAgent", "PPOTrainer", "PPOMemory", "compute_gae", "normalize"] 