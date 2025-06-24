"""
Default configuration for PPO training.
"""

import torch

def get_lr_schedule_kwargs(num_episodes):
    """
    Get learning rate schedule parameters based on number of episodes.
    
    Args:
        num_episodes: Number of episodes for training
        
    Returns:
        Dictionary of learning rate schedule parameters
    """
    return {
        # Linear schedule parameters
        'total_iters': num_episodes,  # Total episodes for linear decay
        'end_factor': 0.0,    # Final learning rate factor (0.0 = decay to 0)
        
        # Cosine schedule parameters
        'eta_min': 1e-6,      # Minimum learning rate for cosine annealing
        
        # Exponential schedule parameters
        'gamma': 0.99,        # Decay factor for exponential decay
        
        # Step schedule parameters
        'step_size': max(1, num_episodes // 5),  # Step size for step decay (every 20% of episodes)
        'gamma': 0.1,         # Decay factor for step decay
        
        # Cosine warm restart parameters
        'T_0': max(1, num_episodes // 5),  # Initial restart period (every 20% of episodes)
        'T_mult': 2,          # Period multiplier after each restart
    }

class PPOConfig:
    """Default hyperparameters for PPO algorithm."""
    
    # Training parameters
    LEARNING_RATE = 3e-4
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    CLIP_EPSILON = 0.2
    VALUE_COEF = 0.5
    ENTROPY_COEF = 0.01
    MAX_GRAD_NORM = 0.5
    
    # Network parameters
    HIDDEN_DIM = 128
    NUM_LAYERS = 2
    ACTIVATION = "tanh"
    
    # Training loop parameters
    BATCH_SIZE = 64
    NUM_EPOCHS = 10
    BUFFER_SIZE = 2048
    TARGET_KL = 0.01
    
    # Environment parameters
    MAX_EPISODE_LENGTH = 1000
    NUM_ENV_STEPS = 2048
    
    # Logging parameters
    LOG_INTERVAL = 10
    SAVE_INTERVAL = 100
    EVAL_INTERVAL = 50
    PRINT_INTERVAL = 100  # Print rewards every N episodes
    
    # Early stopping parameters
    EARLY_STOPPING_PATIENCE = 0  # Disabled early stopping (0 = no early stopping)
    EARLY_STOPPING_THRESHOLD = 0.0  # Not used when early stopping is disabled
    
    # Learning rate scheduling parameters
    LR_SCHEDULE = "cosine"  # Options: "linear", "cosine", "exponential", "step", "cosine_warm_restart", "none"
    
    # Default number of episodes for calculating LR schedule parameters
    DEFAULT_EPISODES = 500
    
    @classmethod
    def get_lr_schedule_kwargs(cls, num_episodes=None):
        """
        Get learning rate schedule parameters.
        
        Args:
            num_episodes: Number of episodes for training. If None, uses DEFAULT_EPISODES.
            
        Returns:
            Dictionary of learning rate schedule parameters
        """
        if num_episodes is None:
            num_episodes = cls.DEFAULT_EPISODES
        return get_lr_schedule_kwargs(num_episodes)
    
    # Device - auto-detect GPU if available
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu" 