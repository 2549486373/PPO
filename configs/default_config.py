"""
Default configuration for PPO training.
"""

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
    HIDDEN_DIM = 64
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
    
    # Device
    DEVICE = "cpu"  # Change to "cuda" if GPU available 