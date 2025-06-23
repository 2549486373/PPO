# PPO (Proximal Policy Optimization)

A PyTorch implementation of the Proximal Policy Optimization (PPO) algorithm for reinforcement learning.

## Features

- Clean, modular implementation of PPO
- Support for both continuous and discrete action spaces
- Configurable hyperparameters
- TensorBoard logging for training visualization
- Multiple environment support (CartPole, LunarLander, etc.)
- **Reward printing every N epochs** for monitoring training progress
- **Automatic model saving every N epochs** for checkpointing
- **Early stopping** with configurable patience and improvement threshold
- **Learning rate annealing** with multiple schedule types (linear, cosine, exponential, step, cosine warm restart)
- Training visualization and plotting tools

## Installation

### Option 1: Using conda

```bash
conda env create -f environment.yml
conda activate ppo
```

### Option 2: Using pip

```bash
pip install -r requirements.txt
```

## Usage

### Basic Training
```python
from ppo.agent import PPOAgent
from ppo.trainer import PPOTrainer
from configs.default_config import PPOConfig

# Create agent and trainer
agent = PPOAgent(
    state_dim=4, 
    action_dim=2, 
    hidden_dim=PPOConfig.HIDDEN_DIM
)
trainer = PPOTrainer(agent, env_name="CartPole-v1")

# Train the agent
trainer.train(episodes=500)
```

### Advanced Training with New Features
```python
from ppo.agent import PPOAgent
from ppo.trainer import PPOTrainer
from configs.default_config import PPOConfig

# Create agent with learning rate scheduling
agent = PPOAgent(
    state_dim=4, 
    action_dim=2, 
    hidden_dim=PPOConfig.HIDDEN_DIM,
    learning_rate=PPOConfig.LEARNING_RATE,
    lr_schedule=PPOConfig.LR_SCHEDULE,  # Options: "linear", "cosine", "exponential", "step", "cosine_warm_restart", "none"
    lr_schedule_kwargs={
        'total_iters': 1000,  # For cosine/linear schedules
        'eta_min': 1e-6,      # Minimum LR for cosine
        'gamma': 0.99,        # Decay factor for exponential
        'step_size': 100,     # Step size for step decay
    }
)

# Create trainer with custom configuration
trainer = PPOTrainer(
    agent=agent,
    env_name="CartPole-v1",
    config={
        'print_interval': 100,        # Print rewards every 100 episodes
        'save_interval': 100,         # Save model every 100 episodes
        'early_stopping_patience': 0, # Early stopping disabled
        'early_stopping_threshold': 0.0 # Not used when early stopping is disabled
    }
)

# Train with automatic saving and early stopping
trainer.train(episodes=1000, save_path="models/cartpole_best.pth")
```

### Learning Rate Schedules

The implementation supports several learning rate scheduling strategies:

#### 1. Linear Decay
```python
agent = PPOAgent(
    state_dim=4,
    action_dim=2,
    hidden_dim=PPOConfig.HIDDEN_DIM,
    lr_schedule="linear",
    lr_schedule_kwargs={
        'total_iters': 1000,  # Total iterations for decay
        'end_factor': 0.0     # Final LR factor (0.0 = decay to 0)
    }
)
```

#### 2. Cosine Annealing
```python
agent = PPOAgent(
    state_dim=4,
    action_dim=2,
    hidden_dim=PPOConfig.HIDDEN_DIM,
    lr_schedule="cosine",
    lr_schedule_kwargs={
        'total_iters': 1000,  # Total iterations
        'eta_min': 1e-6       # Minimum learning rate
    }
)
```

#### 3. Exponential Decay
```python
agent = PPOAgent(
    state_dim=4,
    action_dim=2,
    hidden_dim=PPOConfig.HIDDEN_DIM,
    lr_schedule="exponential",
    lr_schedule_kwargs={
        'gamma': 0.99  # Decay factor per step
    }
)
```

#### 4. Step Decay
```python
agent = PPOAgent(
    state_dim=4,
    action_dim=2,
    hidden_dim=PPOConfig.HIDDEN_DIM,
    lr_schedule="step",
    lr_schedule_kwargs={
        'step_size': 100,  # Decay every N steps
        'gamma': 0.5       # Multiply LR by this factor
    }
)
```

#### 5. Cosine Annealing with Warm Restarts
```python
agent = PPOAgent(
    state_dim=4,
    action_dim=2,
    hidden_dim=PPOConfig.HIDDEN_DIM,
    lr_schedule="cosine_warm_restart",
    lr_schedule_kwargs={
        'T_0': 100,     # Initial restart period
        'T_mult': 2,    # Period multiplier after each restart
        'eta_min': 1e-6 # Minimum learning rate
    }
)
```

### Custom Environment
```python
import gymnasium as gym
from ppo.agent import PPOAgent
from ppo.trainer import PPOTrainer
from configs.default_config import PPOConfig

env = gym.make("LunarLander-v2")
agent = PPOAgent(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.n,
    hidden_dim=PPOConfig.HIDDEN_DIM
)
trainer = PPOTrainer(agent, env=env)
trainer.train(episodes=2000)
```

## New Features

### 1. Reward Printing
The trainer now prints reward statistics every N episodes (configurable via `print_interval`):
```
Episode 100: Avg Reward (last 50): 45.23, Best Reward: 195.00, LR: 3.00e-04
Episode 200: Avg Reward (last 50): 67.89, Best Reward: 195.00, LR: 2.40e-04
```

### 2. Automatic Model Saving
Models are automatically saved every N episodes (configurable via `save_interval`):
- Checkpoint models: `model_episode_100.pth`, `model_episode_200.pth`, etc.
- Best model: Saved whenever a new best reward is achieved

### 3. Early Stopping
Early stopping is currently disabled (patience = 0). When enabled, training automatically stops when no improvement is seen for a specified number of evaluations:
- `early_stopping_patience`: Number of evaluations without improvement
- `early_stopping_threshold`: Minimum improvement required to reset patience
- Helps prevent overfitting and saves training time

### 4. Learning Rate Annealing
Configurable learning rate schedules to improve training stability and convergence:
- **Linear**: Gradual linear decay from initial to final learning rate
- **Cosine**: Smooth cosine annealing with configurable minimum
- **Exponential**: Exponential decay with configurable decay factor
- **Step**: Step-wise decay at regular intervals
- **Cosine Warm Restart**: Cosine annealing with periodic restarts
- **None**: Constant learning rate (no decay)

## Project Structure

```
PPO/
├── ppo/
│   ├── __init__.py
│   ├── agent.py          # PPO agent with actor-critic networks
│   ├── memory.py         # Experience replay buffer
│   ├── trainer.py        # Training loop and logic
│   └── utils.py          # Utility functions
├── configs/
│   └── default_config.py # Default hyperparameters
├── examples/
│   ├── cartpole_example.py
│   └── lunar_lander_example.py
├── test_new_features.py  # Test script for new features
├── test_lr_schedules.py  # Test script for learning rate schedules
├── environment.yml       # Conda environment file
├── requirements.txt      # pip requirements (alternative)
└── README.md
```

## Key Components

1. **PPOAgent**: Implements the actor-critic architecture with separate policy and value networks
2. **PPOMemory**: Manages experience storage and sampling for training
3. **PPOTrainer**: Handles the training loop, environment interaction, and logging
4. **Config**: Centralized hyperparameter management

## Hyperparameters

- Learning rate: 3e-4
- Clipping parameter (ε): 0.2
- Value function coefficient: 0.5
- Entropy coefficient: 0.01
- Number of epochs per update: 10
- Batch size: 64
- Discount factor (γ): 0.99
- GAE lambda: 0.95
- Print interval: 100 episodes
- Save interval: 100 episodes
- Early stopping: Disabled (patience = 0)
- Learning rate schedule: cosine (configurable)
- LR schedule parameters: See `configs/default_config.py`

## Testing Features

### Test New Features:
```bash
python test_new_features.py
```

### Test Learning Rate Schedules:
```bash
python test_lr_schedules.py
```

This will compare different learning rate schedules and generate visualization plots.

## Results

The implementation achieves:
- CartPole-v1: ~500 episodes to solve (195+ score for 100 consecutive episodes)
- LunarLander-v2: ~1000-1500 episodes to achieve 200+ score
