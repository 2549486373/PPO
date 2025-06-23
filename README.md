# PPO (Proximal Policy Optimization)

A PyTorch implementation of the Proximal Policy Optimization (PPO) algorithm for reinforcement learning.

## Features

- Clean, modular implementation of PPO
- Support for both continuous and discrete action spaces
- Configurable hyperparameters
- TensorBoard logging for training visualization
- Multiple environment support (CartPole, LunarLander, etc.)

## Installation

### Option 1: Using Conda (Recommended)

```bash
# Create and activate conda environment
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

# Create agent and trainer
agent = PPOAgent(state_dim=4, action_dim=2, hidden_dim=64)
trainer = PPOTrainer(agent, env_name="CartPole-v1")

# Train the agent
trainer.train(episodes=1000)
```

### Custom Environment
```python
import gymnasium as gym
from ppo.agent import PPOAgent
from ppo.trainer import PPOTrainer

env = gym.make("LunarLander-v2")
agent = PPOAgent(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.n,
    hidden_dim=128
)
trainer = PPOTrainer(agent, env=env)
trainer.train(episodes=2000)
```

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

## Results

The implementation achieves:
- CartPole-v1: ~500 episodes to solve (195+ score for 100 consecutive episodes)
- LunarLander-v2: ~1000-1500 episodes to achieve 200+ score

## License

MIT License