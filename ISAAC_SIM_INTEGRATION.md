# Isaac Sim Integration Guide

This guide explains how to integrate Isaac Sim with your PPO project for high-performance physics simulation.

## Overview

Isaac Sim provides GPU-accelerated physics simulation that can significantly speed up reinforcement learning training. This integration allows you to:

- Use realistic physics simulation for training
- Run multiple environments in parallel on GPU
- Achieve faster training times compared to CPU-based environments
- Use high-fidelity sensor simulation

## Installation

### Prerequisites

1. **NVIDIA GPU** with CUDA support (recommended)
2. **Python 3.8+**
3. **Isaac Sim** (installed via Omniverse Launcher)

### Step 1: Install Isaac Sim

#### Option A: Omniverse Launcher (Recommended)
1. Download and install [Omniverse Launcher](https://www.nvidia.com/en-us/omniverse/)
2. Install Isaac Sim from the Omniverse Launcher
3. Follow the installation instructions provided by NVIDIA

#### Option B: Docker
```bash
docker pull nvcr.io/nvidia/isaac-sim:2023.1.1
docker run --gpus all -it --rm -v $(pwd):/workspace nvcr.io/nvidia/isaac-sim:2023.1.1
```

### Step 2: Install Python Dependencies

#### Using the Installation Script
```bash
python install_isaac_sim.py
```

#### Manual Installation
```bash
# Activate your conda environment
conda activate ppo

# Install Isaac Sim Python packages
pip install omni-isaac-gym>=1.0.0
pip install omni-isaac-sim>=2023.1.0
pip install omni-isaac-gym-envs>=1.0.0

# Install additional dependencies
pip install Pillow>=8.0.0 scipy>=1.7.0
```

### Step 3: Verify Installation

```python
# Test Isaac Sim import
import isaacgym
from isaacgym import gymapi
from isaacgym import gymtorch
print("Isaac Sim installation successful!")
```

## Usage

### Basic Isaac Sim Environment

```python
from ppo.agent import PPOAgent
from ppo.trainer import PPOTrainer
from ppo.isaac_env import create_isaac_env
from configs.default_config import PPOConfig

# Create Isaac Sim environment
env = create_isaac_env(
    env_name="cartpole",
    num_envs=4,
    device="cuda",
    headless=True
)

# Create agent
agent = PPOAgent(
    state_dim=4,
    action_dim=2,
    hidden_dim=PPOConfig.HIDDEN_DIM
)

# Create trainer
trainer = PPOTrainer(agent, env=env)

# Train
trainer.train(episodes=500, save_path="models/isaac_cartpole_best.pth")
```

### Vectorized Isaac Sim Environment (Recommended)

For better performance, use vectorized environments:

```python
from ppo.isaac_vec_env import create_isaac_vec_env

# Create vectorized environment
env = create_isaac_vec_env(
    env_name="cartpole",
    num_envs=8,  # Multiple parallel environments
    device="cuda",
    headless=True
)

# Use with trainer as before
trainer = PPOTrainer(agent, env=env)
```

## Environment Wrappers

### IsaacEnvWrapper

Single environment wrapper for Isaac Sim environments:

```python
from ppo.isaac_env import IsaacEnvWrapper

class MyIsaacEnv(IsaacEnvWrapper):
    def _setup_spaces(self):
        self.observation_space = spaces.Box(low=-1, high=1, shape=(10,))
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,))
    
    def _get_observation(self):
        # Implement observation retrieval
        return np.zeros(10)
    
    def _get_reward(self):
        # Implement reward calculation
        return 0.0
    
    def _get_done(self):
        # Implement done condition
        return False
    
    def _apply_action(self, action):
        # Implement action application
        pass
```

### IsaacVecEnv

Vectorized environment wrapper for parallel training:

```python
from ppo.isaac_vec_env import IsaacVecEnv

class MyIsaacVecEnv(IsaacVecEnv):
    def _setup_spaces(self):
        self.observation_space = spaces.Box(low=-1, high=1, shape=(10,))
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,))
    
    def _get_observations(self):
        # Return observations for all environments
        return np.zeros((self.num_envs, 10))
    
    def _get_rewards(self):
        # Return rewards for all environments
        return np.zeros(self.num_envs)
    
    def _get_dones(self):
        # Return done flags for all environments
        return np.zeros(self.num_envs, dtype=bool)
    
    def _apply_actions(self, actions):
        # Apply actions to all environments
        pass
```

## Performance Optimization

### 1. Use Vectorized Environments

Vectorized environments run multiple simulations in parallel, significantly improving performance:

```python
# Good: Vectorized environment
env = create_isaac_vec_env(num_envs=8)

# Avoid: Single environment
env = create_isaac_env(num_envs=1)
```

### 2. Set Headless Mode

Disable visualization for faster training:

```python
env = create_isaac_env(headless=True)  # Faster
env = create_isaac_env(headless=False)  # Slower, with visualization
```

### 3. Optimize Number of Environments

Adjust `num_envs` based on your GPU memory:

```python
# For 8GB GPU
env = create_isaac_vec_env(num_envs=4)

# For 16GB+ GPU
env = create_isaac_vec_env(num_envs=8)
```

### 4. Use CUDA Device

Always use CUDA for optimal performance:

```python
env = create_isaac_env(device="cuda")
```

## Troubleshooting

### Common Issues

#### 1. Import Errors
```
ImportError: No module named 'isaacgym'
```

**Solution**: Install Isaac Sim from Omniverse Launcher first, then install Python packages.

#### 2. CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```

**Solution**: Reduce `num_envs` or use smaller batch sizes.

#### 3. Physics Simulation Issues
```
Physics simulation not working correctly
```

**Solution**: Check Isaac Sim installation and ensure proper asset loading.

### Performance Tips

1. **Monitor GPU Memory**: Use `nvidia-smi` to monitor GPU usage
2. **Batch Size**: Adjust batch size based on available memory
3. **Environment Count**: Start with 4 environments, increase if memory allows
4. **Headless Mode**: Always use headless mode for training

## Examples

### CartPole with Isaac Sim
```bash
python examples/isaac_cartpole_example.py
```

### Vectorized CartPole
```bash
python examples/isaac_vec_cartpole_example.py
```

## Custom Environments

To create custom Isaac Sim environments:

1. Extend `IsaacEnvWrapper` or `IsaacVecEnv`
2. Implement required methods (`_get_observation`, `_get_reward`, etc.)
3. Load your custom assets in `_create_env`
4. Register your environment with the factory function

Example:
```python
class MyCustomEnv(IsaacVecEnv):
    def _create_env(self, **kwargs):
        # Load custom assets
        asset = self.gym.load_asset(self.sim, "", "path/to/asset.urdf")
        
        # Create environments
        for i in range(self.num_envs):
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, 1)
            actor_handle = self.gym.create_actor(env_handle, asset, pose, "actor", i, 1)
            self.actor_handles.append(actor_handle)
            self.env_handles.append(env_handle)

# Register environment
def create_isaac_vec_env(env_name, **kwargs):
    if env_name == "my_custom":
        return MyCustomEnv(**kwargs)
    # ... other environments
```

## Resources

- [Isaac Sim Documentation](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html)
- [Isaac Gym Documentation](https://developer.nvidia.com/isaac-gym)
- [Omniverse Launcher](https://www.nvidia.com/en-us/omniverse/)
- [PPO Project README](README.md) 