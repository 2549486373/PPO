"""
Isaac Sim Environment Wrapper for PPO Training
"""

import numpy as np
import torch
from typing import Dict, Any, Optional, Tuple
import gymnasium as gym
from gymnasium import spaces

try:
    import isaacgym
    from isaacgym import gymapi
    from isaacgym import gymtorch
    from isaacgymenvs.utils.torch_jit_utils import *
    from isaacgymenvs.tasks.base.vec_task import VecTask
    ISAAC_AVAILABLE = True
except ImportError:
    ISAAC_AVAILABLE = False
    print("Warning: Isaac Sim not available. Install omni-isaac-gym to use Isaac Sim environments.")


class IsaacEnvWrapper(gym.Env):
    """
    Wrapper for Isaac Sim environments to make them compatible with the PPO trainer.
    """
    
    def __init__(self, env_name: str, num_envs: int = 1, device: str = "cuda", 
                 headless: bool = True, **kwargs):
        """
        Initialize Isaac Sim environment wrapper.
        
        Args:
            env_name: Name of the Isaac Sim environment
            num_envs: Number of parallel environments
            device: Device to run on ("cuda" or "cpu")
            headless: Whether to run in headless mode
            **kwargs: Additional environment-specific parameters
        """
        if not ISAAC_AVAILABLE:
            raise ImportError("Isaac Sim is not available. Please install omni-isaac-gym.")
        
        self.env_name = env_name
        self.num_envs = num_envs
        self.device = device
        self.headless = headless
        
        # Initialize Isaac Sim
        self._init_isaac_sim()
        
        # Create environment
        self._create_env(**kwargs)
        
        # Setup observation and action spaces
        self._setup_spaces()
        
        # Initialize state
        self.reset()
    
    def _init_isaac_sim(self):
        """Initialize Isaac Sim gym."""
        self.gym = gymapi.acquire_gym()
        
        # Create sim
        sim_params = gymapi.SimParams()
        sim_params.dt = 1.0 / 60.0
        sim_params.substeps = 2
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        
        # Set physics parameters
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.contact_offset = 0.01
        sim_params.physx.rest_offset = 0.0
        
        # Create sim
        self.sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
        
        if not self.headless:
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        else:
            self.viewer = None
    
    def _create_env(self, **kwargs):
        """Create the specific Isaac Sim environment."""
        # This is a generic implementation - specific environments will override this
        self.env = None
        self.env_handles = []
        
        # Create ground plane
        plane_params = gymapi.PlaneParams()
        self.gym.add_ground(self.sim, plane_params)
        
        # Create environments
        env_spacing = 1.0
        env_lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.0)
        env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
        
        for i in range(self.num_envs):
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, 1)
            self.env_handles.append(env_handle)
    
    def _setup_spaces(self):
        """Setup observation and action spaces."""
        # Default spaces - should be overridden by specific environments
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment."""
        if seed is not None:
            np.random.seed(seed)
        
        # Reset Isaac Sim environment
        self.gym.reset_env_sim(self.sim)
        
        # Get initial observation
        obs = self._get_observation()
        
        return obs, {}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment."""
        # Apply action
        self._apply_action(action)
        
        # Step simulation
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        
        if self.viewer:
            self.gym.sync_frame_time(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, False)
        
        # Get observation, reward, done
        obs = self._get_observation()
        reward = self._get_reward()
        done = self._get_done()
        truncated = False
        info = self._get_info()
        
        return obs, reward, done, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        # Default implementation - should be overridden
        return np.zeros(self.observation_space.shape, dtype=np.float32)
    
    def _get_reward(self) -> float:
        """Get current reward."""
        # Default implementation - should be overridden
        return 0.0
    
    def _get_done(self) -> bool:
        """Check if episode is done."""
        # Default implementation - should be overridden
        return False
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional info."""
        return {}
    
    def _apply_action(self, action: np.ndarray):
        """Apply action to the environment."""
        # Default implementation - should be overridden
        pass
    
    def close(self):
        """Close the environment."""
        if self.viewer:
            self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)


class CartpoleIsaacEnv(IsaacEnvWrapper):
    """Isaac Sim CartPole environment."""
    
    def __init__(self, num_envs: int = 1, device: str = "cuda", headless: bool = True):
        super().__init__("CartPole", num_envs, device, headless)
    
    def _setup_spaces(self):
        """Setup CartPole observation and action spaces."""
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(2)
    
    def _create_env(self, **kwargs):
        """Create CartPole environment."""
        # Create cartpole asset
        asset_root = ""
        cartpole_file = "urdf/cartpole.urdf"
        
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        cartpole_asset = self.gym.load_asset(self.sim, asset_root, cartpole_file, asset_options)
        
        # Create environments
        env_spacing = 2.0
        env_lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.0)
        env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
        
        for i in range(self.num_envs):
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, 1)
            
            # Create cartpole instance
            initial_pose = gymapi.Transform()
            initial_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
            initial_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
            
            cartpole_handle = self.gym.create_actor(env_handle, cartpole_asset, initial_pose, "cartpole", i, 1)
            self.env_handles.append(env_handle)
    
    def _get_observation(self) -> np.ndarray:
        """Get CartPole observation."""
        # This would need to be implemented based on the specific Isaac Sim CartPole setup
        # For now, return a placeholder
        return np.random.randn(4).astype(np.float32)
    
    def _get_reward(self) -> float:
        """Get CartPole reward."""
        # Implement reward calculation based on cartpole state
        return 1.0
    
    def _get_done(self) -> bool:
        """Check if CartPole episode is done."""
        # Implement done condition
        return False
    
    def _apply_action(self, action: np.ndarray):
        """Apply action to CartPole."""
        # Apply force to cart based on action
        pass


def create_isaac_env(env_name: str, num_envs: int = 1, device: str = "cuda", 
                    headless: bool = True, **kwargs) -> IsaacEnvWrapper:
    """
    Factory function to create Isaac Sim environments.
    
    Args:
        env_name: Name of the environment
        num_envs: Number of parallel environments
        device: Device to run on
        headless: Whether to run in headless mode
        **kwargs: Additional environment-specific parameters
    
    Returns:
        IsaacEnvWrapper instance
    """
    if env_name.lower() == "cartpole":
        return CartpoleIsaacEnv(num_envs, device, headless)
    else:
        raise ValueError(f"Unknown Isaac Sim environment: {env_name}") 