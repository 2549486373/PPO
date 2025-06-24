"""
Vectorized Isaac Sim Environment Wrapper for PPO Training
"""

import numpy as np
import torch
from typing import Dict, Any, Optional, Tuple, List
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


class IsaacVecEnv(gym.Env):
    """
    Vectorized Isaac Sim environment wrapper for efficient parallel training.
    """
    
    def __init__(self, env_name: str, num_envs: int = 4, device: str = "cuda", 
                 headless: bool = True, **kwargs):
        """
        Initialize vectorized Isaac Sim environment.
        
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
        if not ISAAC_AVAILABLE:
            raise ImportError("Isaac Sim is not available.")
        
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
        self.env_handles = []
        self.actor_handles = []
        
        # Create ground plane
        plane_params = gymapi.PlaneParams()
        self.gym.add_ground(self.sim, plane_params)
        
        # Create environments
        env_spacing = 2.0
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
        """Reset all environments."""
        if seed is not None:
            np.random.seed(seed)
        
        # Reset Isaac Sim environment
        self.gym.reset_env_sim(self.sim)
        
        # Get initial observation for all environments
        obs = self._get_observations()
        
        return obs, {}
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Take a step in all environments."""
        # Apply actions to all environments
        self._apply_actions(actions)
        
        # Step simulation
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        
        if self.viewer:
            self.gym.sync_frame_time(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, False)
        
        # Get observations, rewards, dones for all environments
        obs = self._get_observations()
        rewards = self._get_rewards()
        dones = self._get_dones()
        truncated = np.zeros(self.num_envs, dtype=bool)
        info = self._get_infos()
        
        return obs, rewards, dones, truncated, info
    
    def _get_observations(self) -> np.ndarray:
        """Get current observations for all environments."""
        # Default implementation - should be overridden
        obs_shape = self.observation_space.shape
        if obs_shape is None:
            obs_shape = (10,)
        return np.zeros((self.num_envs,) + obs_shape, dtype=np.float32)
    
    def _get_rewards(self) -> np.ndarray:
        """Get current rewards for all environments."""
        # Default implementation - should be overridden
        return np.zeros(self.num_envs, dtype=np.float32)
    
    def _get_dones(self) -> np.ndarray:
        """Check if episodes are done for all environments."""
        # Default implementation - should be overridden
        return np.zeros(self.num_envs, dtype=bool)
    
    def _get_infos(self) -> Dict[str, Any]:
        """Get additional info for all environments."""
        return {}
    
    def _apply_actions(self, actions: np.ndarray):
        """Apply actions to all environments."""
        # Default implementation - should be overridden
        pass
    
    def close(self):
        """Close the environment."""
        if self.viewer:
            self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)


class CartpoleIsaacVecEnv(IsaacVecEnv):
    """Vectorized Isaac Sim CartPole environment."""
    
    def __init__(self, num_envs: int = 4, device: str = "cuda", headless: bool = True):
        super().__init__("CartPole", num_envs, device, headless)
    
    def _setup_spaces(self):
        """Setup CartPole observation and action spaces."""
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(2)
    
    def _create_env(self, **kwargs):
        """Create vectorized CartPole environment."""
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
            self.actor_handles.append(cartpole_handle)
            self.env_handles.append(env_handle)
    
    def _get_observations(self) -> np.ndarray:
        """Get CartPole observations for all environments."""
        # This would need to be implemented based on the specific Isaac Sim CartPole setup
        # For now, return placeholders
        return np.random.randn(self.num_envs, 4).astype(np.float32)
    
    def _get_rewards(self) -> np.ndarray:
        """Get CartPole rewards for all environments."""
        # Implement reward calculation based on cartpole state
        return np.ones(self.num_envs, dtype=np.float32)
    
    def _get_dones(self) -> np.ndarray:
        """Check if CartPole episodes are done for all environments."""
        # Implement done condition
        return np.zeros(self.num_envs, dtype=bool)
    
    def _apply_actions(self, actions: np.ndarray):
        """Apply actions to CartPole in all environments."""
        # Apply force to cart based on actions
        pass


def create_isaac_vec_env(env_name: str, num_envs: int = 4, device: str = "cuda", 
                        headless: bool = True, **kwargs) -> IsaacVecEnv:
    """
    Factory function to create vectorized Isaac Sim environments.
    
    Args:
        env_name: Name of the environment
        num_envs: Number of parallel environments
        device: Device to run on
        headless: Whether to run in headless mode
        **kwargs: Additional environment-specific parameters
    
    Returns:
        IsaacVecEnv instance
    """
    if env_name.lower() == "cartpole":
        return CartpoleIsaacVecEnv(num_envs, device, headless)
    else:
        raise ValueError(f"Unknown Isaac Sim environment: {env_name}") 