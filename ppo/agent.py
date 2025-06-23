"""
PPO Agent implementation with actor-critic networks.
"""

import torch
import torch.nn
import torch.nn.functional as F
import numpy as np
import os
import math
from typing import Tuple, Dict, Any, Optional


class ActorCritic(torch.nn.Module):
    """
    Actor-Critic network for PPO.
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=64, num_layers=2, activation="tanh"):
        """
        Initialize Actor-Critic network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Number of hidden units
            num_layers: Number of hidden layers
            activation: Activation function ("tanh", "relu", "leaky_relu")
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Choose activation function
        if activation == "tanh":
            self.activation = torch.tanh
        elif activation == "relu":
            self.activation = F.relu
        elif activation == "leaky_relu":
            self.activation = F.leaky_relu
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Shared layers
        self.shared_layers = torch.nn.ModuleList()
        self.shared_layers.append(torch.nn.Linear(state_dim, hidden_dim))
        
        for _ in range(num_layers - 1):
            self.shared_layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
        
        # Actor (policy) head
        self.actor = torch.nn.Linear(hidden_dim, action_dim)
        
        # Critic (value) head
        self.critic = torch.nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            torch.nn.init.constant_(module.bias, 0)
    
    def forward(self, state):
        """
        Forward pass through the network.
        
        Args:
            state: Input state tensor [batch_size, state_dim]
            
        Returns:
            action_probs: Action probabilities [batch_size, action_dim]
            value: State value [batch_size, 1]
        """
        x = state
        
        # Shared layers
        for layer in self.shared_layers:
            x = self.activation(layer(x))
        
        # Actor and critic heads
        action_logits = self.actor(x)
        value = self.critic(x)
        
        # Convert logits to probabilities
        action_probs = F.softmax(action_logits, dim=-1)
        
        return action_probs, value
    
    def get_action(self, state, deterministic=False):
        """
        Get action from the policy.
        
        Args:
            state: Input state
            deterministic: Whether to use deterministic action selection
            
        Returns:
            action: Selected action
            log_prob: Log probability of the action
            value: State value
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).unsqueeze(0)
        
        # Move state to the same device as the model
        state = state.to(next(self.parameters()).device)
        
        action_probs, value = self.forward(state)
        
        if deterministic:
            action = torch.argmax(action_probs, dim=-1)
        else:
            # Sample action from categorical distribution
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
        
        # Compute log probability
        log_prob = torch.log(action_probs + 1e-8).gather(-1, action.unsqueeze(-1)).squeeze(-1)
        
        return action.item(), log_prob.item(), value.item()


class PPOAgent:
    """
    PPO Agent with actor-critic architecture and configurable learning rate annealing.
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=64, num_layers=2, 
                 learning_rate=3e-4, device="cpu", lr_schedule="cosine", 
                 lr_schedule_kwargs=None):
        """
        Initialize PPO Agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Number of hidden units
            num_layers: Number of hidden layers
            learning_rate: Initial learning rate for optimization
            device: Device to run on ("cpu" or "cuda")
            lr_schedule: Learning rate schedule type ("linear", "cosine", "exponential", "step", "none")
            lr_schedule_kwargs: Additional arguments for the scheduler
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.initial_lr = learning_rate
        self.lr_schedule = lr_schedule
        
        # Create actor-critic network
        self.actor_critic = ActorCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        ).to(device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.actor_critic.parameters(),
            lr=learning_rate,
            eps=1e-5
        )
        
        # Learning rate scheduler
        self.scheduler = self._create_scheduler(lr_schedule, lr_schedule_kwargs or {})
    
    def _create_scheduler(self, schedule_type, kwargs):
        """Create learning rate scheduler based on type."""
        if schedule_type == "none":
            return None
        elif schedule_type == "linear":
            total_iters = kwargs.get('total_iters', 1000)
            end_factor = kwargs.get('end_factor', 0.0)
            return torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=end_factor,
                total_iters=total_iters
            )
        elif schedule_type == "cosine":
            total_iters = kwargs.get('total_iters', 1000)
            eta_min = kwargs.get('eta_min', 0.0)
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_iters,
                eta_min=eta_min
            )
        elif schedule_type == "exponential":
            gamma = kwargs.get('gamma', 0.99)
            return torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=gamma
            )
        elif schedule_type == "step":
            step_size = kwargs.get('step_size', 100)
            gamma = kwargs.get('gamma', 0.1)
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=step_size,
                gamma=gamma
            )
        elif schedule_type == "cosine_warm_restart":
            T_0 = kwargs.get('T_0', 100)
            T_mult = kwargs.get('T_mult', 2)
            eta_min = kwargs.get('eta_min', 0.0)
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=T_0,
                T_mult=T_mult,
                eta_min=eta_min
            )
        else:
            raise ValueError(f"Unsupported learning rate schedule: {schedule_type}")
    
    def get_current_lr(self):
        """Get current learning rate."""
        if self.scheduler is None:
            return self.optimizer.param_groups[0]['lr']
        return self.optimizer.param_groups[0]['lr']
    
    def get_lr_schedule_info(self):
        """Get information about the learning rate schedule."""
        info = {
            'schedule_type': self.lr_schedule,
            'initial_lr': self.initial_lr,
            'current_lr': self.get_current_lr()
        }
        
        if self.scheduler is not None:
            if hasattr(self.scheduler, 'last_epoch'):
                info['last_epoch'] = self.scheduler.last_epoch
            if hasattr(self.scheduler, 'base_lrs'):
                info['base_lrs'] = self.scheduler.base_lrs
        
        return info
    
    def get_action(self, state, deterministic=False):
        """
        Get action from the policy.
        
        Args:
            state: Input state
            deterministic: Whether to use deterministic action selection
            
        Returns:
            action: Selected action
            log_prob: Log probability of the action
            value: State value
        """
        return self.actor_critic.get_action(state, deterministic)
    
    def evaluate_actions(self, states, actions):
        """
        Evaluate actions for computing losses.
        
        Args:
            states: Batch of states [batch_size, state_dim]
            actions: Batch of actions [batch_size]
            
        Returns:
            log_probs: Log probabilities of actions [batch_size]
            values: State values [batch_size, 1]
            entropy: Entropy of policy [batch_size]
        """
        # Ensure states and actions are on the correct device
        states = states.to(self.device)
        actions = actions.to(self.device)
        
        action_probs, values = self.actor_critic(states)
        
        # Get log probabilities for the taken actions
        log_probs = torch.log(action_probs + 1e-8).gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        
        # Compute entropy
        entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=-1)
        
        return log_probs, values, entropy
    
    def update(self, batch, clip_epsilon=0.2, value_coef=0.5, entropy_coef=0.01, max_grad_norm=0.5):
        """
        Update the agent using PPO.
        
        Args:
            batch: Dictionary containing training data
            clip_epsilon: PPO clipping parameter
            value_coef: Value function coefficient
            entropy_coef: Entropy coefficient
            max_grad_norm: Maximum gradient norm for clipping
            
        Returns:
            loss_info: Dictionary containing loss information
        """
        states = batch['states'].to(self.device)
        actions = batch['actions'].to(self.device)
        old_log_probs = batch['old_log_probs'].to(self.device)
        advantages = batch['advantages'].to(self.device)
        returns = batch['returns'].to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Get current policy outputs
        log_probs, values, entropy = self.evaluate_actions(states, actions)
        
        # Compute policy loss with clipping
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Compute value loss
        value_loss = F.mse_loss(values.squeeze(), returns)
        
        # Compute entropy loss
        entropy_loss = -entropy.mean()
        
        # Total loss
        total_loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), max_grad_norm)
        self.optimizer.step()
        
        # Update learning rate
        if self.scheduler is not None:
            self.scheduler.step()
        
        # Compute additional metrics
        with torch.no_grad():
            kl_div = (old_log_probs - log_probs).mean().item()
            clip_fraction = (abs(ratio - 1) > clip_epsilon).float().mean().item()
        
        loss_info = {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'kl_div': kl_div,
            'clip_fraction': clip_fraction,
            'learning_rate': self.get_current_lr()
        }
        
        return loss_info
    
    def save(self, path):
        """Save the agent to a file."""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save only the essential data for compatibility
        checkpoint = {
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'device': self.device,
            'lr_schedule': self.lr_schedule,
            'initial_lr': self.initial_lr
        }
        
        torch.save(checkpoint, path, _use_new_zipfile_serialization=True)
    
    def load(self, path):
        """Load the agent from a file."""
        try:
            # Try loading with weights_only=True first (PyTorch 2.6+ default)
            checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        except Exception as e:
            # If that fails, try with weights_only=False (PyTorch <2.6 behavior)
            print(f"Note: Loading with weights_only=False due to compatibility: {e}")
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        # Check if network architecture matches
        saved_state_dict = checkpoint['actor_critic_state_dict']
        current_state_dict = self.actor_critic.state_dict()
        
        # Check for architecture mismatch
        architecture_mismatch = False
        mismatch_info = []
        
        for key in saved_state_dict.keys():
            if key in current_state_dict:
                if saved_state_dict[key].shape != current_state_dict[key].shape:
                    architecture_mismatch = True
                    mismatch_info.append(f"{key}: {saved_state_dict[key].shape} -> {current_state_dict[key].shape}")
        
        if architecture_mismatch:
            print("❌ Architecture mismatch detected!")
            print("Saved model has different network dimensions:")
            for info in mismatch_info:
                print(f"   {info}")
            print("\nTo fix this, either:")
            print("1. Use the same network architecture (hidden_dim, num_layers) as the saved model")
            print("2. Retrain the model with the current architecture")
            print("3. Create a new agent with matching architecture before loading")
            raise ValueError("Network architecture mismatch")
        
        # Load the model weights
        self.actor_critic.load_state_dict(saved_state_dict)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler if it exists and matches
        if checkpoint.get('scheduler_state_dict') is not None and self.scheduler is not None:
            try:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except Exception as e:
                print(f"Warning: Could not load scheduler state: {e}")
                print("Scheduler will continue from current state")
        
        # Update agent attributes if available
        if 'lr_schedule' in checkpoint:
            self.lr_schedule = checkpoint['lr_schedule']
        if 'initial_lr' in checkpoint:
            self.initial_lr = checkpoint['initial_lr']
        
        print(f"✅ Model loaded successfully from {path}")
        print(f"   Schedule: {self.lr_schedule}")
        print(f"   Initial LR: {self.initial_lr:.2e}")
        print(f"   Current LR: {self.get_current_lr():.2e}") 