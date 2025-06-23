"""
PPO Agent implementation with actor-critic networks.
"""

import torch
import torch.nn
import torch.nn.functional as F
import numpy as np
import os
from typing import Tuple, Dict, Any


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
    PPO Agent with actor-critic architecture.
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=64, num_layers=2, 
                 learning_rate=3e-4, device="cpu"):
        """
        Initialize PPO Agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Number of hidden units
            num_layers: Number of hidden layers
            learning_rate: Learning rate for optimization
            device: Device to run on ("cpu" or "cuda")
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
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
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=1000
        )
    
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
            'clip_fraction': clip_fraction
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
            'scheduler_state_dict': self.scheduler.state_dict(),
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'device': self.device
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
        
        self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict']) 