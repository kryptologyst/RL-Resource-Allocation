"""PPO (Proximal Policy Optimization) implementation for resource allocation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
from dataclasses import dataclass
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class PPOConfig:
    """Configuration for PPO algorithm."""
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    batch_size: int = 64
    device: str = "auto"


class PPONetwork(nn.Module):
    """PPO network with shared feature extractor."""
    
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        hidden_dims: List[int] = [256, 256],
    ):
        super().__init__()
        
        # Shared feature layers
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        
        self.feature_layers = nn.Sequential(*layers)
        
        # Policy head (actor)
        self.policy_head = nn.Linear(prev_dim, action_dim)
        
        # Value head (critic)
        self.value_head = nn.Linear(prev_dim, 1)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network."""
        features = self.feature_layers(state)
        
        # Policy logits
        logits = self.policy_head(features)
        
        # Value estimate
        value = self.value_head(features)
        
        return logits, value
    
    def get_action_and_value(
        self, 
        state: torch.Tensor, 
        action: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action, log probability, entropy, and value."""
        logits, value = self.forward(state)
        
        # Create categorical distribution
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        
        if action is None:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy, value


class PPOBuffer:
    """Buffer for storing PPO rollout data."""
    
    def __init__(self, buffer_size: int, state_dim: int, action_dim: int, device: torch.device):
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        # Storage
        self.states = torch.zeros((buffer_size, state_dim), device=device)
        self.actions = torch.zeros((buffer_size,), device=device, dtype=torch.long)
        self.log_probs = torch.zeros((buffer_size,), device=device)
        self.rewards = torch.zeros((buffer_size,), device=device)
        self.values = torch.zeros((buffer_size,), device=device)
        self.advantages = torch.zeros((buffer_size,), device=device)
        self.returns = torch.zeros((buffer_size,), device=device)
        
        self.ptr = 0
        self.size = 0
    
    def add(
        self, 
        state: np.ndarray, 
        action: int, 
        log_prob: float, 
        reward: float, 
        value: float
    ):
        """Add experience to buffer."""
        self.states[self.ptr] = torch.FloatTensor(state).to(self.device)
        self.actions[self.ptr] = action
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)
    
    def compute_gae(
        self, 
        next_value: float, 
        gamma: float, 
        gae_lambda: float
    ):
        """Compute Generalized Advantage Estimation."""
        advantages = torch.zeros_like(self.rewards)
        last_advantage = 0
        
        for t in reversed(range(self.size)):
            if t == self.size - 1:
                next_non_terminal = 0.0
                next_value_t = next_value
            else:
                next_non_terminal = 1.0
                next_value_t = self.values[t + 1]
            
            delta = self.rewards[t] + gamma * next_value_t * next_non_terminal - self.values[t]
            advantages[t] = last_advantage = delta + gamma * gae_lambda * next_non_terminal * last_advantage
        
        self.returns = advantages + self.values
        self.advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    def get_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Get batch of data for training."""
        indices = torch.randperm(self.size, device=self.device)[:batch_size]
        
        return {
            "states": self.states[indices],
            "actions": self.actions[indices],
            "log_probs": self.log_probs[indices],
            "advantages": self.advantages[indices],
            "returns": self.returns[indices],
        }


class PPOAgent:
    """PPO agent for resource allocation."""
    
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        config: PPOConfig,
        device: Optional[torch.device] = None,
    ):
        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Device setup
        if device is None:
            if config.device == "auto":
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(config.device)
        else:
            self.device = device
        
        # Network
        self.network = PPONetwork(state_dim, action_dim).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=config.learning_rate)
        
        # Buffer
        self.buffer = PPOBuffer(
            buffer_size=2048,  # Standard PPO buffer size
            state_dim=state_dim,
            action_dim=action_dim,
            device=self.device
        )
        
        logger.info(f"PPO Agent initialized on device: {self.device}")
    
    def select_action(self, state: np.ndarray, training: bool = True) -> Tuple[int, float, float]:
        """Select action using current policy."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, log_prob, entropy, value = self.network.get_action_and_value(state_tensor)
            
            return action.item(), log_prob.item(), value.item()
    
    def store_transition(
        self, 
        state: np.ndarray, 
        action: int, 
        log_prob: float, 
        reward: float, 
        value: float
    ):
        """Store transition in buffer."""
        self.buffer.add(state, action, log_prob, reward, value)
    
    def update(self) -> Dict[str, float]:
        """Update the policy using PPO."""
        if self.buffer.size < self.config.batch_size:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        
        # Compute GAE
        with torch.no_grad():
            next_state = self.buffer.states[-1]
            _, _, _, next_value = self.network.get_action_and_value(next_state.unsqueeze(0))
            self.buffer.compute_gae(next_value.item(), self.config.gamma, self.config.gae_lambda)
        
        # PPO updates
        policy_losses = []
        value_losses = []
        entropies = []
        
        for _ in range(self.config.ppo_epochs):
            batch = self.buffer.get_batch(self.config.batch_size)
            
            # Get current policy outputs
            _, new_log_probs, entropy, new_values = self.network.get_action_and_value(
                batch["states"], batch["actions"]
            )
            
            # Compute policy loss
            ratio = torch.exp(new_log_probs - batch["log_probs"])
            surr1 = ratio * batch["advantages"]
            surr2 = torch.clamp(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio) * batch["advantages"]
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Compute value loss
            value_loss = F.mse_loss(new_values.squeeze(), batch["returns"])
            
            # Total loss
            total_loss = policy_loss + self.config.value_loss_coef * value_loss - self.config.entropy_coef * entropy.mean()
            
            # Optimize
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropies.append(entropy.mean().item())
        
        # Clear buffer
        self.buffer.size = 0
        self.buffer.ptr = 0
        
        return {
            "policy_loss": np.mean(policy_losses),
            "value_loss": np.mean(value_losses),
            "entropy": np.mean(entropies),
        }
    
    def save(self, filepath: str):
        """Save agent state."""
        torch.save({
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.config,
        }, filepath)
    
    def load(self, filepath: str):
        """Load agent state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
