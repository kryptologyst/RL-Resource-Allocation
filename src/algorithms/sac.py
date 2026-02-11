"""SAC (Soft Actor-Critic) implementation for resource allocation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
from dataclasses import dataclass
import random
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class SACConfig:
    """Configuration for SAC algorithm."""
    learning_rate: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.2  # Entropy coefficient (auto-tuned if None)
    buffer_size: int = 100000
    batch_size: int = 256
    learning_starts: int = 1000
    train_freq: int = 1
    gradient_steps: int = 1
    target_update_interval: int = 1
    device: str = "auto"


class SACActor(nn.Module):
    """SAC Actor network."""
    
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        hidden_dims: List[int] = [256, 256],
    ):
        super().__init__()
        
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        
        self.feature_layers = nn.Sequential(*layers)
        
        # Output layers for mean and log_std
        self.mean_head = nn.Linear(prev_dim, action_dim)
        self.log_std_head = nn.Linear(prev_dim, action_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            torch.nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the actor."""
        features = self.feature_layers(state)
        
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        
        # Clamp log_std to prevent numerical instability
        log_std = torch.clamp(log_std, -20, 2)
        
        return mean, log_std
    
    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action from the policy."""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # Reparameterization trick
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        
        # Apply tanh squashing
        action = torch.tanh(x_t)
        
        # Compute log probability
        log_prob = normal.log_prob(x_t)
        # Apply tanh correction
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob


class SACCritic(nn.Module):
    """SAC Critic network (Q-function)."""
    
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        hidden_dims: List[int] = [256, 256],
    ):
        super().__init__()
        
        layers = []
        prev_dim = state_dim + action_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        
        self.feature_layers = nn.Sequential(*layers)
        self.q_head = nn.Linear(prev_dim, 1)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            torch.nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass through the critic."""
        x = torch.cat([state, action], dim=1)
        features = self.feature_layers(x)
        q_value = self.q_head(features)
        return q_value


class SACReplayBuffer:
    """Experience replay buffer for SAC."""
    
    def __init__(self, capacity: int, state_dim: int, action_dim: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.state_dim = state_dim
        self.action_dim = action_dim
    
    def add(
        self, 
        state: np.ndarray, 
        action: np.ndarray, 
        reward: float, 
        next_state: np.ndarray, 
        done: bool
    ):
        """Add experience to buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample batch of experiences."""
        batch = random.sample(self.buffer, batch_size)
        
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.FloatTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        return len(self.buffer)


class SACAgent:
    """SAC agent for resource allocation."""
    
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        config: SACConfig,
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
        
        # Networks
        self.actor = SACActor(state_dim, action_dim).to(self.device)
        self.critic1 = SACCritic(state_dim, action_dim).to(self.device)
        self.critic2 = SACCritic(state_dim, action_dim).to(self.device)
        
        # Target networks
        self.target_critic1 = SACCritic(state_dim, action_dim).to(self.device)
        self.target_critic2 = SACCritic(state_dim, action_dim).to(self.device)
        
        # Copy weights to target networks
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.learning_rate)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=config.learning_rate)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=config.learning_rate)
        
        # Entropy coefficient (auto-tune if None)
        if config.alpha is None:
            self.auto_entropy = True
            self.target_entropy = -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=config.learning_rate)
        else:
            self.auto_entropy = False
            self.alpha = config.alpha
        
        # Replay buffer
        self.replay_buffer = SACReplayBuffer(
            config.buffer_size, 
            state_dim, 
            action_dim
        )
        
        # Training variables
        self.step_count = 0
        
        logger.info(f"SAC Agent initialized on device: {self.device}")
    
    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """Select action using current policy."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            if training:
                action, _ = self.actor.sample(state_tensor)
            else:
                mean, _ = self.actor.forward(state_tensor)
                action = torch.tanh(mean)
            
            return action.cpu().numpy().flatten()
    
    def store_transition(
        self, 
        state: np.ndarray, 
        action: np.ndarray, 
        reward: float, 
        next_state: np.ndarray, 
        done: bool
    ):
        """Store transition in replay buffer."""
        self.replay_buffer.add(state, action, reward, next_state, done)
    
    def update(self) -> Dict[str, float]:
        """Update the SAC networks."""
        if len(self.replay_buffer) < self.config.learning_starts:
            return {"actor_loss": 0.0, "critic_loss": 0.0, "alpha": 0.0}
        
        if self.step_count % self.config.train_freq != 0:
            return {"actor_loss": 0.0, "critic_loss": 0.0, "alpha": 0.0}
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.config.batch_size
        )
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Update critics
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            q1_next = self.target_critic1(next_states, next_actions)
            q2_next = self.target_critic2(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next)
            
            alpha = self.log_alpha.exp() if self.auto_entropy else self.alpha
            q_target = rewards.unsqueeze(1) + self.config.gamma * (1 - dones.unsqueeze(1)) * (
                q_next - alpha * next_log_probs
            )
        
        # Critic losses
        q1_current = self.critic1(states, actions)
        q2_current = self.critic2(states, actions)
        
        critic1_loss = F.mse_loss(q1_current, q_target)
        critic2_loss = F.mse_loss(q2_current, q_target)
        
        # Update critics
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # Update actor
        new_actions, log_probs = self.actor.sample(states)
        q1_new = self.critic1(states, new_actions)
        q2_new = self.critic2(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        
        alpha = self.log_alpha.exp() if self.auto_entropy else self.alpha
        actor_loss = (alpha * log_probs - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update alpha (entropy coefficient)
        alpha_loss = 0.0
        if self.auto_entropy:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
        
        # Update target networks
        if self.step_count % self.config.target_update_interval == 0:
            self._soft_update(self.target_critic1, self.critic1)
            self._soft_update(self.target_critic2, self.critic2)
        
        self.step_count += 1
        
        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": (critic1_loss.item() + critic2_loss.item()) / 2,
            "alpha": alpha.item() if self.auto_entropy else self.alpha,
            "alpha_loss": alpha_loss.item() if self.auto_entropy else 0.0,
        }
    
    def _soft_update(self, target: nn.Module, source: nn.Module):
        """Soft update target network."""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.config.tau * param.data + (1 - self.config.tau) * target_param.data
            )
    
    def save(self, filepath: str):
        """Save agent state."""
        checkpoint = {
            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "target_critic1": self.target_critic1.state_dict(),
            "target_critic2": self.target_critic2.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic1_optimizer": self.critic1_optimizer.state_dict(),
            "critic2_optimizer": self.critic2_optimizer.state_dict(),
            "config": self.config,
            "step_count": self.step_count,
        }
        
        if self.auto_entropy:
            checkpoint.update({
                "log_alpha": self.log_alpha,
                "alpha_optimizer": self.alpha_optimizer.state_dict(),
            })
        
        torch.save(checkpoint, filepath)
    
    def load(self, filepath: str):
        """Load agent state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic1.load_state_dict(checkpoint["critic1"])
        self.critic2.load_state_dict(checkpoint["critic2"])
        self.target_critic1.load_state_dict(checkpoint["target_critic1"])
        self.target_critic2.load_state_dict(checkpoint["target_critic2"])
        
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic1_optimizer.load_state_dict(checkpoint["critic1_optimizer"])
        self.critic2_optimizer.load_state_dict(checkpoint["critic2_optimizer"])
        
        self.step_count = checkpoint["step_count"]
        
        if self.auto_entropy and "log_alpha" in checkpoint:
            self.log_alpha = checkpoint["log_alpha"]
            self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])
