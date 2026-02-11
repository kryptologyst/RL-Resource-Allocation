"""Modern RL algorithms for resource allocation.

This module implements state-of-the-art RL algorithms optimized for resource allocation
problems, including DQN variants, PPO, and SAC.
"""

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
class DQNConfig:
    """Configuration for DQN algorithm."""
    learning_rate: float = 3e-4
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: int = 10000
    buffer_size: int = 100000
    batch_size: int = 64
    target_update_freq: int = 1000
    learning_starts: int = 1000
    train_freq: int = 4
    gradient_steps: int = 1
    tau: float = 1.0  # For soft target updates
    use_double_dqn: bool = True
    use_dueling: bool = True
    use_per: bool = False  # Prioritized Experience Replay
    device: str = "auto"


class DQNNetwork(nn.Module):
    """Deep Q-Network with optional dueling architecture."""
    
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        hidden_dims: List[int] = [256, 256],
        use_dueling: bool = True,
    ):
        super().__init__()
        self.use_dueling = use_dueling
        
        # Shared feature layers
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        
        self.feature_layers = nn.Sequential(*layers)
        
        if use_dueling:
            # Dueling DQN: separate value and advantage streams
            self.value_stream = nn.Linear(prev_dim, 1)
            self.advantage_stream = nn.Linear(prev_dim, output_dim)
        else:
            # Standard DQN
            self.q_head = nn.Linear(prev_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        features = self.feature_layers(x)
        
        if self.use_dueling:
            value = self.value_stream(features)
            advantage = self.advantage_stream(features)
            # Dueling formula: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
            q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        else:
            q_values = self.q_head(features)
        
        return q_values


class ReplayBuffer:
    """Experience replay buffer for DQN."""
    
    def __init__(self, capacity: int, state_dim: int, action_dim: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.state_dim = state_dim
        self.action_dim = action_dim
    
    def add(
        self, 
        state: np.ndarray, 
        action: int, 
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
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    """Deep Q-Network agent for resource allocation."""
    
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        config: DQNConfig,
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
        self.q_network = DQNNetwork(
            state_dim, 
            action_dim, 
            use_dueling=config.use_dueling
        ).to(self.device)
        
        self.target_network = DQNNetwork(
            state_dim, 
            action_dim, 
            use_dueling=config.use_dueling
        ).to(self.device)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.q_network.parameters(), 
            lr=config.learning_rate
        )
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            config.buffer_size, 
            state_dim, 
            action_dim
        )
        
        # Training variables
        self.epsilon = config.epsilon_start
        self.step_count = 0
        self.update_count = 0
        
        logger.info(f"DQN Agent initialized on device: {self.device}")
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy."""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def store_transition(
        self, 
        state: np.ndarray, 
        action: int, 
        reward: float, 
        next_state: np.ndarray, 
        done: bool
    ):
        """Store transition in replay buffer."""
        self.replay_buffer.add(state, action, reward, next_state, done)
    
    def update(self) -> Dict[str, float]:
        """Update the Q-network."""
        if len(self.replay_buffer) < self.config.learning_starts:
            return {"loss": 0.0, "q_value": 0.0}
        
        if self.step_count % self.config.train_freq != 0:
            return {"loss": 0.0, "q_value": 0.0}
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.config.batch_size
        )
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Compute current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Compute target Q-values
        with torch.no_grad():
            if self.config.use_double_dqn:
                # Double DQN: use main network to select action, target network to evaluate
                next_actions = self.q_network(next_states).argmax(1, keepdim=True)
                next_q_values = self.target_network(next_states).gather(1, next_actions)
            else:
                # Standard DQN
                next_q_values = self.target_network(next_states).max(1, keepdim=True)[0]
            
            target_q_values = rewards.unsqueeze(1) + self.config.gamma * next_q_values * (~dones).unsqueeze(1)
        
        # Compute loss
        loss = F.mse_loss(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.update_count += 1
        if self.update_count % self.config.target_update_freq == 0:
            if self.config.tau < 1.0:
                # Soft update
                for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
                    target_param.data.copy_(self.config.tau * param.data + (1 - self.config.tau) * target_param.data)
            else:
                # Hard update
                self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        self.epsilon = max(
            self.config.epsilon_end,
            self.config.epsilon_start - (self.config.epsilon_start - self.config.epsilon_end) * self.step_count / self.config.epsilon_decay
        )
        
        self.step_count += 1
        
        return {
            "loss": loss.item(),
            "q_value": current_q_values.mean().item(),
            "epsilon": self.epsilon,
        }
    
    def save(self, filepath: str):
        """Save agent state."""
        torch.save({
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.config,
            "step_count": self.step_count,
            "epsilon": self.epsilon,
        }, filepath)
    
    def load(self, filepath: str):
        """Load agent state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.step_count = checkpoint["step_count"]
        self.epsilon = checkpoint["epsilon"]
