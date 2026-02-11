"""Algorithm factory and utilities for RL agents."""

from __future__ import annotations

import torch
import numpy as np
from typing import Any, Dict, Optional, Type, Union
import logging

from .dqn import DQNAgent, DQNConfig
from .ppo import PPOAgent, PPOConfig
from .sac import SACAgent, SACConfig

logger = logging.getLogger(__name__)


class AlgorithmFactory:
    """Factory for creating RL agents."""
    
    _algorithms = {
        "dqn": (DQNAgent, DQNConfig),
        "ppo": (PPOAgent, PPOConfig),
        "sac": (SACAgent, SACConfig),
    }
    
    @classmethod
    def create_agent(
        cls,
        algorithm: str,
        state_dim: int,
        action_dim: int,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
    ) -> Union[DQNAgent, PPOAgent, SACAgent]:
        """Create an RL agent.
        
        Args:
            algorithm: Algorithm name ("dqn", "ppo", "sac")
            state_dim: State dimension
            action_dim: Action dimension
            config: Algorithm configuration
            device: Device to run on
            
        Returns:
            Configured RL agent
        """
        if algorithm not in cls._algorithms:
            raise ValueError(f"Unknown algorithm: {algorithm}. Available: {list(cls._algorithms.keys())}")
        
        agent_class, config_class = cls._algorithms[algorithm]
        
        # Create config
        if config is None:
            config = {}
        
        agent_config = config_class(**config)
        
        # Create agent
        agent = agent_class(
            state_dim=state_dim,
            action_dim=action_dim,
            config=agent_config,
            device=device,
        )
        
        logger.info(f"Created {algorithm.upper()} agent with config: {agent_config}")
        return agent
    
    @classmethod
    def get_available_algorithms(cls) -> list[str]:
        """Get list of available algorithms."""
        return list(cls._algorithms.keys())
    
    @classmethod
    def register_algorithm(cls, name: str, agent_class: Type, config_class: Type):
        """Register a new algorithm."""
        cls._algorithms[name] = (agent_class, config_class)
        logger.info(f"Registered algorithm: {name}")


def get_device(device: str = "auto") -> torch.device:
    """Get the appropriate device for computation.
    
    Args:
        device: Device specification ("auto", "cpu", "cuda", "mps")
        
    Returns:
        PyTorch device
    """
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device)


def set_seed(seed: int):
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed
    """
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    logger.info(f"Set random seed to {seed}")


def compute_returns(rewards: list[float], gamma: float = 0.99) -> list[float]:
    """Compute discounted returns.
    
    Args:
        rewards: List of rewards
        gamma: Discount factor
        
    Returns:
        List of discounted returns
    """
    returns = []
    running_return = 0.0
    
    for reward in reversed(rewards):
        running_return = reward + gamma * running_return
        returns.insert(0, running_return)
    
    return returns


def compute_gae(
    rewards: list[float],
    values: list[float],
    next_value: float,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> list[float]:
    """Compute Generalized Advantage Estimation.
    
    Args:
        rewards: List of rewards
        values: List of value estimates
        next_value: Value estimate for next state
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        
    Returns:
        List of advantage estimates
    """
    advantages = []
    running_advantage = 0.0
    
    for i in reversed(range(len(rewards))):
        if i == len(rewards) - 1:
            next_non_terminal = 0.0
            next_val = next_value
        else:
            next_non_terminal = 1.0
            next_val = values[i + 1]
        
        delta = rewards[i] + gamma * next_val * next_non_terminal - values[i]
        running_advantage = delta + gamma * gae_lambda * next_non_terminal * running_advantage
        advantages.insert(0, running_advantage)
    
    return advantages
