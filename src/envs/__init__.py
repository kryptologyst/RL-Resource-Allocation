"""Environment wrappers and utilities for resource allocation RL."""

from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Any, Dict, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class ResourceAllocationWrapper(gym.Wrapper):
    """Base wrapper for resource allocation environments."""
    
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.env = env


class NormalizeObservation(gym.ObservationWrapper):
    """Normalize observations to [0, 1] range."""
    
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.obs_low = np.zeros(self.observation_space.shape[0])
        self.obs_high = np.ones(self.observation_space.shape[0])
        
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=self.observation_space.shape,
            dtype=np.float32,
        )
    
    def observation(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observation."""
        return np.clip((obs - self.obs_low) / (self.obs_high - self.obs_low), 0.0, 1.0)


class RewardShaping(gym.RewardWrapper):
    """Apply reward shaping to encourage better behavior."""
    
    def __init__(
        self, 
        env: gym.Env, 
        efficiency_bonus: float = 0.1,
        constraint_penalty: float = 0.5,
    ):
        super().__init__(env)
        self.efficiency_bonus = efficiency_bonus
        self.constraint_penalty = constraint_penalty
    
    def reward(self, reward: float) -> float:
        """Apply reward shaping."""
        # Get additional info for shaping
        info = self.env.unwrapped._get_info()
        
        # Add efficiency bonus
        efficiency = info.get("allocation_efficiency", 0.0)
        shaped_reward = reward + efficiency * self.efficiency_bonus
        
        # Add constraint penalty
        violations = info.get("constraint_violations", 0.0)
        shaped_reward -= violations * self.constraint_penalty
        
        return shaped_reward


class ActionNoise(gym.ActionWrapper):
    """Add noise to actions for exploration."""
    
    def __init__(
        self, 
        env: gym.Env, 
        noise_std: float = 0.1,
        noise_decay: float = 0.995,
    ):
        super().__init__(env)
        self.noise_std = noise_std
        self.noise_decay = noise_decay
        self.current_noise_std = noise_std
    
    def action(self, action: np.ndarray) -> np.ndarray:
        """Add noise to action."""
        noise = np.random.normal(0, self.current_noise_std, action.shape)
        noisy_action = action + noise
        
        # Decay noise over time
        self.current_noise_std *= self.noise_decay
        
        return np.clip(noisy_action, 0.0, 1.0)


class EpisodeLogger(gym.Wrapper):
    """Log episode statistics."""
    
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_costs = []
        self.episode_violations = []
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset and log episode statistics."""
        obs, info = self.env.reset(**kwargs)
        
        # Log previous episode if it exists
        if hasattr(self, '_current_episode_reward'):
            self.episode_rewards.append(self._current_episode_reward)
            self.episode_lengths.append(self._current_episode_length)
            self.episode_costs.append(self._current_episode_cost)
            self.episode_violations.append(self._current_episode_violations)
        
        # Initialize current episode tracking
        self._current_episode_reward = 0.0
        self._current_episode_length = 0
        self._current_episode_cost = 0.0
        self._current_episode_violations = 0.0
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Step and track episode statistics."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        self._current_episode_reward += reward
        self._current_episode_length += 1
        self._current_episode_cost += info.get("total_cost", 0.0)
        self._current_episode_violations += info.get("constraint_violations", 0.0)
        
        return obs, reward, terminated, truncated, info
    
    def get_episode_stats(self) -> Dict[str, Any]:
        """Get episode statistics."""
        if not self.episode_rewards:
            return {}
        
        return {
            "mean_reward": np.mean(self.episode_rewards),
            "std_reward": np.std(self.episode_rewards),
            "mean_length": np.mean(self.episode_lengths),
            "mean_cost": np.mean(self.episode_costs),
            "mean_violations": np.mean(self.episode_violations),
            "num_episodes": len(self.episode_rewards),
        }


def make_resource_allocation_env(
    num_resources: int = 3,
    num_tasks: int = 5,
    episode_length: int = 100,
    normalize_obs: bool = True,
    reward_shaping: bool = True,
    action_noise: bool = False,
    episode_logging: bool = True,
    render_mode: Optional[str] = None,
    seed: Optional[int] = None,
) -> gym.Env:
    """Create a resource allocation environment with optional wrappers.
    
    Args:
        num_resources: Number of resource types
        num_tasks: Number of tasks/services
        episode_length: Maximum steps per episode
        normalize_obs: Whether to normalize observations
        reward_shaping: Whether to apply reward shaping
        action_noise: Whether to add action noise
        episode_logging: Whether to log episode statistics
        render_mode: Rendering mode
        seed: Random seed
        
    Returns:
        Configured environment with wrappers
    """
    from .resource_allocation_env import ResourceAllocationEnv
    
    env = ResourceAllocationEnv(
        num_resources=num_resources,
        num_tasks=num_tasks,
        episode_length=episode_length,
        render_mode=render_mode,
        seed=seed,
    )
    
    if normalize_obs:
        env = NormalizeObservation(env)
    
    if reward_shaping:
        env = RewardShaping(env)
    
    if action_noise:
        env = ActionNoise(env)
    
    if episode_logging:
        env = EpisodeLogger(env)
    
    return env
