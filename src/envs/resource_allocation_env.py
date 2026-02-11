"""Resource Allocation Environment for Reinforcement Learning.

This module implements a realistic resource allocation environment using Gymnasium,
where agents learn to optimally distribute limited resources across multiple tasks
or services to maximize efficiency and minimize costs.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Any, Dict, Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class ResourceAllocationEnv(gym.Env):
    """Resource allocation environment for RL training.
    
    The environment simulates a realistic resource allocation scenario where:
    - Multiple resources (CPU, Memory, Bandwidth) need to be allocated
    - Multiple tasks/services compete for these resources
    - The agent learns optimal allocation strategies
    - Rewards are based on efficiency, cost, and constraint satisfaction
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(
        self,
        num_resources: int = 3,
        num_tasks: int = 5,
        max_allocation_per_resource: float = 1.0,
        episode_length: int = 100,
        reward_scale: float = 1.0,
        cost_penalty: float = 0.1,
        constraint_violation_penalty: float = 10.0,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        """Initialize the resource allocation environment.
        
        Args:
            num_resources: Number of resource types (e.g., CPU, Memory, Bandwidth)
            num_tasks: Number of tasks/services requiring resources
            max_allocation_per_resource: Maximum allocation per resource type
            episode_length: Maximum steps per episode
            reward_scale: Scaling factor for rewards
            cost_penalty: Penalty for resource usage costs
            constraint_violation_penalty: Penalty for constraint violations
            render_mode: Rendering mode for visualization
            seed: Random seed for reproducibility
        """
        super().__init__()
        
        self.num_resources = num_resources
        self.num_tasks = num_tasks
        self.max_allocation_per_resource = max_allocation_per_resource
        self.episode_length = episode_length
        self.reward_scale = reward_scale
        self.cost_penalty = cost_penalty
        self.constraint_violation_penalty = constraint_violation_penalty
        self.render_mode = render_mode
        
        # Action space: allocation matrix [num_tasks x num_resources]
        # Each action represents allocation percentages for each task-resource pair
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(num_tasks, num_resources),
            dtype=np.float32,
        )
        
        # Observation space: [resource_availability, task_demands, current_allocations, step]
        obs_dim = num_resources + num_tasks + (num_tasks * num_resources) + 1
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        
        # Initialize state variables
        self.current_step = 0
        self.resource_availability = np.zeros(num_resources, dtype=np.float32)
        self.task_demands = np.zeros(num_tasks, dtype=np.float32)
        self.current_allocations = np.zeros((num_tasks, num_resources), dtype=np.float32)
        
        # Resource costs and task priorities (static for now)
        self.resource_costs = np.random.uniform(0.1, 1.0, num_resources)
        self.task_priorities = np.random.uniform(0.5, 1.0, num_tasks)
        
        # Performance tracking
        self.total_reward = 0.0
        self.total_cost = 0.0
        self.constraint_violations = 0
        
        self.seed(seed)
        
    def seed(self, seed: Optional[int] = None) -> Tuple[int, ...]:
        """Set the random seed for reproducibility."""
        if seed is not None:
            np.random.seed(seed)
        return super().seed(seed)
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state."""
        if seed is not None:
            self.seed(seed)
            
        # Reset episode variables
        self.current_step = 0
        self.total_reward = 0.0
        self.total_cost = 0.0
        self.constraint_violations = 0
        
        # Initialize resource availability (random for each episode)
        self.resource_availability = np.random.uniform(
            0.5, 1.0, self.num_resources
        ).astype(np.float32)
        
        # Initialize task demands (random for each episode)
        self.task_demands = np.random.uniform(
            0.3, 0.8, self.num_tasks
        ).astype(np.float32)
        
        # Initialize allocations to zero
        self.current_allocations = np.zeros(
            (self.num_tasks, self.num_resources), dtype=np.float32
        )
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        # Validate action
        action = np.clip(action, 0.0, 1.0)
        
        # Update allocations
        self.current_allocations = action.copy()
        
        # Calculate reward components
        efficiency_reward = self._calculate_efficiency_reward()
        cost_penalty = self._calculate_cost_penalty()
        constraint_penalty = self._calculate_constraint_penalty()
        
        # Total reward
        reward = (
            efficiency_reward * self.reward_scale
            - cost_penalty * self.cost_penalty
            - constraint_penalty * self.constraint_violation_penalty
        )
        
        # Update tracking variables
        self.total_reward += reward
        self.total_cost += cost_penalty
        
        # Check termination conditions
        terminated = self.current_step >= self.episode_length - 1
        truncated = False  # No truncation in this environment
        
        # Update step counter
        self.current_step += 1
        
        # Get next observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        obs_parts = [
            self.resource_availability,  # Resource availability
            self.task_demands,  # Task demands
            self.current_allocations.flatten(),  # Current allocations
            np.array([self.current_step / self.episode_length], dtype=np.float32),  # Progress
        ]
        return np.concatenate(obs_parts).astype(np.float32)
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information about the environment state."""
        return {
            "total_reward": self.total_reward,
            "total_cost": self.total_cost,
            "constraint_violations": self.constraint_violations,
            "resource_utilization": self._calculate_resource_utilization(),
            "allocation_efficiency": self._calculate_allocation_efficiency(),
        }
    
    def _calculate_efficiency_reward(self) -> float:
        """Calculate reward based on allocation efficiency."""
        # Reward based on how well allocations match task demands
        efficiency = 0.0
        for task_idx in range(self.num_tasks):
            task_allocation = np.sum(self.current_allocations[task_idx])
            task_demand = self.task_demands[task_idx]
            task_priority = self.task_priorities[task_idx]
            
            # Reward for meeting demand (with priority weighting)
            if task_allocation >= task_demand:
                efficiency += task_priority * 1.0
            else:
                # Partial reward for partial fulfillment
                efficiency += task_priority * (task_allocation / task_demand)
        
        return efficiency / self.num_tasks
    
    def _calculate_cost_penalty(self) -> float:
        """Calculate penalty based on resource usage costs."""
        total_cost = 0.0
        for task_idx in range(self.num_tasks):
            for resource_idx in range(self.num_resources):
                allocation = self.current_allocations[task_idx, resource_idx]
                resource_cost = self.resource_costs[resource_idx]
                total_cost += allocation * resource_cost
        
        return total_cost
    
    def _calculate_constraint_penalty(self) -> float:
        """Calculate penalty for constraint violations."""
        violations = 0.0
        
        # Check resource capacity constraints
        for resource_idx in range(self.num_resources):
            total_allocation = np.sum(self.current_allocations[:, resource_idx])
            available = self.resource_availability[resource_idx]
            
            if total_allocation > available:
                violations += (total_allocation - available) ** 2
        
        self.constraint_violations += violations
        return violations
    
    def _calculate_resource_utilization(self) -> np.ndarray:
        """Calculate resource utilization rates."""
        utilization = np.zeros(self.num_resources)
        for resource_idx in range(self.num_resources):
            total_allocation = np.sum(self.current_allocations[:, resource_idx])
            available = self.resource_availability[resource_idx]
            utilization[resource_idx] = total_allocation / available if available > 0 else 0.0
        
        return utilization
    
    def _calculate_allocation_efficiency(self) -> float:
        """Calculate overall allocation efficiency."""
        return self._calculate_efficiency_reward()
    
    def render(self) -> Optional[np.ndarray]:
        """Render the environment."""
        if self.render_mode == "human":
            print(f"Step: {self.current_step}")
            print(f"Resource Availability: {self.resource_availability}")
            print(f"Task Demands: {self.task_demands}")
            print(f"Current Allocations:\n{self.current_allocations}")
            print(f"Total Reward: {self.total_reward:.3f}")
            print(f"Total Cost: {self.total_cost:.3f}")
            print(f"Constraint Violations: {self.constraint_violations:.3f}")
            print("-" * 50)
        
        elif self.render_mode == "rgb_array":
            # Return a simple visualization as RGB array
            # This is a placeholder - in practice, you'd create a proper visualization
            return np.zeros((100, 100, 3), dtype=np.uint8)
        
        return None
    
    def close(self) -> None:
        """Clean up resources."""
        pass
