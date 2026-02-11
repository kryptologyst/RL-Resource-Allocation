"""Evaluation framework for RL resource allocation agents."""

from __future__ import annotations

import torch
import numpy as np
import gymnasium as gym
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd

from algorithms import AlgorithmFactory, get_device
from envs import make_resource_allocation_env

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluator for RL agents."""
    
    def __init__(
        self,
        agent_path: str,
        config_path: str,
        output_dir: str = "./evaluation_results",
    ):
        """Initialize evaluator.
        
        Args:
            agent_path: Path to saved agent
            config_path: Path to training config
            output_dir: Directory to save evaluation results
        """
        self.agent_path = agent_path
        self.config_path = config_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load config
        with open(config_path, "r") as f:
            self.config = json.load(f)
        
        # Setup device
        self.device = get_device(self.config.get("device", "auto"))
        
        # Initialize environment
        self.env = make_resource_allocation_env(
            num_resources=self.config["num_resources"],
            num_tasks=self.config["num_tasks"],
            episode_length=self.config["episode_length"],
            seed=42,  # Fixed seed for evaluation
        )
        
        # Get environment dimensions
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n if hasattr(self.env.action_space, 'n') else self.env.action_space.shape[0]
        
        # Load agent
        self.agent = AlgorithmFactory.create_agent(
            algorithm=self.config["algorithm"],
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            config=self.config["algorithm_config"],
            device=self.device,
        )
        
        # Load agent weights
        checkpoint = torch.load(self.agent_path, map_location=self.device)
        if "q_network" in checkpoint:  # DQN
            self.agent.q_network.load_state_dict(checkpoint["q_network"])
        elif "network" in checkpoint:  # PPO
            self.agent.network.load_state_dict(checkpoint["network"])
        elif "actor" in checkpoint:  # SAC
            self.agent.actor.load_state_dict(checkpoint["actor"])
        
        logger.info(f"Loaded agent from {agent_path}")
    
    def evaluate(
        self,
        num_episodes: int = 100,
        render: bool = False,
        save_trajectories: bool = True,
    ) -> Dict[str, Any]:
        """Evaluate the agent.
        
        Args:
            num_episodes: Number of episodes to evaluate
            render: Whether to render episodes
            save_trajectories: Whether to save trajectory data
            
        Returns:
            Evaluation results
        """
        logger.info(f"Evaluating agent for {num_episodes} episodes...")
        
        episode_rewards = []
        episode_costs = []
        episode_violations = []
        episode_lengths = []
        trajectories = []
        
        for episode in range(num_episodes):
            trajectory = self._evaluate_episode(render=render)
            
            episode_rewards.append(trajectory["total_reward"])
            episode_costs.append(trajectory["total_cost"])
            episode_violations.append(trajectory["constraint_violations"])
            episode_lengths.append(trajectory["episode_length"])
            
            if save_trajectories:
                trajectories.append(trajectory)
        
        # Compute statistics
        results = self._compute_statistics(
            episode_rewards, episode_costs, episode_violations, episode_lengths
        )
        
        # Save results
        self._save_results(results, trajectories if save_trajectories else None)
        
        # Generate plots
        self._generate_plots(results)
        
        logger.info(f"Evaluation completed. Mean reward: {results['reward_mean']:.3f} ± {results['reward_std']:.3f}")
        
        return results
    
    def _evaluate_episode(self, render: bool = False) -> Dict[str, Any]:
        """Evaluate one episode."""
        state, _ = self.env.reset()
        done = False
        episode_reward = 0.0
        episode_length = 0
        states = [state.copy()]
        actions = []
        rewards = []
        
        while not done:
            # Select action
            if self.config["algorithm"] == "dqn":
                action = self.agent.select_action(state, training=False)
            elif self.config["algorithm"] == "ppo":
                action, _, _ = self.agent.select_action(state, training=False)
            elif self.config["algorithm"] == "sac":
                action = self.agent.select_action(state, training=False)
            
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            actions.append(action)
            rewards.append(reward)
            states.append(next_state.copy())
            
            episode_reward += reward
            episode_length += 1
            state = next_state
            
            if render:
                self.env.render()
        
        return {
            "total_reward": episode_reward,
            "total_cost": info.get("total_cost", 0.0),
            "constraint_violations": info.get("constraint_violations", 0.0),
            "episode_length": episode_length,
            "states": states,
            "actions": actions,
            "rewards": rewards,
        }
    
    def _compute_statistics(
        self,
        rewards: List[float],
        costs: List[float],
        violations: List[float],
        lengths: List[int],
    ) -> Dict[str, Any]:
        """Compute evaluation statistics."""
        # Basic statistics
        reward_mean = np.mean(rewards)
        reward_std = np.std(rewards)
        reward_ci = stats.t.interval(0.95, len(rewards)-1, loc=reward_mean, scale=stats.sem(rewards))
        
        cost_mean = np.mean(costs)
        cost_std = np.std(costs)
        cost_ci = stats.t.interval(0.95, len(costs)-1, loc=cost_mean, scale=stats.sem(costs))
        
        violation_mean = np.mean(violations)
        violation_std = np.std(violations)
        violation_ci = stats.t.interval(0.95, len(violations)-1, loc=violation_mean, scale=stats.sem(violations))
        
        length_mean = np.mean(lengths)
        length_std = np.std(lengths)
        
        # Additional metrics
        success_rate = np.mean([r > 0 for r in rewards])  # Episodes with positive reward
        constraint_satisfaction = np.mean([v == 0 for v in violations])  # Episodes with no violations
        
        # Risk metrics
        reward_cvar_95 = np.mean(np.sort(rewards)[:int(0.05 * len(rewards))])  # CVaR at 95%
        reward_cvar_99 = np.mean(np.sort(rewards)[:int(0.01 * len(rewards))])  # CVaR at 99%
        
        return {
            "reward_mean": reward_mean,
            "reward_std": reward_std,
            "reward_ci": reward_ci,
            "cost_mean": cost_mean,
            "cost_std": cost_std,
            "cost_ci": cost_ci,
            "violation_mean": violation_mean,
            "violation_std": violation_std,
            "violation_ci": violation_ci,
            "length_mean": length_mean,
            "length_std": length_std,
            "success_rate": success_rate,
            "constraint_satisfaction": constraint_satisfaction,
            "reward_cvar_95": reward_cvar_95,
            "reward_cvar_99": reward_cvar_99,
            "num_episodes": len(rewards),
        }
    
    def _save_results(self, results: Dict[str, Any], trajectories: Optional[List[Dict]] = None):
        """Save evaluation results."""
        # Save statistics
        with open(self.output_dir / "evaluation_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # Save trajectories if provided
        if trajectories:
            with open(self.output_dir / "trajectories.json", "w") as f:
                json.dump(trajectories, f, indent=2)
        
        logger.info(f"Saved evaluation results to {self.output_dir}")
    
    def _generate_plots(self, results: Dict[str, Any]):
        """Generate evaluation plots."""
        plt.style.use("seaborn-v0_8")
        
        # Reward distribution
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Load trajectory data for plotting
        try:
            with open(self.output_dir / "trajectories.json", "r") as f:
                trajectories = json.load(f)
            
            rewards = [t["total_reward"] for t in trajectories]
            costs = [t["total_cost"] for t in trajectories]
            violations = [t["constraint_violations"] for t in trajectories]
            
            # Reward distribution
            axes[0, 0].hist(rewards, bins=20, alpha=0.7, color="blue")
            axes[0, 0].axvline(results["reward_mean"], color="red", linestyle="--", label=f"Mean: {results['reward_mean']:.3f}")
            axes[0, 0].set_title("Reward Distribution")
            axes[0, 0].set_xlabel("Episode Reward")
            axes[0, 0].set_ylabel("Frequency")
            axes[0, 0].legend()
            
            # Cost distribution
            axes[0, 1].hist(costs, bins=20, alpha=0.7, color="green")
            axes[0, 1].axvline(results["cost_mean"], color="red", linestyle="--", label=f"Mean: {results['cost_mean']:.3f}")
            axes[0, 1].set_title("Cost Distribution")
            axes[0, 1].set_xlabel("Episode Cost")
            axes[0, 1].set_ylabel("Frequency")
            axes[0, 1].legend()
            
            # Violation distribution
            axes[1, 0].hist(violations, bins=20, alpha=0.7, color="orange")
            axes[1, 0].axvline(results["violation_mean"], color="red", linestyle="--", label=f"Mean: {results['violation_mean']:.3f}")
            axes[1, 0].set_title("Constraint Violations Distribution")
            axes[1, 0].set_xlabel("Episode Violations")
            axes[1, 0].set_ylabel("Frequency")
            axes[1, 0].legend()
            
            # Reward vs Cost scatter
            axes[1, 1].scatter(costs, rewards, alpha=0.6, color="purple")
            axes[1, 1].set_title("Reward vs Cost")
            axes[1, 1].set_xlabel("Episode Cost")
            axes[1, 1].set_ylabel("Episode Reward")
            
        except FileNotFoundError:
            # If no trajectory data, create summary plots
            metrics = ["reward_mean", "cost_mean", "violation_mean", "success_rate"]
            values = [results[m] for m in metrics]
            
            axes[0, 0].bar(metrics, values)
            axes[0, 0].set_title("Evaluation Metrics Summary")
            axes[0, 0].set_ylabel("Value")
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "evaluation_plots.png", dpi=300, bbox_inches="tight")
        plt.close()
        
        logger.info(f"Saved evaluation plots to {self.output_dir / 'evaluation_plots.png'}")
    
    def compare_agents(self, other_agent_paths: List[str], agent_names: List[str]) -> Dict[str, Any]:
        """Compare multiple agents.
        
        Args:
            other_agent_paths: Paths to other agents
            agent_names: Names for the agents
            
        Returns:
            Comparison results
        """
        logger.info(f"Comparing {len(other_agent_paths) + 1} agents...")
        
        all_results = {}
        
        # Evaluate current agent
        current_results = self.evaluate(num_episodes=50)
        all_results[agent_names[0]] = current_results
        
        # Evaluate other agents
        for i, agent_path in enumerate(other_agent_paths):
            # Create temporary evaluator for other agent
            temp_evaluator = Evaluator(
                agent_path=agent_path,
                config_path=self.config_path,
                output_dir=str(self.output_dir / f"agent_{i+1}"),
            )
            
            results = temp_evaluator.evaluate(num_episodes=50)
            all_results[agent_names[i+1]] = results
        
        # Generate comparison plots
        self._generate_comparison_plots(all_results)
        
        return all_results
    
    def _generate_comparison_plots(self, all_results: Dict[str, Dict[str, Any]]):
        """Generate comparison plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        agents = list(all_results.keys())
        
        # Reward comparison
        reward_means = [all_results[agent]["reward_mean"] for agent in agents]
        reward_stds = [all_results[agent]["reward_std"] for agent in agents]
        
        axes[0, 0].bar(agents, reward_means, yerr=reward_stds, capsize=5, alpha=0.7)
        axes[0, 0].set_title("Reward Comparison")
        axes[0, 0].set_ylabel("Mean Reward")
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Cost comparison
        cost_means = [all_results[agent]["cost_mean"] for agent in agents]
        cost_stds = [all_results[agent]["cost_std"] for agent in agents]
        
        axes[0, 1].bar(agents, cost_means, yerr=cost_stds, capsize=5, alpha=0.7, color="green")
        axes[0, 1].set_title("Cost Comparison")
        axes[0, 1].set_ylabel("Mean Cost")
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Violation comparison
        violation_means = [all_results[agent]["violation_mean"] for agent in agents]
        violation_stds = [all_results[agent]["violation_std"] for agent in agents]
        
        axes[1, 0].bar(agents, violation_means, yerr=violation_stds, capsize=5, alpha=0.7, color="orange")
        axes[1, 0].set_title("Constraint Violations Comparison")
        axes[1, 0].set_ylabel("Mean Violations")
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Success rate comparison
        success_rates = [all_results[agent]["success_rate"] for agent in agents]
        
        axes[1, 1].bar(agents, success_rates, alpha=0.7, color="purple")
        axes[1, 1].set_title("Success Rate Comparison")
        axes[1, 1].set_ylabel("Success Rate")
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "agent_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()
        
        logger.info(f"Saved comparison plots to {self.output_dir / 'agent_comparison.png'}")
