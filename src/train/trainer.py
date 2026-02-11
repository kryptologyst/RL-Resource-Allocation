"""Training framework for RL resource allocation agents."""

from __future__ import annotations

import torch
import numpy as np
import gymnasium as gym
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import time
from pathlib import Path
import json
from dataclasses import dataclass, asdict
from tqdm import tqdm

from algorithms import AlgorithmFactory, get_device, set_seed
from envs import make_resource_allocation_env
from utils.logging import setup_logging, log_metrics
from utils.checkpointing import CheckpointManager

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Environment
    env_name: str = "resource_allocation"
    num_resources: int = 3
    num_tasks: int = 5
    episode_length: int = 100
    
    # Algorithm
    algorithm: str = "dqn"
    algorithm_config: Dict[str, Any] = None
    
    # Training
    total_timesteps: int = 100000
    eval_freq: int = 10000
    eval_episodes: int = 10
    save_freq: int = 50000
    log_freq: int = 1000
    
    # Reproducibility
    seed: int = 42
    device: str = "auto"
    
    # Paths
    output_dir: str = "./outputs"
    experiment_name: str = "resource_allocation_experiment"
    
    def __post_init__(self):
        if self.algorithm_config is None:
            self.algorithm_config = {}


class Trainer:
    """Trainer for RL agents."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        
        # Setup logging
        self.logger = setup_logging(
            log_level="INFO",
            log_file=Path(config.output_dir) / f"{config.experiment_name}.log"
        )
        
        # Setup device
        self.device = get_device(config.device)
        
        # Set seed
        set_seed(config.seed)
        
        # Create output directory
        self.output_dir = Path(config.output_dir) / config.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        with open(self.output_dir / "config.json", "w") as f:
            json.dump(asdict(config), f, indent=2)
        
        # Initialize environment
        self.env = make_resource_allocation_env(
            num_resources=config.num_resources,
            num_tasks=config.num_tasks,
            episode_length=config.episode_length,
            seed=config.seed,
        )
        
        # Get environment dimensions
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n if hasattr(self.env.action_space, 'n') else self.env.action_space.shape[0]
        
        # Initialize agent
        self.agent = AlgorithmFactory.create_agent(
            algorithm=config.algorithm,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            config=config.algorithm_config,
            device=self.device,
        )
        
        # Initialize checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            save_dir=self.output_dir / "checkpoints",
            max_checkpoints=5,
        )
        
        # Training metrics
        self.training_metrics = {
            "episode_rewards": [],
            "episode_lengths": [],
            "episode_costs": [],
            "episode_violations": [],
            "losses": [],
            "eval_rewards": [],
            "eval_costs": [],
            "eval_violations": [],
        }
        
        self.logger.info(f"Initialized trainer with config: {config}")
    
    def train(self) -> Dict[str, Any]:
        """Train the agent."""
        self.logger.info("Starting training...")
        
        start_time = time.time()
        episode_count = 0
        timestep_count = 0
        
        # Training loop
        with tqdm(total=self.config.total_timesteps, desc="Training") as pbar:
            while timestep_count < self.config.total_timesteps:
                episode_count += 1
                episode_metrics = self._train_episode()
                
                # Update metrics
                self.training_metrics["episode_rewards"].append(episode_metrics["total_reward"])
                self.training_metrics["episode_lengths"].append(episode_metrics["episode_length"])
                self.training_metrics["episode_costs"].append(episode_metrics["total_cost"])
                self.training_metrics["episode_violations"].append(episode_metrics["constraint_violations"])
                
                timestep_count += episode_metrics["episode_length"]
                pbar.update(episode_metrics["episode_length"])
                
                # Logging
                if episode_count % self.config.log_freq == 0:
                    self._log_training_metrics(episode_count, timestep_count)
                
                # Evaluation
                if timestep_count % self.config.eval_freq == 0:
                    eval_metrics = self._evaluate()
                    self.training_metrics["eval_rewards"].extend(eval_metrics["rewards"])
                    self.training_metrics["eval_costs"].extend(eval_metrics["costs"])
                    self.training_metrics["eval_violations"].extend(eval_metrics["violations"])
                    
                    self.logger.info(
                        f"Evaluation at {timestep_count} timesteps: "
                        f"Reward: {np.mean(eval_metrics['rewards']):.3f} ± {np.std(eval_metrics['rewards']):.3f}, "
                        f"Cost: {np.mean(eval_metrics['costs']):.3f} ± {np.std(eval_metrics['costs']):.3f}, "
                        f"Violations: {np.mean(eval_metrics['violations']):.3f} ± {np.std(eval_metrics['violations']):.3f}"
                    )
                
                # Checkpointing
                if timestep_count % self.config.save_freq == 0:
                    self._save_checkpoint(timestep_count)
        
        # Final evaluation
        final_eval_metrics = self._evaluate()
        
        # Save final model
        self._save_checkpoint(timestep_count, is_final=True)
        
        # Training summary
        training_time = time.time() - start_time
        training_summary = {
            "total_timesteps": timestep_count,
            "total_episodes": episode_count,
            "training_time": training_time,
            "final_eval_reward": np.mean(final_eval_metrics["rewards"]),
            "final_eval_cost": np.mean(final_eval_metrics["costs"]),
            "final_eval_violations": np.mean(final_eval_metrics["violations"]),
        }
        
        self.logger.info(f"Training completed in {training_time:.2f} seconds")
        self.logger.info(f"Training summary: {training_summary}")
        
        return training_summary
    
    def _train_episode(self) -> Dict[str, Any]:
        """Train for one episode."""
        state, _ = self.env.reset()
        done = False
        episode_reward = 0.0
        episode_length = 0
        
        while not done:
            # Select action
            if self.config.algorithm == "dqn":
                action = self.agent.select_action(state, training=True)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # Store transition
                self.agent.store_transition(state, action, reward, next_state, done)
                
                # Update agent
                update_metrics = self.agent.update()
                if update_metrics:
                    self.training_metrics["losses"].append(update_metrics)
            
            elif self.config.algorithm == "ppo":
                action, log_prob, value = self.agent.select_action(state, training=True)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # Store transition
                self.agent.store_transition(state, action, log_prob, reward, value)
                
                # Update agent (PPO updates after episode)
                if done:
                    update_metrics = self.agent.update()
                    if update_metrics:
                        self.training_metrics["losses"].append(update_metrics)
            
            elif self.config.algorithm == "sac":
                action = self.agent.select_action(state, training=True)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # Store transition
                self.agent.store_transition(state, action, reward, next_state, done)
                
                # Update agent
                update_metrics = self.agent.update()
                if update_metrics:
                    self.training_metrics["losses"].append(update_metrics)
            
            episode_reward += reward
            episode_length += 1
            state = next_state
        
        return {
            "total_reward": episode_reward,
            "episode_length": episode_length,
            "total_cost": info.get("total_cost", 0.0),
            "constraint_violations": info.get("constraint_violations", 0.0),
        }
    
    def _evaluate(self) -> Dict[str, List[float]]:
        """Evaluate the agent."""
        eval_rewards = []
        eval_costs = []
        eval_violations = []
        
        for _ in range(self.config.eval_episodes):
            state, _ = self.env.reset()
            done = False
            episode_reward = 0.0
            
            while not done:
                if self.config.algorithm == "dqn":
                    action = self.agent.select_action(state, training=False)
                elif self.config.algorithm == "ppo":
                    action, _, _ = self.agent.select_action(state, training=False)
                elif self.config.algorithm == "sac":
                    action = self.agent.select_action(state, training=False)
                
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                state = next_state
            
            eval_rewards.append(episode_reward)
            eval_costs.append(info.get("total_cost", 0.0))
            eval_violations.append(info.get("constraint_violations", 0.0))
        
        return {
            "rewards": eval_rewards,
            "costs": eval_costs,
            "violations": eval_violations,
        }
    
    def _log_training_metrics(self, episode_count: int, timestep_count: int):
        """Log training metrics."""
        recent_rewards = self.training_metrics["episode_rewards"][-100:]
        recent_costs = self.training_metrics["episode_costs"][-100:]
        recent_violations = self.training_metrics["episode_violations"][-100:]
        
        self.logger.info(
            f"Episode {episode_count} ({timestep_count} timesteps): "
            f"Reward: {np.mean(recent_rewards):.3f} ± {np.std(recent_rewards):.3f}, "
            f"Cost: {np.mean(recent_costs):.3f} ± {np.std(recent_costs):.3f}, "
            f"Violations: {np.mean(recent_violations):.3f} ± {np.std(recent_violations):.3f}"
        )
    
    def _save_checkpoint(self, timestep_count: int, is_final: bool = False):
        """Save checkpoint."""
        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            agent=self.agent,
            timestep=timestep_count,
            metrics=self.training_metrics,
            is_final=is_final,
        )
        
        if is_final:
            self.logger.info(f"Saved final model to {checkpoint_path}")
        else:
            self.logger.info(f"Saved checkpoint at {timestep_count} timesteps")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint."""
        self.agent, timestep, metrics = self.checkpoint_manager.load_checkpoint(
            checkpoint_path, self.agent
        )
        
        self.training_metrics = metrics
        self.logger.info(f"Loaded checkpoint from {checkpoint_path} at {timestep} timesteps")
        
        return timestep
