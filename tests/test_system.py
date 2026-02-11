"""Unit tests for RL Resource Allocation system."""

import pytest
import torch
import numpy as np
import gymnasium as gym
from pathlib import Path
import sys
import tempfile
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from algorithms import AlgorithmFactory, DQNConfig, PPOConfig, SACConfig, get_device, set_seed
from envs import ResourceAllocationEnv, make_resource_allocation_env
from utils.logging import setup_logging, MetricsLogger
from utils.checkpointing import CheckpointManager


class TestEnvironment:
    """Test the resource allocation environment."""
    
    def test_env_creation(self):
        """Test environment creation."""
        env = ResourceAllocationEnv(num_resources=3, num_tasks=5)
        
        assert env.num_resources == 3
        assert env.num_tasks == 5
        assert env.observation_space.shape[0] > 0
        assert env.action_space.shape == (5, 3)  # num_tasks x num_resources
    
    def test_env_reset(self):
        """Test environment reset."""
        env = ResourceAllocationEnv(num_resources=3, num_tasks=5, seed=42)
        obs, info = env.reset()
        
        assert isinstance(obs, np.ndarray)
        assert obs.shape[0] == env.observation_space.shape[0]
        assert isinstance(info, dict)
        assert "total_reward" in info
    
    def test_env_step(self):
        """Test environment step."""
        env = ResourceAllocationEnv(num_resources=3, num_tasks=5, seed=42)
        obs, _ = env.reset()
        
        # Create a valid action (allocation matrix)
        action = np.random.uniform(0, 1, (env.num_tasks, env.num_resources))
        
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        assert isinstance(next_obs, np.ndarray)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
    
    def test_env_wrappers(self):
        """Test environment wrappers."""
        env = make_resource_allocation_env(
            num_resources=3,
            num_tasks=5,
            normalize_obs=True,
            reward_shaping=True,
            episode_logging=True,
        )
        
        obs, _ = env.reset()
        assert isinstance(obs, np.ndarray)
        
        # Test episode logging
        if hasattr(env, 'get_episode_stats'):
            stats = env.get_episode_stats()
            assert isinstance(stats, dict)


class TestAlgorithms:
    """Test RL algorithms."""
    
    def test_dqn_creation(self):
        """Test DQN agent creation."""
        config = DQNConfig()
        agent = AlgorithmFactory.create_agent(
            algorithm="dqn",
            state_dim=10,
            action_dim=5,
            config=config.__dict__,
        )
        
        assert hasattr(agent, 'q_network')
        assert hasattr(agent, 'target_network')
        assert hasattr(agent, 'select_action')
        assert hasattr(agent, 'update')
    
    def test_ppo_creation(self):
        """Test PPO agent creation."""
        config = PPOConfig()
        agent = AlgorithmFactory.create_agent(
            algorithm="ppo",
            state_dim=10,
            action_dim=5,
            config=config.__dict__,
        )
        
        assert hasattr(agent, 'network')
        assert hasattr(agent, 'select_action')
        assert hasattr(agent, 'update')
    
    def test_sac_creation(self):
        """Test SAC agent creation."""
        config = SACConfig()
        agent = AlgorithmFactory.create_agent(
            algorithm="sac",
            state_dim=10,
            action_dim=5,
            config=config.__dict__,
        )
        
        assert hasattr(agent, 'actor')
        assert hasattr(agent, 'critic1')
        assert hasattr(agent, 'critic2')
        assert hasattr(agent, 'select_action')
        assert hasattr(agent, 'update')
    
    def test_dqn_action_selection(self):
        """Test DQN action selection."""
        agent = AlgorithmFactory.create_agent(
            algorithm="dqn",
            state_dim=10,
            action_dim=5,
        )
        
        state = np.random.randn(10)
        action = agent.select_action(state, training=True)
        
        assert isinstance(action, int)
        assert 0 <= action < 5
    
    def test_ppo_action_selection(self):
        """Test PPO action selection."""
        agent = AlgorithmFactory.create_agent(
            algorithm="ppo",
            state_dim=10,
            action_dim=5,
        )
        
        state = np.random.randn(10)
        action, log_prob, value = agent.select_action(state, training=True)
        
        assert isinstance(action, int)
        assert isinstance(log_prob, float)
        assert isinstance(value, float)
        assert 0 <= action < 5
    
    def test_sac_action_selection(self):
        """Test SAC action selection."""
        agent = AlgorithmFactory.create_agent(
            algorithm="sac",
            state_dim=10,
            action_dim=5,
        )
        
        state = np.random.randn(10)
        action = agent.select_action(state, training=True)
        
        assert isinstance(action, np.ndarray)
        assert action.shape == (5,)
        assert np.all(action >= 0) and np.all(action <= 1)


class TestUtilities:
    """Test utility functions."""
    
    def test_device_selection(self):
        """Test device selection."""
        device = get_device("auto")
        assert isinstance(device, torch.device)
        
        device = get_device("cpu")
        assert device.type == "cpu"
    
    def test_seed_setting(self):
        """Test random seed setting."""
        set_seed(42)
        
        # Test that seeds are set
        np_random_state = np.random.get_state()
        torch_random_state = torch.get_rng_state()
        
        assert np_random_state is not None
        assert torch_random_state is not None
    
    def test_logging_setup(self):
        """Test logging setup."""
        logger = setup_logging(log_level="INFO")
        assert isinstance(logger, logging.Logger)
    
    def test_metrics_logger(self):
        """Test metrics logger."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "metrics.log"
            metrics_logger = MetricsLogger(log_file)
            
            metrics_logger.log({"reward": 1.0, "loss": 0.5}, step=100)
            
            assert len(metrics_logger.metrics_history) == 1
            assert metrics_logger.get_latest_metrics()["step"] == 100
    
    def test_checkpoint_manager(self):
        """Test checkpoint manager."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_manager = CheckpointManager(
                save_dir=Path(temp_dir),
                max_checkpoints=3
            )
            
            # Create a mock agent
            class MockAgent:
                def __init__(self):
                    self.q_network = torch.nn.Linear(10, 5)
                    self.target_network = torch.nn.Linear(10, 5)
                    self.optimizer = torch.optim.Adam(self.q_network.parameters())
                    self.step_count = 0
                    self.epsilon = 1.0
            
            agent = MockAgent()
            metrics = {"reward": 1.0, "loss": 0.5}
            
            # Save checkpoint
            checkpoint_path = checkpoint_manager.save_checkpoint(
                agent=agent,
                timestep=1000,
                metrics=metrics
            )
            
            assert checkpoint_path.exists()
            
            # Load checkpoint
            loaded_agent, timestep, loaded_metrics = checkpoint_manager.load_checkpoint(
                checkpoint_path, agent
            )
            
            assert timestep == 1000
            assert loaded_metrics["reward"] == 1.0


class TestIntegration:
    """Integration tests."""
    
    def test_training_loop(self):
        """Test a short training loop."""
        env = ResourceAllocationEnv(num_resources=3, num_tasks=5, episode_length=10)
        agent = AlgorithmFactory.create_agent(
            algorithm="dqn",
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
        )
        
        # Short training loop
        for episode in range(3):
            state, _ = env.reset()
            done = False
            
            while not done:
                action = agent.select_action(state, training=True)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                agent.store_transition(state, action, reward, next_state, done)
                agent.update()
                
                state = next_state
        
        # Agent should have some experience
        assert len(agent.replay_buffer) > 0
    
    def test_evaluation(self):
        """Test evaluation process."""
        env = ResourceAllocationEnv(num_resources=3, num_tasks=5, episode_length=10)
        agent = AlgorithmFactory.create_agent(
            algorithm="dqn",
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
        )
        
        # Evaluate for a few episodes
        rewards = []
        for episode in range(3):
            state, _ = env.reset()
            done = False
            episode_reward = 0.0
            
            while not done:
                action = agent.select_action(state, training=False)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                state = next_state
            
            rewards.append(episode_reward)
        
        assert len(rewards) == 3
        assert all(isinstance(r, (int, float)) for r in rewards)


if __name__ == "__main__":
    pytest.main([__file__])
