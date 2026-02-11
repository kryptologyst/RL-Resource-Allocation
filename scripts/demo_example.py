"""Example script demonstrating RL Resource Allocation system."""

import sys
from pathlib import Path
import numpy as np
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from algorithms import AlgorithmFactory, set_seed
from envs import make_resource_allocation_env


def main():
    """Run a simple demonstration of the RL system."""
    print("🎯 RL Resource Allocation Demo")
    print("=" * 50)
    
    # Set seed for reproducibility
    set_seed(42)
    print("✓ Set random seed for reproducibility")
    
    # Create environment
    print("\n📦 Creating Resource Allocation Environment...")
    env = make_resource_allocation_env(
        num_resources=3,
        num_tasks=5,
        episode_length=50,
        seed=42,
    )
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print(f"✓ Environment created: {state_dim}D state, {action_dim}D action space")
    
    # Create DQN agent
    print("\n🤖 Creating DQN Agent...")
    agent = AlgorithmFactory.create_agent(
        algorithm="dqn",
        state_dim=state_dim,
        action_dim=action_dim,
        config={
            "learning_rate": 1e-3,
            "epsilon_start": 0.9,
            "epsilon_end": 0.01,
            "epsilon_decay": 1000,
            "buffer_size": 10000,
            "batch_size": 32,
        }
    )
    print("✓ DQN agent created")
    
    # Training loop
    print("\n🏋️ Training Agent...")
    num_episodes = 100
    episode_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0.0
        
        while not done:
            # Select action
            action = agent.select_action(state, training=True)
            
            # Take step
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done)
            
            # Update agent
            agent.update()
            
            episode_reward += reward
            state = next_state
        
        episode_rewards.append(episode_reward)
        
        # Print progress
        if (episode + 1) % 20 == 0:
            avg_reward = np.mean(episode_rewards[-20:])
            print(f"  Episode {episode + 1:3d}: Avg Reward = {avg_reward:.3f}")
    
    print(f"✓ Training completed! Final avg reward: {np.mean(episode_rewards[-10:]):.3f}")
    
    # Evaluation
    print("\n📊 Evaluating Trained Agent...")
    eval_rewards = []
    
    for episode in range(10):
        state, _ = env.reset()
        done = False
        episode_reward = 0.0
        
        while not done:
            action = agent.select_action(state, training=False)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            state = next_state
        
        eval_rewards.append(episode_reward)
    
    print(f"✓ Evaluation completed!")
    print(f"  Mean reward: {np.mean(eval_rewards):.3f} ± {np.std(eval_rewards):.3f}")
    print(f"  Min reward: {np.min(eval_rewards):.3f}")
    print(f"  Max reward: {np.max(eval_rewards):.3f}")
    
    # Demonstrate environment info
    print("\n🔍 Environment Information:")
    print(f"  Resource availability: {env.unwrapped.resource_availability}")
    print(f"  Task demands: {env.unwrapped.task_demands}")
    print(f"  Resource costs: {env.unwrapped.resource_costs}")
    print(f"  Task priorities: {env.unwrapped.task_priorities}")
    
    print("\n✅ Demo completed successfully!")
    print("\n⚠️  Remember: This is for educational/research purposes only!")
    print("   Do not use in production environments without proper validation.")


if __name__ == "__main__":
    main()
