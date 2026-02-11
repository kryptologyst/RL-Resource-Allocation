"""Streamlit demo for RL Resource Allocation."""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import json
from pathlib import Path
import sys
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from algorithms import AlgorithmFactory, get_device
from envs import make_resource_allocation_env

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="RL Resource Allocation Demo",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">🎯 RL Resource Allocation Demo</h1>', unsafe_allow_html=True)

# Warning disclaimer
st.markdown("""
<div class="warning-box">
    <h4>⚠️ Important Disclaimer</h4>
    <p><strong>This is a research/educational demonstration only.</strong> This system is NOT intended for production use in real-world resource allocation scenarios. 
    The algorithms and models shown here are for educational and research purposes only.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.header("Configuration")

# Environment parameters
st.sidebar.subheader("Environment Settings")
num_resources = st.sidebar.slider("Number of Resources", 2, 10, 3)
num_tasks = st.sidebar.slider("Number of Tasks", 3, 15, 5)
episode_length = st.sidebar.slider("Episode Length", 50, 200, 100)

# Algorithm selection
st.sidebar.subheader("Algorithm Selection")
algorithm = st.sidebar.selectbox(
    "RL Algorithm",
    ["dqn", "ppo", "sac"],
    help="Choose the reinforcement learning algorithm"
)

# Load pre-trained model option
st.sidebar.subheader("Model Loading")
load_model = st.sidebar.checkbox("Load Pre-trained Model", value=False)
model_path = None

if load_model:
    uploaded_file = st.sidebar.file_uploader(
        "Upload Model File",
        type=['pth'],
        help="Upload a pre-trained model checkpoint"
    )
    if uploaded_file:
        # Save uploaded file temporarily
        model_path = Path("temp_model.pth")
        with open(model_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

# Initialize environment and agent
@st.cache_resource
def initialize_system(num_resources, num_tasks, episode_length, algorithm):
    """Initialize the environment and agent."""
    try:
        # Create environment
        env = make_resource_allocation_env(
            num_resources=num_resources,
            num_tasks=num_tasks,
            episode_length=episode_length,
            seed=42,
        )
        
        # Get environment dimensions
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n if hasattr(env.action_space, 'n') else env.action_space.shape[0]
        
        # Create agent
        agent = AlgorithmFactory.create_agent(
            algorithm=algorithm,
            state_dim=state_dim,
            action_dim=action_dim,
            config={},
            device=get_device("auto"),
        )
        
        return env, agent, state_dim, action_dim
    except Exception as e:
        st.error(f"Error initializing system: {str(e)}")
        return None, None, None, None

# Initialize system
env, agent, state_dim, action_dim = initialize_system(num_resources, num_tasks, episode_length, algorithm)

if env is None or agent is None:
    st.error("Failed to initialize the system. Please check your configuration.")
    st.stop()

# Load model if provided
if load_model and model_path and model_path.exists():
    try:
        checkpoint = torch.load(model_path, map_location="cpu")
        if "q_network" in checkpoint:  # DQN
            agent.q_network.load_state_dict(checkpoint["q_network"])
        elif "network" in checkpoint:  # PPO
            agent.network.load_state_dict(checkpoint["network"])
        elif "actor" in checkpoint:  # SAC
            agent.actor.load_state_dict(checkpoint["actor"])
        
        st.sidebar.success("Model loaded successfully!")
        
        # Clean up temp file
        model_path.unlink()
    except Exception as e:
        st.sidebar.error(f"Error loading model: {str(e)}")

# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs(["🎮 Interactive Demo", "📊 Analysis", "🔬 Algorithm Comparison", "📚 About"])

with tab1:
    st.header("Interactive Resource Allocation Demo")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Environment State")
        
        # Reset environment button
        if st.button("🔄 Reset Environment", type="primary"):
            state, _ = env.reset()
            st.session_state.state = state
            st.session_state.episode_reward = 0.0
            st.session_state.episode_length = 0
            st.session_state.episode_data = []
        
        # Initialize session state
        if "state" not in st.session_state:
            state, _ = env.reset()
            st.session_state.state = state
            st.session_state.episode_reward = 0.0
            st.session_state.episode_length = 0
            st.session_state.episode_data = []
        
        # Display current state
        if st.session_state.state is not None:
            state_data = st.session_state.state
            
            # Parse state components
            resource_availability = state_data[:num_resources]
            task_demands = state_data[num_resources:num_resources + num_tasks]
            current_allocations = state_data[num_resources + num_tasks:num_resources + num_tasks + (num_tasks * num_resources)]
            progress = state_data[-1]
            
            # Reshape allocations
            allocations_matrix = current_allocations.reshape(num_tasks, num_resources)
            
            # Display resource availability
            st.write("**Resource Availability:**")
            resource_df = pd.DataFrame({
                f"Resource {i+1}": [resource_availability[i]] for i in range(num_resources)
            })
            st.dataframe(resource_df, use_container_width=True)
            
            # Display task demands
            st.write("**Task Demands:**")
            task_df = pd.DataFrame({
                f"Task {i+1}": [task_demands[i]] for i in range(num_tasks)
            })
            st.dataframe(task_df, use_container_width=True)
            
            # Display current allocations
            st.write("**Current Allocations:**")
            allocation_df = pd.DataFrame(
                allocations_matrix,
                index=[f"Task {i+1}" for i in range(num_tasks)],
                columns=[f"Resource {i+1}" for i in range(num_resources)]
            )
            st.dataframe(allocation_df, use_container_width=True)
            
            # Progress bar
            st.write("**Episode Progress:**")
            st.progress(progress)
    
    with col2:
        st.subheader("Agent Actions")
        
        # Action selection
        if st.button("🎯 Take Action", type="secondary"):
            if st.session_state.state is not None:
                # Get action from agent
                if algorithm == "dqn":
                    action = agent.select_action(st.session_state.state, training=False)
                elif algorithm == "ppo":
                    action, _, _ = agent.select_action(st.session_state.state, training=False)
                elif algorithm == "sac":
                    action = agent.select_action(st.session_state.state, training=False)
                
                # Take step in environment
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # Update session state
                st.session_state.state = next_state
                st.session_state.episode_reward += reward
                st.session_state.episode_length += 1
                
                # Store episode data
                st.session_state.episode_data.append({
                    "step": st.session_state.episode_length,
                    "action": action,
                    "reward": reward,
                    "total_reward": st.session_state.episode_reward,
                    "cost": info.get("total_cost", 0.0),
                    "violations": info.get("constraint_violations", 0.0),
                })
                
                if done:
                    st.success(f"Episode completed! Total reward: {st.session_state.episode_reward:.3f}")
        
        # Episode statistics
        st.subheader("Episode Statistics")
        
        metric_col1, metric_col2 = st.columns(2)
        
        with metric_col1:
            st.metric("Total Reward", f"{st.session_state.episode_reward:.3f}")
            st.metric("Episode Length", st.session_state.episode_length)
        
        with metric_col2:
            if st.session_state.episode_data:
                latest_data = st.session_state.episode_data[-1]
                st.metric("Total Cost", f"{latest_data['cost']:.3f}")
                st.metric("Violations", f"{latest_data['violations']:.3f}")
    
    # Episode trajectory plot
    if st.session_state.episode_data:
        st.subheader("Episode Trajectory")
        
        episode_df = pd.DataFrame(st.session_state.episode_data)
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Reward per Step", "Cumulative Reward", "Cost per Step", "Violations per Step"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Reward per step
        fig.add_trace(
            go.Scatter(x=episode_df["step"], y=episode_df["reward"], mode="lines+markers", name="Reward"),
            row=1, col=1
        )
        
        # Cumulative reward
        fig.add_trace(
            go.Scatter(x=episode_df["step"], y=episode_df["total_reward"], mode="lines", name="Cumulative Reward"),
            row=1, col=2
        )
        
        # Cost per step
        fig.add_trace(
            go.Scatter(x=episode_df["step"], y=episode_df["cost"], mode="lines+markers", name="Cost"),
            row=2, col=1
        )
        
        # Violations per step
        fig.add_trace(
            go.Scatter(x=episode_df["step"], y=episode_df["violations"], mode="lines+markers", name="Violations"),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Resource Allocation Analysis")
    
    # Run multiple episodes for analysis
    if st.button("🔬 Run Analysis (10 episodes)", type="primary"):
        with st.spinner("Running analysis..."):
            analysis_results = []
            
            for episode in range(10):
                state, _ = env.reset()
                done = False
                episode_reward = 0.0
                episode_data = []
                
                while not done:
                    if algorithm == "dqn":
                        action = agent.select_action(state, training=False)
                    elif algorithm == "ppo":
                        action, _, _ = agent.select_action(state, training=False)
                    elif algorithm == "sac":
                        action = agent.select_action(state, training=False)
                    
                    next_state, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    
                    episode_data.append({
                        "step": len(episode_data) + 1,
                        "action": action,
                        "reward": reward,
                        "cost": info.get("total_cost", 0.0),
                        "violations": info.get("constraint_violations", 0.0),
                    })
                    
                    episode_reward += reward
                    state = next_state
                
                analysis_results.append({
                    "episode": episode + 1,
                    "total_reward": episode_reward,
                    "episode_length": len(episode_data),
                    "total_cost": info.get("total_cost", 0.0),
                    "total_violations": info.get("constraint_violations", 0.0),
                    "data": episode_data,
                })
            
            st.session_state.analysis_results = analysis_results
    
    # Display analysis results
    if "analysis_results" in st.session_state:
        results = st.session_state.analysis_results
        
        # Summary statistics
        st.subheader("Summary Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            rewards = [r["total_reward"] for r in results]
            st.metric("Mean Reward", f"{np.mean(rewards):.3f}", f"±{np.std(rewards):.3f}")
        
        with col2:
            costs = [r["total_cost"] for r in results]
            st.metric("Mean Cost", f"{np.mean(costs):.3f}", f"±{np.std(costs):.3f}")
        
        with col3:
            violations = [r["total_violations"] for r in results]
            st.metric("Mean Violations", f"{np.mean(violations):.3f}", f"±{np.std(violations):.3f}")
        
        with col4:
            lengths = [r["episode_length"] for r in results]
            st.metric("Mean Length", f"{np.mean(lengths):.1f}", f"±{np.std(lengths):.1f}")
        
        # Detailed analysis plots
        st.subheader("Detailed Analysis")
        
        # Episode comparison
        episode_df = pd.DataFrame(results)
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Episode Rewards", "Episode Costs", "Episode Violations", "Episode Lengths"),
        )
        
        fig.add_trace(
            go.Bar(x=episode_df["episode"], y=episode_df["total_reward"], name="Reward"),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=episode_df["episode"], y=episode_df["total_cost"], name="Cost"),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Bar(x=episode_df["episode"], y=episode_df["total_violations"], name="Violations"),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Bar(x=episode_df["episode"], y=episode_df["episode_length"], name="Length"),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Algorithm Comparison")
    
    st.info("This feature would compare different RL algorithms. In a full implementation, you would load multiple pre-trained models and compare their performance.")
    
    # Placeholder for algorithm comparison
    algorithms = ["DQN", "PPO", "SAC"]
    
    # Mock comparison data
    comparison_data = {
        "Algorithm": algorithms,
        "Mean Reward": [0.75, 0.82, 0.79],
        "Mean Cost": [0.45, 0.38, 0.42],
        "Mean Violations": [0.12, 0.08, 0.10],
        "Success Rate": [0.85, 0.92, 0.88],
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    
    st.subheader("Performance Comparison")
    st.dataframe(comparison_df, use_container_width=True)
    
    # Comparison plots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Mean Reward", "Mean Cost", "Mean Violations", "Success Rate"),
    )
    
    fig.add_trace(
        go.Bar(x=comparison_df["Algorithm"], y=comparison_df["Mean Reward"], name="Reward"),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=comparison_df["Algorithm"], y=comparison_df["Mean Cost"], name="Cost"),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Bar(x=comparison_df["Algorithm"], y=comparison_df["Mean Violations"], name="Violations"),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Bar(x=comparison_df["Algorithm"], y=comparison_df["Success Rate"], name="Success Rate"),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("About This Demo")
    
    st.markdown("""
    ## RL Resource Allocation System
    
    This interactive demo showcases a Reinforcement Learning system for resource allocation optimization.
    
    ### Features
    
    - **Multiple RL Algorithms**: DQN, PPO, and SAC implementations
    - **Realistic Environment**: Simulates resource allocation with constraints
    - **Interactive Visualization**: Real-time environment state and agent actions
    - **Performance Analysis**: Comprehensive evaluation metrics and plots
    
    ### Environment Details
    
    The resource allocation environment simulates:
    - Multiple resource types (CPU, Memory, Bandwidth, etc.)
    - Multiple tasks/services competing for resources
    - Resource capacity constraints
    - Task priority and demand variations
    - Cost-based optimization objectives
    
    ### Algorithms Implemented
    
    1. **DQN (Deep Q-Network)**: Value-based method with experience replay
    2. **PPO (Proximal Policy Optimization)**: Policy gradient method with clipping
    3. **SAC (Soft Actor-Critic)**: Off-policy method with entropy regularization
    
    ### Key Metrics
    
    - **Reward**: Efficiency and performance-based rewards
    - **Cost**: Resource usage costs
    - **Violations**: Constraint violation penalties
    - **Success Rate**: Percentage of successful episodes
    
    ### Technical Implementation
    
    - Built with PyTorch for neural networks
    - Uses Gymnasium for environment interface
    - Streamlit for interactive web interface
    - Plotly for dynamic visualizations
    
    ### Safety and Limitations
    
    This system is designed for educational and research purposes only. It should not be used for:
    - Production resource allocation systems
    - Real-world infrastructure management
    - Critical system operations
    
    The algorithms and models are simplified for demonstration purposes and may not perform optimally in real-world scenarios.
    """)
    
    st.subheader("Contact and Support")
    st.markdown("""
    For questions, issues, or contributions, please refer to the project repository.
    
    This demo is part of a larger RL research project focused on resource allocation optimization.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.8rem;">
    RL Resource Allocation Demo | Educational/Research Use Only | Not for Production
</div>
""", unsafe_allow_html=True)
