# RL Resource Allocation: Reinforcement Learning for Resource Optimization

A comprehensive Reinforcement Learning system for resource allocation optimization, featuring state-of-the-art algorithms, realistic environments, and interactive demonstrations.

## ⚠️ Important Disclaimer

**This system is for RESEARCH and EDUCATIONAL purposes ONLY. It is NOT intended for production use in real-world resource allocation scenarios.**

The algorithms, models, and implementations shown here are simplified demonstrations designed for learning and research. They should not be used for:
- Production resource allocation systems
- Real-world infrastructure management  
- Critical system operations
- Financial or healthcare applications

## Features

### Modern RL Algorithms
- **DQN (Deep Q-Network)**: Value-based method with experience replay, double DQN, and dueling architecture
- **PPO (Proximal Policy Optimization)**: Policy gradient method with clipping and GAE
- **SAC (Soft Actor-Critic)**: Off-policy method with entropy regularization

### Realistic Environment
- Multi-resource allocation simulation (CPU, Memory, Bandwidth, etc.)
- Multiple tasks/services competing for resources
- Resource capacity constraints and task priorities
- Cost-based optimization objectives
- Constraint violation penalties

### Comprehensive Evaluation
- Statistical analysis with confidence intervals
- Risk metrics (CVaR, tail events)
- Constraint satisfaction tracking
- Success rate and efficiency metrics
- Comparative analysis between algorithms

### Interactive Demo
- Streamlit web interface for real-time interaction
- Environment state visualization
- Agent action monitoring
- Performance analysis and plotting
- Model loading and evaluation

### Production-Ready Infrastructure
- Hydra/OmegaConf configuration management
- Structured logging and monitoring
- Checkpointing and model persistence
- Reproducible experiments with seeding
- Type hints and comprehensive documentation

## Requirements

- Python 3.10+
- PyTorch 2.0+
- Gymnasium 0.29+
- Streamlit 1.25+
- Additional dependencies in `requirements.txt`

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/kryptologyst/RL-Resource-Allocation.git
   cd RL-Resource-Allocation
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install in development mode (optional)**
   ```bash
   pip install -e .
   ```

## Quick Start

### Training an Agent

Train a DQN agent with default configuration:
```bash
python scripts/train.py
```

Train with custom algorithm:
```bash
python scripts/train.py algorithm=ppo
```

Train with custom environment settings:
```bash
python scripts/train.py env.num_resources=5 env.num_tasks=8 training.total_timesteps=200000
```

### Evaluating an Agent

Evaluate a trained model:
```bash
python scripts/evaluate.py --agent-path outputs/experiment/checkpoints/checkpoint_final.pth --config-path outputs/experiment/config.json --num-episodes 100
```

### Interactive Demo

Launch the Streamlit demo:
```bash
streamlit run demo/streamlit_demo.py
```

## 📁 Project Structure

```
rl-resource-allocation/
├── src/                          # Source code
│   ├── algorithms/              # RL algorithm implementations
│   │   ├── dqn.py              # Deep Q-Network
│   │   ├── ppo.py              # Proximal Policy Optimization
│   │   ├── sac.py              # Soft Actor-Critic
│   │   └── __init__.py         # Algorithm factory
│   ├── envs/                   # Environment implementations
│   │   ├── resource_allocation_env.py
│   │   └── __init__.py
│   ├── train/                  # Training framework
│   │   └── trainer.py
│   ├── eval/                   # Evaluation framework
│   │   └── evaluator.py
│   └── utils/                  # Utility modules
│       ├── logging.py
│       └── checkpointing.py
├── configs/                     # Configuration files
│   ├── config.yaml            # Main configuration
│   ├── algorithm/             # Algorithm configs
│   ├── env/                   # Environment configs
│   └── training/              # Training configs
├── scripts/                    # Training and evaluation scripts
│   ├── train.py
│   └── evaluate.py
├── demo/                       # Interactive demos
│   └── streamlit_demo.py
├── tests/                      # Unit tests
├── assets/                     # Generated assets and plots
├── data/                       # Data storage
├── requirements.txt            # Python dependencies
├── pyproject.toml             # Project configuration
└── README.md                  # This file
```

## 🔧 Configuration

The project uses Hydra for configuration management. Key configuration files:

- `configs/config.yaml`: Main configuration
- `configs/algorithm/`: Algorithm-specific settings
- `configs/env/`: Environment parameters
- `configs/training/`: Training hyperparameters

### Example Configuration Override

```bash
python scripts/train.py \
    algorithm=ppo \
    env.num_resources=4 \
    env.num_tasks=6 \
    training.total_timesteps=500000 \
    training.eval_freq=25000
```

## Evaluation Metrics

The system provides comprehensive evaluation metrics:

### Performance Metrics
- **Mean Reward**: Average episode reward with confidence intervals
- **Success Rate**: Percentage of episodes with positive rewards
- **Sample Efficiency**: Steps required to reach performance thresholds

### Safety and Constraint Metrics
- **Constraint Violations**: Frequency and severity of constraint violations
- **Cost Efficiency**: Resource usage costs and optimization
- **Risk Metrics**: CVaR, tail event analysis, and robustness measures

### Algorithm Comparison
- **Learning Curves**: Training progress over time
- **Final Performance**: Comparative analysis across algorithms
- **Ablation Studies**: Component-wise performance analysis

## Interactive Demo Features

The Streamlit demo provides:

1. **Real-time Environment Interaction**
   - Live environment state visualization
   - Agent action selection and execution
   - Episode progress tracking

2. **Performance Analysis**
   - Episode trajectory plotting
   - Statistical analysis across multiple episodes
   - Algorithm comparison visualizations

3. **Model Management**
   - Pre-trained model loading
   - Configuration adjustment
   - Real-time parameter modification

## Experiments and Ablations

### Supported Experiments

1. **Algorithm Comparison**
   ```bash
   python scripts/train.py algorithm=dqn
   python scripts/train.py algorithm=ppo  
   python scripts/train.py algorithm=sac
   ```

2. **Environment Scaling**
   ```bash
   python scripts/train.py env.num_resources=3 env.num_tasks=5
   python scripts/train.py env.num_resources=5 env.num_tasks=10
   ```

3. **Hyperparameter Sensitivity**
   ```bash
   python scripts/train.py algorithm.dqn.learning_rate=1e-4
   python scripts/train.py algorithm.dqn.learning_rate=1e-3
   ```

### Ablation Studies

- **Exploration Strategies**: Epsilon-greedy vs. entropy regularization
- **Network Architectures**: Standard vs. dueling DQN
- **Experience Replay**: Standard vs. prioritized experience replay
- **Reward Shaping**: Sparse vs. shaped rewards

## Research Applications

This system is designed for research in:

- **Resource Allocation Optimization**: Efficient distribution of limited resources
- **Constraint Satisfaction**: Learning under resource and operational constraints
- **Multi-objective Optimization**: Balancing efficiency, cost, and safety
- **Transfer Learning**: Adapting policies across different resource scenarios
- **Safe RL**: Learning with constraint violations and risk measures

## Safety and Limitations

### Known Limitations

1. **Simplified Environment**: The simulation is simplified for educational purposes
2. **Limited Scalability**: Not optimized for large-scale production systems
3. **No Real-time Constraints**: Does not handle real-time system requirements
4. **Static Environment**: Environment parameters are fixed during training

### Safety Considerations

1. **Constraint Violations**: Monitor and penalize constraint violations appropriately
2. **Risk Assessment**: Use risk metrics to evaluate policy safety
3. **Robustness Testing**: Test policies under various environmental conditions
4. **Human Oversight**: Always maintain human oversight in critical applications

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure code follows the project's style guidelines
5. Submit a pull request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install -e .[dev]

# Run tests
pytest tests/

# Format code
black src/ scripts/ demo/
ruff check src/ scripts/ demo/
```

## References and Further Reading

### Key Papers
- **DQN**: Mnih et al. (2015) "Human-level control through deep reinforcement learning"
- **Double DQN**: Van Hasselt et al. (2016) "Deep reinforcement learning with double Q-learning"
- **Dueling DQN**: Wang et al. (2016) "Dueling network architectures for deep reinforcement learning"
- **PPO**: Schulman et al. (2017) "Proximal policy optimization algorithms"
- **SAC**: Haarnoja et al. (2018) "Soft actor-critic: Off-policy maximum entropy deep reinforcement learning"

### Related Work
- Resource allocation in cloud computing
- Multi-agent resource allocation
- Constrained reinforcement learning
- Safe reinforcement learning

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- PyTorch team for the deep learning framework
- OpenAI Gym/Gymnasium for the RL environment interface
- Hydra team for configuration management
- Streamlit team for the web interface framework
- The broader RL research community for algorithms and methodologies

**Remember: This system is for educational and research purposes only. Do not use in production environments without proper validation and safety measures.**
# RL-Resource-Allocation
