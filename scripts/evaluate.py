"""Evaluation script for RL resource allocation agents."""

import hydra
from omegaconf import DictConfig, OmegaConf
import logging
from pathlib import Path
import sys
import argparse

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from eval.evaluator import Evaluator

logger = logging.getLogger(__name__)


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate RL Resource Allocation Agent")
    parser.add_argument("--agent-path", type=str, required=True, help="Path to saved agent")
    parser.add_argument("--config-path", type=str, required=True, help="Path to training config")
    parser.add_argument("--output-dir", type=str, default="./evaluation_results", help="Output directory")
    parser.add_argument("--num-episodes", type=int, default=100, help="Number of evaluation episodes")
    parser.add_argument("--render", action="store_true", help="Render episodes during evaluation")
    parser.add_argument("--save-trajectories", action="store_true", help="Save trajectory data")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    logger.info("Starting RL Resource Allocation Evaluation")
    logger.info(f"Agent path: {args.agent_path}")
    logger.info(f"Config path: {args.config_path}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Number of episodes: {args.num_episodes}")
    
    # Create evaluator
    evaluator = Evaluator(
        agent_path=args.agent_path,
        config_path=args.config_path,
        output_dir=args.output_dir,
    )
    
    # Evaluate the agent
    results = evaluator.evaluate(
        num_episodes=args.num_episodes,
        render=args.render,
        save_trajectories=args.save_trajectories,
    )
    
    logger.info("Evaluation completed successfully!")
    logger.info(f"Results: {results}")


if __name__ == "__main__":
    main()
