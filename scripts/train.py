"""Main training script for RL resource allocation."""

import hydra
from omegaconf import DictConfig, OmegaConf
import logging
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from train.trainer import Trainer, TrainingConfig
from algorithms import set_seed

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function."""
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    logger.info("Starting RL Resource Allocation Training")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    # Set seed for reproducibility
    if cfg.deterministic:
        set_seed(cfg.seed)
        logger.info(f"Set random seed to {cfg.seed}")
    
    # Convert config to TrainingConfig
    training_config = TrainingConfig(
        env_name=cfg.env._target_.split(".")[-1],
        num_resources=cfg.env.num_resources,
        num_tasks=cfg.env.num_tasks,
        episode_length=cfg.env.episode_length,
        algorithm=cfg.algorithm.algorithm,
        algorithm_config=cfg.algorithm.algorithm_config,
        total_timesteps=cfg.training.total_timesteps,
        eval_freq=cfg.training.eval_freq,
        eval_episodes=cfg.training.eval_episodes,
        save_freq=cfg.training.save_freq,
        log_freq=cfg.training.log_freq,
        seed=cfg.seed,
        device=cfg.device,
        output_dir=cfg.output_dir,
        experiment_name=cfg.experiment_name,
    )
    
    # Create trainer
    trainer = Trainer(training_config)
    
    # Train the agent
    training_summary = trainer.train()
    
    logger.info("Training completed successfully!")
    logger.info(f"Training summary: {training_summary}")


if __name__ == "__main__":
    main()
