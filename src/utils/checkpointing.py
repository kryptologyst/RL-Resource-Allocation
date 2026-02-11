"""Checkpointing utilities for RL training."""

from __future__ import annotations

import torch
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manager for saving and loading model checkpoints."""
    
    def __init__(
        self,
        save_dir: Path,
        max_checkpoints: int = 5,
        checkpoint_prefix: str = "checkpoint",
    ):
        """Initialize checkpoint manager.
        
        Args:
            save_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
            checkpoint_prefix: Prefix for checkpoint files
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.checkpoint_prefix = checkpoint_prefix
        
        logger.info(f"CheckpointManager initialized with save_dir: {self.save_dir}")
    
    def save_checkpoint(
        self,
        agent: Any,
        timestep: int,
        metrics: Dict[str, Any],
        is_final: bool = False,
        additional_data: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save a checkpoint.
        
        Args:
            agent: RL agent to save
            timestep: Current timestep
            metrics: Training metrics
            is_final: Whether this is the final checkpoint
            additional_data: Additional data to save
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint_data = {
            "timestep": timestep,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "agent_state": self._extract_agent_state(agent),
        }
        
        if additional_data:
            checkpoint_data.update(additional_data)
        
        # Determine checkpoint filename
        if is_final:
            checkpoint_name = f"{self.checkpoint_prefix}_final.pth"
        else:
            checkpoint_name = f"{self.checkpoint_prefix}_step_{timestep}.pth"
        
        checkpoint_path = self.save_dir / checkpoint_name
        
        # Save checkpoint
        torch.save(checkpoint_data, checkpoint_path)
        
        # Save metadata
        metadata_path = checkpoint_path.with_suffix(".json")
        with open(metadata_path, "w") as f:
            json.dump({
                "timestep": timestep,
                "timestamp": checkpoint_data["timestamp"],
                "is_final": is_final,
            }, f, indent=2)
        
        # Clean up old checkpoints
        if not is_final:
            self._cleanup_old_checkpoints()
        
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        return checkpoint_path
    
    def load_checkpoint(
        self,
        checkpoint_path: Path,
        agent: Any,
    ) -> Tuple[Any, int, Dict[str, Any]]:
        """Load a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            agent: Agent to load state into
            
        Returns:
            Tuple of (agent, timestep, metrics)
        """
        checkpoint_data = torch.load(checkpoint_path, map_location="cpu")
        
        # Load agent state
        self._load_agent_state(agent, checkpoint_data["agent_state"])
        
        timestep = checkpoint_data["timestep"]
        metrics = checkpoint_data["metrics"]
        
        logger.info(f"Loaded checkpoint from {checkpoint_path} at timestep {timestep}")
        return agent, timestep, metrics
    
    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get the latest checkpoint file.
        
        Returns:
            Path to latest checkpoint or None
        """
        checkpoint_files = list(self.save_dir.glob(f"{self.checkpoint_prefix}_step_*.pth"))
        
        if not checkpoint_files:
            return None
        
        # Sort by timestep
        checkpoint_files.sort(key=lambda x: int(x.stem.split("_")[-1]))
        return checkpoint_files[-1]
    
    def get_final_checkpoint(self) -> Optional[Path]:
        """Get the final checkpoint file.
        
        Returns:
            Path to final checkpoint or None
        """
        final_checkpoint = self.save_dir / f"{self.checkpoint_prefix}_final.pth"
        
        if final_checkpoint.exists():
            return final_checkpoint
        
        return None
    
    def list_checkpoints(self) -> list[Path]:
        """List all available checkpoints.
        
        Returns:
            List of checkpoint paths
        """
        checkpoint_files = list(self.save_dir.glob(f"{self.checkpoint_prefix}_*.pth"))
        checkpoint_files.sort(key=lambda x: x.stat().st_mtime)
        return checkpoint_files
    
    def _extract_agent_state(self, agent: Any) -> Dict[str, Any]:
        """Extract agent state for saving.
        
        Args:
            agent: RL agent
            
        Returns:
            Dictionary containing agent state
        """
        agent_state = {}
        
        # Extract state based on agent type
        if hasattr(agent, "q_network"):  # DQN
            agent_state.update({
                "q_network": agent.q_network.state_dict(),
                "target_network": agent.target_network.state_dict(),
                "optimizer": agent.optimizer.state_dict(),
                "step_count": agent.step_count,
                "epsilon": agent.epsilon,
            })
        elif hasattr(agent, "network"):  # PPO
            agent_state.update({
                "network": agent.network.state_dict(),
                "optimizer": agent.optimizer.state_dict(),
            })
        elif hasattr(agent, "actor"):  # SAC
            agent_state.update({
                "actor": agent.actor.state_dict(),
                "critic1": agent.critic1.state_dict(),
                "critic2": agent.critic2.state_dict(),
                "target_critic1": agent.target_critic1.state_dict(),
                "target_critic2": agent.target_critic2.state_dict(),
                "actor_optimizer": agent.actor_optimizer.state_dict(),
                "critic1_optimizer": agent.critic1_optimizer.state_dict(),
                "critic2_optimizer": agent.critic2_optimizer.state_dict(),
                "step_count": agent.step_count,
            })
            
            if hasattr(agent, "log_alpha"):
                agent_state["log_alpha"] = agent.log_alpha
                agent_state["alpha_optimizer"] = agent.alpha_optimizer.state_dict()
        
        return agent_state
    
    def _load_agent_state(self, agent: Any, agent_state: Dict[str, Any]):
        """Load agent state from checkpoint.
        
        Args:
            agent: RL agent to load state into
            agent_state: Agent state dictionary
        """
        # Load state based on agent type
        if hasattr(agent, "q_network"):  # DQN
            agent.q_network.load_state_dict(agent_state["q_network"])
            agent.target_network.load_state_dict(agent_state["target_network"])
            agent.optimizer.load_state_dict(agent_state["optimizer"])
            agent.step_count = agent_state["step_count"]
            agent.epsilon = agent_state["epsilon"]
        elif hasattr(agent, "network"):  # PPO
            agent.network.load_state_dict(agent_state["network"])
            agent.optimizer.load_state_dict(agent_state["optimizer"])
        elif hasattr(agent, "actor"):  # SAC
            agent.actor.load_state_dict(agent_state["actor"])
            agent.critic1.load_state_dict(agent_state["critic1"])
            agent.critic2.load_state_dict(agent_state["critic2"])
            agent.target_critic1.load_state_dict(agent_state["target_critic1"])
            agent.target_critic2.load_state_dict(agent_state["target_critic2"])
            agent.actor_optimizer.load_state_dict(agent_state["actor_optimizer"])
            agent.critic1_optimizer.load_state_dict(agent_state["critic1_optimizer"])
            agent.critic2_optimizer.load_state_dict(agent_state["critic2_optimizer"])
            agent.step_count = agent_state["step_count"]
            
            if "log_alpha" in agent_state:
                agent.log_alpha = agent_state["log_alpha"]
                agent.alpha_optimizer.load_state_dict(agent_state["alpha_optimizer"])
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints to maintain max_checkpoints limit."""
        checkpoint_files = list(self.save_dir.glob(f"{self.checkpoint_prefix}_step_*.pth"))
        
        if len(checkpoint_files) <= self.max_checkpoints:
            return
        
        # Sort by timestep
        checkpoint_files.sort(key=lambda x: int(x.stem.split("_")[-1]))
        
        # Remove oldest checkpoints
        files_to_remove = checkpoint_files[:-self.max_checkpoints]
        
        for file_path in files_to_remove:
            file_path.unlink()
            # Also remove metadata file
            metadata_path = file_path.with_suffix(".json")
            if metadata_path.exists():
                metadata_path.unlink()
            
            logger.info(f"Removed old checkpoint: {file_path}")
    
    def copy_checkpoint(self, source_path: Path, dest_path: Path):
        """Copy a checkpoint to a new location.
        
        Args:
            source_path: Source checkpoint path
            dest_path: Destination checkpoint path
        """
        shutil.copy2(source_path, dest_path)
        
        # Also copy metadata if it exists
        source_metadata = source_path.with_suffix(".json")
        dest_metadata = dest_path.with_suffix(".json")
        
        if source_metadata.exists():
            shutil.copy2(source_metadata, dest_metadata)
        
        logger.info(f"Copied checkpoint from {source_path} to {dest_path}")
    
    def get_checkpoint_info(self, checkpoint_path: Path) -> Dict[str, Any]:
        """Get information about a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Dictionary with checkpoint information
        """
        metadata_path = checkpoint_path.with_suffix(".json")
        
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                return json.load(f)
        
        # Fallback: load checkpoint data
        checkpoint_data = torch.load(checkpoint_path, map_location="cpu")
        return {
            "timestep": checkpoint_data.get("timestep", "unknown"),
            "timestamp": checkpoint_data.get("timestamp", "unknown"),
            "is_final": False,
        }
