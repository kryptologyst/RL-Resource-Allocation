"""Logging utilities for RL training."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional
import json
import time
from datetime import datetime


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[Path] = None,
    log_format: Optional[str] = None,
) -> logging.Logger:
    """Setup logging configuration.
    
    Args:
        log_level: Logging level
        log_file: Path to log file
        log_format: Custom log format
        
    Returns:
        Configured logger
    """
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_formatter = logging.Formatter(log_format)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_formatter = logging.Formatter(log_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def log_metrics(
    metrics: Dict[str, Any],
    step: int,
    logger: Optional[logging.Logger] = None,
    prefix: str = "",
):
    """Log metrics to console and file.
    
    Args:
        metrics: Dictionary of metrics to log
        step: Current step/timestep
        logger: Logger instance
        prefix: Prefix for log messages
    """
    if logger is None:
        logger = logging.getLogger()
    
    log_msg = f"Step {step}"
    if prefix:
        log_msg = f"{prefix} - {log_msg}"
    
    for key, value in metrics.items():
        if isinstance(value, float):
            log_msg += f" | {key}: {value:.4f}"
        else:
            log_msg += f" | {key}: {value}"
    
    logger.info(log_msg)


class MetricsLogger:
    """Logger for training metrics."""
    
    def __init__(self, log_file: Optional[Path] = None):
        """Initialize metrics logger.
        
        Args:
            log_file: Path to metrics log file
        """
        self.log_file = log_file
        self.metrics_history = []
        
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
    
    def log(self, metrics: Dict[str, Any], step: int):
        """Log metrics.
        
        Args:
            metrics: Dictionary of metrics
            step: Current step
        """
        log_entry = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            **metrics,
        }
        
        self.metrics_history.append(log_entry)
        
        # Save to file
        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
    
    def get_metrics(self, metric_name: str) -> list[Any]:
        """Get history of a specific metric.
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            List of metric values
        """
        return [entry.get(metric_name) for entry in self.metrics_history if metric_name in entry]
    
    def get_latest_metrics(self) -> Optional[Dict[str, Any]]:
        """Get the latest metrics entry.
        
        Returns:
            Latest metrics dictionary or None
        """
        if self.metrics_history:
            return self.metrics_history[-1]
        return None
    
    def save_summary(self, output_file: Path):
        """Save metrics summary to file.
        
        Args:
            output_file: Path to save summary
        """
        if not self.metrics_history:
            return
        
        summary = {
            "total_entries": len(self.metrics_history),
            "first_entry": self.metrics_history[0],
            "last_entry": self.metrics_history[-1],
            "metrics_history": self.metrics_history,
        }
        
        with open(output_file, "w") as f:
            json.dump(summary, f, indent=2)
