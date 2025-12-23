"""Logging utilities for browser automation agent."""

import sys
from typing import Optional
from loguru import logger


def get_logger(name: Optional[str] = None):
    """Get a configured logger instance.

    Args:
        name: Optional module name for the logger

    Returns:
        Configured loguru logger
    """
    # Remove default handler
    logger.remove()

    # Add custom handler with formatting
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )

    logger.add(
        sys.stderr,
        format=log_format,
        level="INFO",
        colorize=True,
    )

    # Add file handler for persistent logging
    logger.add(
        "/tmp/browser_agent.log",
        rotation="10 MB",
        retention="7 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
    )

    if name:
        return logger.bind(name=name)
    return logger


class StepLogger:
    """Logger for tracking agent steps with structured output."""

    def __init__(self, task_id: str):
        self.task_id = task_id
        self.step_count = 0
        self.logger = get_logger("StepLogger")

    def log_step_start(self, step_name: str, details: dict = None):
        """Log the start of a step.

        Args:
            step_name: Name of the step (observe, parse, plan, act, verify)
            details: Optional details dictionary
        """
        self.step_count += 1
        self.logger.info(
            f"[Task {self.task_id}] Step {self.step_count}: {step_name} started",
            extra={"details": details or {}},
        )

    def log_step_end(self, step_name: str, success: bool, details: dict = None):
        """Log the end of a step.

        Args:
            step_name: Name of the step
            success: Whether the step succeeded
            details: Optional details dictionary
        """
        status = "completed" if success else "failed"
        level = "info" if success else "warning"
        getattr(self.logger, level)(
            f"[Task {self.task_id}] Step {self.step_count}: {step_name} {status}",
            extra={"details": details or {}},
        )

    def log_action(self, action_type: str, target: str, result: str):
        """Log an executed action.

        Args:
            action_type: Type of action (click, type, etc.)
            target: Target element or URL
            result: Result of the action
        """
        self.logger.info(
            f"[Task {self.task_id}] Action: {action_type} on {target} -> {result}"
        )

    def log_error(self, error: str, recoverable: bool = True):
        """Log an error.

        Args:
            error: Error message
            recoverable: Whether the error is recoverable
        """
        level = "warning" if recoverable else "error"
        getattr(self.logger, level)(
            f"[Task {self.task_id}] Error: {error} (recoverable: {recoverable})"
        )
