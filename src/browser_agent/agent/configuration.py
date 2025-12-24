"""Configuration management for Browser Agent."""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv, find_dotenv

# Load .env file (override=True to ensure .env values take precedence over system env vars)
load_dotenv(find_dotenv(), override=True)


def _env_or_default(name: str, default: str) -> str:
    """Get environment variable or return default."""
    v = os.getenv(name)
    return v if (v is not None and str(v).strip() != "") else default


# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
WEIGHTS_DIR = PROJECT_ROOT / _env_or_default("OMNIPARSER_WEIGHTS_DIR", "weights")

# LLM Configuration
LLM_TYPE = os.getenv("LLM_TYPE", "azure_openai")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
LLM_API_VERSION = os.getenv("LLM_API_VERSION", "2025-01-01-preview")
LLM_MODEL_NAME = _env_or_default("LLM_MODEL_NAME", "gpt-4o")

# OmniParser Configuration
OMNIPARSER_WEIGHTS_DIR = Path(_env_or_default("OMNIPARSER_WEIGHTS_DIR", str(WEIGHTS_DIR)))
OMNIPARSER_CONF_THRESHOLD = float(os.getenv("OMNIPARSER_CONF_THRESHOLD", "0.3"))
OMNIPARSER_IOU_THRESHOLD = float(os.getenv("OMNIPARSER_IOU_THRESHOLD", "0.5"))
OMNIPARSER_MAX_DETECTIONS = int(os.getenv("OMNIPARSER_MAX_DETECTIONS", "100"))

# Browser Configuration
BROWSER_HEADLESS = os.getenv("BROWSER_HEADLESS", "true").lower() == "true"
BROWSER_SLOW_MO = int(os.getenv("BROWSER_SLOW_MO", "0"))
BROWSER_TIMEOUT = int(os.getenv("BROWSER_TIMEOUT", "30000"))

# Agent Configuration
AGENT_MAX_STEPS = int(os.getenv("AGENT_MAX_STEPS", "50"))
AGENT_MAX_RETRIES = int(os.getenv("AGENT_MAX_RETRIES", "3"))

# Screenshot Configuration
SCREENSHOT_DIR = Path(_env_or_default("SCREENSHOT_DIR", "/tmp/browser_agent_screenshots"))


class AgentConfig:
    """Configuration container for browser automation agent."""

    def __init__(
        self,
        llm_type: str = LLM_TYPE,
        model_name: str = LLM_MODEL_NAME,
        azure_endpoint: Optional[str] = AZURE_OPENAI_ENDPOINT,
        azure_api_key: Optional[str] = AZURE_OPENAI_API_KEY,
        openai_api_key: Optional[str] = OPENAI_API_KEY,
        api_version: str = LLM_API_VERSION,
        weights_dir: Path = OMNIPARSER_WEIGHTS_DIR,
        conf_threshold: float = OMNIPARSER_CONF_THRESHOLD,
        max_detections: int = OMNIPARSER_MAX_DETECTIONS,
        headless: bool = BROWSER_HEADLESS,
        slow_mo: int = BROWSER_SLOW_MO,
        timeout: int = BROWSER_TIMEOUT,
        max_steps: int = AGENT_MAX_STEPS,
        max_retries: int = AGENT_MAX_RETRIES,
        screenshot_dir: Path = SCREENSHOT_DIR,
    ):
        # LLM settings
        self.llm_type = llm_type
        self.model_name = model_name
        self.azure_endpoint = azure_endpoint
        self.azure_api_key = azure_api_key
        self.openai_api_key = openai_api_key
        self.api_version = api_version

        # OmniParser settings
        self.weights_dir = weights_dir
        self.conf_threshold = conf_threshold
        self.max_detections = max_detections

        # Browser settings
        self.headless = headless
        self.slow_mo = slow_mo
        self.timeout = timeout

        # Agent settings
        self.max_steps = max_steps
        self.max_retries = max_retries
        self.screenshot_dir = screenshot_dir

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "llm_type": self.llm_type,
            "model_name": self.model_name,
            "azure_endpoint": self.azure_endpoint,
            "api_version": self.api_version,
            "weights_dir": str(self.weights_dir),
            "conf_threshold": self.conf_threshold,
            "max_detections": self.max_detections,
            "headless": self.headless,
            "slow_mo": self.slow_mo,
            "timeout": self.timeout,
            "max_steps": self.max_steps,
            "max_retries": self.max_retries,
            "screenshot_dir": str(self.screenshot_dir),
        }


# Default configuration instance
default_config = AgentConfig()
