"""
Browser Automation Agent

A visual browser automation agent combining OmniParser + GPT + Playwright.
"""

from browser_agent.agent.wrapper import BrowserAutomationAgent
from browser_agent.agent.graph import create_browser_agent_graph

__version__ = "0.1.0"
__all__ = ["BrowserAutomationAgent", "create_browser_agent_graph"]
