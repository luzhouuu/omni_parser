"""Data models for browser automation."""

from browser_agent.models.ui_element import UIElement, ElementType
from browser_agent.models.action import (
    ClickAction,
    TypeAction,
    ScrollAction,
    NavigateAction,
    SelectAction,
    WaitAction,
    UploadAction,
    DownloadAction,
    ScreenshotAction,
    DoneAction,
    AgentAction,
)
from browser_agent.models.observation import Observation

__all__ = [
    "UIElement",
    "ElementType",
    "ClickAction",
    "TypeAction",
    "ScrollAction",
    "NavigateAction",
    "SelectAction",
    "WaitAction",
    "UploadAction",
    "DownloadAction",
    "ScreenshotAction",
    "DoneAction",
    "AgentAction",
    "Observation",
]
