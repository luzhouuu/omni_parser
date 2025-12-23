"""Action schemas for GPT tool outputs."""

from typing import Union, Literal, Optional
from pydantic import BaseModel, Field


class ClickAction(BaseModel):
    """Click on a UI element."""

    action_type: Literal["click"] = "click"
    element_id: str = Field(description="ID of element to click (e.g., 'elem_005')")
    click_type: Literal["single", "double", "right"] = Field(
        default="single",
        description="Type of click: single, double, or right-click"
    )


class TypeAction(BaseModel):
    """Type text into an input field."""

    action_type: Literal["type"] = "type"
    text: str = Field(description="Text to type")
    element_id: Optional[str] = Field(
        default=None,
        description="Target element ID. If None, types into currently focused element."
    )
    clear_first: bool = Field(
        default=False,
        description="Whether to clear existing text before typing"
    )


class ScrollAction(BaseModel):
    """Scroll the page."""

    action_type: Literal["scroll"] = "scroll"
    direction: Literal["up", "down", "left", "right"] = Field(
        description="Direction to scroll"
    )
    amount: int = Field(
        default=300,
        description="Amount to scroll in pixels"
    )


class NavigateAction(BaseModel):
    """Navigate to a URL."""

    action_type: Literal["navigate"] = "navigate"
    url: str = Field(description="The URL to navigate to")


class SelectAction(BaseModel):
    """Select an option from a dropdown."""

    action_type: Literal["select"] = "select"
    element_id: str = Field(description="The dropdown element ID")
    value: str = Field(description="The option value or visible text to select")


class WaitAction(BaseModel):
    """Wait for a condition."""

    action_type: Literal["wait"] = "wait"
    condition: Literal["time", "element", "url_change"] = Field(
        description="What to wait for: time (seconds), element (to appear), or url_change"
    )
    value: str = Field(
        description="Time in seconds, element ID to wait for, or expected URL pattern"
    )


class UploadAction(BaseModel):
    """Upload a file."""

    action_type: Literal["upload"] = "upload"
    element_id: str = Field(description="The file input element ID")
    file_path: str = Field(description="Path to the file to upload")


class DownloadAction(BaseModel):
    """Trigger a file download."""

    action_type: Literal["download"] = "download"
    element_id: str = Field(description="The download button/link element ID")
    save_as: Optional[str] = Field(
        default=None,
        description="Optional filename to save the download as"
    )


class ScreenshotAction(BaseModel):
    """Take a screenshot of the current state."""

    action_type: Literal["screenshot"] = "screenshot"
    annotation: Optional[str] = Field(
        default=None,
        description="Optional annotation or reason for the screenshot"
    )


class DoneAction(BaseModel):
    """Signal task completion."""

    action_type: Literal["done"] = "done"
    success: bool = Field(description="Whether the task was completed successfully")
    message: str = Field(description="Summary of what was accomplished or why it failed")


# Union type for all possible actions
AgentAction = Union[
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
]


def parse_action(action_dict: dict) -> AgentAction:
    """Parse a dictionary into the appropriate action type.

    Args:
        action_dict: Dictionary containing action_type and parameters

    Returns:
        Parsed action object

    Raises:
        ValueError: If action_type is unknown
    """
    action_type = action_dict.get("action_type")

    type_map = {
        "click": ClickAction,
        "type": TypeAction,
        "scroll": ScrollAction,
        "navigate": NavigateAction,
        "select": SelectAction,
        "wait": WaitAction,
        "upload": UploadAction,
        "download": DownloadAction,
        "screenshot": ScreenshotAction,
        "done": DoneAction,
    }

    if action_type not in type_map:
        raise ValueError(f"Unknown action type: {action_type}")

    return type_map[action_type](**action_dict)
