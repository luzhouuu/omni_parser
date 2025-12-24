"""Tool definitions for GPT to interact with the browser.

These tools are bound to the LLM and provide a structured interface
for browser automation actions.
"""

from typing import Optional, Literal, List
from pydantic import BaseModel, Field
from langchain_core.tools import tool


# Tool Input Schemas

class ClickInput(BaseModel):
    """Input schema for click tool."""
    element_id: str = Field(
        description="The ID of the element to click (e.g., 'elem_005')"
    )
    click_type: Literal["single", "double", "right"] = Field(
        default="single",
        description="Type of click: single (default), double, or right-click"
    )


class ClickAtInput(BaseModel):
    """Input schema for click_at tool (coordinate-based)."""
    x: int = Field(
        description="X coordinate in CSS pixels"
    )
    y: int = Field(
        description="Y coordinate in CSS pixels"
    )
    click_type: Literal["single", "double", "right"] = Field(
        default="single",
        description="Type of click: single (default), double, or right-click"
    )


class TypeInput(BaseModel):
    """Input schema for type tool."""
    text: str = Field(
        description="The text to type"
    )
    element_id: Optional[str] = Field(
        default=None,
        description="Element to type into. If None, types into currently focused element."
    )
    clear_first: bool = Field(
        default=False,
        description="Whether to clear existing text before typing"
    )


class ScrollInput(BaseModel):
    """Input schema for scroll tool."""
    direction: Literal["up", "down", "left", "right"] = Field(
        description="Direction to scroll"
    )
    amount: int = Field(
        default=300,
        description="Amount to scroll in pixels (default: 300)"
    )


class NavigateInput(BaseModel):
    """Input schema for navigate tool."""
    url: str = Field(
        description="The URL to navigate to"
    )


class SelectInput(BaseModel):
    """Input schema for select tool."""
    element_id: str = Field(
        description="The dropdown element ID"
    )
    value: str = Field(
        description="The option value or visible text to select"
    )


class WaitInput(BaseModel):
    """Input schema for wait tool."""
    condition: Literal["time", "element", "url_change"] = Field(
        description="What to wait for: time (seconds), element (to appear), or url_change"
    )
    value: str = Field(
        description="Time in seconds, element ID to wait for, or expected URL pattern"
    )


class UploadInput(BaseModel):
    """Input schema for upload tool."""
    element_id: str = Field(
        description="The file input element ID"
    )
    file_path: str = Field(
        description="Path to the file to upload"
    )


class DownloadInput(BaseModel):
    """Input schema for download tool."""
    element_id: str = Field(
        description="The download button/link element ID"
    )
    save_as: Optional[str] = Field(
        default=None,
        description="Optional filename to save the download as"
    )


class ScreenshotInput(BaseModel):
    """Input schema for screenshot tool."""
    annotation: Optional[str] = Field(
        default=None,
        description="Optional annotation or reason for the screenshot"
    )


class ClickAllMatchingInput(BaseModel):
    """Input schema for click_all_matching tool."""
    text_pattern: str = Field(
        description="Text pattern to match (e.g., '下载全文', 'Download')"
    )
    delay_between_clicks: float = Field(
        default=1.0,
        description="Delay in seconds between each click (default: 1.0)"
    )
    scroll_and_find: bool = Field(
        default=True,
        description="Whether to scroll page and find more matching elements (default: True)"
    )
    max_scroll_attempts: int = Field(
        default=10,
        description="Maximum number of scroll attempts per page (default: 10)"
    )
    paginate: bool = Field(
        default=False,
        description="Whether to click next page button and continue processing on subsequent pages (default: False)"
    )
    next_page_pattern: str = Field(
        default="下一页",
        description="Text pattern to find next page button (default: '下一页')"
    )
    max_pages: int = Field(
        default=50,
        description="Maximum number of pages to process when paginate=True (default: 50)"
    )


class HandleLoginInput(BaseModel):
    """Input schema for handle_login tool."""
    pass  # No parameters needed - credentials come from environment variables


class DoneInput(BaseModel):
    """Input schema for done tool."""
    success: bool = Field(
        description="Whether the task was completed successfully"
    )
    message: str = Field(
        description="Summary of what was accomplished or why it failed"
    )


# Tool Definitions

@tool("click", args_schema=ClickInput)
def click_tool(element_id: str, click_type: str = "single") -> str:
    """Click on a UI element identified by its element ID.

    Use this tool to interact with buttons, links, checkboxes, and other clickable elements.
    The element_id should match one from the detected elements list.

    Args:
        element_id: The ID of the element to click (e.g., 'elem_005')
        click_type: Type of click - 'single' (default), 'double', or 'right'

    Returns:
        Result message indicating success or failure
    """
    # This is a placeholder - actual execution happens in act_node
    return f"Click {click_type} on element {element_id}"


@tool("click_at", args_schema=ClickAtInput)
def click_at_tool(x: int, y: int, click_type: str = "single") -> str:
    """Click at specific screen coordinates.

    Use this tool when you can see where to click in the screenshot but element_id mapping is uncertain.
    The coordinates should be in CSS pixels, matching the positions shown in the element list.

    IMPORTANT: Use coordinates from the element list or estimate based on visual position.
    For example, if a button appears at approximately x=500, y=100 in the screenshot, use those values.

    Args:
        x: X coordinate in CSS pixels
        y: Y coordinate in CSS pixels
        click_type: Type of click - 'single' (default), 'double', or 'right'

    Returns:
        Result message indicating success or failure
    """
    return f"Click {click_type} at coordinates ({x}, {y})"


@tool("type", args_schema=TypeInput)
def type_tool(
    text: str,
    element_id: Optional[str] = None,
    clear_first: bool = False,
) -> str:
    """Type text into an input field or text area.

    Use this tool to enter text into forms, search boxes, or any text input.
    If element_id is provided, the element will be clicked first to focus it.

    Args:
        text: The text to type
        element_id: Optional element to type into. If None, types into currently focused element.
        clear_first: Whether to clear existing text before typing

    Returns:
        Result message indicating success or failure
    """
    target = f" into element {element_id}" if element_id else ""
    return f"Type '{text[:50]}...'{target}"


@tool("scroll", args_schema=ScrollInput)
def scroll_tool(direction: str, amount: int = 300) -> str:
    """Scroll the page in a specified direction.

    Use this tool when you need to see more content that's not currently visible.
    Common use cases: scrolling to find elements, loading more content, navigating long pages.

    Args:
        direction: Direction to scroll - 'up', 'down', 'left', or 'right'
        amount: Amount to scroll in pixels (default: 300)

    Returns:
        Result message indicating the scroll action
    """
    return f"Scroll {direction} by {amount} pixels"


@tool("navigate", args_schema=NavigateInput)
def navigate_tool(url: str) -> str:
    """Navigate the browser to a specific URL.

    Use this tool to go to a new page or website.
    The URL should be a complete, valid URL including the protocol (https://).

    Args:
        url: The URL to navigate to

    Returns:
        Result message indicating navigation status
    """
    return f"Navigate to {url}"


@tool("select", args_schema=SelectInput)
def select_tool(element_id: str, value: str) -> str:
    """Select an option from a dropdown menu.

    Use this tool to choose an option from a select/dropdown element.
    The value can be either the option's value attribute or its visible text.

    Args:
        element_id: The dropdown element ID
        value: The option value or visible text to select

    Returns:
        Result message indicating selection status
    """
    return f"Select '{value}' from dropdown {element_id}"


@tool("wait", args_schema=WaitInput)
def wait_tool(condition: str, value: str) -> str:
    """Wait for a condition before proceeding.

    Use this tool when you need to wait for:
    - A specific amount of time (condition='time', value=seconds)
    - An element to appear (condition='element', value=element_id)
    - URL to change (condition='url_change', value=expected_pattern)

    Args:
        condition: What to wait for - 'time', 'element', or 'url_change'
        value: Depends on condition - seconds, element ID, or URL pattern

    Returns:
        Result message indicating wait completion
    """
    return f"Wait for {condition}: {value}"


@tool("upload", args_schema=UploadInput)
def upload_tool(element_id: str, file_path: str) -> str:
    """Upload a file to a file input element.

    Use this tool to upload files to file input fields.
    The element_id should point to an <input type="file"> element.

    Args:
        element_id: The file input element ID
        file_path: Path to the file to upload

    Returns:
        Result message indicating upload status
    """
    return f"Upload '{file_path}' to element {element_id}"


@tool("download", args_schema=DownloadInput)
def download_tool(element_id: str, save_as: Optional[str] = None) -> str:
    """Trigger a file download by clicking a download element.

    Use this tool to download files by clicking on download links or buttons.
    The file will be saved to the downloads directory.

    Args:
        element_id: The download button/link element ID
        save_as: Optional filename to save the download as

    Returns:
        Result message with download path
    """
    filename = f" as '{save_as}'" if save_as else ""
    return f"Download from element {element_id}{filename}"


@tool("screenshot", args_schema=ScreenshotInput)
def screenshot_tool(annotation: Optional[str] = None) -> str:
    """Take a screenshot of the current page state.

    Use this tool to capture the current state of the page.
    Useful for debugging or recording progress.

    Args:
        annotation: Optional annotation or reason for the screenshot

    Returns:
        Result message with screenshot path
    """
    reason = f" ({annotation})" if annotation else ""
    return f"Screenshot captured{reason}"


@tool("click_all_matching", args_schema=ClickAllMatchingInput)
def click_all_matching_tool(
    text_pattern: str,
    delay_between_clicks: float = 1.0,
    scroll_and_find: bool = True,
    max_scroll_attempts: int = 10,
    paginate: bool = False,
    next_page_pattern: str = "下一页",
    max_pages: int = 50,
) -> str:
    """Click all UI elements whose text matches the given pattern, with auto-scroll and pagination.

    Use this tool when you need to click multiple elements with similar text,
    such as clicking all "下载全文" (Download Full Text) buttons on a page.

    The tool will:
    1. Find all elements containing the text_pattern on current view
    2. Click each one sequentially with a delay between clicks
    3. Scroll down to find more matching elements on current page
    4. If paginate=True, click the next page button and repeat on new pages
    5. Report total elements clicked across all pages

    IMPORTANT: This is useful for batch operations like downloading multiple files.
    Set paginate=True to process multiple pages automatically.

    Args:
        text_pattern: Text to search for in element text (e.g., '下载全文', 'Download')
        delay_between_clicks: Seconds to wait between each click (default: 1.0)
        scroll_and_find: Whether to scroll and find more elements (default: True)
        max_scroll_attempts: Maximum scroll attempts per page (default: 10)
        paginate: Whether to click next page and continue on new pages (default: False)
        next_page_pattern: Text pattern for next page button (default: '下一页')
        max_pages: Maximum pages to process when paginating (default: 50)

    Returns:
        Result message with count of elements clicked
    """
    return f"Click all elements matching '{text_pattern}'"


@tool("handle_login", args_schema=HandleLoginInput)
def handle_login_tool() -> str:
    """Handle login popup automatically using pre-configured credentials.

    Use this tool when a login popup or login form appears on the page.
    This typically happens when trying to access protected content (like downloads).

    The tool will:
    1. Close the current browser session
    2. Execute the login script with credentials from environment variables
    3. Restart the browser and return to the original page

    IMPORTANT: Use this when you see a login form with password input field.
    The credentials are configured via WANFANG_USERNAME and WANFANG_PASSWORD environment variables.

    Returns:
        Result message indicating login status
    """
    return "Handle login using stored credentials"


@tool("done", args_schema=DoneInput)
def done_tool(success: bool, message: str) -> str:
    """Signal that the task is complete.

    Use this tool when you have completed the task (successfully or not).
    Provide a summary of what was accomplished.

    IMPORTANT: Only call this when the task is truly complete or cannot be completed.

    Args:
        success: Whether the task was completed successfully
        message: Summary of what was accomplished or why it failed

    Returns:
        Completion message
    """
    status = "successfully" if success else "with failure"
    return f"Task completed {status}: {message}"


def get_all_tools() -> List:
    """Get all available tools.

    Returns:
        List of all tool functions
    """
    return [
        click_tool,
        click_at_tool,
        type_tool,
        scroll_tool,
        navigate_tool,
        select_tool,
        wait_tool,
        upload_tool,
        download_tool,
        screenshot_tool,
        click_all_matching_tool,
        handle_login_tool,
        done_tool,
    ]


def get_tool_descriptions() -> str:
    """Get formatted descriptions of all tools.

    Returns:
        Formatted string with tool descriptions
    """
    tools = get_all_tools()
    descriptions = []

    for tool_func in tools:
        name = tool_func.name
        desc = tool_func.description.split("\n")[0]  # First line only
        descriptions.append(f"- {name}: {desc}")

    return "\n".join(descriptions)
