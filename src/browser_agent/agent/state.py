"""Agent state definition for browser automation workflow."""

from typing import Annotated, List, Optional, Dict, Any, TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
import operator


class BrowserAgentState(TypedDict, total=False):
    """State for the browser automation agent.

    This state is passed between nodes in the LangGraph workflow.
    It contains all information needed for the Observe-Parse-Plan-Act-Verify loop.
    """

    # Task definition
    task_goal: str
    """The goal or objective to accomplish."""

    task_id: str
    """Unique identifier for this task execution."""

    max_steps: int
    """Maximum number of steps before stopping."""

    current_step: int
    """Current step number in the execution."""

    start_url: str
    """Initial URL to start from."""

    # Browser state
    current_url: str
    """Current page URL."""

    page_title: str
    """Current page title."""

    viewport_width: int
    """Viewport width in CSS pixels."""

    viewport_height: int
    """Viewport height in CSS pixels."""

    dpr: float
    """Device pixel ratio."""

    # Observation state
    screenshot_path: Optional[str]
    """Path to the current screenshot file."""

    screenshot_base64: Optional[str]
    """Base64 encoded screenshot."""

    previous_screenshot_path: Optional[str]
    """Path to the previous screenshot (for comparison)."""

    # Parsed elements
    ui_elements: List[Dict[str, Any]]
    """List of detected UI elements as dictionaries."""

    element_map: Dict[str, Dict[str, Any]]
    """Dictionary mapping element_id to element info."""

    # Planning state
    current_plan: Optional[str]
    """Current reasoning/plan from the LLM."""

    next_action: Optional[Dict[str, Any]]
    """The next action to execute."""

    action_history: Annotated[List[Dict[str, Any]], operator.add]
    """History of all executed actions."""

    # Verification state
    verification_result: Optional[Dict[str, Any]]
    """Result of the last action verification."""

    success_indicators: List[str]
    """List of indicators that signal task completion."""

    # Control flow
    is_complete: bool
    """Whether the task is complete."""

    should_retry: bool
    """Whether the current action should be retried."""

    retry_count: int
    """Number of retries for the current action."""

    # Login handling
    login_required: bool
    """Whether login popup was detected and login is needed."""

    # Error handling
    error: Optional[str]
    """Current error message, if any."""

    error_history: Annotated[List[str], operator.add]
    """History of errors encountered."""

    # Messages for LLM interactions
    messages: Annotated[List[BaseMessage], add_messages]
    """Conversation messages for the LLM."""

    # Configuration
    config: Optional[Dict[str, Any]]
    """Runtime configuration dictionary."""

    # Internal references (not serialized)
    _browser_controller: Optional[Any]
    """Reference to the browser controller instance."""

    _omni_parser: Optional[Any]
    """Reference to the OmniParser instance."""


def create_initial_state(
    task_goal: str,
    start_url: str,
    task_id: Optional[str] = None,
    max_steps: int = 50,
    success_indicators: Optional[List[str]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> BrowserAgentState:
    """Create an initial state for a new task.

    Args:
        task_goal: The goal or objective to accomplish
        start_url: Initial URL to start from
        task_id: Optional unique task identifier
        max_steps: Maximum number of steps
        success_indicators: Optional list of completion indicators
        config: Optional runtime configuration

    Returns:
        Initialized BrowserAgentState
    """
    import uuid

    return BrowserAgentState(
        # Task definition
        task_goal=task_goal,
        task_id=task_id or str(uuid.uuid4())[:8],
        max_steps=max_steps,
        current_step=0,
        start_url=start_url,
        # Browser state
        current_url="",
        page_title="",
        viewport_width=1280,
        viewport_height=720,
        dpr=1.0,
        # Observation state
        screenshot_path=None,
        screenshot_base64=None,
        previous_screenshot_path=None,
        # Parsed elements
        ui_elements=[],
        element_map={},
        # Planning state
        current_plan=None,
        next_action=None,
        action_history=[],
        # Verification state
        verification_result=None,
        success_indicators=success_indicators or [],
        # Control flow
        is_complete=False,
        should_retry=False,
        retry_count=0,
        # Login handling
        login_required=False,
        # Error handling
        error=None,
        error_history=[],
        # Messages
        messages=[],
        # Configuration
        config=config or {},
        # Internal references
        _browser_controller=None,
        _omni_parser=None,
    )


def state_summary(state: BrowserAgentState) -> str:
    """Generate a human-readable summary of the current state.

    Args:
        state: Current agent state

    Returns:
        Formatted summary string
    """
    lines = [
        f"=== Task: {state.get('task_id', 'unknown')} ===",
        f"Goal: {state.get('task_goal', 'unknown')[:100]}",
        f"Step: {state.get('current_step', 0)}/{state.get('max_steps', 50)}",
        f"URL: {state.get('current_url', 'N/A')}",
        f"Title: {state.get('page_title', 'N/A')[:50]}",
        f"Elements: {len(state.get('ui_elements', []))}",
        f"Actions: {len(state.get('action_history', []))}",
        f"Complete: {state.get('is_complete', False)}",
    ]

    if state.get("error"):
        lines.append(f"Error: {state['error'][:100]}")

    if state.get("next_action"):
        action = state["next_action"]
        lines.append(f"Next: {action.get('action_type', 'unknown')}")

    return "\n".join(lines)
