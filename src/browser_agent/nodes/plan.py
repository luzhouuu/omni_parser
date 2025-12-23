"""Plan node - uses GPT to determine next action."""

import base64
from typing import Dict, Any, List
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from browser_agent.agent.state import BrowserAgentState
from browser_agent.tools.definitions import get_all_tools
from browser_agent.utils.llm_utils import create_llm_client
from browser_agent.utils.log_utils import get_logger
from browser_agent.prompts.system_prompt import SYSTEM_PROMPT
from browser_agent.prompts.planning_prompt import PLANNING_PROMPT

logger = get_logger(__name__)


def format_elements_for_prompt(elements: List[Dict[str, Any]], max_elements: int = 50) -> str:
    """Format UI elements for inclusion in prompt.

    Args:
        elements: List of element dictionaries
        max_elements: Maximum number of elements to include

    Returns:
        Formatted string for prompt
    """
    if not elements:
        return "No elements detected."

    # Sort by y position first (top to bottom), then x (left to right)
    sorted_elements = sorted(
        elements,
        key=lambda e: (e.get("center_y", 0), e.get("center_x", 0)),
    )[:max_elements]

    lines = []
    for elem in sorted_elements:
        elem_id = elem.get("element_id", "unknown")
        elem_type = elem.get("element_type", "other")
        text = elem.get("text") or elem.get("description") or "UI element"
        confidence = elem.get("confidence", 0)
        center_x = elem.get("center_x", 0)
        center_y = elem.get("center_y", 0)

        # Truncate long text
        if len(text) > 50:
            text = text[:47] + "..."

        # Include position to help LLM match visual elements
        lines.append(f'[{elem_id}] {elem_type} at ({center_x:.0f}, {center_y:.0f}): "{text}"')

    return "\n".join(lines)


def format_action_history(history: List[Dict[str, Any]], max_items: int = 10) -> str:
    """Format action history for inclusion in prompt.

    Args:
        history: List of action dictionaries
        max_items: Maximum number of items to include

    Returns:
        Formatted string for prompt
    """
    if not history:
        return "No previous actions."

    recent = history[-max_items:]
    lines = []

    for i, action in enumerate(recent, 1):
        action_type = action.get("action_type", "unknown")
        target = action.get("element_id") or action.get("url") or action.get("text", "")[:30]
        result = action.get("result", "")[:50]

        lines.append(f"{i}. {action_type} on {target} -> {result}")

    return "\n".join(lines)


def load_screenshot_base64(screenshot_path: str) -> str:
    """Load screenshot as base64 string.

    Args:
        screenshot_path: Path to screenshot file

    Returns:
        Base64 encoded string
    """
    if not screenshot_path or not Path(screenshot_path).exists():
        return ""

    with open(screenshot_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


async def plan_node(state: BrowserAgentState) -> Dict[str, Any]:
    """Use GPT to determine next action based on goal and current state.

    This node:
    1. Builds context from task_goal, current state, and element list
    2. Calls LLM with tool definitions
    3. Parses structured output for next_action
    4. Updates current_plan with reasoning

    Args:
        state: Current agent state

    Returns:
        State updates with next_action and current_plan
    """
    logger.info(f"[Step {state.get('current_step', 0)}] Planning next action...")

    config = state.get("config", {})

    # Create LLM client with tools
    try:
        llm = create_llm_client(
            llm_type=config.get("llm_type"),
            model_name=config.get("model_name"),
            azure_endpoint=config.get("azure_endpoint"),
            azure_api_key=config.get("azure_api_key"),
            openai_api_key=config.get("openai_api_key"),
            temperature=0,
        )

        # Bind tools
        tools = get_all_tools()
        llm_with_tools = llm.bind_tools(tools, tool_choice="required")

    except Exception as e:
        logger.error(f"Failed to create LLM client: {e}")
        return {
            "error": f"LLM error: {str(e)}",
            "next_action": None,
        }

    # Build prompt context
    task_goal = state.get("task_goal", "No goal specified")
    current_url = state.get("current_url", "N/A")
    page_title = state.get("page_title", "N/A")
    elements = state.get("ui_elements", [])
    action_history = state.get("action_history", [])
    success_indicators = state.get("success_indicators", [])

    elements_str = format_elements_for_prompt(elements)
    history_str = format_action_history(action_history)
    indicators_str = ", ".join(success_indicators) if success_indicators else "None specified"

    # Format system prompt
    system_prompt = SYSTEM_PROMPT.format(
        task_goal=task_goal,
        success_indicators=indicators_str,
    )

    # Format planning prompt
    planning_prompt = PLANNING_PROMPT.format(
        current_url=current_url,
        page_title=page_title,
        element_list=elements_str,
        action_history=history_str,
    )

    # Build messages with screenshot for visual context
    screenshot_path = state.get("screenshot_path")
    screenshot_base64 = load_screenshot_base64(screenshot_path) if screenshot_path else ""

    if screenshot_base64:
        # Multimodal message with text and image
        human_content = [
            {
                "type": "text",
                "text": planning_prompt + "\n\n**IMPORTANT**: Look at the screenshot above to visually identify UI elements. Match element_ids with their visual positions. For search boxes, look for input fields. For search buttons, look for buttons labeled '搜索', '检索', 'Search', or similar.",
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{screenshot_base64}",
                },
            },
        ]
        logger.info("Including screenshot in LLM prompt for visual context")
    else:
        human_content = planning_prompt
        logger.warning("No screenshot available for visual context")

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_content),
    ]

    # Call LLM
    try:
        response = await llm_with_tools.ainvoke(messages)

        # Extract tool call from response
        if response.tool_calls:
            tool_call = response.tool_calls[0]
            action_type = tool_call["name"]
            action_args = tool_call["args"]

            next_action = {
                "action_type": action_type,
                **action_args,
            }

            # Extract reasoning if available
            current_plan = response.content if response.content else f"Executing {action_type}"

            logger.info(f"Planned action: {action_type}")
            logger.debug(f"Action args: {action_args}")

            return {
                "next_action": next_action,
                "current_plan": current_plan,
                "should_retry": False,
                "retry_count": 0,
            }

        else:
            # No tool call - this shouldn't happen with tool_choice="required"
            logger.warning("LLM did not return a tool call")
            return {
                "error": "LLM did not return an action",
                "next_action": None,
            }

    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return {
            "error": f"LLM error: {str(e)}",
            "next_action": None,
        }
