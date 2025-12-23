"""Observe node - captures current browser state."""

from typing import Dict, Any

from browser_agent.agent.state import BrowserAgentState
from browser_agent.core.browser_controller import BrowserController
from browser_agent.utils.log_utils import get_logger
from browser_agent.utils.image_utils import save_screenshot

logger = get_logger(__name__)


async def observe_node(state: BrowserAgentState) -> Dict[str, Any]:
    """Capture current browser state: screenshot, URL, title.

    This node:
    1. Gets or creates a browser controller
    2. Navigates to start_url if this is the first step
    3. Takes a screenshot of the current page
    4. Gets URL, title, viewport info

    Args:
        state: Current agent state

    Returns:
        State updates with screenshot and page info
    """
    logger.info(f"[Step {state.get('current_step', 0) + 1}] Observing page...")

    # Get or create browser controller
    browser = state.get("_browser_controller")
    config = state.get("config", {})

    if browser is None:
        logger.info("Starting browser...")
        browser = BrowserController(
            headless=config.get("headless", True),
            slow_mo=config.get("slow_mo", 0),
            timeout=config.get("timeout", 30000),
            viewport_width=config.get("viewport_width", 1280),
            viewport_height=config.get("viewport_height", 720),
            user_data_dir=config.get("user_data_dir"),
        )
        await browser.start()

        # Navigate to start URL if this is the first step
        start_url = state.get("start_url")
        if start_url and state.get("current_step", 0) == 0:
            await browser.navigate(start_url)

    # Store previous screenshot path for comparison
    previous_screenshot = state.get("screenshot_path")

    # Capture screenshot
    screenshot_bytes, screenshot_b64 = await browser.capture_screenshot()

    # Save screenshot
    step = state.get("current_step", 0) + 1
    task_id = state.get("task_id", "task")
    filename = f"{task_id}_step_{step:03d}.png"
    screenshot_path = save_screenshot(screenshot_bytes, filename)

    # Get page info
    page_info = await browser.get_page_info()

    logger.info(
        f"Observed: {page_info['title'][:50]}... at {page_info['url'][:50]}..."
    )

    return {
        "screenshot_path": screenshot_path,
        "screenshot_base64": screenshot_b64,
        "previous_screenshot_path": previous_screenshot,
        "current_url": page_info["url"],
        "page_title": page_info["title"],
        "viewport_width": page_info["viewport"]["width"],
        "viewport_height": page_info["viewport"]["height"],
        "dpr": page_info["dpr"],
        "current_step": step,
        "_browser_controller": browser,
        "error": None,  # Clear any previous error
    }
