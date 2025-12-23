"""Parse node - extracts UI elements from screenshot using OmniParser."""

from typing import Dict, Any

from browser_agent.agent.state import BrowserAgentState
from browser_agent.core.omni_parser import create_omni_parser
from browser_agent.utils.log_utils import get_logger

logger = get_logger(__name__)


async def parse_node(state: BrowserAgentState) -> Dict[str, Any]:
    """Parse screenshot using OmniParser to extract UI elements.

    This node:
    1. Loads the screenshot from the observe step
    2. Runs OmniParser detection and captioning
    3. Maps detections to UIElement objects
    4. Builds element_map for quick lookup

    Args:
        state: Current agent state

    Returns:
        State updates with parsed elements
    """
    logger.info(f"[Step {state.get('current_step', 0)}] Parsing screenshot...")

    screenshot_path = state.get("screenshot_path")
    if not screenshot_path:
        logger.error("No screenshot path available")
        return {
            "ui_elements": [],
            "element_map": {},
            "error": "No screenshot to parse",
        }

    # Get or create OmniParser
    parser = state.get("_omni_parser")
    config = state.get("config", {})

    if parser is None:
        parser = create_omni_parser(
            weights_dir=config.get("weights_dir"),
            use_mock=config.get("use_mock_parser", False),
        )

    # Parse screenshot
    try:
        elements, element_map = parser.parse_screenshot_to_elements(
            image_path=screenshot_path,
            viewport_width=state.get("viewport_width", 1280),
            viewport_height=state.get("viewport_height", 720),
            dpr=state.get("dpr", 1.0),
            use_ocr=True,  # Use OCR for Chinese text recognition
            caption_all=False,  # Disable slow Florence-2 captioning
        )

        # Convert UIElements to dictionaries for state storage
        ui_elements = [elem.to_dict() for elem in elements]
        element_map_dict = {k: v.to_dict() for k, v in element_map.items()}

        logger.info(f"Parsed {len(ui_elements)} UI elements")

        # Log some element info for debugging
        for elem in elements[:5]:  # First 5 elements
            logger.debug(f"  {elem.element_id}: {elem.text[:30] if elem.text else 'N/A'}...")

        return {
            "ui_elements": ui_elements,
            "element_map": element_map_dict,
            "_omni_parser": parser,
        }

    except Exception as e:
        logger.error(f"Failed to parse screenshot: {e}")
        return {
            "ui_elements": [],
            "element_map": {},
            "error": f"Parse error: {str(e)}",
        }
