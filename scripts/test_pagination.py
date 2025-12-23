#!/usr/bin/env python3
"""Test pagination functionality.

Usage:
    python scripts/test_pagination.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from browser_agent.core.browser_controller import BrowserController
from browser_agent.core.omni_parser import create_omni_parser
from browser_agent.nodes.act import execute_click_all_matching
from browser_agent.utils.log_utils import get_logger

logger = get_logger(__name__)


async def test_pagination():
    """Test pagination with click_all_matching."""

    url = "https://lczl.med.wanfangdata.com.cn/ADR?pagesize=10"

    print("=" * 70)
    print("Pagination Test")
    print("=" * 70)
    print(f"URL: {url}")
    print("Task: Click '全文下载' on each page, paginate through 3 pages")
    print("=" * 70)
    print()

    # Initialize browser
    browser = BrowserController(headless=False, viewport_width=1280, viewport_height=720)
    await browser.start()

    # Initialize OmniParser
    parser = create_omni_parser(use_mock=False)

    try:
        # Navigate to URL
        print("[1] Navigating to page...")
        await browser.navigate(url)
        await asyncio.sleep(2)

        # Take initial screenshot and parse
        print("[2] Taking screenshot and parsing UI elements...")
        screenshot_bytes, _ = await browser.capture_screenshot()

        from browser_agent.utils.image_utils import save_screenshot
        screenshot_path = save_screenshot(screenshot_bytes, "pagination_test_initial.png")

        elements, element_map = parser.parse_screenshot_to_elements(
            image_path=screenshot_path,
            viewport_width=1280,
            viewport_height=720,
            use_ocr=True,
        )
        element_map_dict = {k: v.to_dict() for k, v in element_map.items()}

        print(f"    Found {len(elements)} UI elements")

        # Show some elements
        for elem in elements[:10]:
            text = elem.text or elem.description or "N/A"
            print(f"    - [{elem.element_id}] {elem.element_type.value}: {text[:40]}")

        # Execute click_all_matching with pagination
        print()
        print("[3] Starting click_all_matching with pagination...")
        print("    Pattern: '全文下载'")
        print("    Paginate: True")
        print("    Max pages: 3")
        print()

        result = await execute_click_all_matching(
            browser=browser,
            element_map=element_map_dict,
            text_pattern="全文下载",
            delay_between_clicks=0.5,
            parser=parser,
            scroll_and_find=True,
            max_scroll_attempts=3,
            viewport_height=720,
            paginate=True,
            next_page_pattern="下一页",
            max_pages=3,
        )

        print()
        print("=" * 70)
        print("Results")
        print("=" * 70)
        print(f"Success: {result['success']}")
        print(f"Result: {result['result']}")
        print(f"Clicked count: {result.get('clicked_count', 0)}")
        print(f"Scroll attempts: {result.get('scroll_attempts', 0)}")
        print(f"Pages processed: {result.get('pages_processed', 0)}")

        # Take final screenshot
        print()
        print("[4] Taking final screenshot...")
        final_screenshot_bytes, _ = await browser.capture_screenshot()
        final_path = save_screenshot(final_screenshot_bytes, "pagination_test_final.png")
        print(f"    Saved to: {final_path}")

        return result

    finally:
        await browser.close()
        print()
        print("Browser closed.")


if __name__ == "__main__":
    result = asyncio.run(test_pagination())
    sys.exit(0 if result.get("success") else 1)
