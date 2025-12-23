#!/usr/bin/env python3
"""Debug OCR recognition to see what text is being detected."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from browser_agent.core.browser_controller import BrowserController
from browser_agent.core.omni_parser import create_omni_parser
from browser_agent.utils.image_utils import save_screenshot


async def debug_ocr():
    """Debug OCR recognition."""

    url = "https://lczl.med.wanfangdata.com.cn/ADR?pagesize=10"

    print("=" * 70)
    print("OCR Debug Test")
    print("=" * 70)

    browser = BrowserController(headless=False, viewport_width=1280, viewport_height=720)
    await browser.start()
    parser = create_omni_parser(use_mock=False)

    try:
        await browser.navigate(url)
        await asyncio.sleep(2)

        # Scroll down to see more content
        await browser.scroll("down", 300)
        await asyncio.sleep(1)

        screenshot_bytes, _ = await browser.capture_screenshot()
        screenshot_path = save_screenshot(screenshot_bytes, "debug_ocr.png")

        print(f"\nScreenshot saved: {screenshot_path}")
        print("\n--- Parsing with OCR ---\n")

        elements, element_map = parser.parse_screenshot_to_elements(
            image_path=screenshot_path,
            viewport_width=1280,
            viewport_height=720,
            use_ocr=True,
        )

        print(f"Total elements: {len(elements)}\n")

        # Show all elements with their text
        print("All detected elements:")
        print("-" * 70)
        for elem in elements:
            text = elem.text or elem.description or "N/A"
            x, y = elem.page_center
            print(f"[{elem.element_id}] ({x:4.0f}, {y:4.0f}) {elem.element_type.value:10} | {text[:50]}")

        # Search for specific patterns
        print("\n" + "=" * 70)
        print("Searching for specific patterns:")
        print("=" * 70)

        patterns = ["全文", "下载", "下一页", "尾页", "页"]
        for pattern in patterns:
            matches = [e for e in elements if pattern in (e.text or "") or pattern in (e.description or "")]
            print(f"\n'{pattern}': {len(matches)} matches")
            for m in matches:
                text = m.text or m.description
                x, y = m.page_center
                print(f"  - [{m.element_id}] ({x:.0f}, {y:.0f}): {text[:40]}")

    finally:
        await browser.close()


if __name__ == "__main__":
    asyncio.run(debug_ocr())
