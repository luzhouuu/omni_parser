#!/usr/bin/env python3
"""Test pagination using Playwright selectors."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from browser_agent.core.browser_controller import BrowserController
from browser_agent.utils.image_utils import save_screenshot


async def test_pagination_selector():
    """Test pagination using text-based selectors."""

    url = "https://lczl.med.wanfangdata.com.cn/ADR?pagesize=10"

    print("=" * 70)
    print("Pagination Test (using Playwright selectors)")
    print("=" * 70)
    print(f"URL: {url}")
    print("=" * 70)
    print()

    browser = BrowserController(headless=False, viewport_width=1280, viewport_height=720)
    await browser.start()
    page = browser.page

    try:
        # Navigate to URL
        print("[1] Navigating to page...")
        await browser.navigate(url)
        await asyncio.sleep(2)

        # Take initial screenshot
        screenshot_bytes, _ = await browser.capture_screenshot()
        save_screenshot(screenshot_bytes, "selector_page1.png")
        print("    Page 1 screenshot saved")

        page_info = await browser.get_page_info()
        print(f"    URL: {page_info['url']}")

        # Try to find "下一页" button using different selectors
        print()
        print("[2] Looking for '下一页' button...")

        # Try text-based selector
        next_page_selectors = [
            "text=下一页",
            "a:has-text('下一页')",
            "button:has-text('下一页')",
            ".pagination >> text=下一页",
            "//a[contains(text(), '下一页')]",
            "//span[contains(text(), '下一页')]",
        ]

        clicked = False
        for selector in next_page_selectors:
            try:
                element = page.locator(selector).first
                if await element.count() > 0:
                    print(f"    Found with selector: {selector}")

                    # Get bounding box
                    bbox = await element.bounding_box()
                    if bbox:
                        print(f"    Position: x={bbox['x']:.0f}, y={bbox['y']:.0f}, w={bbox['width']:.0f}, h={bbox['height']:.0f}")

                    # Click it
                    print("    Clicking...")
                    await element.click()
                    clicked = True
                    break
            except Exception as e:
                continue

        if not clicked:
            print("    Could not find '下一页' with standard selectors")
            print("    Trying to find any pagination links...")

            # Get all links on page and find one containing pagination keywords
            links = await page.query_selector_all("a")
            for link in links:
                text = await link.inner_text()
                if "下一页" in text or "next" in text.lower():
                    print(f"    Found link with text: {text}")
                    bbox = await link.bounding_box()
                    if bbox:
                        print(f"    Position: x={bbox['x']:.0f}, y={bbox['y']:.0f}")
                    await link.click()
                    clicked = True
                    break

        await asyncio.sleep(2)

        # Take screenshot after click
        screenshot_bytes, _ = await browser.capture_screenshot()
        save_screenshot(screenshot_bytes, "selector_page2.png")

        page_info = await browser.get_page_info()
        print(f"\n    After click URL: {page_info['url']}")

        if clicked:
            print("\n    SUCCESS: Clicked pagination button!")
        else:
            print("\n    FAILED: Could not find pagination button")

        # Also try to find "全文下载" links
        print()
        print("[3] Looking for '全文下载' links...")

        download_selectors = [
            "text=全文下载",
            "a:has-text('全文下载')",
            "//a[contains(text(), '全文下载')]",
        ]

        for selector in download_selectors:
            try:
                elements = page.locator(selector)
                count = await elements.count()
                if count > 0:
                    print(f"    Found {count} elements with selector: {selector}")

                    # Get first one's position
                    first = elements.first
                    bbox = await first.bounding_box()
                    if bbox:
                        print(f"    First element position: x={bbox['x']:.0f}, y={bbox['y']:.0f}")
                    break
            except Exception as e:
                continue

        print()
        print("=" * 70)
        print("Test Complete")
        print("=" * 70)

        return clicked

    finally:
        await browser.close()
        print("\nBrowser closed.")


if __name__ == "__main__":
    success = asyncio.run(test_pagination_selector())
    sys.exit(0 if success else 1)
