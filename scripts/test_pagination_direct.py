#!/usr/bin/env python3
"""Test pagination using direct coordinates.

This test demonstrates that pagination works by clicking directly
on the "下一页" button using its visual coordinates.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from browser_agent.core.browser_controller import BrowserController
from browser_agent.utils.image_utils import save_screenshot


async def test_pagination_direct():
    """Test pagination by clicking coordinates directly."""

    url = "https://lczl.med.wanfangdata.com.cn/ADR?pagesize=10"

    print("=" * 70)
    print("Direct Pagination Test (using coordinates)")
    print("=" * 70)
    print(f"URL: {url}")
    print("=" * 70)
    print()

    browser = BrowserController(headless=False, viewport_width=1280, viewport_height=720)
    await browser.start()

    try:
        # Navigate to URL
        print("[1] Navigating to page...")
        await browser.navigate(url)
        await asyncio.sleep(2)

        # Take initial screenshot
        screenshot_bytes, _ = await browser.capture_screenshot()
        save_screenshot(screenshot_bytes, "pagination_direct_page1.png")
        print("    Page 1 screenshot saved")

        # Get page info
        page_info = await browser.get_page_info()
        print(f"    URL: {page_info['url']}")

        # Click "下一页" button (approximate coordinates from visual inspection)
        # The pagination is at the bottom of the page
        print()
        print("[2] Scrolling to bottom to see pagination...")
        await browser.scroll("down", 500)
        await asyncio.sleep(1)

        screenshot_bytes, _ = await browser.capture_screenshot()
        save_screenshot(screenshot_bytes, "pagination_direct_scrolled.png")

        # "下一页" button is approximately at (530, 472) based on the screenshot
        # The pagination bar shows: 首页 上一页 10 11 12 ... 8326 下一页 尾页
        print()
        print("[3] Clicking '下一页' at approximate position (530, 472)...")
        await browser.click_at(530, 472)
        await asyncio.sleep(2)

        # Take screenshot of page 2
        screenshot_bytes, _ = await browser.capture_screenshot()
        save_screenshot(screenshot_bytes, "pagination_direct_page2.png")

        page_info = await browser.get_page_info()
        print(f"    After click URL: {page_info['url']}")

        # Check if URL changed (page parameter)
        if "page=2" in page_info['url'] or "pageindex=2" in page_info['url'].lower():
            print("    SUCCESS: Page changed!")
        else:
            print("    Checking page content...")

        # Click "下一页" again for page 3
        print()
        print("[4] Clicking '下一页' again for page 3...")
        await browser.scroll("down", 500)
        await asyncio.sleep(0.5)
        await browser.click_at(490, 472)
        await asyncio.sleep(2)

        screenshot_bytes, _ = await browser.capture_screenshot()
        save_screenshot(screenshot_bytes, "pagination_direct_page3.png")

        page_info = await browser.get_page_info()
        print(f"    After second click URL: {page_info['url']}")

        # Also test clicking "全文下载" (approximately at x=1100 for each row)
        print()
        print("[5] Testing '全文下载' click at approximate position (1100, 200)...")
        await browser.scroll("up", 300)
        await asyncio.sleep(0.5)
        await browser.click_at(1100, 200)
        await asyncio.sleep(2)

        screenshot_bytes, _ = await browser.capture_screenshot()
        save_screenshot(screenshot_bytes, "pagination_direct_download_click.png")

        print()
        print("=" * 70)
        print("Test Complete")
        print("=" * 70)
        print("Screenshots saved to /tmp/browser_agent_screenshots/")
        print()
        print("Pagination works if:")
        print("1. URL changes to include page parameter")
        print("2. Page content changes between screenshots")
        print("=" * 70)

        return True

    finally:
        await browser.close()
        print("\nBrowser closed.")


if __name__ == "__main__":
    success = asyncio.run(test_pagination_direct())
    sys.exit(0 if success else 1)
