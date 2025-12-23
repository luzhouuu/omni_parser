#!/usr/bin/env python3
"""Simple test script - no LLM required.

This script demonstrates basic browser automation:
1. Open a website
2. Take screenshots
3. Perform actions (click, type)

Usage:
    python scripts/simple_test.py
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from browser_agent.core.browser_controller import BrowserController


async def main():
    print("=" * 60)
    print("Browser Automation - Simple Test")
    print("=" * 60)

    # Create browser controller
    browser = BrowserController(
        headless=False,  # Show browser window
        viewport_width=1280,
        viewport_height=720,
    )

    try:
        # Start browser
        print("\n1. Starting browser...")
        await browser.start()
        print("   ✓ Browser started")

        # Navigate to Baidu
        print("\n2. Navigating to Baidu...")
        await browser.navigate("https://www.baidu.com")
        page_info = await browser.get_page_info()
        print(f"   ✓ Page loaded: {page_info['title']}")

        # Take screenshot
        print("\n3. Taking screenshot...")
        screenshot_bytes, _ = await browser.capture_screenshot()
        screenshot_path = "/tmp/browser_agent_screenshots/simple_test_1.png"
        Path(screenshot_path).parent.mkdir(parents=True, exist_ok=True)
        with open(screenshot_path, "wb") as f:
            f.write(screenshot_bytes)
        print(f"   ✓ Screenshot saved: {screenshot_path}")

        # Type in search box
        print("\n4. Typing in search box...")
        # Click on search box (approximate center position for Baidu)
        await browser.click_at(640, 350)
        await asyncio.sleep(0.5)
        await browser.type_text("不良反应")
        print("   ✓ Typed '不良反应'")

        # Take another screenshot
        print("\n5. Taking screenshot after typing...")
        screenshot_bytes, _ = await browser.capture_screenshot()
        screenshot_path = "/tmp/browser_agent_screenshots/simple_test_2.png"
        with open(screenshot_path, "wb") as f:
            f.write(screenshot_bytes)
        print(f"   ✓ Screenshot saved: {screenshot_path}")

        # Click search button
        print("\n6. Clicking search button...")
        await browser.click_at(900, 350)  # Approximate position of search button
        await asyncio.sleep(2)  # Wait for results

        # Final screenshot
        print("\n7. Taking final screenshot...")
        page_info = await browser.get_page_info()
        print(f"   Current URL: {page_info['url'][:60]}...")
        print(f"   Current Title: {page_info['title'][:40]}...")

        screenshot_bytes, _ = await browser.capture_screenshot()
        screenshot_path = "/tmp/browser_agent_screenshots/simple_test_3.png"
        with open(screenshot_path, "wb") as f:
            f.write(screenshot_bytes)
        print(f"   ✓ Screenshot saved: {screenshot_path}")

        print("\n" + "=" * 60)
        print("Test completed successfully!")
        print("=" * 60)
        print("\nScreenshots saved to /tmp/browser_agent_screenshots/")

        # Keep browser open for a moment to see the result
        print("\nBrowser will close in 5 seconds...")
        await asyncio.sleep(5)

    finally:
        print("\nClosing browser...")
        await browser.close()
        print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
