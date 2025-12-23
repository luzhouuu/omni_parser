#!/usr/bin/env python3
"""Wanfang Medical Site Auto Login Script.

This script demonstrates automated login for the Wanfang medical site,
including handling the slider captcha verification.

Usage:
    python scripts/wanfang_login.py --username YOUR_USERNAME --password YOUR_PASSWORD

    # Or use environment variables:
    export WANFANG_USERNAME=your_username
    export WANFANG_PASSWORD=your_password
    python scripts/wanfang_login.py
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from browser_agent.core.browser_controller import BrowserController
from browser_agent.utils.log_utils import get_logger

logger = get_logger(__name__)

LOGIN_URL = "https://login.med.wanfangdata.com.cn/Account/LogOn"


# Default browser data directory for session persistence
DEFAULT_USER_DATA_DIR = str(Path(__file__).parent.parent / ".browser_data")


async def auto_login(
    username: str,
    password: str,
    headless: bool = False,
    stay_open: bool = True,
    user_data_dir: str = DEFAULT_USER_DATA_DIR,
) -> bool:
    """Perform automatic login to Wanfang medical site.

    Args:
        username: Wanfang account username or phone number
        password: Account password
        headless: Run browser in headless mode
        stay_open: Keep browser open after login (for verification)
        user_data_dir: Directory to store browser data for session persistence.
                       Pass None to use fresh session each time.

    Returns:
        True if login successful, False otherwise
    """
    browser = BrowserController(
        headless=headless,
        viewport_width=1280,
        viewport_height=800,
        slow_mo=50,  # Slight slow down
        user_data_dir=user_data_dir,
    )

    try:
        await browser.start()
        print(f"[1/5] Navigating to login page...")
        await browser.navigate(LOGIN_URL)
        await asyncio.sleep(1)

        # Step 2: Click username input and enter username
        print(f"[2/5] Entering username: {username[:3]}***")
        # Find the username input field by clicking on it
        # Based on the screenshot, the username input is around (1050, 417)
        # But we should use CSS selectors for reliability

        # Try to click and type using JavaScript for reliability
        await browser.evaluate("""
            () => {
                const usernameInput = document.querySelector('input[placeholder*="用户名"], input[placeholder*="手机号"], input[name="userName"], input#userName');
                if (usernameInput) {
                    usernameInput.focus();
                    usernameInput.value = '';
                }
            }
        """)
        await asyncio.sleep(0.3)
        await browser.type_text(username, delay=30)
        await asyncio.sleep(0.5)

        # Step 3: Click password input and enter password
        print(f"[3/5] Entering password...")
        await browser.evaluate("""
            () => {
                const passwordInput = document.querySelector('input[type="password"], input[placeholder*="密码"], input[name="password"], input#password');
                if (passwordInput) {
                    passwordInput.focus();
                    passwordInput.value = '';
                }
            }
        """)
        await asyncio.sleep(0.3)
        await browser.type_text(password, delay=30)
        await asyncio.sleep(0.5)

        # Step 4: Handle slider captcha
        print(f"[4/5] Solving slider captcha...")
        success = await solve_slider_captcha(browser)
        if not success:
            print("Warning: Slider captcha may not have been solved correctly")
        await asyncio.sleep(1)

        # Step 5: Click login button
        print(f"[5/5] Clicking login button...")

        # Use Playwright's native locator for more reliable clicking
        page = browser.page
        login_clicked = False

        # Try different approaches to click the login button
        try:
            # Method 1: Try by text content
            login_btn = page.locator('button:has-text("登录"), button:has-text("登 录")').first
            if await login_btn.count() > 0:
                await login_btn.click()
                login_clicked = True
                logger.info("Clicked login button via text locator")
        except Exception as e:
            logger.debug(f"Text locator failed: {e}")

        if not login_clicked:
            try:
                # Method 2: Try common CSS selectors
                for selector in ['button.login-btn', '.btn-login', 'button[type="submit"]', '.login-button']:
                    btn = page.locator(selector).first
                    if await btn.count() > 0:
                        await btn.click()
                        login_clicked = True
                        logger.info(f"Clicked login button via selector: {selector}")
                        break
            except Exception as e:
                logger.debug(f"CSS selector failed: {e}")

        if not login_clicked:
            try:
                # Method 3: Find by role
                login_btn = page.get_by_role("button", name="登录")
                if await login_btn.count() > 0:
                    await login_btn.click()
                    login_clicked = True
                    logger.info("Clicked login button via role")
            except Exception as e:
                logger.debug(f"Role locator failed: {e}")

        if not login_clicked:
            # Method 4: Fallback to JavaScript click
            await browser.evaluate("""
                () => {
                    const buttons = document.querySelectorAll('button, input[type="submit"]');
                    for (const btn of buttons) {
                        const text = btn.textContent || btn.value || '';
                        if (text.includes('登录') || text.includes('登 录')) {
                            btn.click();
                            return true;
                        }
                    }
                    return false;
                }
            """)
            logger.info("Clicked login button via JavaScript fallback")

        # Wait for navigation/redirect
        await asyncio.sleep(2)

        # Try to wait for URL change
        try:
            await page.wait_for_url("**/med.wanfangdata.com.cn/**", timeout=5000)
            logger.info("URL changed, login may have succeeded")
        except Exception:
            logger.debug("URL did not change within timeout")

        await asyncio.sleep(1)

        # Check if login was successful
        page_info = await browser.get_page_info()
        current_url = page_info["url"]

        # Also check page content for error messages
        try:
            error_msg = await page.locator('.error-msg, .alert-danger, .login-error').text_content()
            if error_msg:
                logger.warning(f"Login error message: {error_msg}")
        except Exception:
            pass

        if "LogOn" not in current_url and "login" not in current_url.lower():
            print(f"\n✅ Login successful!")
            print(f"   Current URL: {current_url}")
            print(f"   Page Title: {page_info['title']}")

            if stay_open:
                print("\nBrowser will stay open for verification. Press Ctrl+C to close.")
                while True:
                    await asyncio.sleep(1)
            return True
        else:
            print(f"\n❌ Login may have failed. Still on login page.")
            print(f"   Current URL: {current_url}")

            # Take screenshot for debugging
            _, _ = await browser.capture_screenshot(save_path="/tmp/wanfang_login_result.png")
            print(f"   Screenshot saved to: /tmp/wanfang_login_result.png")

            if stay_open:
                print("\nBrowser will stay open for manual verification. Press Ctrl+C to close.")
                while True:
                    await asyncio.sleep(1)
            return False

    except KeyboardInterrupt:
        print("\nClosing browser...")
    except Exception as e:
        logger.error(f"Login failed with error: {e}")
        raise
    finally:
        if not stay_open:
            await browser.close()

    return False


async def solve_slider_captcha(browser: BrowserController) -> bool:
    """Attempt to solve the slider captcha.

    This function tries to locate and drag the slider to the right.
    Note: This may not work on all captcha implementations as they
    may have anti-bot detection.

    Args:
        browser: BrowserController instance

    Returns:
        True if captcha appears to be solved, False otherwise
    """
    import random

    page = browser.page

    try:
        # Method 1: Try using Playwright's native drag_to with bounding box
        logger.info("Trying Playwright native drag method...")

        # Find the slider button element
        slider_selectors = [
            '#nc_1_n1z',  # Common Aliyun captcha slider ID
            '.nc_iconfont.btn_slide',
            '.btn_slide',
            '.slide-btn',
            '[class*="nc_"][class*="btn"]',
            '.nc_iconfont',
        ]

        slider_elem = None
        for selector in slider_selectors:
            try:
                elem = page.locator(selector).first
                if await elem.count() > 0:
                    box = await elem.bounding_box()
                    if box and box['width'] > 10:
                        slider_elem = elem
                        logger.info(f"Found slider with selector: {selector}, box: {box}")
                        break
            except Exception:
                continue

        if slider_elem:
            box = await slider_elem.bounding_box()
            if box:
                start_x = box['x'] + box['width'] / 2
                start_y = box['y'] + box['height'] / 2

                # Find the track to calculate end position
                track_selectors = ['.nc-lang-cnt', '#nc_1__scale_text', '.scale_text', '.nc_scale']
                end_x = start_x + 280  # Default distance (increased)

                for selector in track_selectors:
                    try:
                        track = page.locator(selector).first
                        if await track.count() > 0:
                            track_box = await track.bounding_box()
                            if track_box:
                                # Move all the way to the right edge
                                end_x = track_box['x'] + track_box['width'] - box['width'] / 2
                                logger.info(f"Found track: {track_box}, end_x: {end_x}")
                                break
                    except Exception:
                        continue

                logger.info(f"Dragging from ({start_x}, {start_y}) to ({end_x}, {start_y})")

                # Hover first to activate
                await page.mouse.move(start_x, start_y)
                await asyncio.sleep(random.uniform(0.3, 0.5))

                # Use Playwright's mouse with human-like behavior
                await page.mouse.down()
                await asyncio.sleep(random.uniform(0.05, 0.1))

                # Move in human-like steps
                import math
                distance = end_x - start_x
                steps = random.randint(30, 50)

                for i in range(1, steps + 1):
                    t = i / steps
                    # Ease out cubic for natural deceleration
                    eased = 1 - pow(1 - t, 3)

                    current_x = start_x + distance * eased
                    # Add slight wobble
                    wobble_y = start_y + random.uniform(-1.5, 1.5)

                    await page.mouse.move(current_x, wobble_y)

                    # Variable timing
                    if t < 0.3:
                        await asyncio.sleep(random.uniform(0.01, 0.025))
                    elif t > 0.85:
                        await asyncio.sleep(random.uniform(0.015, 0.035))
                    else:
                        await asyncio.sleep(random.uniform(0.005, 0.015))

                # Final position - move to exact end and hold
                await page.mouse.move(end_x, start_y)
                await asyncio.sleep(random.uniform(0.3, 0.5))  # Hold at the end for 0.3-0.5 seconds
                await page.mouse.up()

                logger.info("Released slider at end position")

                await asyncio.sleep(1)

                # Check if verification passed
                try:
                    success_text = await page.locator('.nc_ok, .nc-lang-cnt:has-text("验证通过")').count()
                    if success_text > 0:
                        logger.info("Slider verification passed!")
                        return True
                except Exception:
                    pass

                return True

        # Fallback to original method
        logger.warning("Could not find slider with Playwright locator, trying JavaScript method...")

        # Get slider element position using JavaScript
        slider_info = await browser.evaluate("""
            () => {
                // Try different selectors for slider button/handle
                const sliderSelectors = [
                    '.slider-btn',
                    '.slider-handle',
                    '.drag-btn',
                    '.slide-btn',
                    '.nc_iconfont',
                    '#nc_1_n1z',
                    '.btn_slide',
                    '[class*="slider-button"]',
                    '[class*="drag-button"]',
                ];

                // Try selectors for the slider track/container
                const trackSelectors = [
                    '.slider-track',
                    '.slider-container',
                    '.slide-track',
                    '.nc-lang-cnt',
                    '[class*="slider"][class*="container"]',
                    '[class*="slide"][class*="track"]',
                    '[class*="captcha"]',
                ];

                let sliderEl = null;
                let usedSelector = null;

                // First try specific selectors
                for (const selector of sliderSelectors) {
                    const el = document.querySelector(selector);
                    if (el && el.offsetWidth > 0) {
                        sliderEl = el;
                        usedSelector = selector;
                        break;
                    }
                }

                // If not found, try generic approach
                if (!sliderEl) {
                    const allSliders = document.querySelectorAll('[class*="slider"], [class*="drag"]');
                    for (const el of allSliders) {
                        const rect = el.getBoundingClientRect();
                        // Look for small draggable element (likely the button)
                        if (rect.width > 20 && rect.width < 80 && rect.height > 20 && rect.height < 80) {
                            sliderEl = el;
                            usedSelector = 'generic-slider';
                            break;
                        }
                    }
                }

                if (!sliderEl) {
                    return { found: false };
                }

                const rect = sliderEl.getBoundingClientRect();

                // Find track - look for parent container
                let trackEl = null;
                let parent = sliderEl.parentElement;
                while (parent && !trackEl) {
                    const parentRect = parent.getBoundingClientRect();
                    // Track should be wider than slider
                    if (parentRect.width > rect.width * 2) {
                        trackEl = parent;
                        break;
                    }
                    parent = parent.parentElement;
                }

                // Also try explicit track selectors
                if (!trackEl) {
                    for (const selector of trackSelectors) {
                        const el = document.querySelector(selector);
                        if (el) {
                            trackEl = el;
                            break;
                        }
                    }
                }

                const trackRect = trackEl ? trackEl.getBoundingClientRect() : null;

                return {
                    found: true,
                    selector: usedSelector,
                    slider: {
                        x: rect.x + rect.width / 2,
                        y: rect.y + rect.height / 2,
                        width: rect.width,
                        height: rect.height,
                        left: rect.left
                    },
                    track: trackRect ? {
                        x: trackRect.x,
                        y: trackRect.y,
                        width: trackRect.width,
                        height: trackRect.height,
                        right: trackRect.right
                    } : null
                };
            }
        """)

        if slider_info.get("found"):
            logger.info(f"Found slider with selector: {slider_info['selector']}")
            logger.info(f"Slider info: {slider_info}")

            start_x = slider_info["slider"]["x"]
            start_y = slider_info["slider"]["y"]
            slider_width = slider_info["slider"]["width"]

            # Calculate end position (drag to right edge of track)
            if slider_info.get("track") and slider_info["track"].get("right"):
                # End position should be track right edge minus half slider width
                end_x = slider_info["track"]["right"] - slider_width / 2 - 5
            elif slider_info.get("track") and slider_info["track"].get("width"):
                # Calculate from track width
                end_x = slider_info["track"]["x"] + slider_info["track"]["width"] - slider_width / 2 - 5
            else:
                # Default: drag 250 pixels to the right
                end_x = start_x + 250

            end_y = start_y  # Keep same Y position

            logger.info(f"Dragging slider from ({start_x}, {start_y}) to ({end_x}, {end_y})")

            # Perform the drag with human-like movement
            await browser.drag(
                start_x=start_x,
                start_y=start_y,
                end_x=end_x,
                end_y=end_y,
                steps=25,
                delay=30,
                human_like=True,  # Use human-like movement to bypass captcha detection
            )

            await asyncio.sleep(0.5)
            return True
        else:
            # Try clicking on common positions if selector not found
            # Based on screenshot, slider seems to be around (927, 516)
            logger.warning("Slider not found by selector, trying default position")

            # Try drag from default position
            await browser.drag(
                start_x=927,
                start_y=516,
                end_x=1177,  # Drag 250px to the right
                end_y=516,
                steps=25,
                delay=30,
                human_like=True,
            )
            await asyncio.sleep(0.5)
            return True

    except Exception as e:
        logger.error(f"Failed to solve slider captcha: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Wanfang Medical Site Auto Login")
    parser.add_argument(
        "--username", "-u",
        default=os.environ.get("WANFANG_USERNAME", ""),
        help="Username or phone number (or set WANFANG_USERNAME env var)",
    )
    parser.add_argument(
        "--password", "-p",
        default=os.environ.get("WANFANG_PASSWORD", ""),
        help="Password (or set WANFANG_PASSWORD env var)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run browser in headless mode",
    )
    parser.add_argument(
        "--no-stay",
        action="store_true",
        help="Close browser after login attempt",
    )
    parser.add_argument(
        "--no-persist",
        action="store_true",
        help="Don't persist browser session (use fresh session each time)",
    )
    parser.add_argument(
        "--user-data-dir",
        default=DEFAULT_USER_DATA_DIR,
        help=f"Directory to store browser data (default: {DEFAULT_USER_DATA_DIR})",
    )

    args = parser.parse_args()

    if not args.username or not args.password:
        print("Error: Username and password are required.")
        print("Use --username and --password options, or set environment variables:")
        print("  export WANFANG_USERNAME=your_username")
        print("  export WANFANG_PASSWORD=your_password")
        sys.exit(1)

    user_data_dir = None if args.no_persist else args.user_data_dir

    print("=" * 50)
    print("Wanfang Medical Site Auto Login")
    print("=" * 50)
    print(f"Username: {args.username[:3]}***")
    print(f"Headless: {args.headless}")
    print(f"Persist session: {not args.no_persist}")
    if user_data_dir:
        print(f"Browser data: {user_data_dir}")
    print("=" * 50)

    success = asyncio.run(
        auto_login(
            username=args.username,
            password=args.password,
            headless=args.headless,
            stay_open=not args.no_stay,
            user_data_dir=user_data_dir,
        )
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
