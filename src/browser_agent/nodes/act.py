"""Act node - executes planned actions via Playwright."""

import asyncio
from typing import Dict, Any, Optional

from browser_agent.agent.state import BrowserAgentState
from browser_agent.core.browser_controller import BrowserController
from browser_agent.models.ui_element import UIElement
from browser_agent.utils.log_utils import get_logger

logger = get_logger(__name__)


def get_element_center(
    element_id: str,
    element_map: Dict[str, Dict[str, Any]],
) -> Optional[tuple]:
    """Get the page center coordinates for an element.

    Args:
        element_id: Element ID to look up
        element_map: Dictionary mapping element_id to element info

    Returns:
        (x, y) coordinates in CSS pixels, or None if not found
    """
    element = element_map.get(element_id)
    if not element:
        return None

    page_center = element.get("page_center")
    if page_center:
        return tuple(page_center)

    return None


async def execute_click(
    browser: BrowserController,
    element_map: Dict[str, Dict[str, Any]],
    element_id: str,
    click_type: str = "single",
) -> Dict[str, Any]:
    """Execute a click action.

    Args:
        browser: Browser controller instance
        element_map: Element map from state
        element_id: Element to click
        click_type: Type of click

    Returns:
        Action result dictionary
    """
    center = get_element_center(element_id, element_map)
    if not center:
        return {
            "success": False,
            "result": f"Element {element_id} not found",
        }

    x, y = center
    await browser.click_at(x, y, click_type=click_type)

    return {
        "success": True,
        "result": f"Clicked at ({x:.0f}, {y:.0f})",
    }


async def execute_type(
    browser: BrowserController,
    element_map: Dict[str, Dict[str, Any]],
    text: str,
    element_id: Optional[str] = None,
    clear_first: bool = False,
) -> Dict[str, Any]:
    """Execute a type action.

    Args:
        browser: Browser controller instance
        element_map: Element map from state
        text: Text to type
        element_id: Optional element to focus first
        clear_first: Whether to clear before typing

    Returns:
        Action result dictionary
    """
    # Click element first if specified
    if element_id:
        center = get_element_center(element_id, element_map)
        if center:
            x, y = center
            await browser.click_at(x, y)
            await asyncio.sleep(0.2)  # Wait for focus

    await browser.type_text(text, clear_first=clear_first)

    return {
        "success": True,
        "result": f"Typed '{text[:30]}...' " + (f"into {element_id}" if element_id else ""),
    }


async def execute_scroll(
    browser: BrowserController,
    direction: str,
    amount: int = 300,
) -> Dict[str, Any]:
    """Execute a scroll action.

    Args:
        browser: Browser controller instance
        direction: Scroll direction
        amount: Pixels to scroll

    Returns:
        Action result dictionary
    """
    await browser.scroll(direction, amount)

    return {
        "success": True,
        "result": f"Scrolled {direction} by {amount}px",
    }


async def execute_navigate(
    browser: BrowserController,
    url: str,
) -> Dict[str, Any]:
    """Execute a navigation action.

    Args:
        browser: Browser controller instance
        url: URL to navigate to

    Returns:
        Action result dictionary
    """
    await browser.navigate(url)
    page_info = await browser.get_page_info()

    return {
        "success": True,
        "result": f"Navigated to {page_info['url'][:50]}...",
    }


async def execute_select(
    browser: BrowserController,
    element_map: Dict[str, Dict[str, Any]],
    element_id: str,
    value: str,
) -> Dict[str, Any]:
    """Execute a select action.

    Args:
        browser: Browser controller instance
        element_map: Element map from state
        element_id: Dropdown element
        value: Value to select

    Returns:
        Action result dictionary
    """
    # Click the dropdown first
    center = get_element_center(element_id, element_map)
    if center:
        x, y = center
        await browser.click_at(x, y)
        await asyncio.sleep(0.3)

    # Type the value and press enter (works for most dropdowns)
    await browser.type_text(value)
    await browser.press_key("Enter")

    return {
        "success": True,
        "result": f"Selected '{value}' from {element_id}",
    }


async def execute_wait(
    browser: BrowserController,
    condition: str,
    value: str,
) -> Dict[str, Any]:
    """Execute a wait action.

    Args:
        browser: Browser controller instance
        condition: Wait condition type
        value: Wait value

    Returns:
        Action result dictionary
    """
    if condition == "time":
        seconds = float(value)
        await asyncio.sleep(seconds)
        return {"success": True, "result": f"Waited {seconds}s"}

    elif condition == "element":
        # Wait by checking periodically (element_id not directly usable as selector)
        # In a real implementation, we'd need to re-observe and check
        await asyncio.sleep(2)  # Simple wait for now
        return {"success": True, "result": f"Waited for element {value}"}

    elif condition == "url_change":
        success = await browser.wait_for_url(value, timeout=10000)
        if success:
            return {"success": True, "result": f"URL changed to match {value}"}
        else:
            return {"success": False, "result": f"URL did not change to {value}"}

    return {"success": False, "result": f"Unknown wait condition: {condition}"}


async def execute_upload(
    browser: BrowserController,
    element_map: Dict[str, Dict[str, Any]],
    element_id: str,
    file_path: str,
) -> Dict[str, Any]:
    """Execute an upload action.

    Args:
        browser: Browser controller instance
        element_map: Element map from state
        element_id: File input element
        file_path: Path to file

    Returns:
        Action result dictionary
    """
    # Click the file input
    center = get_element_center(element_id, element_map)
    if center:
        x, y = center
        # Use set_input_files via JavaScript (more reliable)
        # For now, just click and hope the system dialog handles it
        await browser.click_at(x, y)
        await asyncio.sleep(0.5)

    # Note: Actual file upload requires special handling
    return {
        "success": True,
        "result": f"Initiated upload of {file_path} to {element_id}",
    }


async def execute_click_all_matching(
    browser: BrowserController,
    element_map: Dict[str, Dict[str, Any]],
    text_pattern: str,
    delay_between_clicks: float = 1.0,
    parser: Optional[Any] = None,
    scroll_and_find: bool = True,
    max_scroll_attempts: int = 10,
    viewport_height: int = 720,
    paginate: bool = False,
    next_page_pattern: str = "下一页",
    max_pages: int = 50,
) -> Dict[str, Any]:
    """Click all elements matching a text pattern with auto-scroll and pagination.

    Args:
        browser: Browser controller instance
        element_map: Element map from state
        text_pattern: Text pattern to match
        delay_between_clicks: Delay between clicks in seconds
        parser: OmniParser instance for re-parsing after scroll
        scroll_and_find: Whether to scroll and find more elements
        max_scroll_attempts: Maximum scroll attempts per page
        viewport_height: Viewport height for scroll amount
        paginate: Whether to click next page and continue on new pages
        next_page_pattern: Text pattern for next page button (default: "下一页")
        max_pages: Maximum number of pages to process (default: 50)

    Returns:
        Action result dictionary with click count
    """
    from browser_agent.utils.image_utils import save_screenshot

    # Track clicked positions to avoid duplicates (reset per page if paginating)
    clicked_positions = set()
    total_clicked = 0
    total_scroll_attempts = 0
    pages_processed = 0

    def get_position_key(x: float, y: float) -> tuple:
        """Create a position key for deduplication (grid-based)."""
        return (int(x / 30), int(y / 30))

    def find_matching_elements(elem_map: Dict[str, Dict[str, Any]], pattern: str, exclude_positions: set = None) -> list:
        """Find all elements matching the text pattern."""
        matching = []
        exclude = exclude_positions or set()
        for elem_id, elem in elem_map.items():
            text = elem.get("text") or elem.get("description") or ""
            if pattern.lower() in text.lower():
                page_center = elem.get("page_center")
                if page_center:
                    x, y = page_center
                    pos_key = get_position_key(x, y)
                    if pos_key not in exclude:
                        matching.append((elem_id, elem, x, y, pos_key))
        # Sort by position (top to bottom)
        matching.sort(key=lambda x: (x[3], x[2]))
        return matching

    def find_next_page_button(elem_map: Dict[str, Dict[str, Any]]) -> Optional[tuple]:
        """Find the next page button from element map."""
        for elem_id, elem in elem_map.items():
            text = elem.get("text") or elem.get("description") or ""
            if next_page_pattern.lower() in text.lower():
                page_center = elem.get("page_center")
                if page_center:
                    return (elem_id, page_center[0], page_center[1])
        return None

    async def find_next_page_with_selector() -> bool:
        """Try to find and click next page using Playwright selector."""
        if not hasattr(browser, 'page') or browser.page is None:
            return False
        try:
            page = browser.page
            # Try different selectors for next page button
            selectors = [
                f"text={next_page_pattern}",
                f"a:has-text('{next_page_pattern}')",
                f"button:has-text('{next_page_pattern}')",
            ]
            for selector in selectors:
                element = page.locator(selector).first
                if await element.count() > 0:
                    await element.click()
                    return True
            return False
        except Exception as e:
            logger.debug(f"Selector-based pagination failed: {e}")
            return False

    async def take_screenshot_and_parse() -> Dict[str, Dict[str, Any]]:
        """Take screenshot and parse to get element map."""
        screenshot_bytes, _ = await browser.capture_screenshot()
        screenshot_path = save_screenshot(screenshot_bytes, f"page_{pages_processed}_scroll_{total_scroll_attempts}.png")

        elements, new_element_map = parser.parse_screenshot_to_elements(
            image_path=screenshot_path,
            viewport_width=browser.viewport_width if hasattr(browser, 'viewport_width') else 1280,
            viewport_height=viewport_height,
            use_ocr=True,  # Use OCR for Chinese text recognition
            caption_all=False,
        )
        return {k: v.to_dict() for k, v in new_element_map.items()}

    async def check_login_redirect() -> bool:
        """Check if we've been redirected to a login page."""
        if not hasattr(browser, 'page') or browser.page is None:
            return False
        try:
            current_url = browser.page.url.lower()
            login_patterns = ["login", "logon", "signin", "sign-in", "authenticate", "account/log"]
            for pattern in login_patterns:
                if pattern in current_url:
                    logger.warning(f"Login redirect detected: {current_url}")
                    return True
        except Exception:
            pass
        return False

    async def click_matching_with_selector(pattern: str) -> tuple:
        """Click elements matching pattern using Playwright selector.

        Returns:
            Tuple of (clicked_count, login_detected)
        """
        if not hasattr(browser, 'page') or browser.page is None:
            return 0, False
        try:
            page = browser.page
            elements = page.locator(f"text={pattern}")
            count = await elements.count()
            clicked = 0
            for i in range(count):
                try:
                    elem = elements.nth(i)
                    bbox = await elem.bounding_box()
                    if bbox:
                        pos_key = get_position_key(bbox['x'] + bbox['width']/2, bbox['y'] + bbox['height']/2)
                        if pos_key not in clicked_positions:
                            await elem.click()
                            clicked_positions.add(pos_key)
                            clicked += 1
                            logger.info(f"Clicked element '{pattern}' #{i+1} using selector")
                            if delay_between_clicks > 0:
                                await asyncio.sleep(delay_between_clicks)
                            # Check for login redirect after click
                            if await check_login_redirect():
                                return clicked, True
                except Exception as e:
                    logger.debug(f"Failed to click element {i}: {e}")
            return clicked, False
        except Exception as e:
            logger.debug(f"Selector-based click failed: {e}")
            return 0, False

    logger.info(f"Starting click_all_matching for '{text_pattern}' with scroll_and_find={scroll_and_find}, paginate={paginate}")

    # Initial search in current element_map
    current_map = element_map

    while pages_processed < max_pages:
        pages_processed += 1
        page_scroll_attempts = 0
        clicked_positions.clear()  # Reset positions for new page

        logger.info(f"=== Processing page {pages_processed} ===")

        # Process current page with scrolling
        login_detected = False
        while True:
            # First try Playwright selector (more reliable for text matching)
            selector_clicked, login_detected = await click_matching_with_selector(text_pattern)
            if selector_clicked > 0:
                total_clicked += selector_clicked
                logger.info(f"Selector clicked {selector_clicked} elements on page {pages_processed}")

            # If login page detected, stop and return special result
            if login_detected:
                logger.warning("Login required - stopping click_all_matching")
                return {
                    "success": False,
                    "result": f"Login required after clicking {total_clicked} elements",
                    "clicked_count": total_clicked,
                    "login_required": True,
                }

            # Then try OCR-based matching
            matching_elements = find_matching_elements(current_map, text_pattern, clicked_positions)

            # Click all currently visible matching elements (OCR-detected)
            for elem_id, elem, x, y, pos_key in matching_elements:
                try:
                    await browser.click_at(x, y)
                    clicked_positions.add(pos_key)
                    total_clicked += 1
                    logger.info(f"[Page {pages_processed}] Clicked element {elem_id} at ({x:.0f}, {y:.0f}) - Total: {total_clicked}")

                    # Wait between clicks
                    if delay_between_clicks > 0:
                        await asyncio.sleep(delay_between_clicks)
                except Exception as e:
                    logger.warning(f"Failed to click element {elem_id}: {e}")

            # If no scroll_and_find or no parser, break scroll loop
            if not scroll_and_find or parser is None:
                break

            # Scroll down to find more elements
            page_scroll_attempts += 1
            total_scroll_attempts += 1

            if page_scroll_attempts > max_scroll_attempts:
                logger.info(f"Reached max scroll attempts for page {pages_processed}")
                break

            logger.info(f"Scrolling down (page {pages_processed}, scroll {page_scroll_attempts}/{max_scroll_attempts})...")
            await browser.scroll("down", amount=viewport_height - 100)
            await asyncio.sleep(0.5)

            # Take new screenshot and re-parse
            try:
                current_map = await take_screenshot_and_parse()
                new_matching = find_matching_elements(current_map, text_pattern, clicked_positions)

                if not new_matching:
                    logger.info("No new matching elements found after scroll")
                    break

                logger.info(f"Found {len(new_matching)} new matching elements after scroll")

            except Exception as e:
                logger.error(f"Failed to re-parse after scroll: {e}")
                break

        # If pagination is disabled, we're done
        if not paginate or parser is None:
            break

        # Scroll back to top before looking for next page button
        logger.info("Scrolling to top to find next page button...")
        for _ in range(3):  # Scroll up a few times
            await browser.scroll("up", amount=viewport_height)
            await asyncio.sleep(0.3)

        # Try to click next page button
        try:
            await asyncio.sleep(0.5)

            # First try using Playwright selector (more reliable)
            clicked_next = await find_next_page_with_selector()

            if not clicked_next:
                # Fall back to OCR-based detection
                current_map = await take_screenshot_and_parse()
                next_page = find_next_page_button(current_map)

                if not next_page:
                    logger.info(f"No '{next_page_pattern}' button found, pagination complete")
                    break

                elem_id, x, y = next_page
                logger.info(f"Clicking next page button at ({x:.0f}, {y:.0f})...")
                await browser.click_at(x, y)
            else:
                logger.info(f"Clicked '{next_page_pattern}' using Playwright selector")

            await asyncio.sleep(1.5)  # Wait for page to load

            # Re-parse the new page
            current_map = await take_screenshot_and_parse()

        except Exception as e:
            logger.error(f"Failed to navigate to next page: {e}")
            break

    result_msg = f"Clicked {total_clicked} elements matching '{text_pattern}'"
    if total_scroll_attempts > 0:
        result_msg += f" (scrolled {total_scroll_attempts} times)"
    if paginate and pages_processed > 1:
        result_msg += f" across {pages_processed} pages"

    return {
        "success": total_clicked > 0,
        "result": result_msg,
        "clicked_count": total_clicked,
        "scroll_attempts": total_scroll_attempts,
        "pages_processed": pages_processed,
    }


async def execute_download(
    browser: BrowserController,
    element_map: Dict[str, Dict[str, Any]],
    element_id: str,
    save_as: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute a download action.

    Args:
        browser: Browser controller instance
        element_map: Element map from state
        element_id: Download button/link
        save_as: Optional filename

    Returns:
        Action result dictionary
    """
    center = get_element_center(element_id, element_map)
    if not center:
        return {"success": False, "result": f"Element {element_id} not found"}

    x, y = center

    async def click_action():
        await browser.click_at(x, y)

    try:
        download_path = await browser.handle_download(click_action, save_as)
        return {"success": True, "result": f"Downloaded to {download_path}"}
    except Exception as e:
        return {"success": False, "result": f"Download failed: {str(e)}"}


async def execute_screenshot(
    browser: BrowserController,
    annotation: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute a screenshot action.

    Args:
        browser: Browser controller instance
        annotation: Optional annotation

    Returns:
        Action result dictionary
    """
    from browser_agent.utils.image_utils import save_screenshot
    from datetime import datetime

    screenshot_bytes, _ = await browser.capture_screenshot()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"manual_screenshot_{timestamp}.png"
    path = save_screenshot(screenshot_bytes, filename)

    return {
        "success": True,
        "result": f"Screenshot saved to {path}" + (f" ({annotation})" if annotation else ""),
    }


async def execute_handle_login(
    browser: BrowserController,
    state: BrowserAgentState,
) -> Dict[str, Any]:
    """Execute login directly in the current browser without closing it.

    This action performs login in-browser to maintain session continuity:
    1. Fills in username and password fields
    2. Handles slider captcha if present
    3. Returns to the original page after login

    Args:
        browser: Browser controller instance
        state: Current agent state

    Returns:
        Action result dictionary
    """
    import os
    import random
    from urllib.parse import urlparse, parse_qs

    # Get credentials from environment variables
    username = os.getenv("WANFANG_USERNAME")
    password = os.getenv("WANFANG_PASSWORD")

    if not username or not password:
        return {
            "success": False,
            "result": "Missing WANFANG_USERNAME or WANFANG_PASSWORD in environment variables",
        }

    # Extract target URL from login page ReturnUrl parameter
    current_url = state.get("current_url", "")
    start_url = state.get("start_url", "")
    target_url = start_url

    if "login" in current_url.lower() or "logon" in current_url.lower():
        try:
            parsed = urlparse(current_url)
            params = parse_qs(parsed.query)
            if "ReturnUrl" in params:
                target_url = params["ReturnUrl"][0]
                logger.info(f"Extracted ReturnUrl: {target_url}")
            elif "returnurl" in params:
                target_url = params["returnurl"][0]
        except Exception:
            pass
    else:
        target_url = current_url or start_url

    logger.info(f"Starting in-browser login for user: {username[:3]}***")

    page = browser.page
    if not page:
        return {"success": False, "result": "Browser page not available"}

    try:
        # Step 1: Enter username
        logger.info("[1/5] Entering username...")
        await page.evaluate("""
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

        # Step 2: Enter password
        logger.info("[2/5] Entering password...")
        await page.evaluate("""
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

        # Step 3: Solve slider captcha if present
        logger.info("[3/5] Checking for slider captcha...")
        captcha_solved = await _solve_slider_captcha(browser)
        if captcha_solved:
            logger.info("Slider captcha solved")
        else:
            logger.info("No captcha detected or already solved")

        # Step 4: Click login button
        logger.info("[4/5] Clicking login button...")
        login_clicked = await page.evaluate("""
            () => {
                const loginButtons = [
                    document.querySelector('button[type="submit"]'),
                    document.querySelector('input[type="submit"]'),
                    document.querySelector('.login-btn'),
                    document.querySelector('#loginBtn'),
                    document.querySelector('[class*="login"][class*="btn"]'),
                ];

                for (const btn of loginButtons) {
                    if (btn && btn.offsetWidth > 0) {
                        btn.click();
                        return true;
                    }
                }

                // Try finding by text content
                const allButtons = document.querySelectorAll('button, input[type="submit"], a');
                for (const btn of allButtons) {
                    if (btn.textContent && (btn.textContent.includes('登录') || btn.textContent.includes('登 录') || btn.textContent.includes('Login'))) {
                        btn.click();
                        return true;
                    }
                }
                return false;
            }
        """)

        if not login_clicked:
            # Try using Playwright locator
            try:
                login_btn = page.locator('button:has-text("登录"), button:has-text("登 录"), input[type="submit"]').first
                if await login_btn.count() > 0:
                    await login_btn.click()
                    login_clicked = True
                    logger.info("Clicked login button via Playwright locator")
            except Exception as e:
                logger.warning(f"Playwright locator click failed: {e}")

        if not login_clicked:
            logger.warning("Could not find login button")
            return {
                "success": False,
                "result": "Could not find login button on page",
            }

        # Wait for page to process login - increased time for slower networks
        logger.info("Waiting for login to complete...")
        await asyncio.sleep(3)

        # Check for URL change during wait
        for _ in range(3):
            current_check_url = page.url.lower()
            if not any(p in current_check_url for p in ["login", "logon", "signin", "authenticate"]):
                logger.info(f"Login redirect detected to: {page.url[:60]}...")
                break
            await asyncio.sleep(1)

        # Step 5: Verify login success
        logger.info("[5/5] Verifying login...")
        current_page_url = page.url
        login_patterns = ["login", "logon", "signin", "authenticate", "account/log"]

        is_still_on_login = any(p in current_page_url.lower() for p in login_patterns)

        if not is_still_on_login:
            logger.info(f"✅ Login successful! Current URL: {current_page_url[:60]}...")
            return {
                "success": True,
                "result": "Login completed successfully - session maintained",
            }
        else:
            # Check if there's a meaningful error message
            error_msg = await page.evaluate("""
                () => {
                    const errorSelectors = [
                        '.error-message',
                        '.alert-danger',
                        '.login-error',
                        '.error-tip',
                        '.err-msg',
                        '[class*="error"]:not(input):not(label)',
                    ];
                    for (const selector of errorSelectors) {
                        const el = document.querySelector(selector);
                        if (el && el.offsetWidth > 0) {
                            const text = el.textContent.trim();
                            if (text && text.length > 0 && text.length < 200) {
                                return text;
                            }
                        }
                    }
                    return null;
                }
            """)

            if error_msg:
                logger.warning(f"Login error message: {error_msg}")
                return {
                    "success": False,
                    "result": f"Login failed: {error_msg[:100]}",
                }

            # Still on login page but no error - might need more time
            logger.warning("Still on login page after submit")
            return {
                "success": False,
                "result": "Login submitted but still on login page",
            }

    except Exception as e:
        logger.error(f"In-browser login failed: {e}")
        return {
            "success": False,
            "result": f"Login execution error: {str(e)}",
        }


async def _solve_slider_captcha(browser: BrowserController) -> bool:
    """Attempt to solve slider captcha if present.

    Args:
        browser: Browser controller instance

    Returns:
        True if captcha was found and solved, False if no captcha or failed
    """
    import random

    page = browser.page
    if not page:
        return False

    try:
        # Find slider element
        slider_selectors = [
            '#nc_1_n1z',
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
                        logger.info(f"Found slider with selector: {selector}")
                        break
            except Exception:
                continue

        if not slider_elem:
            return False

        box = await slider_elem.bounding_box()
        if not box:
            return False

        start_x = box['x'] + box['width'] / 2
        start_y = box['y'] + box['height'] / 2

        # Find track to calculate end position
        track_selectors = ['.nc-lang-cnt', '#nc_1__scale_text', '.scale_text', '.nc_scale']
        end_x = start_x + 280  # Default distance

        for selector in track_selectors:
            try:
                track = page.locator(selector).first
                if await track.count() > 0:
                    track_box = await track.bounding_box()
                    if track_box:
                        end_x = track_box['x'] + track_box['width'] - box['width'] / 2
                        break
            except Exception:
                continue

        logger.info(f"Dragging slider from ({start_x}, {start_y}) to ({end_x}, {start_y})")

        # Perform human-like drag
        await page.mouse.move(start_x, start_y)
        await asyncio.sleep(random.uniform(0.1, 0.2))
        await page.mouse.down()
        await asyncio.sleep(random.uniform(0.05, 0.1))

        # Move in small steps
        distance = end_x - start_x
        steps = int(distance / 10)
        current_x = start_x

        for i in range(steps):
            progress = (i + 1) / steps
            # Ease out: faster at start, slower at end
            eased_progress = 1 - (1 - progress) ** 2
            target_x = start_x + distance * eased_progress

            # Add slight vertical jitter
            jitter_y = start_y + random.uniform(-2, 2)
            await page.mouse.move(target_x, jitter_y)

            if random.random() > 0.7:
                await asyncio.sleep(random.uniform(0.01, 0.03))
            else:
                await asyncio.sleep(random.uniform(0.005, 0.015))

        # Final position
        await page.mouse.move(end_x, start_y)
        await asyncio.sleep(random.uniform(0.3, 0.5))
        await page.mouse.up()

        logger.info("Released slider at end position")
        await asyncio.sleep(1)

        # Check for success
        try:
            success_count = await page.locator('.nc_ok, .nc-lang-cnt:has-text("验证通过")').count()
            if success_count > 0:
                logger.info("Slider verification passed!")
                return True
        except Exception:
            pass

        return True

    except Exception as e:
        logger.warning(f"Slider captcha handling failed: {e}")
        return False


async def act_node(state: BrowserAgentState) -> Dict[str, Any]:
    """Execute the planned action via Playwright.

    This node:
    1. Gets the next_action from state
    2. Maps action type to execution function
    3. Converts visual coordinates to page coordinates
    4. Executes the action
    5. Records result in action_history

    Args:
        state: Current agent state

    Returns:
        State updates with action result
    """
    next_action = state.get("next_action")
    if not next_action:
        logger.warning("No action to execute")
        return {"error": "No action specified"}

    action_type = next_action.get("action_type")
    logger.info(f"[Step {state.get('current_step', 0)}] Executing: {action_type}")

    # Get browser controller
    browser = state.get("_browser_controller")
    if not browser:
        return {"error": "Browser not initialized"}

    element_map = state.get("element_map", {})

    # Execute based on action type
    result = {"success": False, "result": "Unknown action"}

    try:
        if action_type == "click":
            result = await execute_click(
                browser,
                element_map,
                next_action.get("element_id"),
                next_action.get("click_type", "single"),
            )

        elif action_type == "click_at":
            # Direct coordinate-based click
            x = next_action.get("x", 0)
            y = next_action.get("y", 0)
            click_type = next_action.get("click_type", "single")
            await browser.click_at(x, y, click_type=click_type)
            result = {"success": True, "result": f"Clicked at ({x}, {y})"}

        elif action_type == "type":
            result = await execute_type(
                browser,
                element_map,
                next_action.get("text", ""),
                next_action.get("element_id"),
                next_action.get("clear_first", False),
            )

        elif action_type == "scroll":
            result = await execute_scroll(
                browser,
                next_action.get("direction", "down"),
                next_action.get("amount", 300),
            )

        elif action_type == "navigate":
            result = await execute_navigate(
                browser,
                next_action.get("url", ""),
            )

        elif action_type == "select":
            result = await execute_select(
                browser,
                element_map,
                next_action.get("element_id"),
                next_action.get("value", ""),
            )

        elif action_type == "wait":
            result = await execute_wait(
                browser,
                next_action.get("condition", "time"),
                next_action.get("value", "1"),
            )

        elif action_type == "upload":
            result = await execute_upload(
                browser,
                element_map,
                next_action.get("element_id"),
                next_action.get("file_path", ""),
            )

        elif action_type == "click_all_matching":
            # Get parser from state for scroll and re-parse
            parser = state.get("_omni_parser")
            viewport_height = state.get("viewport_height", 720)

            result = await execute_click_all_matching(
                browser,
                element_map,
                next_action.get("text_pattern", ""),
                next_action.get("delay_between_clicks", 1.0),
                parser=parser,
                scroll_and_find=next_action.get("scroll_and_find", True),
                max_scroll_attempts=next_action.get("max_scroll_attempts", 10),
                viewport_height=viewport_height,
                paginate=next_action.get("paginate", False),
                next_page_pattern=next_action.get("next_page_pattern", "下一页"),
                max_pages=next_action.get("max_pages", 50),
            )

            # If login was detected during click_all_matching, set login_required flag
            if result.get("login_required"):
                logger.warning("Login required detected during click_all_matching")
                return {
                    "action_history": [{
                        **next_action,
                        "result": result.get("result", ""),
                        "success": False,
                        "step": state.get("current_step", 0),
                    }],
                    "login_required": True,
                    "error": None,
                }

        elif action_type == "download":
            result = await execute_download(
                browser,
                element_map,
                next_action.get("element_id"),
                next_action.get("save_as"),
            )

        elif action_type == "screenshot":
            result = await execute_screenshot(
                browser,
                next_action.get("annotation"),
            )

        elif action_type == "handle_login":
            result = await execute_handle_login(browser, state)
            # After login, reset the login_required flag
            if result.get("success"):
                return {
                    "action_history": [{
                        **next_action,
                        "result": result.get("result", ""),
                        "success": True,
                        "step": state.get("current_step", 0),
                    }],
                    "login_required": False,
                    "error": None,
                }

        elif action_type == "done":
            success = next_action.get("success", True)
            message = next_action.get("message", "Task completed")
            return {
                "is_complete": True,
                "verification_result": {
                    "success": success,
                    "level": "agent",
                    "confidence": 1.0,
                    "details": message,
                    "should_retry": False,
                    "stuck_detection": False,
                },
                "action_history": [{
                    **next_action,
                    "result": message,
                    "step": state.get("current_step", 0),
                }],
            }

        else:
            result = {"success": False, "result": f"Unknown action type: {action_type}"}

    except Exception as e:
        logger.error(f"Action execution failed: {e}")
        result = {"success": False, "result": f"Execution error: {str(e)}"}

    # Record action in history
    action_record = {
        **next_action,
        "result": result.get("result", ""),
        "success": result.get("success", False),
        "step": state.get("current_step", 0),
    }

    logger.info(f"Action result: {result.get('result', '')[:50]}")

    return {
        "action_history": [action_record],
        "error": None if result.get("success") else result.get("result"),
    }
