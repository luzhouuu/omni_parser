"""Browser controller using Playwright for browser automation."""

import asyncio
from typing import Optional, Tuple, Dict, Any, List
from pathlib import Path

from playwright.async_api import async_playwright, Browser, BrowserContext, Page, Playwright

from browser_agent.agent.configuration import (
    BROWSER_HEADLESS,
    BROWSER_SLOW_MO,
    BROWSER_TIMEOUT,
    SCREENSHOT_DIR,
)
from browser_agent.utils.log_utils import get_logger
from browser_agent.utils.image_utils import encode_image_base64, save_screenshot

logger = get_logger(__name__)


class BrowserController:
    """Manages Playwright browser instance and page interactions.

    This class provides a high-level API for browser automation,
    including navigation, screenshot capture, and user interactions.
    """

    def __init__(
        self,
        headless: bool = BROWSER_HEADLESS,
        slow_mo: int = BROWSER_SLOW_MO,
        timeout: int = BROWSER_TIMEOUT,
        viewport_width: int = 1280,
        viewport_height: int = 720,
        user_data_dir: Optional[str] = None,
    ):
        """Initialize the browser controller.

        Args:
            headless: Run browser in headless mode
            slow_mo: Slow down operations by this many milliseconds
            timeout: Default timeout for operations in milliseconds
            viewport_width: Browser viewport width
            viewport_height: Browser viewport height
            user_data_dir: Directory to store browser data (cookies, localStorage, etc.)
                          If provided, browser session will persist across restarts.
                          If None, uses a fresh incognito-like session each time.
        """
        self._headless = headless
        self._slow_mo = slow_mo
        self._timeout = timeout
        self._viewport_width = viewport_width
        self._viewport_height = viewport_height
        self._user_data_dir = user_data_dir

        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        self._dpr: float = 1.0
        self._download_dir: Path = Path(SCREENSHOT_DIR) / "downloads"

    @property
    def page(self) -> Optional[Page]:
        """Get the current page instance."""
        return self._page

    @property
    def dpr(self) -> float:
        """Get the device pixel ratio."""
        return self._dpr

    async def start(self) -> None:
        """Initialize browser with appropriate settings."""
        logger.info("Starting browser...")

        self._playwright = await async_playwright().start()

        # Create download directory
        self._download_dir.mkdir(parents=True, exist_ok=True)

        # Common context options
        context_options = {
            "viewport": {
                "width": self._viewport_width,
                "height": self._viewport_height,
            },
            "accept_downloads": True,
            "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "locale": "zh-CN",
        }

        if self._user_data_dir:
            # Use persistent context - keeps cookies, localStorage, etc.
            user_data_path = Path(self._user_data_dir)
            user_data_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Using persistent browser data at: {user_data_path}")

            self._context = await self._playwright.chromium.launch_persistent_context(
                user_data_dir=str(user_data_path),
                headless=self._headless,
                slow_mo=self._slow_mo,
                **context_options,
            )
            # Persistent context may already have pages
            self._page = self._context.pages[0] if self._context.pages else await self._context.new_page()
        else:
            # Use non-persistent context (fresh session each time)
            self._browser = await self._playwright.chromium.launch(
                headless=self._headless,
                slow_mo=self._slow_mo,
            )
            self._context = await self._browser.new_context(**context_options)
            self._page = await self._context.new_page()

        # Set default timeout
        self._context.set_default_timeout(self._timeout)

        # Apply stealth measures to hide automation detection
        await self._page.add_init_script("""
            // Hide webdriver flag
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            });

            // Override plugins to look normal
            Object.defineProperty(navigator, 'plugins', {
                get: () => [1, 2, 3, 4, 5]
            });

            // Override languages
            Object.defineProperty(navigator, 'languages', {
                get: () => ['zh-CN', 'zh', 'en']
            });

            // Hide chrome runtime
            window.chrome = {
                runtime: {}
            };

            // Override permissions query
            const originalQuery = window.navigator.permissions.query;
            window.navigator.permissions.query = (parameters) => (
                parameters.name === 'notifications' ?
                    Promise.resolve({ state: Notification.permission }) :
                    originalQuery(parameters)
            );
        """)

        # Get device pixel ratio
        self._dpr = await self._page.evaluate("window.devicePixelRatio")

        logger.info(
            f"Browser started: viewport={self._viewport_width}x{self._viewport_height}, "
            f"dpr={self._dpr}, headless={self._headless}"
        )

    async def navigate(self, url: str, wait_until: str = "domcontentloaded") -> None:
        """Navigate to URL and wait for load.

        Args:
            url: URL to navigate to
            wait_until: When to consider navigation complete
                ("domcontentloaded", "load", "networkidle")
        """
        if not self._page:
            raise RuntimeError("Browser not started. Call start() first.")

        logger.info(f"Navigating to: {url}")
        await self._page.goto(url, wait_until=wait_until)
        logger.info(f"Navigation complete: {await self._page.title()}")

    async def capture_screenshot(
        self,
        full_page: bool = False,
        save_path: Optional[str] = None,
    ) -> Tuple[bytes, str]:
        """Capture screenshot of the current page.

        Args:
            full_page: Capture full page (including scrollable area)
            save_path: Optional path to save screenshot

        Returns:
            Tuple of (screenshot_bytes, base64_encoded_string)
        """
        if not self._page:
            raise RuntimeError("Browser not started. Call start() first.")

        screenshot_bytes = await self._page.screenshot(
            full_page=full_page,
            type="png",
        )

        base64_string = encode_image_base64(screenshot_bytes)

        if save_path:
            with open(save_path, "wb") as f:
                f.write(screenshot_bytes)
            logger.debug(f"Screenshot saved to: {save_path}")
        else:
            # Auto-save to default location
            save_path = save_screenshot(screenshot_bytes)

        return screenshot_bytes, base64_string

    async def click_at(
        self,
        x: float,
        y: float,
        click_type: str = "single",
        delay: int = 0,
    ) -> None:
        """Click at CSS pixel coordinates.

        Args:
            x: X coordinate in CSS pixels
            y: Y coordinate in CSS pixels
            click_type: Type of click ("single", "double", "right")
            delay: Delay between mousedown and mouseup in milliseconds
        """
        if not self._page:
            raise RuntimeError("Browser not started. Call start() first.")

        logger.debug(f"Clicking at ({x}, {y}) with type={click_type}")

        if click_type == "double":
            await self._page.mouse.dblclick(x, y)
        elif click_type == "right":
            await self._page.mouse.click(x, y, button="right")
        else:
            await self._page.mouse.click(x, y, delay=delay)

    async def type_text(
        self,
        text: str,
        delay: int = 50,
        clear_first: bool = False,
    ) -> None:
        """Type text into the currently focused element.

        Args:
            text: Text to type
            delay: Delay between keystrokes in milliseconds
            clear_first: Clear existing content before typing
        """
        if not self._page:
            raise RuntimeError("Browser not started. Call start() first.")

        if clear_first:
            # Select all and delete
            await self._page.keyboard.press("Control+a")
            await self._page.keyboard.press("Backspace")

        logger.debug(f"Typing text: {text[:50]}..." if len(text) > 50 else f"Typing text: {text}")
        await self._page.keyboard.type(text, delay=delay)

    async def press_key(self, key: str) -> None:
        """Press a keyboard key.

        Args:
            key: Key to press (e.g., "Enter", "Tab", "Escape")
        """
        if not self._page:
            raise RuntimeError("Browser not started. Call start() first.")

        logger.debug(f"Pressing key: {key}")
        await self._page.keyboard.press(key)

    async def drag(
        self,
        start_x: float,
        start_y: float,
        end_x: float,
        end_y: float,
        steps: int = 10,
        delay: int = 50,
        human_like: bool = False,
    ) -> None:
        """Drag from one point to another (useful for sliders/captchas).

        Args:
            start_x: Starting X coordinate in CSS pixels
            start_y: Starting Y coordinate in CSS pixels
            end_x: Ending X coordinate in CSS pixels
            end_y: Ending Y coordinate in CSS pixels
            steps: Number of intermediate steps for smooth drag
            delay: Delay in milliseconds between steps
            human_like: Use human-like movement with randomness
        """
        import random

        if not self._page:
            raise RuntimeError("Browser not started. Call start() first.")

        logger.debug(f"Dragging from ({start_x}, {start_y}) to ({end_x}, {end_y}), human_like={human_like}")

        # Move to start position and press
        await self._page.mouse.move(start_x, start_y)
        await asyncio.sleep(random.uniform(0.1, 0.3) if human_like else 0.05)
        await self._page.mouse.down()
        await asyncio.sleep(random.uniform(0.05, 0.15) if human_like else 0.02)

        if human_like:
            # Human-like movement with ease-in-out and randomness
            total_distance = end_x - start_x
            actual_steps = random.randint(steps - 5, steps + 10)

            for i in range(1, actual_steps + 1):
                # Ease-in-out using sine curve
                import math
                t = i / actual_steps
                # Slow start, fast middle, slow end
                eased_t = (1 - math.cos(t * math.pi)) / 2

                # Add slight overshoot near the end
                if t > 0.9:
                    overshoot = random.uniform(0, 5)
                else:
                    overshoot = 0

                current_x = start_x + total_distance * eased_t + overshoot

                # Add Y-axis wobble (human hands aren't perfectly steady)
                y_wobble = random.uniform(-2, 2)
                current_y = start_y + y_wobble

                await self._page.mouse.move(current_x, current_y)

                # Variable delay (humans don't move at constant speed)
                if t < 0.2:
                    # Slower at start
                    step_delay = random.uniform(0.02, 0.05)
                elif t > 0.8:
                    # Slower at end
                    step_delay = random.uniform(0.02, 0.04)
                else:
                    # Faster in middle
                    step_delay = random.uniform(0.005, 0.02)

                await asyncio.sleep(step_delay)

            # Final position adjustment
            await self._page.mouse.move(end_x, start_y)
            await asyncio.sleep(random.uniform(0.1, 0.2))
        else:
            # Simple linear movement
            for i in range(1, steps + 1):
                progress = i / steps
                current_x = start_x + (end_x - start_x) * progress
                current_y = start_y + (end_y - start_y) * progress
                await self._page.mouse.move(current_x, current_y)
                await asyncio.sleep(delay / 1000)

        # Release
        await self._page.mouse.up()
        logger.debug("Drag completed")

    async def scroll(
        self,
        direction: str,
        amount: int = 300,
    ) -> None:
        """Scroll page in specified direction.

        Args:
            direction: Direction to scroll ("up", "down", "left", "right")
            amount: Amount to scroll in pixels
        """
        if not self._page:
            raise RuntimeError("Browser not started. Call start() first.")

        delta_x, delta_y = 0, 0
        if direction == "down":
            delta_y = amount
        elif direction == "up":
            delta_y = -amount
        elif direction == "right":
            delta_x = amount
        elif direction == "left":
            delta_x = -amount

        logger.debug(f"Scrolling {direction} by {amount}px")
        await self._page.mouse.wheel(delta_x, delta_y)

        # Wait for scroll to complete
        await asyncio.sleep(0.3)

    async def wait_for_element(
        self,
        selector: str,
        timeout: Optional[int] = None,
        state: str = "visible",
    ) -> bool:
        """Wait for an element to appear.

        Args:
            selector: CSS selector for the element
            timeout: Timeout in milliseconds
            state: State to wait for ("attached", "detached", "visible", "hidden")

        Returns:
            True if element found, False if timeout
        """
        if not self._page:
            raise RuntimeError("Browser not started. Call start() first.")

        try:
            await self._page.wait_for_selector(
                selector,
                timeout=timeout or self._timeout,
                state=state,
            )
            return True
        except Exception:
            return False

    async def wait_for_url(
        self,
        url_pattern: str,
        timeout: Optional[int] = None,
    ) -> bool:
        """Wait for URL to match pattern.

        Args:
            url_pattern: URL pattern to match (can be substring)
            timeout: Timeout in milliseconds

        Returns:
            True if URL matches, False if timeout
        """
        if not self._page:
            raise RuntimeError("Browser not started. Call start() first.")

        try:
            await self._page.wait_for_url(
                f"**{url_pattern}**",
                timeout=timeout or self._timeout,
            )
            return True
        except Exception:
            return False

    async def select_option(
        self,
        selector: str,
        value: str,
    ) -> None:
        """Select an option from a dropdown.

        Args:
            selector: CSS selector for the select element
            value: Option value or label to select
        """
        if not self._page:
            raise RuntimeError("Browser not started. Call start() first.")

        logger.debug(f"Selecting '{value}' from {selector}")
        await self._page.select_option(selector, value)

    async def upload_file(
        self,
        selector: str,
        file_path: str,
    ) -> None:
        """Upload a file to a file input.

        Args:
            selector: CSS selector for the file input
            file_path: Path to the file to upload
        """
        if not self._page:
            raise RuntimeError("Browser not started. Call start() first.")

        logger.debug(f"Uploading file: {file_path}")
        await self._page.set_input_files(selector, file_path)

    async def handle_download(
        self,
        action_callback,
        save_as: Optional[str] = None,
    ) -> str:
        """Handle file download triggered by an action.

        Args:
            action_callback: Async function that triggers the download
            save_as: Optional filename to save as

        Returns:
            Path to the downloaded file
        """
        if not self._page:
            raise RuntimeError("Browser not started. Call start() first.")

        async with self._page.expect_download() as download_info:
            await action_callback()

        download = await download_info.value

        if save_as:
            save_path = self._download_dir / save_as
        else:
            save_path = self._download_dir / download.suggested_filename

        await download.save_as(save_path)
        logger.info(f"File downloaded to: {save_path}")
        return str(save_path)

    async def get_page_info(self) -> Dict[str, Any]:
        """Get current page information.

        Returns:
            Dictionary with url, title, viewport, and dpr
        """
        if not self._page:
            raise RuntimeError("Browser not started. Call start() first.")

        url = self._page.url
        title = await self._page.title()

        return {
            "url": url,
            "title": title,
            "viewport": {
                "width": self._viewport_width,
                "height": self._viewport_height,
            },
            "dpr": self._dpr,
        }

    async def get_element_at_point(
        self,
        x: float,
        y: float,
    ) -> Optional[Dict[str, Any]]:
        """Get element information at a specific point.

        This is useful for Vision→DOM alignment.

        Args:
            x: X coordinate in CSS pixels
            y: Y coordinate in CSS pixels

        Returns:
            Element information dict or None
        """
        if not self._page:
            raise RuntimeError("Browser not started. Call start() first.")

        result = await self._page.evaluate(
            """
            ([x, y]) => {
                const el = document.elementFromPoint(x, y);
                if (!el) return null;
                const rect = el.getBoundingClientRect();
                return {
                    tagName: el.tagName,
                    id: el.id,
                    className: el.className,
                    text: el.innerText?.substring(0, 100),
                    rect: {
                        x: rect.x,
                        y: rect.y,
                        width: rect.width,
                        height: rect.height
                    }
                };
            }
            """,
            [x, y],
        )
        return result

    async def evaluate(self, script: str, arg: Any = None) -> Any:
        """Execute JavaScript in the page context.

        Args:
            script: JavaScript code to execute
            arg: Optional argument to pass to the script

        Returns:
            Result of the script execution
        """
        if not self._page:
            raise RuntimeError("Browser not started. Call start() first.")

        if arg is not None:
            return await self._page.evaluate(script, arg)
        return await self._page.evaluate(script)

    async def reload(self) -> None:
        """Reload the current page."""
        if not self._page:
            raise RuntimeError("Browser not started. Call start() first.")

        logger.debug("Reloading page")
        await self._page.reload()

    async def go_back(self) -> None:
        """Navigate back in history."""
        if not self._page:
            raise RuntimeError("Browser not started. Call start() first.")

        logger.debug("Going back")
        await self._page.go_back()

    async def close(self) -> None:
        """Clean up browser resources."""
        logger.info("Closing browser...")

        if self._page:
            await self._page.close()
            self._page = None

        if self._context:
            await self._context.close()
            self._context = None

        if self._browser:
            await self._browser.close()
            self._browser = None

        if self._playwright:
            await self._playwright.stop()
            self._playwright = None

        logger.info("Browser closed")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
