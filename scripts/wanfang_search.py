#!/usr/bin/env python3
"""Wanfang Medical Literature Search Automation Script.

This script automates the process of searching for medical literature on Wanfang,
including setting search filters, entering search queries, and modifying date ranges.

Usage:
    python scripts/wanfang_search.py --query '((("得理多") OR "卡马西平") OR "Tegretol") OR "Carbamazepine"'

    # With custom date range:
    python scripts/wanfang_search.py --query '((("得理多") OR "卡马西平") OR "Tegretol") OR "Carbamazepine"' --start-date 2025/10/15 --end-date 2025/11/26
    python scripts/wanfang_search.py \
    --query '((("得理多") OR "卡马西平") OR "Tegretol") OR "Carbamazepine"' \
    --start-date 2025/10/15 \
    --end-date 2025/11/26 \
    --export \
    --no-stay


"""

import argparse
import asyncio
import os
import random
import sys
from pathlib import Path
from datetime import datetime

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from urllib.parse import quote

from browser_agent.core.browser_controller import BrowserController
from browser_agent.utils.log_utils import get_logger

logger = get_logger(__name__)

# Wanfang medical search page (default)
DEFAULT_SEARCH_URL = "https://med.wanfangdata.com.cn/Paper"
SEARCH_RESULT_URL = "https://med.wanfangdata.com.cn/Paper/Search"

# Default browser data directory for session persistence
DEFAULT_USER_DATA_DIR = str(Path(__file__).parent.parent / ".browser_data")

# Default search query
DEFAULT_QUERY = '((("得理多") OR "卡马西平") OR "Tegretol") OR "Carbamazepine"'


def build_search_url(
    query: str,
    start_year: str = "2019",
    end_year: str = None,
    resource_type: str = "chinese",
    page: int = 1,
    per_page: int = 50,
) -> str:
    """Build Wanfang search URL with filters as URL parameters.

    Args:
        query: Search query formula
        start_year: Start year for filter
        end_year: End year for filter (None = * for unlimited)
        resource_type: "chinese", "foreign", or "all"
        page: Page number (1-based)
        per_page: Number of results per page

    Returns:
        Complete search URL with encoded parameters
    """
    # Build year filter: 年份_fl=2009-* or 2009-2025
    if end_year:
        year_param = f"{start_year}-{end_year}"
    else:
        year_param = f"{start_year}-*"  # * means unlimited (至今)

    # Build resource type filter (include 学位论文 and 会议论文)
    resource_map = {
        "chinese": "(中文期刊 OR 学位论文 OR 会议论文)",
        "foreign": "(外文期刊)",
        "all": "(中文期刊 OR 外文期刊 OR 学位论文 OR 会议论文)",
    }
    resource_param = resource_map.get(resource_type, "(中文期刊 OR 学位论文 OR 会议论文)")

    # Build URL with encoded parameters (no 主题= prefix)
    url = (
        f"{SEARCH_RESULT_URL}"
        f"?q={quote(query)}"
        f"&年份_fl={quote(year_param)}"
        f"&资源类型_fl={quote(resource_param)}"
        f"&SearchMode=Professional"
        f"&n={per_page}"
        f"&p={page}"
    )

    return url


async def perform_search(
    url: str = DEFAULT_SEARCH_URL,
    query: str = DEFAULT_QUERY,
    start_year: str = "2019",
    end_year: str = None,
    resource_type: str = "chinese",
    headless: bool = False,
    stay_open: bool = True,
    user_data_dir: str = DEFAULT_USER_DATA_DIR,
    export_results: bool = False,
    export_dir: str = "/tmp/wanfang_exports",
    max_articles: int = 0,
) -> bool:
    """Perform automated search on Wanfang medical database.

    Args:
        url: Search page URL
        query: Search query formula
        start_year: Start year for search filter (e.g., "2019")
        end_year: End year for search filter (e.g., "2025"), None keeps default
        resource_type: Resource type - "chinese", "foreign", or "all"
        headless: Run browser in headless mode
        stay_open: Keep browser open after search
        user_data_dir: Directory to store browser data
        export_results: Enable export to Excel
        export_dir: Directory to save exported files
        max_articles: Maximum articles to export (0 = unlimited)

    Returns:
        True if search successful, False otherwise
    """
    browser = BrowserController(
        headless=headless,
        viewport_width=1400,
        viewport_height=900,
        slow_mo=50,
        user_data_dir=user_data_dir,
    )

    try:
        await browser.start()
        page = browser.page

        # Build search URL with all filters as parameters
        search_url = build_search_url(
            query=query,
            start_year=start_year,
            end_year=end_year,
            resource_type=resource_type,
        )

        # Step 1: Navigate directly to search results with filters
        year_range_str = f"{start_year} - {end_year}" if end_year else f"{start_year} - {datetime.now().year}"
        resource_type_label = {"chinese": "中文期刊", "foreign": "外文期刊", "all": "全部"}
        print(f"[1/2] Searching with filters:")
        print(f"      Query: {query[:50]}..." if len(query) > 50 else f"      Query: {query}")
        print(f"      Year: {year_range_str}")
        print(f"      Resource: {resource_type_label.get(resource_type, resource_type)}")
        print(f"[2/2] Navigating to search results...")
        await browser.navigate(search_url)
        await asyncio.sleep(3)  # Wait for results to load

        # Check for captcha and handle if present
        if await check_for_captcha(page):
            print(f"[!] Captcha detected, solving...")
            await solve_slider_captcha(browser)
            await asyncio.sleep(2)

        # Export results (if enabled) - directly start export without additional filtering
        if export_results:
            print(f"\n[Export] Starting export process...")
            exported_count = await export_search_results(
                browser=browser,
                export_dir=export_dir,
                max_articles=max_articles,
                query=query,
                start_year=start_year,
                end_year=end_year,
                resource_type=resource_type,
            )
            print(f"   Exported {exported_count} articles")

        # Capture result screenshot
        _, _ = await browser.capture_screenshot(save_path="/tmp/wanfang_search_result.png")
        print(f"\n✅ Search completed!")
        print(f"   Screenshot saved to: /tmp/wanfang_search_result.png")

        # Get current URL and title
        page_info = await browser.get_page_info()
        print(f"   Current URL: {page_info['url']}")
        print(f"   Page Title: {page_info['title']}")

        if stay_open:
            print("\nBrowser will stay open. Press Ctrl+C to close.")
            while True:
                await asyncio.sleep(1)

        return True

    except KeyboardInterrupt:
        print("\nClosing browser...")
    except Exception as e:
        logger.error(f"Search failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        if not stay_open:
            await browser.close()

    return False


async def select_resource_type(page, resource_type: str = "chinese"):
    """Select resource type by checking/unchecking checkboxes.

    The checkboxes have specific IDs:
    - #typeCnPeriodical - 中文期刊
    - #typeEnPeriodical - 外文期刊

    Args:
        page: Playwright page object
        resource_type: One of:
            - "chinese": Only Chinese journals (uncheck 外文期刊)
            - "foreign": Only foreign journals (uncheck 中文期刊)
            - "all": Both Chinese and foreign journals (keep both checked)
    """
    try:
        if resource_type == "all":
            logger.info("Keeping both 中文期刊 and 外文期刊 checked")
            return True

        # Determine which checkbox to uncheck based on resource type
        # chinese -> uncheck typeEnPeriodical (外文期刊)
        # foreign -> uncheck typeCnPeriodical (中文期刊)
        uncheck_id = "typeEnPeriodical" if resource_type == "chinese" else "typeCnPeriodical"
        uncheck_label = "外文期刊" if resource_type == "chinese" else "中文期刊"

        result = await page.evaluate("""
            (checkboxId) => {
                const checkbox = document.getElementById(checkboxId);
                if (checkbox) {
                    if (checkbox.checked) {
                        checkbox.checked = false;
                        // Trigger all relevant events
                        checkbox.dispatchEvent(new Event('change', { bubbles: true }));
                        checkbox.dispatchEvent(new Event('input', { bubbles: true }));
                        checkbox.dispatchEvent(new MouseEvent('click', { bubbles: true }));
                        return {success: true, method: 'by_id', wasChecked: true};
                    } else {
                        return {success: true, method: 'by_id', wasChecked: false, note: 'already unchecked'};
                    }
                }
                return {success: false, method: 'not_found'};
            }
        """, uncheck_id)

        if result and result.get('success'):
            if result.get('wasChecked'):
                logger.info(f"Unchecked '{uncheck_label}' (#{uncheck_id})")
            else:
                logger.info(f"'{uncheck_label}' was already unchecked")
            return True

        logger.warning(f"Could not find checkbox #{uncheck_id}")
        return False

    except Exception as e:
        logger.warning(f"Failed to select resource type: {e}")
        return False


async def enter_search_query(page, query: str):
    """Enter the search query into the search input field."""
    try:
        # Try common selectors for search input
        selectors = [
            'input[placeholder*="检索"]',
            'input[placeholder*="搜索"]',
            'input.search-input',
            'input[type="text"]',
            '#searchInput',
            '.search-box input',
            'textarea.search-input',
        ]

        for selector in selectors:
            try:
                elem = page.locator(selector).first
                if await elem.count() > 0:
                    await elem.click()
                    # Use JavaScript to set value directly (much faster than fill)
                    await elem.evaluate('(el, q) => { el.value = q; el.dispatchEvent(new Event("input", {bubbles: true})); }', query)
                    logger.info(f"Entered query via selector: {selector}")
                    return True
            except Exception:
                continue

        # Fallback: JavaScript to find and fill input
        result = await page.evaluate("""
            (query) => {
                const inputs = document.querySelectorAll('input[type="text"], textarea');
                for (const input of inputs) {
                    const placeholder = input.placeholder || '';
                    if (placeholder.includes('检索') || placeholder.includes('搜索') ||
                        input.className.includes('search')) {
                        input.focus();
                        input.value = query;
                        input.dispatchEvent(new Event('input', { bubbles: true }));
                        return true;
                    }
                }
                // Try first visible text input
                for (const input of inputs) {
                    if (input.offsetWidth > 0 && input.offsetHeight > 0) {
                        input.focus();
                        input.value = query;
                        input.dispatchEvent(new Event('input', { bubbles: true }));
                        return true;
                    }
                }
                return false;
            }
        """, query)

        if result:
            logger.info("Entered query via JavaScript")
            return True

        logger.error("Could not find search input field")
        return False

    except Exception as e:
        logger.error(f"Failed to enter search query: {e}")
        return False


async def set_year_filter(page, start_year: str, end_year: str = None):
    """Set the year filter using custom dropdown selects.

    The year selectors have specific IDs:
    - #startDateList - 起始年份选择器
    - #endDateList - 结束年份选择器

    Structure: <div id="startDateList"><h4><span>开始</span></h4><ul><li>2025</li>...</ul></div>

    Args:
        page: Playwright page object
        start_year: Start year (e.g., "2019")
        end_year: End year (e.g., "2025"), if None keeps default "结束"
    """
    try:
        async def select_year_from_dropdown(selector_id: str, year: str) -> bool:
            """Select a year from a custom dropdown.

            Args:
                selector_id: ID of the dropdown container (startDateList or endDateList)
                year: Year to select
            """
            # Step 1: Click the dropdown header to open the list
            opened = await page.evaluate("""
                (selectorId) => {
                    const container = document.getElementById(selectorId);
                    if (!container) return {success: false, error: 'Container not found'};

                    // Click on h4 or span to open dropdown
                    const trigger = container.querySelector('h4') || container.querySelector('span');
                    if (trigger) {
                        trigger.click();
                        return {success: true};
                    }
                    return {success: false, error: 'Trigger not found'};
                }
            """, selector_id)

            if not opened or not opened.get('success'):
                logger.warning(f"Could not open dropdown #{selector_id}: {opened}")
                return False

            # Wait for dropdown to open
            await asyncio.sleep(0.3)

            # Step 2: Click the target year in the list
            selected = await page.evaluate("""
                (args) => {
                    const selectorId = args.selectorId;
                    const year = args.year;

                    const container = document.getElementById(selectorId);
                    if (!container) return {success: false, error: 'Container not found'};

                    // Find and click the year in the ul > li list
                    const items = container.querySelectorAll('li');
                    for (const item of items) {
                        if (item.textContent.trim() === year) {
                            item.click();
                            // Trigger change event on the container
                            container.dispatchEvent(new Event('change', { bubbles: true }));
                            return {success: true, year: year};
                        }
                    }
                    return {success: false, error: 'Year not found in list', available: Array.from(items).slice(0,5).map(i => i.textContent)};
                }
            """, {"selectorId": selector_id, "year": year})

            await asyncio.sleep(0.2)

            if selected and selected.get('success'):
                return True
            else:
                logger.warning(f"Could not select year: {selected}")
                return False

        # Set start year
        start_success = await select_year_from_dropdown("startDateList", start_year)
        if start_success:
            logger.info(f"Set start year to {start_year}")
        else:
            logger.warning(f"Could not set start year to {start_year}")

        # Set end year if provided
        if end_year:
            await asyncio.sleep(0.3)
            end_success = await select_year_from_dropdown("endDateList", end_year)
            if end_success:
                logger.info(f"Set end year to {end_year}")
            else:
                logger.warning(f"Could not set end year to {end_year}")

        return start_success

    except Exception as e:
        logger.warning(f"Failed to set year filter: {e}")
        return False


async def click_search_button(page):
    """Click the search button."""
    try:
        # Try common selectors for search button
        selectors = [
            'button:has-text("检索")',
            'button:has-text("搜索")',
            'button.search-btn',
            'input[type="submit"]',
            '.search-button',
            '#searchBtn',
            'button[type="submit"]',
        ]

        for selector in selectors:
            try:
                elem = page.locator(selector).first
                if await elem.count() > 0:
                    await elem.click()
                    await asyncio.sleep(0.5)
                    logger.info(f"Clicked search button via selector: {selector}")
                    return True
            except Exception:
                continue

        # Fallback: JavaScript
        result = await page.evaluate("""
            () => {
                const buttons = document.querySelectorAll('button, input[type="submit"], a');
                for (const btn of buttons) {
                    const text = btn.textContent || btn.value || '';
                    if (text.includes('检索') || text.includes('搜索')) {
                        btn.click();
                        return true;
                    }
                }
                return false;
            }
        """)

        if result:
            await asyncio.sleep(0.5)
            logger.info("Clicked search button via JavaScript")
            return True

        logger.error("Could not find search button")
        return False

    except Exception as e:
        logger.error(f"Failed to click search button: {e}")
        return False


async def is_login_page(page) -> bool:
    """Check if current page is the login page."""
    try:
        title = await page.title()
        if "登录" in title:
            return True
        # Check URL
        url = page.url
        if "login" in url.lower() or "logon" in url.lower():
            return True
        return False
    except Exception:
        return False


async def perform_page_login(browser, page, username: str = None, password: str = None) -> bool:
    """Perform login on the login page.

    Credentials are read from environment variables WANFANG_USERNAME and WANFANG_PASSWORD.
    """
    # Get credentials from environment
    if username is None:
        username = os.environ.get("WANFANG_USERNAME", "")
    if password is None:
        password = os.environ.get("WANFANG_PASSWORD", "")

    if not username or not password:
        logger.error("Missing WANFANG_USERNAME or WANFANG_PASSWORD in environment")
        return False

    try:
        logger.info(f"Performing login with username: {username[:3]}***...")

        # Fill username
        username_input = page.locator('input[name="username"], input[type="text"], #username').first
        if await username_input.count() > 0:
            await username_input.fill(username)
            logger.info("Filled username")

        # Fill password
        password_input = page.locator('input[name="password"], input[type="password"], #password').first
        if await password_input.count() > 0:
            await password_input.fill(password)
            logger.info("Filled password")

        await asyncio.sleep(0.5)

        # Solve captcha if present
        if await check_for_captcha(page):
            await solve_slider_captcha(browser)
            await asyncio.sleep(1)

        # Click login button
        login_btn = page.locator('button.btn-login, button:has-text("登录"), input[type="submit"]').first
        if await login_btn.count() > 0:
            await login_btn.click()
            logger.info("Clicked login button")

            # Wait for login to complete (page should redirect)
            try:
                await page.wait_for_url("**/Paper/**", timeout=10000)
                logger.info("Login successful, redirected to search page")
            except Exception:
                # Fallback: wait and check if still on login page
                await asyncio.sleep(3)
                if not await is_login_page(page):
                    logger.info("Login appears successful")
                else:
                    logger.warning("Still on login page after clicking login")

        return True
    except Exception as e:
        logger.error(f"Login failed: {e}")
        return False


async def check_for_captcha(page) -> bool:
    """Check if captcha verification page is displayed."""
    try:
        # Check page title or content for captcha indicators
        title = await page.title()
        if "人机校验" in title or "安全验证" in title:
            return True

        # Check for captcha elements
        captcha_selectors = [
            'text=安全验证',
            'text=请通过校验',
            'text=请按住滑块',
            '.nc_iconfont',
            '#nc_1_n1z',
        ]

        for selector in captcha_selectors:
            try:
                elem = page.locator(selector).first
                if await elem.count() > 0:
                    return True
            except Exception:
                continue

        return False
    except Exception:
        return False


async def solve_slider_captcha(browser: BrowserController) -> bool:
    """Attempt to solve the slider captcha.

    Args:
        browser: BrowserController instance

    Returns:
        True if captcha appears to be solved, False otherwise
    """
    page = browser.page

    try:
        logger.info("Attempting to solve slider captcha...")
        await asyncio.sleep(0.5)

        # Find the slider button element
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

        if slider_elem:
            box = await slider_elem.bounding_box()
            if box:
                start_x = box['x'] + box['width'] / 2
                start_y = box['y'] + box['height'] / 2

                # Find the track to calculate end position
                track_selectors = ['.nc-lang-cnt', '#nc_1__scale_text', '.scale_text', '.nc_scale']
                end_x = start_x + 280  # Default distance

                for selector in track_selectors:
                    try:
                        track = page.locator(selector).first
                        if await track.count() > 0:
                            track_box = await track.bounding_box()
                            if track_box:
                                end_x = track_box['x'] + track_box['width'] - box['width'] / 2
                                logger.info(f"Found track, end_x: {end_x}")
                                break
                    except Exception:
                        continue

                logger.info(f"Dragging from ({start_x}, {start_y}) to ({end_x}, {start_y})")

                # Hover first to activate
                await page.mouse.move(start_x, start_y)
                await asyncio.sleep(random.uniform(0.3, 0.5))

                # Mouse down
                await page.mouse.down()
                await asyncio.sleep(random.uniform(0.05, 0.1))

                # Move in human-like steps
                distance = end_x - start_x
                steps = random.randint(30, 50)

                for i in range(1, steps + 1):
                    t = i / steps
                    # Ease out cubic for natural deceleration
                    eased = 1 - pow(1 - t, 3)

                    current_x = start_x + distance * eased
                    wobble_y = start_y + random.uniform(-1.5, 1.5)

                    await page.mouse.move(current_x, wobble_y)

                    # Variable timing
                    if t < 0.3:
                        await asyncio.sleep(random.uniform(0.01, 0.025))
                    elif t > 0.85:
                        await asyncio.sleep(random.uniform(0.015, 0.035))
                    else:
                        await asyncio.sleep(random.uniform(0.005, 0.015))

                # Final position
                await page.mouse.move(end_x, start_y)
                await asyncio.sleep(random.uniform(0.3, 0.5))
                await page.mouse.up()

                logger.info("Released slider at end position")
                await asyncio.sleep(1)

                # Check if verification passed
                try:
                    success = await page.locator('.nc_ok, .nc-lang-cnt:has-text("验证通过")').count()
                    if success > 0:
                        logger.info("Slider verification passed!")
                        return True
                except Exception:
                    pass

                return True

        logger.warning("Could not find slider element")
        return False

    except Exception as e:
        logger.error(f"Failed to solve slider captcha: {e}")
        return False


# ============== Export Functions ==============

async def get_total_results(page) -> int:
    """Get total number of search results."""
    try:
        # Try to find result count text like "共 52 条结果"
        result = await page.evaluate("""
            () => {
                const text = document.body.innerText;
                // Match patterns like "共 52 条" or "共52条"
                const match = text.match(/共\\s*(\\d+)\\s*条/);
                if (match) {
                    return parseInt(match[1], 10);
                }
                return 0;
            }
        """)
        logger.info(f"Total results: {result}")
        return result
    except Exception as e:
        logger.warning(f"Failed to get total results: {e}")
        return 0


async def is_page_empty(page) -> bool:
    """Check if current page has no results.

    Detects empty page by:
    1. "共 0 条" text
    2. "抱歉，系统没有检索到相关记录" message
    """
    try:
        return await page.evaluate("""
            () => {
                const text = document.body.innerText;
                // Check for "共 0 条"
                if (/共\\s*0\\s*条/.test(text)) return true;
                // Check for "抱歉" + "没有检索到" message
                if (text.includes('抱歉') && text.includes('没有检索到')) return true;
                return false;
            }
        """)
    except Exception:
        return False


async def select_page_size(page, size: str = "50") -> bool:
    """Select the number of articles to display per page.

    Wanfang uses <a> links with href containing &n=XX parameter for page size.
    Example: <a href="/Paper/Search?q=...&n=50">50篇</a>
    """
    try:
        # Method 1: Find <a> link with exact text "50篇"
        selectors = [
            f'a:has-text("{size}篇")',
            f'li a:has-text("{size}篇")',
            f'a[href*="n={size}"]',
        ]

        for selector in selectors:
            try:
                elems = page.locator(selector)
                count = await elems.count()
                for i in range(count):
                    elem = elems.nth(i)
                    text = await elem.text_content()
                    # Ensure exact match (e.g., "50篇" not "150篇")
                    if text and text.strip() == f"{size}篇":
                        await elem.click()
                        await asyncio.sleep(1)
                        logger.info(f"Selected page size {size} via selector: {selector}")
                        return True
            except Exception:
                continue

        # Method 2: Use JavaScript to find and click the link
        result = await page.evaluate("""
            (size) => {
                // Find <a> tags with href containing n=XX
                const links = document.querySelectorAll('a[href*="n=' + size + '"]');
                for (const link of links) {
                    const text = (link.textContent || '').trim();
                    if (text === size + '篇') {
                        link.click();
                        return true;
                    }
                }

                // Fallback: find by text content
                const allLinks = document.querySelectorAll('a');
                for (const link of allLinks) {
                    const text = (link.textContent || '').trim();
                    if (text === size + '篇') {
                        link.click();
                        return true;
                    }
                }
                return false;
            }
        """, size)

        if result:
            await asyncio.sleep(1)
            logger.info(f"Selected page size {size} via JavaScript")
            return True

        logger.warning(f"Could not select page size {size}")
        return False

    except Exception as e:
        logger.warning(f"Failed to select page size: {e}")
        return False


async def select_all_articles(page) -> bool:
    """Select all articles on the current page."""
    try:
        selectors = [
            'text=全选',
            'input[type="checkbox"]:has-text("全选")',
            '.select-all',
            'label:has-text("全选")',
        ]

        for selector in selectors:
            try:
                elem = page.locator(selector).first
                if await elem.count() > 0:
                    await elem.click()
                    logger.info(f"Selected all articles: {selector}")
                    return True
            except Exception:
                continue

        # Fallback: JavaScript
        result = await page.evaluate("""
            () => {
                const elements = document.querySelectorAll('*');
                for (const el of elements) {
                    if (el.textContent && el.textContent.trim() === '全选') {
                        el.click();
                        return true;
                    }
                }
                // Try checkbox
                const checkbox = document.querySelector('input[type="checkbox"]');
                if (checkbox) {
                    checkbox.click();
                    return true;
                }
                return false;
            }
        """)

        if result:
            logger.info("Selected all articles via JavaScript")
            return True

        logger.warning("Could not find select all button")
        return False

    except Exception as e:
        logger.warning(f"Failed to select all articles: {e}")
        return False


async def click_reference_export(page) -> bool:
    """Click '批量导出' to open export dialog or navigate to export page.

    The Wanfang UI may vary - sometimes it opens a dialog, sometimes it navigates.
    """
    try:
        # Try Playwright selector first for better reliability
        selectors = [
            'text=批量导出',
            'button:has-text("批量导出")',
            'a:has-text("批量导出")',
            'span:has-text("批量导出")',
        ]

        clicked = False
        for selector in selectors:
            try:
                elem = page.locator(selector).first
                if await elem.count() > 0:
                    # Hover first to ensure visibility
                    await elem.hover()
                    await asyncio.sleep(0.1)
                    await elem.click()
                    logger.info(f"Clicked batch export button: {selector}")
                    clicked = True
                    break
            except Exception as e:
                logger.debug(f"Selector {selector} failed: {e}")
                continue

        if not clicked:
            # Fallback: JavaScript
            result = await page.evaluate("""
                () => {
                    const elements = document.querySelectorAll('button, a, span, div');
                    for (const el of elements) {
                        const text = (el.textContent || '').trim();
                        if (text === '批量导出') {
                            el.click();
                            return 'exact';
                        }
                    }
                    for (const el of elements) {
                        const text = (el.textContent || '').trim();
                        if (text.includes('批量导出')) {
                            el.click();
                            return 'contains';
                        }
                    }
                    return null;
                }
            """)
            if result:
                logger.info(f"Clicked batch export via JavaScript: {result}")
                clicked = True

        if clicked:
            # Wait for UI response - either dialog or page change
            await asyncio.sleep(random.uniform(1.5, 2.5))  # 模拟真人点击后等待
            return True

        logger.warning("Could not find batch export button")
        return False

    except Exception as e:
        logger.warning(f"Failed to click batch export: {e}")
        return False


async def export_to_excel(browser: BrowserController, export_dir: str, batch_num: int) -> str:
    """Click '导出到Excel' and download the file.

    Optimized version: Wait for dialog, then use JavaScript click with expect_download.
    """
    import os

    page = browser.page

    # Ensure export directory exists
    os.makedirs(export_dir, exist_ok=True)
    download_path = f"/tmp/browser_agent_screenshots/downloads/wanfang_export_batch_{batch_num}.xlsx"

    try:
        # Wait briefly for export dialog to appear after clicking 批量导出
        await asyncio.sleep(0.5)

        # Set up download handler and click Excel button
        async with page.expect_download(timeout=20000) as download_info:
            # Use JavaScript to find and click Excel button
            clicked = await page.evaluate("""
                () => {
                    // Priority order: look for Excel export buttons (MUST contain "Excel")
                    const patterns = ['导出到Excel', '导出Excel', '导出至Excel'];
                    const elements = document.querySelectorAll('button, a, span, div, li, input');

                    // First pass: exact matches
                    for (const pattern of patterns) {
                        for (const el of elements) {
                            const text = (el.textContent || '').trim();
                            if (text === pattern) {
                                if (el.offsetParent !== null) {
                                    el.click();
                                    return 'exact:' + pattern;
                                }
                            }
                        }
                    }

                    // Second pass: contains pattern
                    for (const pattern of patterns) {
                        for (const el of elements) {
                            const text = (el.textContent || '').trim();
                            if (text.includes(pattern)) {
                                if (el.offsetParent !== null) {
                                    el.click();
                                    return 'contains:' + pattern;
                                }
                            }
                        }
                    }

                    // Third pass: any element with "Excel" (case insensitive)
                    for (const el of elements) {
                        const text = (el.textContent || '').trim();
                        if (text.toLowerCase().includes('excel') && el.offsetParent !== null) {
                            el.click();
                            return 'excel:' + text.substring(0, 30);
                        }
                    }

                    return null;
                }
            """)

            if clicked:
                logger.info(f"Clicked Excel export button: {clicked}")
            else:
                logger.warning("Could not find Excel export button via JavaScript")
                # Try Playwright selectors as backup
                selectors = [
                    'text=导出到Excel',
                    'text=导出Excel',
                    'button:has-text("Excel")',
                    'a:has-text("Excel")',
                ]
                for selector in selectors:
                    try:
                        elem = page.locator(selector).first
                        if await elem.count() > 0 and await elem.is_visible():
                            await elem.click()
                            logger.info(f"Clicked Excel button via Playwright: {selector}")
                            break
                    except Exception:
                        continue

            # Dismiss any alert that might appear
            await dismiss_export_alert(page)

        # Wait for download to complete
        download = await download_info.value
        await download.save_as(download_path)
        logger.info(f"Exported to: {download_path}")

        # Close dialog
        await close_export_dialog(page)

        return download_path

    except Exception as e:
        logger.warning(f"Failed to export to Excel: {e}")
        # Try to close any open dialog
        await page.keyboard.press("Escape")
        return ""


async def dismiss_export_alert(page) -> bool:
    """Dismiss the '导出记录数量超额' alert dialog by clicking '确定'.

    This alert appears when the number of records exceeds export limits.
    We need to click '确定' to acknowledge and continue.
    """
    try:
        await asyncio.sleep(0.1)  # Brief wait for dialog to appear (reduced from 0.5s)

        # Look for the confirm button in the alert dialog
        confirm_selectors = [
            'button:has-text("确定")',
            'button:has-text("确认")',
            'a:has-text("确定")',
            'span:has-text("确定")',
            '.modal button:has-text("确定")',
            '.dialog button:has-text("确定")',
            '[class*="modal"] button:has-text("确定")',
            '[class*="dialog"] button:has-text("确定")',
        ]

        for selector in confirm_selectors:
            try:
                elem = page.locator(selector).first
                if await elem.count() > 0 and await elem.is_visible():
                    await elem.click()
                    logger.info(f"Dismissed export alert dialog: {selector}")
                    await asyncio.sleep(0.1)  # Reduced from 0.3s
                    return True
            except Exception:
                continue

        # Fallback: JavaScript click
        result = await page.evaluate("""
            () => {
                // Look for modal/dialog with "确定" button
                const buttons = document.querySelectorAll('button, a, span');
                for (const btn of buttons) {
                    const text = (btn.textContent || '').trim();
                    if (text === '确定' || text === '确认') {
                        // Check if this button is inside a modal/dialog
                        const parent = btn.closest('.modal, .dialog, [class*="modal"], [class*="dialog"], [role="dialog"]');
                        if (parent || btn.offsetParent !== null) {
                            btn.click();
                            return true;
                        }
                    }
                }
                return false;
            }
        """)

        if result:
            logger.info("Dismissed export alert via JavaScript")
            await asyncio.sleep(0.1)  # Reduced from 0.3s
            return True

        return False

    except Exception as e:
        logger.debug(f"No export alert to dismiss: {e}")
        return False


async def close_export_dialog(page) -> bool:
    """Close the export dialog/panel after exporting.

    IMPORTANT: This function should NOT navigate away from the current page
    as we need to preserve the current pagination state.
    """
    try:
        # NOTE: Removed redundant dismiss_export_alert() call here
        # export_to_excel() already calls it before download

        # Try to close any modal/dialog
        close_selectors = [
            'button:has-text("关闭")',
            'button:has-text("×")',
            'a:has-text("关闭")',
            '.close',
            '.modal-close',
            '[aria-label="Close"]',
            'button.close',
        ]

        for selector in close_selectors:
            try:
                elem = page.locator(selector).first
                if await elem.count() > 0:
                    await elem.click()
                    logger.info(f"Closed export dialog: {selector}")
                    return True
            except Exception:
                continue

        # Try pressing Escape to close any dialog
        await page.keyboard.press("Escape")
        logger.info("Pressed Escape to close dialog")

        # Try clicking outside the dialog to close it
        await page.mouse.click(100, 100)

        # NOTE: Removed page.go_back() as it resets pagination to page 1
        # The export dialog on Wanfang is typically a modal overlay
        # that doesn't change the underlying page URL

        return True

    except Exception as e:
        logger.warning(f"Failed to close export dialog: {e}")
        return False


async def get_selection_count(page) -> int:
    """Get the current selection count from "已选 X/200" display."""
    try:
        result = await page.evaluate("""
            () => {
                // Look for "已选 X/200" pattern
                const text = document.body.innerText;
                const match = text.match(/已选\\s*(\\d+)\\s*[/／]/);
                if (match) {
                    return parseInt(match[1], 10);
                }
                // Alternative: look for selection counter element
                const counters = document.querySelectorAll('[class*="select"], [class*="count"], .batch-bar, .selection-bar');
                for (const counter of counters) {
                    const m = counter.textContent.match(/已选\\s*(\\d+)/);
                    if (m) return parseInt(m[1], 10);
                }
                return -1;  // Not found
            }
        """)
        return result
    except Exception as e:
        logger.debug(f"Could not get selection count: {e}")
        return -1


async def clear_selection(page) -> bool:
    """Clear current article selection.

    This function will:
    1. Check current selection count
    2. Try multiple methods to click "清除" button
    3. Verify that selection was actually cleared
    """
    try:
        # First, check current selection count
        before_count = await get_selection_count(page)
        logger.info(f"Selection count before clear: {before_count}")

        if before_count == 0:
            logger.info("Selection already empty, no need to clear")
            return True

        # Method 1: Playwright selectors for "清除" button
        selectors = [
            # Direct text match
            'text=清除',
            # In a batch operation bar
            '.batch-bar >> text=清除',
            '.selection-bar >> text=清除',
            '[class*="batch"] >> text=清除',
            '[class*="select"] >> text=清除',
            # Button/link elements
            'button:has-text("清除")',
            'a:has-text("清除")',
            'span:has-text("清除"):not(:has(*))',  # Leaf span nodes only
            '.clear-btn',
            '.clear-selection',
            '[class*="clear"]',
        ]

        clicked = False
        for selector in selectors:
            try:
                elem = page.locator(selector).first
                if await elem.count() > 0 and await elem.is_visible():
                    await elem.click()
                    logger.info(f"Clicked clear button: {selector}")
                    clicked = True
                    break
            except Exception as e:
                logger.debug(f"Selector {selector} failed: {e}")
                continue

        # Method 2: JavaScript click with more precise targeting
        if not clicked:
            result = await page.evaluate("""
                () => {
                    // Look for "清除" in the batch operation bar
                    const batchBars = document.querySelectorAll('.batch-bar, .selection-bar, [class*="batch"], [class*="action"], .result-header, .toolbar');
                    for (const bar of batchBars) {
                        const clearBtn = Array.from(bar.querySelectorAll('a, button, span')).find(
                            el => el.textContent && el.textContent.trim() === '清除'
                        );
                        if (clearBtn) {
                            clearBtn.click();
                            return {success: true, location: 'batch-bar'};
                        }
                    }

                    // Fallback: find any "清除" text that's clickable
                    const allElements = document.querySelectorAll('a, button, span, div');
                    for (const el of allElements) {
                        if (el.textContent && el.textContent.trim() === '清除') {
                            // Check if it's a leaf node or has click handler
                            if (el.children.length === 0 || el.onclick || el.tagName === 'A' || el.tagName === 'BUTTON') {
                                el.click();
                                return {success: true, location: 'generic'};
                            }
                        }
                    }

                    return {success: false, location: null};
                }
            """)

            if result and result.get('success'):
                logger.info(f"Cleared selection via JavaScript ({result.get('location')})")
                clicked = True

        if not clicked:
            logger.warning("Could not find clear button on page")
            # Try pressing Escape to close any dialogs that might be blocking
            await page.keyboard.press("Escape")
            await asyncio.sleep(0.2)
            return False

        # Wait for clear action to take effect
        await asyncio.sleep(0.5)

        # Verify that selection was cleared
        after_count = await get_selection_count(page)
        logger.info(f"Selection count after clear: {after_count}")

        if after_count == 0:
            logger.info("Selection successfully cleared!")
            return True
        elif after_count > 0 and after_count < before_count:
            logger.warning(f"Selection partially cleared: {before_count} -> {after_count}")
            return False
        elif after_count == before_count:
            logger.warning(f"Clear did not work, count still {after_count}")
            return False
        else:
            # after_count is -1 (couldn't read) - assume success if we clicked
            return True

    except Exception as e:
        logger.warning(f"Failed to clear selection: {e}")
        return False


async def go_to_next_page(page) -> bool:
    """Navigate to the next page of search results.

    Wanfang pagination format: 共40页 | 首页 | < | 1 2 3 4 5 | > | 跳至 __ 页

    Returns:
        True if successfully navigated to next page, False if on last page or error
    """
    try:
        # First check if we're on the last page by parsing "共X页" and current page
        is_last = await page.evaluate("""
            () => {
                const text = document.body.innerText;
                // Find total pages "共40页"
                const totalMatch = text.match(/共(\\d+)页/);
                if (!totalMatch) return false;
                const totalPages = parseInt(totalMatch[1], 10);

                // Find current active page (highlighted number)
                const activeBtn = document.querySelector('button[style*="background"], a[style*="background"], .active, [class*="current"]');
                if (activeBtn) {
                    const current = parseInt(activeBtn.textContent, 10);
                    if (!isNaN(current) && current >= totalPages) {
                        return true;  // On last page
                    }
                }
                return false;
            }
        """)

        if is_last:
            logger.info("Already on last page")
            return False

        # Click the ">" button for next page
        selectors = [
            'button:has-text(">")',
            'a:has-text(">")',
            'span:has-text(">")',
        ]

        for selector in selectors:
            try:
                # Find element with exactly ">" text (not ">>" or other)
                elems = page.locator(selector)
                count = await elems.count()
                for i in range(count):
                    elem = elems.nth(i)
                    text = await elem.text_content()
                    if text and text.strip() == '>':
                        # Check if disabled
                        is_disabled = await elem.get_attribute("disabled")
                        class_name = await elem.get_attribute("class") or ""
                        if is_disabled or "disabled" in class_name:
                            logger.info("Next button is disabled, on last page")
                            return False

                        await elem.click()
                        logger.info("Clicked '>' to go to next page")
                        return True
            except Exception:
                continue

        # Fallback: JavaScript - click the ">" button
        result = await page.evaluate("""
            () => {
                // Find all buttons/links and look for ">"
                const elements = document.querySelectorAll('button, a, span');
                for (const el of elements) {
                    const text = (el.textContent || '').trim();
                    if (text === '>') {
                        if (!el.disabled && !el.className.includes('disabled')) {
                            el.click();
                            return true;
                        }
                    }
                }
                return false;
            }
        """)

        if result:
            logger.info("Navigated to next page via JavaScript")
            return True

        logger.info("No more pages available")
        return False

    except Exception as e:
        logger.warning(f"Failed to navigate to next page: {e}")
        return False


async def export_search_results(
    browser: BrowserController,
    export_dir: str,
    max_articles: int = 0,
    query: str = "",
    start_year: str = "2019",
    end_year: str = None,
    resource_type: str = "chinese",
) -> int:
    """Export all search results using URL-based pagination.

    Simple workflow:
    1. Build base URL with search parameters
    2. For each page: navigate to URL with p=N → select all → export
    3. Merge all batch files

    Args:
        browser: BrowserController instance
        export_dir: Directory to save exported files
        max_articles: Maximum articles to export (0 = unlimited)
        query: Search query for URL building
        start_year: Start year filter
        end_year: End year filter
        resource_type: Resource type filter

    Returns:
        Total number of articles exported
    """
    page = browser.page

    # Get total results from current page
    total = await get_total_results(page)
    print(f"   Total results: {total}")

    if total == 0:
        logger.warning("No results to export")
        return 0

    # Apply max_articles limit if specified
    if max_articles > 0:
        total = min(total, max_articles)
        print(f"   Limited to {max_articles} articles")

    exported = 0
    current_page = 1
    articles_per_page = 50

    # Calculate total pages
    total_pages = (total + articles_per_page - 1) // articles_per_page
    print(f"   Total pages to export: {total_pages}")

    # Export each page using URL-based pagination
    while current_page <= total_pages:
        # Build URL for current page
        page_url = build_search_url(
            query=query,
            start_year=start_year,
            end_year=end_year,
            resource_type=resource_type,
            page=current_page,
            per_page=articles_per_page,
        )

        # Navigate to the page (skip for page 1 if already there)
        if current_page > 1:
            print(f"   [Page {current_page}/{total_pages}] Navigating to page {current_page}...")
            await browser.navigate(page_url)
            await asyncio.sleep(random.uniform(2.0, 3.5))  # 增加延迟，模拟真人

            # Check if redirected to login page
            if await is_login_page(page):
                print(f"   [!] Login page detected, performing login...")
                await perform_page_login(browser, page)
                # Navigate back to the search page after login
                await browser.navigate(page_url)
                await asyncio.sleep(random.uniform(2.0, 3.5))  # 增加延迟
            # Check for captcha after navigation
            elif await check_for_captcha(page):
                print(f"   [!] Captcha detected, solving...")
                await solve_slider_captcha(browser)
                await asyncio.sleep(random.uniform(1.5, 2.5))  # 增加延迟

        # Wait for content to load
        try:
            await page.wait_for_selector('.paper-list, .result-list, [class*="paper"], [class*="result"]', timeout=3000)  # Reduced from 5000ms
        except Exception:
            pass

        # Check if page is empty (no results)
        if await is_page_empty(page):
            print(f"   [Page {current_page}/{total_pages}] Page is empty, stopping pagination")
            break

        # Step 1: Clear selection first (important: max 200 selection limit)
        # Check selection count before clearing
        selection_count = await get_selection_count(page)
        if selection_count > 0:
            print(f"   [Page {current_page}/{total_pages}] Step 1: Clearing {selection_count} selected items...")
        else:
            print(f"   [Page {current_page}/{total_pages}] Step 1: Checking selection status...")

        clear_success = await clear_selection(page)

        # Retry clear if it failed and there were selections
        if not clear_success and selection_count > 0:
            print(f"   [Page {current_page}/{total_pages}] Retrying clear...")
            await asyncio.sleep(0.2)  # Reduced from 0.5s
            # Try refreshing the page to reset selection state
            await browser.navigate(page_url)
            await asyncio.sleep(0.5)  # Reduced from 1s
            # Try clear again
            clear_success = await clear_selection(page)
            if not clear_success:
                print(f"   [!] Warning: Could not clear selection, may hit 200 limit")

        await asyncio.sleep(random.uniform(0.8, 1.5))  # 模拟真人操作间隔

        # Step 2: Select all articles on this page
        print(f"   [Page {current_page}/{total_pages}] Step 2: Selecting all articles...")
        await select_all_articles(page)
        await asyncio.sleep(random.uniform(0.8, 1.5))  # 模拟真人操作间隔

        # Verify selection count after selecting
        new_count = await get_selection_count(page)
        if new_count > 0:
            print(f"   [Page {current_page}/{total_pages}] Selected {new_count} articles")

        # Step 3: Export this page
        print(f"   [Page {current_page}/{total_pages}] Step 3: Opening export dialog...")
        await click_reference_export(page)

        print(f"   [Page {current_page}/{total_pages}] Exporting to Excel...")
        file_path = await export_to_excel(browser, export_dir, current_page)

        if file_path:
            remaining = total - exported
            articles_this_page = min(articles_per_page, remaining)
            exported += articles_this_page
            print(f"   [Page {current_page}/{total_pages}] Saved: {file_path} (~{articles_this_page} articles)")
        else:
            print(f"   [Page {current_page}/{total_pages}] Warning: Export failed")

        # Check if we're done
        if current_page >= total_pages:
            print(f"   Reached last page (page {current_page})")
            break

        current_page += 1

    print(f"   Export complete: ~{exported} articles across {current_page} page(s)")

    # Post-processing: Merge all batch files into one
    if current_page >= 1:
        print(f"\n   [Post-processing] Merging {current_page} batch files...")
        merged_file, unique_rows, header_row = merge_excel_files(export_dir, current_page)
        if merged_file:
            print(f"   Merged file saved to: {merged_file}")

            # Compare with history and filter new articles
            print(f"\n   [Post-processing] Comparing with download history...")
            new_count, pending_file = filter_and_update_history(unique_rows, header_row)
            if pending_file:
                print(f"   New articles to download: {new_count}")
                print(f"   Pending file saved to: {pending_file}")

    return exported


def merge_excel_files(export_dir: str, batch_count: int) -> tuple:
    """Merge multiple batch Excel files into a single file.

    Uses zipfile to read Excel XML content and merge without pandas/openpyxl.

    Args:
        export_dir: Directory containing batch files
        batch_count: Number of batch files to merge

    Returns:
        Tuple of (path to merged file, unique_rows list, header_row list)
    """
    import zipfile
    import xml.etree.ElementTree as ET
    import shutil

    try:
        # Default download directory
        download_dir = "/tmp/browser_agent_screenshots/downloads"

        # Collect all batch files
        batch_files = []
        for i in range(1, batch_count + 1):
            file_path = f"{download_dir}/wanfang_export_batch_{i}.xlsx"
            if os.path.exists(file_path):
                batch_files.append(file_path)

        if not batch_files:
            logger.warning("No batch files found to merge")
            return "", [], []

        if len(batch_files) == 1:
            # Only one file, read and return its data
            pass  # Continue to read the single file

        # Read all data from batch files
        all_rows = []
        header_row = None
        title_row = None

        for file_idx, file_path in enumerate(batch_files):
            try:
                with zipfile.ZipFile(file_path, 'r') as zf:
                    # Read shared strings
                    shared_strings = []
                    if 'xl/sharedStrings.xml' in zf.namelist():
                        ss_content = zf.read('xl/sharedStrings.xml').decode('utf-8')
                        # Parse shared strings
                        ss_root = ET.fromstring(ss_content)
                        ns = {'main': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
                        for si in ss_root.findall('.//main:si', ns):
                            t_elem = si.find('.//main:t', ns)
                            if t_elem is not None and t_elem.text:
                                shared_strings.append(t_elem.text)
                            else:
                                shared_strings.append('')

                    # Read worksheet
                    sheet_content = zf.read('xl/worksheets/sheet1.xml').decode('utf-8')
                    sheet_root = ET.fromstring(sheet_content)

                    # Parse rows
                    for row in sheet_root.findall('.//{http://schemas.openxmlformats.org/spreadsheetml/2006/main}row'):
                        row_num = int(row.get('r', 0))
                        cells = []

                        for cell in row.findall('.//{http://schemas.openxmlformats.org/spreadsheetml/2006/main}c'):
                            cell_type = cell.get('t', '')
                            v_elem = cell.find('.//{http://schemas.openxmlformats.org/spreadsheetml/2006/main}v')

                            if v_elem is not None and v_elem.text:
                                if cell_type == 's':  # Shared string
                                    idx = int(v_elem.text)
                                    cells.append(shared_strings[idx] if idx < len(shared_strings) else '')
                                else:  # Number or other
                                    cells.append(v_elem.text)
                            else:
                                cells.append('')

                        # First file: capture title and header rows
                        if file_idx == 0:
                            if row_num == 1:
                                title_row = cells
                            elif row_num == 2:
                                header_row = cells
                            else:
                                all_rows.append(cells)
                        else:
                            # Skip title and header rows for subsequent files
                            if row_num > 2:
                                all_rows.append(cells)

            except Exception as e:
                logger.warning(f"Error reading {file_path}: {e}")
                continue

        # Remove duplicates based on article title
        # Column order: 序号(0), 作者(1), 篇名(2), 刊名(3), ...
        seen_titles = set()
        unique_rows = []
        for row in all_rows:
            if len(row) > 2:
                title = row[2]  # 篇名 is at index 2
                if title and title not in seen_titles:
                    seen_titles.add(title)
                    unique_rows.append(row)
            else:
                unique_rows.append(row)

        print(f"   Total rows: {len(all_rows)}, Unique articles: {len(unique_rows)}")

        # Write merged CSV file (simpler than recreating xlsx)
        merged_csv_path = f"{download_dir}/wanfang_merged.csv"
        with open(merged_csv_path, 'w', encoding='utf-8-sig') as f:
            # Write header
            if header_row:
                f.write(','.join(f'"{c}"' for c in header_row) + '\n')

            # Write data rows with proper sequence numbers
            for idx, row in enumerate(unique_rows, 1):
                # Update sequence number (first column)
                if row:
                    row[0] = str(idx)
                f.write(','.join(f'"{c}"' for c in row) + '\n')

        print(f"   Removed {len(all_rows) - len(unique_rows)} duplicate articles")
        return merged_csv_path, unique_rows, header_row

    except Exception as e:
        logger.error(f"Failed to merge Excel files: {e}")
        import traceback
        traceback.print_exc()
        return "", [], []


# Project data directory for history tracking
DATA_DIR = Path(__file__).parent.parent / "data"
HISTORY_CSV = DATA_DIR / "downloaded_history.csv"


def filter_and_update_history(unique_rows: list, header_row: list) -> tuple:
    """Compare with download history and filter out already downloaded articles.

    Args:
        unique_rows: List of article rows from current export
        header_row: Header row with column names

    Returns:
        Tuple of (new_article_count, pending_csv_path)
    """
    try:
        # Ensure data directory exists
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        # Load existing history
        existing_titles = set()
        if HISTORY_CSV.exists():
            with open(HISTORY_CSV, 'r', encoding='utf-8-sig') as f:
                for line in f:
                    if line.strip() and not line.startswith('"序号"'):
                        parts = line.strip().split('","')
                        if len(parts) > 3:
                            # Title is at index 2 (篇名)
                            title = parts[2].strip('"')
                            existing_titles.add(title)
            print(f"   Loaded {len(existing_titles)} articles from history")

        # Filter out already downloaded articles
        # Column order: 序号(0), 作者(1), 篇名(2), 刊名(3), ...
        new_rows = []
        for row in unique_rows:
            if len(row) > 2:
                title = row[2]  # 篇名 is at index 2
                if title and title not in existing_titles:
                    new_rows.append(row)

        print(f"   Found {len(new_rows)} new articles (not in history)")

        if not new_rows:
            print(f"   No new articles to download")
            return 0, ""

        pending_csv_path = DATA_DIR / "pending_download.csv"

        # Write new articles to pending CSV
        with open(pending_csv_path, 'w', encoding='utf-8-sig') as f:
            # Write header
            if header_row:
                f.write(','.join(f'"{c}"' for c in header_row) + '\n')

            # Write data rows with proper sequence numbers
            for idx, row in enumerate(new_rows, 1):
                if row:
                    row[0] = str(idx)
                f.write(','.join(f'"{c}"' for c in row) + '\n')

        # NOTE: Do NOT update history here!
        # History should only be updated when download succeeds (in wanfang_download.py)
        # This prevents marking articles as "downloaded" when they haven't been
        print(f"   Pending download list: {pending_csv_path}")

        return len(new_rows), str(pending_csv_path)

    except Exception as e:
        logger.error(f"Failed to filter and update history: {e}")
        import traceback
        traceback.print_exc()
        return 0, ""


def main():
    parser = argparse.ArgumentParser(description="Wanfang Medical Literature Search")
    parser.add_argument(
        "--url", "-u",
        default=DEFAULT_SEARCH_URL,
        help=f"Search page URL (default: {DEFAULT_SEARCH_URL})",
    )
    parser.add_argument(
        "--query", "-q",
        default=DEFAULT_QUERY,
        help=f"Search query formula (default: {DEFAULT_QUERY})",
    )
    parser.add_argument(
        "--start-year", "-s",
        default="2019",
        help="Start year for search filter (default: 2019)",
    )
    parser.add_argument(
        "--end-year", "-e",
        default=None,
        help="End year for search filter (default: None, keeps '结束')",
    )
    parser.add_argument(
        "--resource-type", "-r",
        default="chinese",
        choices=["chinese", "foreign", "all"],
        help="Resource type: chinese (中文期刊), foreign (外文期刊), all (both) (default: chinese)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run browser in headless mode",
    )
    parser.add_argument(
        "--no-stay",
        action="store_true",
        help="Close browser after search",
    )
    parser.add_argument(
        "--no-persist",
        action="store_true",
        help="Don't persist browser session",
    )
    parser.add_argument(
        "--user-data-dir",
        default=DEFAULT_USER_DATA_DIR,
        help=f"Directory to store browser data (default: {DEFAULT_USER_DATA_DIR})",
    )
    parser.add_argument(
        "--export", "-x",
        action="store_true",
        help="Enable export mode to download search results as Excel",
    )
    parser.add_argument(
        "--export-dir",
        default="/tmp/wanfang_exports",
        help="Directory to save exported files (default: /tmp/wanfang_exports)",
    )
    parser.add_argument(
        "--max-articles", "-m",
        type=int,
        default=0,
        help="Maximum articles to export (default: 0 = unlimited)",
    )

    args = parser.parse_args()

    user_data_dir = None if args.no_persist else args.user_data_dir

    print("=" * 60)
    print("Wanfang Medical Literature Search")
    print("=" * 60)
    print(f"URL: {args.url}")
    print(f"Query: {args.query[:50]}..." if len(args.query) > 50 else f"Query: {args.query}")
    print(f"Year Range: {args.start_year} - {args.end_year or '结束'}")
    print(f"Resource Type: {args.resource_type}")
    print(f"Headless: {args.headless}")
    if args.export:
        print(f"Export: Enabled")
        print(f"Export Dir: {args.export_dir}")
        if args.max_articles > 0:
            print(f"Max Articles: {args.max_articles}")
    print("=" * 60)

    success = asyncio.run(
        perform_search(
            url=args.url,
            query=args.query,
            start_year=args.start_year,
            end_year=args.end_year,
            resource_type=args.resource_type,
            headless=args.headless,
            stay_open=not args.no_stay,
            user_data_dir=user_data_dir,
            export_results=args.export,
            export_dir=args.export_dir,
            max_articles=args.max_articles,
        )
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
