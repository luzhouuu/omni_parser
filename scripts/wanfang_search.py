#!/usr/bin/env python3
"""Wanfang Medical Literature Search Automation Script.

This script automates the process of searching for medical literature on Wanfang,
including setting search filters, entering search queries, and modifying date ranges.

Usage:
    python scripts/wanfang_search.py --query '((("得理多") OR "卡马西平") OR "Tegretol") OR "Carbamazepine"'

    # With custom date range:
    python scripts/wanfang_search.py --query '((("得理多") OR "卡马西平") OR "Tegretol") OR "Carbamazepine"' --start-date 2025/10/15 --end-date 2025/11/26
"""

import argparse
import asyncio
import os
import random
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from browser_agent.core.browser_controller import BrowserController
from browser_agent.utils.log_utils import get_logger

logger = get_logger(__name__)

# Wanfang medical search page (default)
DEFAULT_SEARCH_URL = "https://med.wanfangdata.com.cn/Paper"

# Default browser data directory for session persistence
DEFAULT_USER_DATA_DIR = str(Path(__file__).parent.parent / ".browser_data")

# Default search query
DEFAULT_QUERY = '((("得理多") OR "卡马西平") OR "Tegretol") OR "Carbamazepine"'


async def perform_search(
    url: str = DEFAULT_SEARCH_URL,
    query: str = DEFAULT_QUERY,
    year_filter: str = "2019",
    start_date: str = None,
    end_date: str = None,
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
        year_filter: Year filter for initial search (e.g., "2019")
        start_date: Start date for post-search filter (format: YYYY/MM/DD)
        end_date: End date for post-search filter (format: YYYY/MM/DD)
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

        # Step 1: Navigate to search page
        print(f"[1/6] Navigating to: {url}")
        await browser.navigate(url)
        await asyncio.sleep(2)

        # Step 2: Select "中外期刊" (Chinese and Foreign Journals)
        print(f"[2/6] Selecting '中外期刊' category...")
        await select_journal_category(page)
        await asyncio.sleep(0.5)

        # Step 3: Enter search query
        print(f"[3/6] Entering search query...")
        await enter_search_query(page, query)
        await asyncio.sleep(0.5)

        # Step 4: Set year filter
        print(f"[4/6] Setting year filter to {year_filter}...")
        await set_year_filter(page, year_filter)
        await asyncio.sleep(0.5)

        # Step 5: Click search button
        print(f"[5/6] Clicking search button...")
        await click_search_button(page)
        await asyncio.sleep(3)  # Wait for results to load

        # Check for captcha and handle if present
        if await check_for_captcha(page):
            print(f"[5.5/6] Captcha detected, solving...")
            await solve_slider_captcha(browser)
            await asyncio.sleep(2)

        # Step 6: Modify date range in results page (if specified)
        if start_date and end_date:
            print(f"[6/7] Setting date range: {start_date} - {end_date}..." if export_results else f"[6/6] Setting date range: {start_date} - {end_date}...")
            await set_result_date_range(page, start_date, end_date)
            await asyncio.sleep(2)
        else:
            print(f"[6/7] Skipping date range modification (not specified)" if export_results else f"[6/6] Skipping date range modification (not specified)")

        # Step 7: Export results (if enabled)
        if export_results:
            print(f"[7/7] Exporting search results to Excel...")
            exported_count = await export_search_results(browser, export_dir, max_articles)
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


async def select_journal_category(page):
    """Select '中外期刊' category."""
    try:
        # Try to find and click the journal category tab/button
        selectors = [
            'text=中外期刊',
            '[data-type="中外期刊"]',
            '.tab:has-text("中外期刊")',
            'a:has-text("中外期刊")',
            'span:has-text("中外期刊")',
            'div:has-text("中外期刊")',
        ]

        for selector in selectors:
            try:
                elem = page.locator(selector).first
                if await elem.count() > 0:
                    await elem.click()
                    await asyncio.sleep(0.5)
                    logger.info(f"Clicked journal category with selector: {selector}")
                    return True
            except Exception:
                continue

        # Fallback: Try JavaScript
        result = await page.evaluate("""
            () => {
                const elements = document.querySelectorAll('*');
                for (const el of elements) {
                    if (el.textContent && el.textContent.trim() === '中外期刊') {
                        el.click();
                        return true;
                    }
                }
                // Try partial match
                for (const el of elements) {
                    if (el.textContent && el.textContent.includes('中外期刊') &&
                        (el.tagName === 'A' || el.tagName === 'SPAN' || el.tagName === 'DIV' || el.tagName === 'BUTTON')) {
                        el.click();
                        return true;
                    }
                }
                return false;
            }
        """)

        if result:
            await asyncio.sleep(0.5)
            logger.info("Clicked journal category via JavaScript")
            return True

        logger.warning("Could not find '中外期刊' category, continuing anyway")
        return False

    except Exception as e:
        logger.warning(f"Failed to select journal category: {e}")
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
                    await asyncio.sleep(0.5)
                    await elem.fill('')  # Clear existing content
                    await asyncio.sleep(0.5)
                    await elem.fill(query)
                    await asyncio.sleep(0.5)
                    logger.info(f"Entered query via selector: {selector}")
                    return True
            except Exception:
                continue

        # Fallback: JavaScript
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
            await asyncio.sleep(0.5)
            logger.info("Entered query via JavaScript")
            return True

        logger.error("Could not find search input field")
        return False

    except Exception as e:
        logger.error(f"Failed to enter search query: {e}")
        return False


async def set_year_filter(page, year: str):
    """Set the year filter for the search."""
    try:
        # Try to find year input/select
        selectors = [
            f'input[placeholder*="年"]',
            'input.year-input',
            '#yearFilter',
            'select.year-select',
            '[data-field="year"] input',
        ]

        for selector in selectors:
            try:
                elem = page.locator(selector).first
                if await elem.count() > 0:
                    await elem.click()
                    await asyncio.sleep(0.5)
                    await elem.fill('')
                    await asyncio.sleep(0.5)
                    await elem.fill(year)
                    await asyncio.sleep(0.5)
                    logger.info(f"Set year filter via selector: {selector}")
                    return True
            except Exception:
                continue

        # Try finding by label text "出版年份" or "发布年份"
        result = await page.evaluate("""
            (year) => {
                // Look for labels containing year-related text
                const labels = document.querySelectorAll('label, span, div');
                for (const label of labels) {
                    if (label.textContent && (label.textContent.includes('出版年份') ||
                        label.textContent.includes('发布年份') || label.textContent.includes('年份'))) {
                        // Find nearby input
                        const parent = label.closest('div, li, tr');
                        if (parent) {
                            const input = parent.querySelector('input');
                            if (input) {
                                input.focus();
                                input.value = year;
                                input.dispatchEvent(new Event('input', { bubbles: true }));
                                return true;
                            }
                        }
                    }
                }
                return false;
            }
        """, year)

        if result:
            await asyncio.sleep(0.5)
            logger.info("Set year filter via JavaScript")
            return True

        logger.warning(f"Could not find year filter, continuing without setting year")
        return False

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


async def set_result_date_range(page, start_date: str, end_date: str):
    """Set the date range filter on the results page.

    Args:
        page: Playwright page object
        start_date: Start date in format YYYY/MM/DD (will extract year)
        end_date: End date in format YYYY/MM/DD (will extract year)
    """
    try:
        # Extract years from dates
        start_year = start_date.split('/')[0] if '/' in start_date else start_date[:4]
        end_year = end_date.split('/')[0] if '/' in end_date else end_date[:4]

        # First, scroll the left sidebar to find "年份" filter
        # Try to find the sidebar/filter container
        sidebar_selectors = [
            '.search-filter',
            '.filter-wrap',
            '.left-side',
            '[class*="filter"]',
            '[class*="sidebar"]',
        ]

        for selector in sidebar_selectors:
            try:
                elem = page.locator(selector).first
                if await elem.count() > 0:
                    await elem.evaluate('el => el.scrollTop = 300')
                    await asyncio.sleep(0.3)
                    break
            except Exception:
                continue

        # Look for "年份" section and its input fields
        year_section_found = False
        filter_selectors = [
            'text=年份',
            'span:has-text("年份")',
            'div:has-text("年份")',
            'text=发布时间',
            '[data-filter="publishDate"]',
            '.filter-item:has-text("发布时间")',
            'span:has-text("发布时间")',
        ]

        # Click on "年份" section to ensure it's expanded
        for selector in filter_selectors:
            try:
                elem = page.locator(selector).first
                if await elem.count() > 0:
                    # Scroll element into view first
                    await elem.scroll_into_view_if_needed()
                    await asyncio.sleep(0.3)
                    logger.info(f"Found year filter section: {selector}")
                    break
            except Exception:
                continue

        # Try to set year range using JavaScript - Wanfang uses slider with two input boxes
        # The inputs are next to "年份" text and show values like "1998" and "2025"
        result = await page.evaluate("""
            (years) => {
                const startYear = years.start;
                const endYear = years.end;

                // Method 1: Find inputs near "年份" text
                const yearLabels = Array.from(document.querySelectorAll('*')).filter(
                    el => el.textContent && el.textContent.trim() === '年份'
                );

                for (const label of yearLabels) {
                    // Look for parent container with inputs
                    let parent = label.parentElement;
                    for (let i = 0; i < 5 && parent; i++) {
                        const inputs = parent.querySelectorAll('input[type="text"], input:not([type])');
                        if (inputs.length >= 2) {
                            // Found the year range inputs
                            // First input is start year, second is end year
                            inputs[0].value = startYear;
                            inputs[0].dispatchEvent(new Event('input', { bubbles: true }));
                            inputs[0].dispatchEvent(new Event('change', { bubbles: true }));

                            inputs[1].value = endYear;
                            inputs[1].dispatchEvent(new Event('input', { bubbles: true }));
                            inputs[1].dispatchEvent(new Event('change', { bubbles: true }));

                            console.log('Set year range via parent search:', startYear, '-', endYear);
                            return {success: true, method: 'parent_search'};
                        }
                        parent = parent.parentElement;
                    }
                }

                // Method 2: Find by input value pattern (4-digit year)
                const allInputs = Array.from(document.querySelectorAll('input'));
                const yearInputs = allInputs.filter(input => {
                    const val = input.value;
                    return val && /^\\d{4}$/.test(val) && parseInt(val) >= 1900 && parseInt(val) <= 2100;
                });

                if (yearInputs.length >= 2) {
                    // Sort by position (left to right)
                    yearInputs.sort((a, b) => {
                        const rectA = a.getBoundingClientRect();
                        const rectB = b.getBoundingClientRect();
                        return rectA.left - rectB.left;
                    });

                    yearInputs[0].value = startYear;
                    yearInputs[0].dispatchEvent(new Event('input', { bubbles: true }));
                    yearInputs[0].dispatchEvent(new Event('change', { bubbles: true }));

                    yearInputs[1].value = endYear;
                    yearInputs[1].dispatchEvent(new Event('input', { bubbles: true }));
                    yearInputs[1].dispatchEvent(new Event('change', { bubbles: true }));

                    console.log('Set year range via value pattern:', startYear, '-', endYear);
                    return {success: true, method: 'value_pattern'};
                }

                // Method 3: Find slider component and its inputs
                const sliders = document.querySelectorAll('[class*="slider"], [class*="range"]');
                for (const slider of sliders) {
                    const inputs = slider.querySelectorAll('input');
                    if (inputs.length >= 2) {
                        inputs[0].value = startYear;
                        inputs[0].dispatchEvent(new Event('input', { bubbles: true }));
                        inputs[0].dispatchEvent(new Event('change', { bubbles: true }));

                        inputs[1].value = endYear;
                        inputs[1].dispatchEvent(new Event('input', { bubbles: true }));
                        inputs[1].dispatchEvent(new Event('change', { bubbles: true }));

                        console.log('Set year range via slider:', startYear, '-', endYear);
                        return {success: true, method: 'slider'};
                    }
                }

                return {success: false, method: 'none'};
            }
        """, {"start": start_year, "end": end_year})

        if result and result.get('success'):
            await asyncio.sleep(0.5)
            logger.info(f"Set year range {start_year}-{end_year} via JavaScript ({result.get('method')})")
        else:
            logger.warning(f"Could not set year range via JavaScript")

        # Try to click apply/confirm button if exists
        apply_selectors = [
            'button:has-text("确定")',
            'button:has-text("应用")',
            'button:has-text("筛选")',
            '.date-confirm',
        ]

        for selector in apply_selectors:
            try:
                elem = page.locator(selector).first
                if await elem.count() > 0:
                    await elem.click()
                    await asyncio.sleep(1)  # Wait for filter to apply
                    logger.info(f"Clicked apply button: {selector}")
                    return True
            except Exception:
                continue

        return result and result.get('success', False)

    except Exception as e:
        logger.warning(f"Failed to set date range: {e}")
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

async def filter_chinese_journals(page) -> bool:
    """Filter results to show only Chinese journals (中文期刊)."""
    try:
        selectors = [
            'text=中文期刊',
            '.filter-item:has-text("中文期刊")',
            'label:has-text("中文期刊")',
            'a:has-text("中文期刊")',
        ]

        for selector in selectors:
            try:
                elem = page.locator(selector).first
                if await elem.count() > 0:
                    await elem.click()
                    await asyncio.sleep(0.5)
                    logger.info(f"Filtered by Chinese journals: {selector}")
                    return True
            except Exception:
                continue

        # Fallback: JavaScript
        result = await page.evaluate("""
            () => {
                const elements = document.querySelectorAll('*');
                for (const el of elements) {
                    if (el.textContent && el.textContent.trim() === '中文期刊') {
                        el.click();
                        return true;
                    }
                }
                return false;
            }
        """)

        if result:
            await asyncio.sleep(0.5)
            logger.info("Filtered by Chinese journals via JavaScript")
            return True

        logger.warning("Could not find Chinese journals filter")
        return False

    except Exception as e:
        logger.warning(f"Failed to filter Chinese journals: {e}")
        return False


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
                    await asyncio.sleep(0.5)
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
            await asyncio.sleep(0.5)
            logger.info("Selected all articles via JavaScript")
            return True

        logger.warning("Could not find select all button")
        return False

    except Exception as e:
        logger.warning(f"Failed to select all articles: {e}")
        return False


async def click_reference_export(page) -> bool:
    """Click '批量导出' then select '参考文献' from dropdown menu."""
    try:
        # Step 1: Click or hover on '批量导出' to show dropdown
        batch_export_selectors = [
            'text=批量导出',
            'button:has-text("批量导出")',
            'a:has-text("批量导出")',
            'span:has-text("批量导出")',
        ]

        clicked_batch = False
        for selector in batch_export_selectors:
            try:
                elem = page.locator(selector).first
                if await elem.count() > 0:
                    await elem.hover()  # Hover to show dropdown
                    await asyncio.sleep(0.5)
                    await elem.click()
                    await asyncio.sleep(0.5)
                    logger.info(f"Clicked batch export button: {selector}")
                    clicked_batch = True
                    break
            except Exception:
                continue

        if not clicked_batch:
            # Try JavaScript
            result = await page.evaluate("""
                () => {
                    const elements = document.querySelectorAll('*');
                    for (const el of elements) {
                        if (el.textContent && el.textContent.trim() === '批量导出') {
                            el.click();
                            return true;
                        }
                    }
                    return false;
                }
            """)
            if result:
                await asyncio.sleep(0.5)
                clicked_batch = True
                logger.info("Clicked batch export via JavaScript")

        if not clicked_batch:
            logger.warning("Could not find batch export button")
            return False

        # Step 2: Click '参考文献' from dropdown menu
        await asyncio.sleep(0.5)  # Wait for dropdown to appear

        ref_selectors = [
            'text=参考文献',
            'a:has-text("参考文献")',
            'li:has-text("参考文献")',
            'div:has-text("参考文献")',
        ]

        for selector in ref_selectors:
            try:
                # Get all matching elements and find the one in dropdown (not the main button)
                elems = page.locator(selector)
                count = await elems.count()
                for i in range(count):
                    elem = elems.nth(i)
                    text = await elem.text_content()
                    # Make sure it's exactly "参考文献" not "批量导出" containing it
                    if text and text.strip() == '参考文献':
                        await elem.click()
                        await asyncio.sleep(1)
                        logger.info(f"Clicked reference export option: {selector}")
                        return True
            except Exception:
                continue

        # Fallback: JavaScript
        result = await page.evaluate("""
            () => {
                const elements = document.querySelectorAll('a, li, div, span');
                for (const el of elements) {
                    if (el.textContent && el.textContent.trim() === '参考文献') {
                        el.click();
                        return true;
                    }
                }
                return false;
            }
        """)

        if result:
            await asyncio.sleep(1)
            logger.info("Clicked reference option via JavaScript")
            return True

        logger.warning("Could not find reference option in dropdown")
        return False

    except Exception as e:
        logger.warning(f"Failed to click reference export: {e}")
        return False


async def export_to_excel(browser: BrowserController, export_dir: str, batch_num: int) -> str:
    """Click '导出到Excel' and download the file."""
    import os

    page = browser.page

    # Ensure export directory exists
    os.makedirs(export_dir, exist_ok=True)

    try:
        # Wait a bit more for dialog to fully load
        await asyncio.sleep(1)

        # Find and click '导出到Excel' button - try multiple selectors
        selectors = [
            'text=导出到Excel',
            'text=导出Excel',
            'text=导出至Excel',
            'button:has-text("Excel")',
            'a:has-text("Excel")',
            'span:has-text("导出到Excel")',
            'div:has-text("导出到Excel")',
        ]

        for selector in selectors:
            try:
                elem = page.locator(selector).first
                if await elem.count() > 0:
                    # Handle download
                    async def click_export():
                        await elem.click()

                    file_path = await browser.handle_download(
                        click_export,
                        save_as=f"wanfang_export_batch_{batch_num}.xlsx"
                    )
                    logger.info(f"Exported to: {file_path}")
                    await asyncio.sleep(0.5)

                    # Close the export dialog after download
                    await close_export_dialog(page)

                    return file_path
            except Exception as e:
                logger.debug(f"Selector {selector} failed: {e}")
                continue

        # Fallback: Try JavaScript click with more patterns
        result = await page.evaluate("""
            () => {
                // Look for Excel export button in any element
                const elements = document.querySelectorAll('button, a, span, div, li');
                for (const el of elements) {
                    const text = el.textContent || '';
                    if (text.includes('导出到Excel') || text.includes('导出Excel') ||
                        text.includes('导出至Excel') || (text.includes('Excel') && text.includes('导出'))) {
                        el.click();
                        return true;
                    }
                }
                // Try finding by class or other attributes
                const excelBtn = document.querySelector('[class*="excel"], [class*="export"]');
                if (excelBtn && excelBtn.textContent.includes('Excel')) {
                    excelBtn.click();
                    return true;
                }
                return false;
            }
        """)

        if result:
            await asyncio.sleep(3)  # Wait for download
            logger.info("Clicked Excel export via JavaScript")
            await close_export_dialog(page)
            return f"{export_dir}/wanfang_export_batch_{batch_num}.xlsx"

        logger.warning("Could not find Excel export button")
        return ""

    except Exception as e:
        logger.warning(f"Failed to export to Excel: {e}")
        return ""


async def close_export_dialog(page) -> bool:
    """Close the export dialog/panel after exporting.

    IMPORTANT: This function should NOT navigate away from the current page
    as we need to preserve the current pagination state.
    """
    try:
        await asyncio.sleep(0.5)

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
                    await asyncio.sleep(0.5)
                    logger.info(f"Closed export dialog: {selector}")
                    return True
            except Exception:
                continue

        # Try pressing Escape to close any dialog
        await page.keyboard.press("Escape")
        await asyncio.sleep(0.5)
        logger.info("Pressed Escape to close dialog")

        # Try clicking outside the dialog to close it
        await page.mouse.click(100, 100)
        await asyncio.sleep(0.5)

        # NOTE: Removed page.go_back() as it resets pagination to page 1
        # The export dialog on Wanfang is typically a modal overlay
        # that doesn't change the underlying page URL

        return True

    except Exception as e:
        logger.warning(f"Failed to close export dialog: {e}")
        return False


async def clear_selection(page) -> bool:
    """Clear current article selection."""
    try:
        selectors = [
            'text=清除',
            'button:has-text("清除")',
            '.clear-btn',
            'a:has-text("清除")',
        ]

        for selector in selectors:
            try:
                elem = page.locator(selector).first
                if await elem.count() > 0:
                    await elem.click()
                    await asyncio.sleep(0.5)
                    logger.info(f"Cleared selection: {selector}")
                    return True
            except Exception:
                continue

        # Fallback: JavaScript
        result = await page.evaluate("""
            () => {
                const elements = document.querySelectorAll('button, a, span');
                for (const el of elements) {
                    if (el.textContent && el.textContent.trim() === '清除') {
                        el.click();
                        return true;
                    }
                }
                return false;
            }
        """)

        if result:
            await asyncio.sleep(0.5)
            logger.info("Cleared selection via JavaScript")
            return True

        logger.warning("Could not find clear button")
        return False

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
                        await asyncio.sleep(1)
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
            await asyncio.sleep(1)
            logger.info("Navigated to next page via JavaScript")
            return True

        logger.info("No more pages available")
        return False

    except Exception as e:
        logger.warning(f"Failed to navigate to next page: {e}")
        return False


async def export_search_results(browser: BrowserController, export_dir: str, max_articles: int = 0) -> int:
    """Export all search results by going through every page.

    Simple workflow:
    1. Filter by Chinese journals
    2. On each page: select all → export → clear
    3. Go to next page, repeat until last page

    Args:
        browser: BrowserController instance
        export_dir: Directory to save exported files
        max_articles: Maximum articles to export (0 = unlimited)

    Returns:
        Total number of articles exported
    """
    page = browser.page

    # Step 1: Filter by Chinese journals
    print("   Filtering by 中文期刊...")
    await filter_chinese_journals(page)
    await asyncio.sleep(1)

    # Step 2: Get total results after filtering
    total = await get_total_results(page)
    print(f"   Total Chinese journal results: {total}")

    if total == 0:
        logger.warning("No results to export")
        return 0

    # Apply max_articles limit if specified
    if max_articles > 0:
        total = min(total, max_articles)
        print(f"   Limited to {max_articles} articles")

    # Step 3: Select 50 per page
    print("   Setting page size to 50...")
    await select_page_size(page, "50")
    await asyncio.sleep(1)

    # Save the search results URL to return to after export
    search_results_url = page.url
    print(f"   Search results URL: {search_results_url}")

    exported = 0
    current_page = 1
    articles_per_page = 50

    # Calculate total pages
    total_pages = (total + articles_per_page - 1) // articles_per_page
    print(f"   Total pages to export: ~{total_pages}")

    # Keep going through pages until the last one
    while current_page <= total_pages:
        # Wait for page to be ready
        await asyncio.sleep(1)

        # Verify we're on the correct page by checking the current page number
        actual_page = await page.evaluate("""
            () => {
                // Find the active/current page number button
                const activeBtn = document.querySelector('button[style*="background"], .active, [class*="current"], [class*="active"]');
                if (activeBtn) {
                    const num = parseInt(activeBtn.textContent, 10);
                    if (!isNaN(num)) return num;
                }
                // Try finding by different method - highlighted pagination number
                const pageBtns = document.querySelectorAll('.pagination button, .page-list button, [class*="page"] button');
                for (const btn of pageBtns) {
                    const style = window.getComputedStyle(btn);
                    if (style.backgroundColor !== 'rgba(0, 0, 0, 0)' && style.backgroundColor !== 'transparent') {
                        const num = parseInt(btn.textContent, 10);
                        if (!isNaN(num)) return num;
                    }
                }
                return 1;
            }
        """)
        print(f"   [Page {current_page}/{total_pages}] Current page detected: {actual_page}")

        print(f"   [Page {current_page}/{total_pages}] Selecting all articles...")
        await select_all_articles(page)
        await asyncio.sleep(0.5)

        # Export this page
        print(f"   [Page {current_page}/{total_pages}] Opening export dialog...")
        await click_reference_export(page)
        await asyncio.sleep(1)

        print(f"   [Page {current_page}/{total_pages}] Exporting to Excel...")
        file_path = await export_to_excel(browser, export_dir, current_page)

        if file_path:
            # Estimate articles on this page
            remaining = total - exported
            articles_this_page = min(articles_per_page, remaining)
            exported += articles_this_page
            print(f"   [Page {current_page}/{total_pages}] Saved: {file_path} (~{articles_this_page} articles)")
        else:
            print(f"   [Page {current_page}/{total_pages}] Warning: Export failed, continuing to next page...")

        # Check if we're done
        if current_page >= total_pages:
            print(f"   Reached last page (page {current_page})")
            break

        # IMPORTANT: Do NOT navigate back to search_results_url as it resets to page 1
        # Instead, just close the export dialog and stay on the current page
        # The close_export_dialog function should handle this properly

        # Clear selection before going to next page
        print(f"   [Page {current_page}/{total_pages}] Clearing selection...")
        await clear_selection(page)
        await asyncio.sleep(0.5)

        # Go to next page FIRST, then loop back to export
        print(f"   [Page {current_page}/{total_pages}] Going to next page...")
        has_next = await go_to_next_page(page)

        if not has_next:
            print(f"   No more pages available (stopped at page {current_page})")
            break

        await asyncio.sleep(2)  # Wait longer for page content to load
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

        # Generate timestamp for pending file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pending_csv_path = DATA_DIR / f"pending_download_{timestamp}.csv"

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

        # Update history with all unique articles (append new ones)
        with open(HISTORY_CSV, 'a', encoding='utf-8-sig') as f:
            # Write header if file is new
            if not existing_titles and header_row:
                f.write(','.join(f'"{c}"' for c in header_row) + '\n')

            # Append new articles to history
            for idx, row in enumerate(new_rows, len(existing_titles) + 1):
                if row:
                    row[0] = str(idx)
                f.write(','.join(f'"{c}"' for c in row) + '\n')

        print(f"   Updated history: {HISTORY_CSV}")

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
        "--year", "-y",
        default="2019",
        help="Year filter for initial search (default: 2019)",
    )
    parser.add_argument(
        "--start-date", "-s",
        default=None,
        help="Start date for results filter (format: YYYY/MM/DD, e.g., 2025/10/15)",
    )
    parser.add_argument(
        "--end-date", "-e",
        default=None,
        help="End date for results filter (format: YYYY/MM/DD, e.g., 2025/11/26)",
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
    print(f"Year Filter: {args.year}")
    if args.start_date and args.end_date:
        print(f"Date Range: {args.start_date} - {args.end_date}")
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
            year_filter=args.year,
            start_date=args.start_date,
            end_date=args.end_date,
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
