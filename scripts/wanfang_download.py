#!/usr/bin/env python3
"""Wanfang Medical Paper Download Script.

This script automates downloading papers from Wanfang Medical database
based on a pending CSV file. For each paper:
1. Search by title (题名)
2. Click the first search result
3. Download the PDF
4. Move to data/papers directory
5. Update pending CSV and download history

Usage:
    # Default: reads from data/pending_download.csv
    python scripts/wanfang_download.py

    # With options:
    python scripts/wanfang_download.py --max-papers 10 --delay 3

    # Specify custom CSV:
    python scripts/wanfang_download.py --csv data/custom.csv
"""

import argparse
import asyncio
import csv
import os
import random
import re
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from browser_agent.core.browser_controller import BrowserController
from browser_agent.utils.log_utils import get_logger

logger = get_logger(__name__)

# URLs
SEARCH_URL = "https://med.wanfangdata.com.cn/Paper"
LOGIN_URL = "https://login.med.wanfangdata.com.cn/Account/LogOn"

# Directories
DEFAULT_USER_DATA_DIR = str(Path(__file__).parent.parent / ".browser_data")
DATA_DIR = Path(__file__).parent.parent / "data"
PAPERS_DIR = DATA_DIR / "papers"
PENDING_CSV = DATA_DIR / "pending_download.csv"
HISTORY_CSV = DATA_DIR / "downloaded_history.csv"
FAILED_CSV = DATA_DIR / "download_failed.csv"


def sanitize_filename(title: str, max_length: int = 100) -> str:
    """Sanitize a paper title for use as a filename.

    Args:
        title: Paper title
        max_length: Maximum filename length

    Returns:
        Sanitized filename (without extension)
    """
    # Remove invalid characters
    invalid_chars = r'[\\/:*?"<>|]'
    sanitized = re.sub(invalid_chars, '', title)

    # Replace multiple spaces with single space
    sanitized = re.sub(r'\s+', ' ', sanitized).strip()

    # Truncate if too long
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length].rsplit(' ', 1)[0]

    return sanitized


def read_pending_csv(csv_path: str) -> list:
    """Read pending papers from CSV file.

    Args:
        csv_path: Path to the pending CSV file

    Returns:
        List of paper dicts with keys: 序号, 作者, 篇名, 刊名, 年, 卷, 期, 页码, DOI
    """
    papers = []
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            papers.append(row)
    return papers


def write_csv(csv_path: str, papers: list, fieldnames: list):
    """Write papers to a CSV file.

    Args:
        csv_path: Path to write the CSV
        papers: List of paper dicts
        fieldnames: Column names
    """
    with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(papers)


def append_to_history(paper: dict, download_path: str):
    """Append a downloaded paper to the history CSV.

    Args:
        paper: Paper dict
        download_path: Path where the paper was saved
    """
    HISTORY_CSV.parent.mkdir(parents=True, exist_ok=True)

    # Add download info
    paper_copy = paper.copy()
    paper_copy['下载时间'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    paper_copy['文件路径'] = download_path

    fieldnames = list(paper.keys()) + ['下载时间', '文件路径']

    # Check if file exists to determine if we need header
    file_exists = HISTORY_CSV.exists()

    with open(HISTORY_CSV, 'a', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        if not file_exists:
            writer.writeheader()
        writer.writerow(paper_copy)


def append_to_failed(paper: dict, reason: str):
    """Append a failed paper to the failed CSV.

    Args:
        paper: Paper dict
        reason: Failure reason
    """
    FAILED_CSV.parent.mkdir(parents=True, exist_ok=True)

    paper_copy = paper.copy()
    paper_copy['失败时间'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    paper_copy['失败原因'] = reason

    fieldnames = list(paper.keys()) + ['失败时间', '失败原因']

    file_exists = FAILED_CSV.exists()

    with open(FAILED_CSV, 'a', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        if not file_exists:
            writer.writeheader()
        writer.writerow(paper_copy)


async def wait_for_manual_login(page, timeout: int = 300) -> bool:
    """Wait for user to manually login in the browser.

    Args:
        page: Playwright page object
        timeout: Maximum wait time in seconds (default 5 minutes)

    Returns:
        True if login detected, False if timeout
    """
    import sys
    import select

    print(f"   (Timeout: {timeout}s)")

    start_time = asyncio.get_event_loop().time()
    while True:
        # Check if user pressed Enter
        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            sys.stdin.readline()
            print("   Continuing...")
            return True

        # Check login status periodically
        try:
            is_logged_in = await check_login_status(page)
            if is_logged_in:
                print("   ✅ Login detected!")
                return True
        except Exception:
            pass

        # Check timeout
        elapsed = asyncio.get_event_loop().time() - start_time
        if elapsed > timeout:
            print("   ⏰ Timeout waiting for login")
            return False

        await asyncio.sleep(2)


async def check_login_status(page) -> bool:
    """Check if user is logged in.

    Args:
        page: Playwright page object

    Returns:
        True if logged in, False otherwise
    """
    try:
        # Check for login-related elements or user info
        result = await page.evaluate("""
            () => {
                // Look for logout button or user info as sign of being logged in
                const logoutBtn = document.querySelector('a[href*="LogOff"], .logout, .user-info, .username, .user-name');
                if (logoutBtn) return {loggedIn: true, reason: 'found_logout'};

                // Check page content for user-related text
                const bodyText = document.body.innerText;
                if (bodyText.includes('退出') || bodyText.includes('我的') || bodyText.includes('账户')) {
                    return {loggedIn: true, reason: 'found_user_text'};
                }

                // Check if login link is prominently visible (means not logged in)
                const loginLink = document.querySelector('a[href*="LogOn"], .login-btn, .btn-login');
                if (loginLink) {
                    // Check if the login link is visible
                    const rect = loginLink.getBoundingClientRect();
                    if (rect.width > 0 && rect.height > 0) {
                        return {loggedIn: false, reason: 'found_login_link'};
                    }
                }

                // Default to assuming logged in if no clear indicator
                return {loggedIn: true, reason: 'no_clear_indicator'};
            }
        """)
        logger.info(f"Login status check: {result}")
        return result.get('loggedIn', True)
    except Exception as e:
        logger.warning(f"Login check failed: {e}, assuming logged in")
        return True


async def perform_login(browser: BrowserController, username: str, password: str) -> bool:
    """Perform login to Wanfang.

    Args:
        browser: BrowserController instance
        username: Username
        password: Password

    Returns:
        True if login successful
    """
    page = browser.page

    print("   Navigating to login page...")
    await browser.navigate(LOGIN_URL)
    await asyncio.sleep(1)

    print("   Entering credentials...")
    # Enter username
    await page.evaluate("""
        () => {
            const input = document.querySelector('input[placeholder*="用户名"], input[placeholder*="手机号"], input[name="userName"]');
            if (input) {
                input.focus();
                input.value = '';
            }
        }
    """)
    await asyncio.sleep(0.3)
    await browser.type_text(username, delay=30)
    await asyncio.sleep(0.5)

    # Enter password
    await page.evaluate("""
        () => {
            const input = document.querySelector('input[type="password"]');
            if (input) {
                input.focus();
                input.value = '';
            }
        }
    """)
    await asyncio.sleep(0.3)
    await browser.type_text(password, delay=30)
    await asyncio.sleep(0.5)

    # Solve slider captcha
    print("   Solving captcha...")
    captcha_solved = await solve_slider_captcha(browser)
    print(f"   Captcha solved: {captcha_solved}")

    # Wait for captcha verification to complete
    await asyncio.sleep(2)

    # Click login button with multiple attempts
    print("   Clicking login button...")

    # First, try to find the button and get its info
    btn_info = await page.evaluate("""
        () => {
            // Look for login button with multiple selectors
            const selectors = [
                'button.login-btn',
                'button.btn-login',
                'input[type="submit"]',
                'button[type="submit"]',
                '.login-form button',
                'form button',
                'button',
            ];

            for (const selector of selectors) {
                const buttons = document.querySelectorAll(selector);
                for (const btn of buttons) {
                    const text = (btn.textContent || btn.value || '').trim();
                    if (text.includes('登录') || text.includes('登 录') || text === '登录') {
                        const rect = btn.getBoundingClientRect();
                        return {
                            found: true,
                            text: text,
                            x: rect.x + rect.width / 2,
                            y: rect.y + rect.height / 2,
                            width: rect.width,
                            height: rect.height,
                            disabled: btn.disabled,
                            selector: selector,
                        };
                    }
                }
            }
            return {found: false};
        }
    """)

    print(f"   Button info: {btn_info}")

    if btn_info.get('found'):
        # Wait if button is disabled
        if btn_info.get('disabled'):
            print("   Button is disabled, waiting...")
            await asyncio.sleep(2)

        # Click using mouse coordinates (more reliable)
        x = btn_info.get('x', 0)
        y = btn_info.get('y', 0)
        if x > 0 and y > 0:
            print(f"   Clicking at ({x}, {y})...")
            await page.mouse.click(x, y)
            await asyncio.sleep(0.5)

            # Also try JavaScript click as backup
            await page.evaluate("""
                () => {
                    const buttons = document.querySelectorAll('button, input[type="submit"]');
                    for (const btn of buttons) {
                        const text = (btn.textContent || btn.value || '').trim();
                        if (text.includes('登录') || text.includes('登 录') || text === '登录') {
                            btn.click();
                            return true;
                        }
                    }
                    return false;
                }
            """)
    else:
        print("   Warning: Login button not found!")
        # Try clicking by text using Playwright locator
        try:
            login_btn = page.locator('button:has-text("登录"), input[value*="登录"]').first
            if await login_btn.count() > 0:
                print("   Found button via Playwright locator, clicking...")
                await login_btn.click()
        except Exception as e:
            print(f"   Failed to click via locator: {e}")

    # Wait for page navigation after login
    print("   Waiting for login redirect...")

    # Try multiple times to detect successful login
    for attempt in range(10):  # Try for up to 30 seconds
        await asyncio.sleep(3)
        current_url = page.url

        # Check if we've left the login page
        if "LogOn" not in current_url and "login" not in current_url.lower():
            print(f"   ✅ Login successful! Redirected to: {current_url[:50]}...")
            # Extra wait to let page fully load
            await asyncio.sleep(2)
            return True

        # Check if still on login page - maybe need to re-solve captcha or re-click
        if attempt == 3:
            print("   Still on login page, checking for errors...")
            error_text = await page.evaluate("""
                () => {
                    const errors = document.querySelectorAll('.error, .alert, [class*="error"], [class*="alert"]');
                    for (const err of errors) {
                        if (err.textContent && err.textContent.trim()) {
                            return err.textContent.trim();
                        }
                    }
                    return null;
                }
            """)
            if error_text:
                print(f"   Found error: {error_text}")

        print(f"   Attempt {attempt + 1}/10: Still on login page...")

    print("   Login may have failed after waiting")
    return False


async def solve_slider_captcha(browser: BrowserController) -> bool:
    """Solve slider captcha.

    Args:
        browser: BrowserController instance

    Returns:
        True if solved
    """
    page = browser.page

    try:
        slider_selectors = [
            '#nc_1_n1z',
            '.nc_iconfont.btn_slide',
            '.btn_slide',
            '.slide-btn',
        ]

        slider_elem = None
        for selector in slider_selectors:
            try:
                elem = page.locator(selector).first
                if await elem.count() > 0:
                    box = await elem.bounding_box()
                    if box and box['width'] > 10:
                        slider_elem = elem
                        break
            except Exception:
                continue

        if slider_elem:
            box = await slider_elem.bounding_box()
            if box:
                start_x = box['x'] + box['width'] / 2
                start_y = box['y'] + box['height'] / 2

                # Find track for end position
                end_x = start_x + 280
                track_selectors = ['.nc-lang-cnt', '#nc_1__scale_text', '.scale_text']
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

                # Perform drag
                await page.mouse.move(start_x, start_y)
                await asyncio.sleep(random.uniform(0.3, 0.5))
                await page.mouse.down()
                await asyncio.sleep(0.1)

                distance = end_x - start_x
                steps = random.randint(30, 50)

                for i in range(1, steps + 1):
                    t = i / steps
                    eased = 1 - pow(1 - t, 3)
                    current_x = start_x + distance * eased
                    wobble_y = start_y + random.uniform(-1.5, 1.5)
                    await page.mouse.move(current_x, wobble_y)
                    await asyncio.sleep(random.uniform(0.005, 0.02))

                await page.mouse.move(end_x, start_y)
                await asyncio.sleep(random.uniform(0.3, 0.5))
                await page.mouse.up()

                await asyncio.sleep(1)
                return True

        return False

    except Exception as e:
        logger.warning(f"Failed to solve captcha: {e}")
        return False


async def search_paper_by_title(browser: BrowserController, title: str) -> bool:
    """Search for a paper by its title using URL.

    Args:
        browser: BrowserController instance
        title: Paper title

    Returns:
        True if search results found
    """
    from urllib.parse import quote

    page = browser.page

    # Build search URL directly
    search_query = f"题名={title}"
    search_url = f"https://med.wanfangdata.com.cn/Paper/Search?q={quote(search_query)}"

    print(f"      Navigating to search URL...")
    await browser.navigate(search_url)
    await asyncio.sleep(3)

    # Handle captcha if present
    if await check_for_captcha(page):
        print("      Captcha detected, solving...")
        await solve_slider_captcha(browser)
        await asyncio.sleep(2)

    # Check if results were found
    results_count = await page.evaluate("""
        () => {
            // Check for "下载全文" buttons by text content
            const links = document.querySelectorAll('a');
            let downloadCount = 0;
            for (const link of links) {
                const text = link.textContent || '';
                if (text.includes('下载全文') || text.includes('全文下载')) {
                    downloadCount++;
                }
            }
            if (downloadCount > 0) return downloadCount;

            // Check for result items
            const results = document.querySelectorAll('.paper-item, .result-item, [class*="paper"], [class*="result"]');
            if (results.length > 0) return results.length;

            // Check for "no results" message
            const noResults = document.body.innerText;
            if (noResults.includes('没有找到') || noResults.includes('无结果') || noResults.includes('0条')) {
                return 0;
            }

            // Check for article links
            const links2 = document.querySelectorAll('a[href*="Paper/Detail"], a[href*="paper/"]');
            return links2.length;
        }
    """)

    print(f"      Found {results_count} results")
    return results_count > 0


async def check_for_captcha(page) -> bool:
    """Check if captcha is present."""
    try:
        title = await page.title()
        if "人机校验" in title or "安全验证" in title:
            return True

        captcha_selectors = [
            'text=安全验证',
            'text=请通过校验',
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


async def check_for_login_popup(page) -> bool:
    """Check if a login popup or redirect to login page occurred."""
    try:
        # Check URL for login redirect
        current_url = page.url
        if "LogOn" in current_url or "login" in current_url.lower():
            return True

        # Check for login modal/popup
        result = await page.evaluate("""
            () => {
                // Check for login modal
                const modal = document.querySelector('.login-modal, .modal-login, [class*="login-dialog"]');
                if (modal && modal.offsetParent !== null) return true;

                // Check for login form in popup
                const loginForm = document.querySelector('form[action*="login"], form[action*="LogOn"]');
                if (loginForm && loginForm.offsetParent !== null) return true;

                // Check for iframe with login
                const iframe = document.querySelector('iframe[src*="login"], iframe[src*="LogOn"]');
                if (iframe) return true;

                return false;
            }
        """)
        return result
    except Exception:
        return False


async def handle_login_popup(browser: BrowserController, username: str, password: str) -> bool:
    """Handle login popup by performing login."""
    page = browser.page

    try:
        # If redirected to login page, perform full login
        if "LogOn" in page.url or "login" in page.url.lower():
            return await perform_login(browser, username, password)

        # Try to find and fill login form in popup
        await page.evaluate("""
            () => {
                const userInput = document.querySelector('input[placeholder*="用户名"], input[placeholder*="手机号"], input[name="userName"]');
                if (userInput) userInput.focus();
            }
        """)
        await asyncio.sleep(0.3)
        await browser.type_text(username, delay=30)

        await page.evaluate("""
            () => {
                const passInput = document.querySelector('input[type="password"]');
                if (passInput) passInput.focus();
            }
        """)
        await asyncio.sleep(0.3)
        await browser.type_text(password, delay=30)

        # Try to solve captcha if present
        await solve_slider_captcha(browser)
        await asyncio.sleep(0.5)

        # Click login button
        await page.evaluate("""
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
        await asyncio.sleep(2)
        return True

    except Exception as e:
        logger.warning(f"Handle login popup failed: {e}")
        return False


async def download_paper_from_search(
    browser: BrowserController,
    paper_title: str,
    username: str = "",
    password: str = "",
) -> str:
    """Download the paper PDF directly from search results page.

    Args:
        browser: BrowserController instance
        paper_title: Paper title for naming the file
        username: Wanfang username for re-login if needed
        password: Wanfang password for re-login if needed

    Returns:
        Path to downloaded file, or empty string if failed
    """
    page = browser.page

    # Ensure papers directory exists
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)

    print("      Looking for '下载全文' button...")

    # Prepare filename
    safe_title = sanitize_filename(paper_title)
    filename = f"{safe_title}.pdf"

    # Try to find "下载全文" button in search results (first result)
    try:
        # Look for the first "下载全文" link (Playwright selector syntax)
        download_btn = page.locator('a:has-text("下载全文")').first
        if await download_btn.count() > 0:
            print("      Found '下载全文' button, clicking...")

            try:
                async def click_download():
                    await download_btn.click()

                file_path = await browser.handle_download(
                    click_download,
                    save_as=filename
                )

                # Move to papers directory
                final_path = PAPERS_DIR / filename
                if Path(file_path).exists():
                    import shutil
                    shutil.move(file_path, final_path)
                    print(f"      Downloaded: {final_path}")
                    return str(final_path)

            except Exception as e:
                logger.warning(f"Download via button failed: {e}")
                # Check if login popup appeared
                await asyncio.sleep(1)
                if await check_for_login_popup(page):
                    print("      Login popup detected, re-authenticating...")
                    if username and password:
                        await handle_login_popup(browser, username, password)
                        # Retry download
                        return await download_paper_from_search(browser, paper_title, "", "")

    except Exception as e:
        logger.debug(f"Could not find download button: {e}")

    # Fallback: try JavaScript to find and click download
    print("      Trying JavaScript fallback...")
    result = await page.evaluate("""
        () => {
            // Find first "下载全文" link
            const links = document.querySelectorAll('a');
            for (const link of links) {
                const text = link.textContent || '';
                if (text.includes('下载全文') || text.includes('全文下载')) {
                    return {found: true, href: link.href};
                }
            }
            return {found: false};
        }
    """)

    if result.get('found') and result.get('href'):
        print(f"      Found download link: {result['href'][:50]}...")
        try:
            download_link = page.locator(f'a[href="{result["href"]}"]').first
            if await download_link.count() > 0:
                async def click_download():
                    await download_link.click()

                file_path = await browser.handle_download(
                    click_download,
                    save_as=filename
                )

                final_path = PAPERS_DIR / filename
                if Path(file_path).exists():
                    import shutil
                    shutil.move(file_path, final_path)
                    print(f"      Downloaded: {final_path}")
                    return str(final_path)

        except Exception as e:
            logger.warning(f"Fallback download failed: {e}")

    return ""


async def download_papers(
    csv_path: str,
    username: str = "",
    password: str = "",
    max_papers: int = 0,
    delay: float = 2.0,
    headless: bool = False,
    stay_open: bool = False,
    user_data_dir: str = DEFAULT_USER_DATA_DIR,
    skip_login_check: bool = False,
) -> dict:
    """Main function to download papers from pending CSV.

    Args:
        csv_path: Path to pending CSV file
        username: Wanfang username (optional, for auto-login)
        password: Wanfang password (optional, for auto-login)
        max_papers: Maximum papers to download (0 = unlimited)
        delay: Delay between downloads in seconds
        headless: Run browser in headless mode
        stay_open: Keep browser open after completion
        user_data_dir: Browser data directory

    Returns:
        Dict with success_count, failed_count, skipped_count
    """
    # Read pending papers
    papers = read_pending_csv(csv_path)
    total = len(papers)

    if max_papers > 0:
        papers = papers[:max_papers]

    print(f"\n📚 Starting download of {len(papers)} papers (from {total} total)")
    print("=" * 60)

    # Track results
    success_count = 0
    failed_count = 0
    remaining_papers = []

    # Get fieldnames from first paper
    fieldnames = list(papers[0].keys()) if papers else []

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

        # Navigate to site first
        await browser.navigate(SEARCH_URL)
        await asyncio.sleep(2)

        # Check login and login if needed
        if not skip_login_check:
            is_logged_in = await check_login_status(page)
            if not is_logged_in and username and password:
                print("\n🔐 Not logged in, performing login...")
                login_success = await perform_login(browser, username, password)
                if not login_success:
                    print("❌ Auto-login failed.")
                    print("   Please login manually in the browser window...")
                    print("   Waiting for login (press Enter when done, or Ctrl+C to cancel)...")
                    await wait_for_manual_login(page)
            elif not is_logged_in:
                print("\n⚠️ Not logged in.")
                print("   Please login manually in the browser window...")
                print("   Navigating to login page...")
                await browser.navigate(LOGIN_URL)
                print("   Waiting for login (press Enter when done, or Ctrl+C to cancel)...")
                await wait_for_manual_login(page)
        else:
            print("\n⏭️ Skipping login check...")

        # Process each paper
        for idx, paper in enumerate(papers, 1):
            title = paper.get('篇名', '').strip()

            if not title:
                print(f"\n[{idx}/{len(papers)}] ⚠️ Empty title, skipping")
                remaining_papers.append(paper)
                continue

            print(f"\n[{idx}/{len(papers)}] 📄 {title[:50]}...")

            try:
                # Search for the paper
                found = await search_paper_by_title(browser, title)

                if not found:
                    print("      ❌ No results found")
                    append_to_failed(paper, "Paper not found in search")
                    failed_count += 1
                    remaining_papers.append(paper)
                    continue

                # Download directly from search results page
                download_path = await download_paper_from_search(browser, title, username, password)

                if download_path:
                    print(f"      ✅ Success: {Path(download_path).name}")
                    append_to_history(paper, download_path)
                    success_count += 1
                else:
                    print("      ❌ Download failed")
                    append_to_failed(paper, "Download failed")
                    failed_count += 1
                    remaining_papers.append(paper)

            except Exception as e:
                logger.error(f"Error processing paper: {e}")
                append_to_failed(paper, str(e))
                failed_count += 1
                remaining_papers.append(paper)

            # Delay between papers
            if idx < len(papers):
                await asyncio.sleep(delay + random.uniform(0, 1))

        # Update pending CSV with remaining papers
        if remaining_papers:
            write_csv(csv_path, remaining_papers, fieldnames)
            print(f"\n📝 Updated pending CSV with {len(remaining_papers)} remaining papers")
        else:
            # All done, could delete or keep empty
            write_csv(csv_path, [], fieldnames)
            print(f"\n✨ All papers processed! Pending CSV is now empty.")

        # Summary
        print("\n" + "=" * 60)
        print("📊 Summary:")
        print(f"   ✅ Success: {success_count}")
        print(f"   ❌ Failed: {failed_count}")
        print(f"   📁 Papers saved to: {PAPERS_DIR}")
        print(f"   📋 History: {HISTORY_CSV}")
        if failed_count > 0:
            print(f"   ⚠️ Failed log: {FAILED_CSV}")
        print("=" * 60)

        if stay_open:
            print("\nBrowser will stay open. Press Ctrl+C to close.")
            while True:
                await asyncio.sleep(1)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if not stay_open:
            await browser.close()

    return {
        "success": success_count,
        "failed": failed_count,
        "skipped": len(papers) - success_count - failed_count,
    }


def main():
    parser = argparse.ArgumentParser(description="Wanfang Paper Download Script")
    parser.add_argument(
        "--csv", "-c",
        default=str(PENDING_CSV),
        help=f"Path to pending download CSV file (default: {PENDING_CSV})",
    )
    parser.add_argument(
        "--username", "-u",
        default=os.environ.get("WANFANG_USERNAME", ""),
        help="Wanfang username (or set WANFANG_USERNAME env var)",
    )
    parser.add_argument(
        "--password", "-p",
        default=os.environ.get("WANFANG_PASSWORD", ""),
        help="Wanfang password (or set WANFANG_PASSWORD env var)",
    )
    parser.add_argument(
        "--max-papers", "-m",
        type=int,
        default=0,
        help="Maximum papers to download (0 = unlimited)",
    )
    parser.add_argument(
        "--delay", "-d",
        type=float,
        default=2.0,
        help="Delay between downloads in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run browser in headless mode",
    )
    parser.add_argument(
        "--stay-open",
        action="store_true",
        help="Keep browser open after completion",
    )
    parser.add_argument(
        "--no-persist",
        action="store_true",
        help="Don't persist browser session",
    )
    parser.add_argument(
        "--user-data-dir",
        default=DEFAULT_USER_DATA_DIR,
        help=f"Browser data directory (default: {DEFAULT_USER_DATA_DIR})",
    )
    parser.add_argument(
        "--skip-login-check",
        action="store_true",
        help="Skip login status check and proceed directly to download",
    )

    args = parser.parse_args()

    # Validate CSV path
    if not Path(args.csv).exists():
        print(f"Error: CSV file not found: {args.csv}")
        sys.exit(1)

    user_data_dir = None if args.no_persist else args.user_data_dir

    print("=" * 60)
    print("📚 Wanfang Paper Download Script")
    print("=" * 60)
    print(f"CSV: {args.csv}")
    print(f"Max papers: {args.max_papers if args.max_papers > 0 else 'unlimited'}")
    print(f"Delay: {args.delay}s")
    print(f"Headless: {args.headless}")
    print(f"Output: {PAPERS_DIR}")
    print("=" * 60)

    result = asyncio.run(
        download_papers(
            csv_path=args.csv,
            username=args.username,
            password=args.password,
            max_papers=args.max_papers,
            delay=args.delay,
            headless=args.headless,
            stay_open=args.stay_open,
            user_data_dir=user_data_dir,
            skip_login_check=args.skip_login_check,
        )
    )

    sys.exit(0 if result["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
