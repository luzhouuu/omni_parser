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
    python scripts/wanfang_download.py --csv data/pending_download_20251225_125945.csv

    # With options:
    python scripts/wanfang_download.py --csv data/pending.csv --max-papers 10 --delay 3
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
                const logoutBtn = document.querySelector('a[href*="LogOff"], .logout, .user-info, .username');
                if (logoutBtn) return true;

                // Check if login link is visible (means not logged in)
                const loginLink = document.querySelector('a[href*="LogOn"], .login-btn');
                if (loginLink) return false;

                // Default to assuming logged in if no clear indicator
                return true;
            }
        """)
        return result
    except Exception:
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
    await solve_slider_captcha(browser)
    await asyncio.sleep(1)

    # Click login
    print("   Clicking login button...")
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

    # Check if login succeeded
    current_url = page.url
    if "LogOn" not in current_url and "login" not in current_url.lower():
        print("   Login successful!")
        return True
    else:
        print("   Login may have failed")
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
    """Search for a paper by its title.

    Args:
        browser: BrowserController instance
        title: Paper title

    Returns:
        True if search results found
    """
    page = browser.page

    # Navigate to search page
    await browser.navigate(SEARCH_URL)
    await asyncio.sleep(2)

    # Select 题名 search type
    print("      Selecting title search mode...")
    await page.evaluate("""
        () => {
            // Look for dropdown or select that controls search type
            const selects = document.querySelectorAll('select, .dropdown, [role="listbox"]');
            for (const sel of selects) {
                const options = sel.querySelectorAll('option');
                for (const opt of options) {
                    if (opt.textContent.includes('题名')) {
                        sel.value = opt.value;
                        sel.dispatchEvent(new Event('change', { bubbles: true }));
                        return true;
                    }
                }
            }

            // Try clicking a dropdown and selecting 题名
            const dropdownTriggers = document.querySelectorAll('[class*="dropdown"], [class*="select"]');
            for (const trigger of dropdownTriggers) {
                trigger.click();
            }
            return false;
        }
    """)
    await asyncio.sleep(0.5)

    # Try to click 题名 option if dropdown is visible
    try:
        title_option = page.locator('text=题名').first
        if await title_option.count() > 0:
            await title_option.click()
            await asyncio.sleep(0.5)
    except Exception:
        pass

    # Enter search query
    print("      Entering search query...")
    search_input_found = await page.evaluate("""
        (title) => {
            const inputs = document.querySelectorAll('input[type="text"], textarea');
            for (const input of inputs) {
                const placeholder = input.placeholder || '';
                const className = input.className || '';
                if (placeholder.includes('检索') || placeholder.includes('搜索') ||
                    className.includes('search') || input.offsetWidth > 200) {
                    input.focus();
                    input.value = title;
                    input.dispatchEvent(new Event('input', { bubbles: true }));
                    return true;
                }
            }
            // Try first visible text input
            for (const input of inputs) {
                if (input.offsetWidth > 0 && input.offsetHeight > 0) {
                    input.focus();
                    input.value = title;
                    input.dispatchEvent(new Event('input', { bubbles: true }));
                    return true;
                }
            }
            return false;
        }
    """, title)

    if not search_input_found:
        logger.warning("Could not find search input")
        return False

    await asyncio.sleep(0.5)

    # Click search button
    print("      Clicking search button...")
    await page.evaluate("""
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

    await asyncio.sleep(3)

    # Handle captcha if present
    if await check_for_captcha(page):
        print("      Captcha detected, solving...")
        await solve_slider_captcha(browser)
        await asyncio.sleep(2)

    # Check if results were found
    results_count = await page.evaluate("""
        () => {
            // Check for result items
            const results = document.querySelectorAll('.paper-item, .result-item, [class*="paper"], [class*="result"]');
            if (results.length > 0) return results.length;

            // Check for "no results" message
            const noResults = document.body.innerText;
            if (noResults.includes('没有找到') || noResults.includes('无结果') || noResults.includes('0条')) {
                return 0;
            }

            // Check for article links
            const links = document.querySelectorAll('a[href*="Paper/Detail"], a[href*="paper/"]');
            return links.length;
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


async def click_first_result(browser: BrowserController) -> bool:
    """Click the first search result.

    Args:
        browser: BrowserController instance

    Returns:
        True if clicked successfully
    """
    page = browser.page

    print("      Clicking first result...")

    # Try to click the first result link
    clicked = await page.evaluate("""
        () => {
            // Try specific paper detail links
            const detailLinks = document.querySelectorAll('a[href*="Paper/Detail"], a[href*="/Paper/"]');
            if (detailLinks.length > 0) {
                detailLinks[0].click();
                return true;
            }

            // Try result title links
            const titleLinks = document.querySelectorAll('.paper-title a, .result-title a, h3 a, h4 a');
            if (titleLinks.length > 0) {
                titleLinks[0].click();
                return true;
            }

            // Try any link in result container
            const resultItems = document.querySelectorAll('.paper-item, .result-item, [class*="result"]');
            if (resultItems.length > 0) {
                const link = resultItems[0].querySelector('a');
                if (link) {
                    link.click();
                    return true;
                }
            }

            return false;
        }
    """)

    if clicked:
        await asyncio.sleep(3)  # Wait for detail page to load
        return True

    # Fallback: try Playwright locator
    try:
        result_link = page.locator('a[href*="Paper"]').first
        if await result_link.count() > 0:
            await result_link.click()
            await asyncio.sleep(3)
            return True
    except Exception:
        pass

    return False


async def download_paper(browser: BrowserController, paper_title: str) -> str:
    """Download the paper PDF from the detail page.

    Args:
        browser: BrowserController instance
        paper_title: Paper title for naming the file

    Returns:
        Path to downloaded file, or empty string if failed
    """
    page = browser.page

    # Ensure papers directory exists
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)

    print("      Looking for download button...")

    # Try to find and click download button
    download_selectors = [
        'a:has-text("下载")',
        'a:has-text("PDF")',
        'button:has-text("下载")',
        '.download-btn',
        '[class*="download"]',
        'a[href*="download"]',
    ]

    for selector in download_selectors:
        try:
            elem = page.locator(selector).first
            if await elem.count() > 0:
                # Check if it's a PDF download link
                href = await elem.get_attribute('href') or ''
                text = await elem.text_content() or ''

                if 'pdf' in href.lower() or 'download' in href.lower() or '下载' in text:
                    print(f"      Found download button: {selector}")

                    # Prepare filename
                    safe_title = sanitize_filename(paper_title)
                    filename = f"{safe_title}.pdf"

                    try:
                        async def click_download():
                            await elem.click()

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
                        logger.warning(f"Download failed: {e}")
                        continue

        except Exception as e:
            logger.debug(f"Selector {selector} failed: {e}")
            continue

    # Fallback: try JavaScript to find and click download
    result = await page.evaluate("""
        () => {
            const elements = document.querySelectorAll('a, button');
            for (const el of elements) {
                const text = el.textContent || '';
                const href = el.href || '';
                if ((text.includes('下载') || text.includes('PDF') || text.includes('全文')) &&
                    (href.includes('pdf') || href.includes('download') || el.onclick)) {
                    el.click();
                    return {clicked: true, href: href};
                }
            }
            return {clicked: false};
        }
    """)

    if result.get('clicked'):
        await asyncio.sleep(3)
        print("      Clicked download via JavaScript, waiting...")
        # Check downloads directory for new file
        # This is a fallback - the file may have been downloaded

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
        is_logged_in = await check_login_status(page)
        if not is_logged_in and username and password:
            print("\n🔐 Not logged in, performing login...")
            login_success = await perform_login(browser, username, password)
            if not login_success:
                print("❌ Login failed. Please login manually first using wanfang_login.py")
                return {"success": 0, "failed": 0, "skipped": len(papers)}

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

                # Click first result
                clicked = await click_first_result(browser)
                if not clicked:
                    print("      ❌ Could not click result")
                    append_to_failed(paper, "Could not click search result")
                    failed_count += 1
                    remaining_papers.append(paper)
                    continue

                # Download the paper
                download_path = await download_paper(browser, title)

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
        required=True,
        help="Path to pending download CSV file",
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
        )
    )

    sys.exit(0 if result["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
