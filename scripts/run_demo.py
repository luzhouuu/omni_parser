#!/usr/bin/env python3
"""Demo script for browser automation agent.

This script demonstrates the browser automation agent
with a simple task.

Usage:
    # Run with mock parser (no GPU needed)
    python scripts/run_demo.py --mock

    # Run with real OmniParser (requires weights)
    python scripts/run_demo.py --url "https://example.com" --goal "Find and click the contact link"

    # Run on the Wanfang medical site (from user example)
    python scripts/run_demo.py --url "https://iczt.med.wanfangdata.com.cn" --goal "点击不良反应专题链接" --mock
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from browser_agent import BrowserAutomationAgent
from browser_agent.agent.state import state_summary


async def run_demo(
    url: str,
    goal: str,
    max_steps: int = 20,
    success_indicators: list = None,
    use_mock: bool = True,
    headless: bool = False,
    user_data_dir: str = None,
):
    """Run the browser automation demo.

    Args:
        url: Starting URL
        goal: Task goal description
        max_steps: Maximum steps to take
        success_indicators: List of success indicators
        use_mock: Use mock parser (no model weights needed)
        headless: Run browser in headless mode
        user_data_dir: Directory for browser data persistence
    """
    print("=" * 60)
    print("Browser Automation Agent Demo")
    print("=" * 60)
    print(f"URL: {url}")
    print(f"Goal: {goal}")
    print(f"Max Steps: {max_steps}")
    print(f"Mock Parser: {use_mock}")
    print(f"Headless: {headless}")
    if user_data_dir:
        print(f"Browser Data: {user_data_dir}")
    print("=" * 60)
    print()

    # Create agent
    agent = BrowserAutomationAgent(use_mock_parser=use_mock)

    # Define step callback for progress monitoring
    async def on_step(state):
        print(f"\n--- Step {state.get('current_step', 0)} ---")
        print(f"URL: {state.get('current_url', 'N/A')[:60]}")
        print(f"Title: {state.get('page_title', 'N/A')[:40]}")
        print(f"Elements: {len(state.get('ui_elements', []))}")

        action = state.get('next_action')
        if action:
            print(f"Action: {action.get('action_type', 'N/A')}")

    # Execute task with callbacks
    print("Starting automation...")
    print()

    result = await agent.execute_task_with_callback(
        task_goal=goal,
        start_url=url,
        on_step=on_step,
        max_steps=max_steps,
        success_indicators=success_indicators or [],
        headless=headless,
        user_data_dir=user_data_dir,
    )

    # Print results
    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Success: {result.success}")
    print(f"Message: {result.message}")
    print(f"Steps Taken: {result.steps_taken}")
    print(f"Final URL: {result.final_url}")
    print(f"Final Title: {result.final_title}")

    if result.errors:
        print(f"\nErrors:")
        for error in result.errors:
            print(f"  - {error}")

    if result.action_history:
        print(f"\nAction History ({len(result.action_history)} actions):")
        for i, action in enumerate(result.action_history, 1):
            action_type = action.get('action_type', 'unknown')
            target = action.get('element_id') or action.get('url') or action.get('text', '')[:20]
            success = "✓" if action.get('success') else "✗"
            print(f"  {i}. {success} {action_type}: {target}")

    if result.screenshots:
        print(f"\nScreenshots:")
        for screenshot in result.screenshots[-3:]:  # Last 3
            print(f"  - {screenshot}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Browser Automation Agent Demo"
    )
    parser.add_argument(
        "--url",
        default="https://example.com",
        help="Starting URL (default: https://example.com)",
    )
    parser.add_argument(
        "--goal",
        default="Find and describe the main content of this page",
        help="Task goal description",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=20,
        help="Maximum steps (default: 20)",
    )
    parser.add_argument(
        "--success",
        action="append",
        help="Success indicator (can be specified multiple times)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock parser (no model weights needed)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run browser in headless mode",
    )
    parser.add_argument(
        "--wanfang",
        action="store_true",
        help="Run Wanfang medical site demo",
    )
    parser.add_argument(
        "--user-data-dir",
        default=str(Path(__file__).parent.parent / ".browser_data"),
        help="Directory for browser data persistence (default: .browser_data)",
    )
    parser.add_argument(
        "--no-persist",
        action="store_true",
        help="Don't persist browser session",
    )

    args = parser.parse_args()

    # Wanfang preset
    if args.wanfang:
        args.url = "https://iczt.med.wanfangdata.com.cn"
        args.goal = "在学科导航页面，找到并点击'不良反应'专题链接，进入不良反应专题页面"
        args.success = ["不良反应专题", "ADR"]

    # Determine user_data_dir
    user_data_dir = None if args.no_persist else args.user_data_dir

    # Run demo
    result = asyncio.run(
        run_demo(
            url=args.url,
            goal=args.goal,
            max_steps=args.max_steps,
            success_indicators=args.success,
            use_mock=args.mock,
            headless=args.headless,
            user_data_dir=user_data_dir,
        )
    )

    # Exit code
    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
