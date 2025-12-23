#!/usr/bin/env python3
"""Full pipeline test: Screenshot -> OmniParser -> Florence-2 -> GPT -> Playwright.

Usage:
    python scripts/test_full_pipeline.py
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from browser_agent import BrowserAutomationAgent
from browser_agent.utils.log_utils import get_logger

logger = get_logger(__name__)


async def test_full_pipeline():
    """Test the complete pipeline."""

    # Test configuration
    url = "https://lczl.med.wanfangdata.com.cn/ADR?pagesize=1"
    goal = "点击页面上任意一条记录右侧的'全文下载'链接"

    print("=" * 70)
    print("Full Pipeline Test")
    print("=" * 70)
    print(f"URL: {url}")
    print(f"Goal: {goal}")
    print("Pipeline: Screenshot -> OmniParser -> Florence-2 -> GPT -> Playwright")
    print("=" * 70)
    print()

    # Create agent with real OmniParser (not mock)
    print("[1/5] Initializing agent with OmniParser...")
    agent = BrowserAutomationAgent(use_mock_parser=False)
    print("      Agent initialized.")
    print()

    # Step callback to show progress
    step_count = [0]

    async def on_step(state):
        step_count[0] += 1
        step = state.get('current_step', 0)

        print(f"\n--- Step {step} ---")
        print(f"[2/5] Screenshot: {state.get('screenshot_path', 'N/A')}")

        elements = state.get('ui_elements', [])
        print(f"[3/5] OmniParser detected: {len(elements)} UI elements")

        # Show some elements
        for i, elem in enumerate(elements[:5]):
            elem_type = elem.get('element_type', 'unknown')
            text = elem.get('text', '')[:30] or elem.get('description', '')[:30]
            print(f"      - [{elem.get('element_id', 'N/A')}] {elem_type}: {text}")
        if len(elements) > 5:
            print(f"      ... and {len(elements) - 5} more elements")

        action = state.get('next_action')
        if action:
            action_type = action.get('action_type', 'N/A')
            print(f"[4/5] GPT planned action: {action_type}")
            print(f"      Action details: {action}")

        plan = state.get('current_plan')
        if plan:
            print(f"      Plan: {plan[:100]}...")

    # Execute task
    print("[5/5] Starting Playwright execution...")
    print()

    try:
        result = await agent.execute_task_with_callback(
            task_goal=goal,
            start_url=url,
            on_step=on_step,
            max_steps=10,  # Increase steps for better success rate
            success_indicators=["全文下载", "下载"],
            headless=False,  # Show browser
        )

        print()
        print("=" * 70)
        print("Test Results")
        print("=" * 70)
        print(f"Success: {result.success}")
        print(f"Message: {result.message}")
        print(f"Steps Taken: {result.steps_taken}")
        print(f"Final URL: {result.final_url}")
        print(f"Final Title: {result.final_title}")

        if result.errors:
            print("\nErrors:")
            for error in result.errors:
                print(f"  - {error}")

        if result.action_history:
            print(f"\nAction History ({len(result.action_history)} actions):")
            for i, action in enumerate(result.action_history, 1):
                action_type = action.get('action_type', 'unknown')
                target = action.get('element_id') or action.get('url') or action.get('text', '')[:20]
                success_mark = "OK" if action.get('success') else "FAIL"
                print(f"  {i}. [{success_mark}] {action_type}: {target}")

        if result.screenshots:
            print(f"\nScreenshots saved:")
            for screenshot in result.screenshots[-3:]:
                print(f"  - {screenshot}")

        return result

    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    result = asyncio.run(test_full_pipeline())
    sys.exit(0 if result.success else 1)
