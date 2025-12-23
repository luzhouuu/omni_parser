"""Verify node - checks if action succeeded and detects completion."""

from typing import Dict, Any, List, Optional
import hashlib

from browser_agent.agent.state import BrowserAgentState
from browser_agent.models.observation import VerificationResult
from browser_agent.utils.log_utils import get_logger

logger = get_logger(__name__)


class ActionVerifier:
    """Multi-strategy action verification."""

    def __init__(self):
        self._previous_state_hash: Optional[str] = None
        self._action_repeat_count: int = 0
        self._max_repeats: int = 3

    def _compute_state_hash(self, state: BrowserAgentState) -> str:
        """Compute a hash of relevant state for loop detection."""
        url = state.get("current_url", "")
        title = state.get("page_title", "")
        action = state.get("next_action", {})
        action_type = action.get("action_type", "")

        # Include coordinates for click_at actions
        if action_type == "click_at":
            x = action.get("x", 0)
            y = action.get("y", 0)
            target = f"({x},{y})"
        else:
            target = action.get("element_id", "") or action.get("url", "") or action.get("text", "")

        combined = f"{url}|{title}|{action_type}|{target}"
        return hashlib.md5(combined.encode()).hexdigest()[:8]

    def check_immediate_changes(
        self,
        before_url: str,
        after_url: str,
        before_title: str,
        after_title: str,
        action_type: str,
    ) -> VerificationResult:
        """Check for immediate state changes.

        Args:
            before_url: URL before action
            after_url: URL after action
            before_title: Title before action
            after_title: Title after action
            action_type: Type of action performed

        Returns:
            VerificationResult
        """
        changes = []

        if before_url != after_url:
            changes.append("URL changed")

        if before_title != after_title:
            changes.append("Page title changed")

        # For navigate actions, URL change is expected
        if action_type == "navigate" and "URL changed" in changes:
            return VerificationResult(
                success=True,
                level="immediate",
                confidence=0.95,
                details=f"Navigation successful: {after_url[:50]}",
                should_retry=False,
                stuck_detection=False,
            )

        if changes:
            return VerificationResult(
                success=True,
                level="immediate",
                confidence=0.8,
                details="; ".join(changes),
                should_retry=False,
                stuck_detection=False,
            )

        return VerificationResult(
            success=False,
            level="immediate",
            confidence=0.0,
            details="No immediate changes detected",
            should_retry=True,
            stuck_detection=False,
        )

    def check_visual_changes(
        self,
        before_path: Optional[str],
        after_path: Optional[str],
    ) -> VerificationResult:
        """Compare screenshots for visual changes.

        Args:
            before_path: Path to before screenshot
            after_path: Path to after screenshot

        Returns:
            VerificationResult
        """
        if not before_path or not after_path:
            return VerificationResult(
                success=False,
                level="visual",
                confidence=0.0,
                details="Missing screenshots for comparison",
                should_retry=True,
                stuck_detection=False,
            )

        try:
            from browser_agent.utils.image_utils import images_are_similar

            with open(before_path, "rb") as f:
                before_bytes = f.read()
            with open(after_path, "rb") as f:
                after_bytes = f.read()

            if not images_are_similar(before_bytes, after_bytes):
                return VerificationResult(
                    success=True,
                    level="visual",
                    confidence=0.7,
                    details="Visual change detected in screenshot",
                    should_retry=False,
                    stuck_detection=False,
                )

        except Exception as e:
            logger.warning(f"Visual comparison failed: {e}")

        return VerificationResult(
            success=False,
            level="visual",
            confidence=0.0,
            details="No significant visual change",
            should_retry=True,
            stuck_detection=False,
        )

    def check_success_indicators(
        self,
        current_url: str,
        page_title: str,
        indicators: List[str],
    ) -> VerificationResult:
        """Check for presence of success indicators.

        Args:
            current_url: Current page URL
            page_title: Current page title
            indicators: List of success indicator strings

        Returns:
            VerificationResult
        """
        if not indicators:
            return VerificationResult(
                success=False,
                level="indicator",
                confidence=0.0,
                details="No success indicators defined",
                should_retry=True,
                stuck_detection=False,
            )

        found = []
        for indicator in indicators:
            indicator_lower = indicator.lower()
            if indicator_lower in page_title.lower():
                found.append(f"'{indicator}' in title")
            elif indicator_lower in current_url.lower():
                found.append(f"'{indicator}' in URL")

        if found:
            return VerificationResult(
                success=True,
                level="indicator",
                confidence=0.9,
                details=f"Success indicators found: {', '.join(found)}",
                should_retry=False,
                stuck_detection=False,
            )

        return VerificationResult(
            success=False,
            level="indicator",
            confidence=0.0,
            details="No success indicators found",
            should_retry=True,
            stuck_detection=False,
        )

    def detect_stuck_state(self, state: BrowserAgentState) -> bool:
        """Detect if agent is stuck in a loop.

        Args:
            state: Current agent state

        Returns:
            True if stuck, False otherwise
        """
        current_hash = self._compute_state_hash(state)

        if current_hash == self._previous_state_hash:
            self._action_repeat_count += 1
        else:
            self._action_repeat_count = 0
            self._previous_state_hash = current_hash

        return self._action_repeat_count >= self._max_repeats


# Global verifier instance
_verifier = ActionVerifier()


async def verify_node(state: BrowserAgentState) -> Dict[str, Any]:
    """Verify if action succeeded and check completion criteria.

    This node:
    1. Checks for URL/title changes
    2. Compares screenshots for visual changes
    3. Evaluates success indicators
    4. Detects stuck/loop states
    5. Updates is_complete flag

    Args:
        state: Current agent state

    Returns:
        State updates with verification result
    """
    logger.info(f"[Step {state.get('current_step', 0)}] Verifying action...")

    # Skip verification if already complete
    if state.get("is_complete"):
        return {}

    action_history = state.get("action_history", [])
    if not action_history:
        return {}

    last_action = action_history[-1] if action_history else {}
    action_type = last_action.get("action_type", "unknown")

    # Get before/after state for comparison
    # Note: In this simplified version, we don't have before URL/title stored
    # In a full implementation, we'd store these in the observe node
    current_url = state.get("current_url", "")
    page_title = state.get("page_title", "")
    previous_screenshot = state.get("previous_screenshot_path")
    current_screenshot = state.get("screenshot_path")
    success_indicators = state.get("success_indicators", [])

    # Level 1: Check immediate changes (always passes for type/scroll actions)
    if action_type in ["type", "scroll", "wait", "screenshot"]:
        result = VerificationResult(
            success=True,
            level="immediate",
            confidence=0.8,
            details=f"{action_type} action completed",
            should_retry=False,
            stuck_detection=False,
        )
    else:
        # For click/navigate, check for state changes
        result = _verifier.check_immediate_changes(
            before_url="",  # Would need to store this
            after_url=current_url,
            before_title="",
            after_title=page_title,
            action_type=action_type,
        )

    # Level 2: Check visual changes if immediate check failed
    if not result.success and previous_screenshot and current_screenshot:
        visual_result = _verifier.check_visual_changes(
            previous_screenshot,
            current_screenshot,
        )
        if visual_result.success:
            result = visual_result

    # Level 3: Check success indicators
    indicator_result = _verifier.check_success_indicators(
        current_url,
        page_title,
        success_indicators,
    )
    if indicator_result.success:
        logger.info(f"Success indicators found: {indicator_result.details}")
        return {
            "is_complete": True,
            "verification_result": indicator_result.to_dict(),
        }

    # Level 4: Check for stuck state
    stuck = _verifier.detect_stuck_state(state)
    if stuck:
        logger.warning("Agent appears to be stuck in a loop")
        result = VerificationResult(
            success=False,
            level="stuck",
            confidence=0.9,
            details="Agent stuck in a loop - same state repeated 3+ times",
            should_retry=False,
            stuck_detection=True,
        )
        return {
            "is_complete": True,
            "verification_result": result.to_dict(),
        }

    # Check max steps
    current_step = state.get("current_step", 0)
    max_steps = state.get("max_steps", 50)
    if current_step >= max_steps:
        logger.warning(f"Max steps ({max_steps}) reached")
        result = VerificationResult(
            success=False,
            level="max_steps",
            confidence=1.0,
            details=f"Maximum steps ({max_steps}) reached",
            should_retry=False,
            stuck_detection=False,
        )
        return {
            "is_complete": True,
            "verification_result": result.to_dict(),
        }

    logger.info(f"Verification: {result.level} - {result.details[:50]}")

    return {
        "verification_result": result.to_dict(),
        "should_retry": result.should_retry,
    }
