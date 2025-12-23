"""Observation model for browser state snapshots."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from browser_agent.models.ui_element import UIElement


@dataclass
class Observation:
    """Represents a snapshot of the browser state.

    Attributes:
        url: Current page URL
        title: Current page title
        screenshot_path: Path to saved screenshot file
        screenshot_base64: Base64 encoded screenshot
        viewport_width: Viewport width in CSS pixels
        viewport_height: Viewport height in CSS pixels
        dpr: Device pixel ratio
        elements: List of detected UI elements
        element_map: Dictionary mapping element_id to UIElement
        timestamp: Unix timestamp when observation was taken
    """

    url: str
    title: str
    screenshot_path: Optional[str] = None
    screenshot_base64: Optional[str] = None
    viewport_width: int = 1280
    viewport_height: int = 720
    dpr: float = 1.0
    elements: List[UIElement] = field(default_factory=list)
    element_map: Dict[str, UIElement] = field(default_factory=dict)
    timestamp: float = 0.0

    def get_element(self, element_id: str) -> Optional[UIElement]:
        """Get element by ID.

        Args:
            element_id: The element ID to look up

        Returns:
            UIElement if found, None otherwise
        """
        return self.element_map.get(element_id)

    def get_elements_by_type(self, element_type: str) -> List[UIElement]:
        """Get all elements of a specific type.

        Args:
            element_type: The element type to filter by

        Returns:
            List of matching UIElements
        """
        return [e for e in self.elements if e.element_type.value == element_type]

    def get_interactable_elements(self) -> List[UIElement]:
        """Get all interactable elements.

        Returns:
            List of interactable UIElements
        """
        return [e for e in self.elements if e.is_interactable]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "url": self.url,
            "title": self.title,
            "screenshot_path": self.screenshot_path,
            "viewport_width": self.viewport_width,
            "viewport_height": self.viewport_height,
            "dpr": self.dpr,
            "elements": [e.to_dict() for e in self.elements],
            "timestamp": self.timestamp,
        }

    def to_prompt_context(self, max_elements: int = 50) -> str:
        """Format observation for inclusion in GPT prompt.

        Args:
            max_elements: Maximum number of elements to include

        Returns:
            Formatted string for prompt context
        """
        lines = [
            f"## Current Page",
            f"URL: {self.url}",
            f"Title: {self.title}",
            f"Viewport: {self.viewport_width}x{self.viewport_height}",
            "",
            f"## Detected Elements ({len(self.elements)} total)",
        ]

        # Sort by confidence and take top elements
        sorted_elements = sorted(
            self.elements, key=lambda e: e.confidence, reverse=True
        )[:max_elements]

        for elem in sorted_elements:
            lines.append(elem.to_prompt_string())

        return "\n".join(lines)


@dataclass
class VerificationResult:
    """Result of action verification.

    Attributes:
        success: Whether the action succeeded
        level: Verification level that determined success
        confidence: Confidence score (0.0 to 1.0)
        details: Human-readable details
        should_retry: Whether the action should be retried
        stuck_detection: Whether a stuck/loop state was detected
    """

    success: bool
    level: str  # "immediate", "visual", "semantic", "indicator"
    confidence: float
    details: str
    should_retry: bool = False
    stuck_detection: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "level": self.level,
            "confidence": self.confidence,
            "details": self.details,
            "should_retry": self.should_retry,
            "stuck_detection": self.stuck_detection,
        }
