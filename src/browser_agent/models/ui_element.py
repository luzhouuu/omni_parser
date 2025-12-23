"""UI Element data model for parsed screenshot elements."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List, Tuple


class ElementType(str, Enum):
    """Types of UI elements detected by OmniParser."""

    BUTTON = "button"
    INPUT = "input"
    LINK = "link"
    DROPDOWN = "dropdown"
    CHECKBOX = "checkbox"
    RADIO = "radio"
    TEXT = "text"
    IMAGE = "image"
    ICON = "icon"
    TAB = "tab"
    MENU = "menu"
    OTHER = "other"


@dataclass
class UIElement:
    """Represents a detected UI element from a screenshot.

    Attributes:
        element_id: Unique identifier for this element (e.g., "elem_001")
        element_type: Type of the element (button, input, link, etc.)
        bbox: Bounding box in image pixels [x1, y1, x2, y2]
        center: Center point in image pixels (cx, cy)
        page_center: Center point in CSS pixels (cx, cy), adjusted for DPR
        text: OCR text or caption describing the element
        description: More detailed description from Florence model
        confidence: Detection confidence score (0.0 to 1.0)
        is_interactable: Whether this element can be clicked/interacted with
        attributes: Additional metadata (class_id, etc.)
    """

    element_id: str
    element_type: ElementType
    bbox: List[float]
    center: Tuple[float, float]
    page_center: Tuple[float, float]
    text: Optional[str] = None
    description: Optional[str] = None
    confidence: float = 0.0
    is_interactable: bool = True
    attributes: Dict[str, Any] = field(default_factory=dict)

    @property
    def width(self) -> float:
        """Width of the bounding box."""
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        """Height of the bounding box."""
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> float:
        """Area of the bounding box."""
        return self.width * self.height

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "element_id": self.element_id,
            "element_type": self.element_type.value,
            "bbox": self.bbox,
            "center": list(self.center),
            "page_center": list(self.page_center),
            "text": self.text,
            "description": self.description,
            "confidence": self.confidence,
            "is_interactable": self.is_interactable,
            "attributes": self.attributes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UIElement":
        """Create UIElement from dictionary."""
        return cls(
            element_id=data["element_id"],
            element_type=ElementType(data["element_type"]),
            bbox=data["bbox"],
            center=tuple(data["center"]),
            page_center=tuple(data["page_center"]),
            text=data.get("text"),
            description=data.get("description"),
            confidence=data.get("confidence", 0.0),
            is_interactable=data.get("is_interactable", True),
            attributes=data.get("attributes", {}),
        )

    def to_prompt_string(self) -> str:
        """Format element for inclusion in GPT prompt."""
        display_text = self.text or self.description or "No description"
        return (
            f"[{self.element_id}] {self.element_type.value}: "
            f'"{display_text}" (confidence: {self.confidence:.2f})'
        )


def classify_element_type(caption: str) -> ElementType:
    """Infer element type from caption text.

    Args:
        caption: The caption or description of the element

    Returns:
        Inferred ElementType
    """
    caption_lower = caption.lower()

    # Button indicators
    if any(w in caption_lower for w in ["button", "submit", "click", "btn", "save", "cancel", "ok", "confirm"]):
        return ElementType.BUTTON

    # Input indicators
    if any(w in caption_lower for w in ["input", "field", "text box", "textbox", "search", "enter"]):
        return ElementType.INPUT

    # Link indicators
    if any(w in caption_lower for w in ["link", "href", "navigate", "go to"]):
        return ElementType.LINK

    # Dropdown indicators
    if any(w in caption_lower for w in ["dropdown", "select", "menu", "combobox", "picker"]):
        return ElementType.DROPDOWN

    # Checkbox indicators
    if any(w in caption_lower for w in ["checkbox", "check box", "toggle"]):
        return ElementType.CHECKBOX

    # Radio indicators
    if any(w in caption_lower for w in ["radio", "option"]):
        return ElementType.RADIO

    # Tab indicators
    if any(w in caption_lower for w in ["tab", "panel"]):
        return ElementType.TAB

    # Icon indicators
    if any(w in caption_lower for w in ["icon", "logo", "image", "avatar"]):
        return ElementType.ICON

    # Text indicators
    if any(w in caption_lower for w in ["text", "label", "heading", "title", "paragraph"]):
        return ElementType.TEXT

    return ElementType.OTHER
