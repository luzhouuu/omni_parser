"""Element mapping and coordinate conversion utilities."""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from browser_agent.models.ui_element import UIElement
from browser_agent.utils.log_utils import get_logger

logger = get_logger(__name__)


class DPRHandler:
    """Handle device pixel ratio and screenshot scaling.

    This class manages coordinate transformations between:
    - Image pixels (screenshot coordinates)
    - CSS pixels (page coordinates used by Playwright)
    """

    @staticmethod
    def get_dpr_from_page(page) -> float:
        """Get device pixel ratio from page (sync wrapper).

        Args:
            page: Playwright page object

        Returns:
            Device pixel ratio
        """
        # This should be called with await in async context
        return 1.0  # Default, actual value set during observation

    @staticmethod
    def scale_coordinates(
        x: float,
        y: float,
        from_dpr: float,
        to_dpr: float,
    ) -> Tuple[float, float]:
        """Scale coordinates between different DPR contexts.

        Args:
            x: X coordinate
            y: Y coordinate
            from_dpr: Source DPR
            to_dpr: Target DPR

        Returns:
            Scaled (x, y) coordinates
        """
        scale = to_dpr / from_dpr
        return x * scale, y * scale

    @staticmethod
    def image_to_page_coords(
        x: float,
        y: float,
        image_width: int,
        image_height: int,
        viewport_width: int,
        viewport_height: int,
        dpr: float = 1.0,
    ) -> Tuple[float, float]:
        """Convert image pixel coordinates to page CSS pixel coordinates.

        Args:
            x: X coordinate in image pixels
            y: Y coordinate in image pixels
            image_width: Width of the screenshot image
            image_height: Height of the screenshot image
            viewport_width: Viewport width in CSS pixels
            viewport_height: Viewport height in CSS pixels
            dpr: Device pixel ratio

        Returns:
            (x, y) in CSS pixels
        """
        # Screenshot is captured at DPR resolution
        # So image_width = viewport_width * dpr (approximately)

        # Calculate the actual scale factor
        scale_x = viewport_width / (image_width / dpr)
        scale_y = viewport_height / (image_height / dpr)

        # Convert image coordinates to CSS pixels
        css_x = (x / dpr) * scale_x
        css_y = (y / dpr) * scale_y

        return css_x, css_y

    @staticmethod
    def page_to_image_coords(
        x: float,
        y: float,
        image_width: int,
        image_height: int,
        viewport_width: int,
        viewport_height: int,
        dpr: float = 1.0,
    ) -> Tuple[float, float]:
        """Convert page CSS pixel coordinates to image pixel coordinates.

        Args:
            x: X coordinate in CSS pixels
            y: Y coordinate in CSS pixels
            image_width: Width of the screenshot image
            image_height: Height of the screenshot image
            viewport_width: Viewport width in CSS pixels
            viewport_height: Viewport height in CSS pixels
            dpr: Device pixel ratio

        Returns:
            (x, y) in image pixels
        """
        # Reverse of image_to_page_coords
        scale_x = image_width / viewport_width
        scale_y = image_height / viewport_height

        img_x = x * scale_x
        img_y = y * scale_y

        return img_x, img_y

    @staticmethod
    def adjust_bbox_for_dpr(
        bbox: List[float],
        image_dpr: float,
        target_dpr: float = 1.0,
    ) -> List[float]:
        """Adjust bounding box for DPR differences.

        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            image_dpr: DPR of the source image
            target_dpr: Target DPR (usually 1.0 for CSS pixels)

        Returns:
            Adjusted bounding box
        """
        scale = target_dpr / image_dpr
        return [coord * scale for coord in bbox]


class ElementMapper:
    """Maps OmniParser bboxes to stable element references.

    This class handles:
    - Assigning unique IDs to detected elements
    - Computing center points
    - Converting between coordinate systems
    """

    def __init__(
        self,
        viewport_width: int = 1280,
        viewport_height: int = 720,
        dpr: float = 1.0,
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
    ):
        """Initialize the element mapper.

        Args:
            viewport_width: Viewport width in CSS pixels
            viewport_height: Viewport height in CSS pixels
            dpr: Device pixel ratio
            image_width: Screenshot image width in pixels
            image_height: Screenshot image height in pixels
        """
        self._viewport_width = viewport_width
        self._viewport_height = viewport_height
        self._dpr = dpr
        self._image_width = image_width or int(viewport_width * dpr)
        self._image_height = image_height or int(viewport_height * dpr)
        self._element_counter = 0

    def set_image_dimensions(self, width: int, height: int) -> None:
        """Update image dimensions (call after capturing screenshot).

        Args:
            width: Image width in pixels
            height: Image height in pixels
        """
        self._image_width = width
        self._image_height = height

    def bbox_to_center(self, bbox: List[float]) -> Tuple[float, float]:
        """Convert bbox [x1, y1, x2, y2] to center point (cx, cy).

        Args:
            bbox: Bounding box coordinates

        Returns:
            Center point (cx, cy) in image pixels
        """
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        return cx, cy

    def image_center_to_page_center(
        self,
        center: Tuple[float, float],
    ) -> Tuple[float, float]:
        """Convert center point from image pixels to CSS pixels.

        Args:
            center: (cx, cy) in image pixels

        Returns:
            (cx, cy) in CSS pixels
        """
        return DPRHandler.image_to_page_coords(
            center[0],
            center[1],
            self._image_width,
            self._image_height,
            self._viewport_width,
            self._viewport_height,
            self._dpr,
        )

    def assign_id(self) -> str:
        """Generate a unique element ID.

        Returns:
            Unique element ID (e.g., "elem_001")
        """
        self._element_counter += 1
        return f"elem_{self._element_counter:03d}"

    def reset_counter(self) -> None:
        """Reset the element ID counter."""
        self._element_counter = 0

    def process_detections(
        self,
        detections: List[Dict[str, Any]],
    ) -> Tuple[List[UIElement], Dict[str, UIElement]]:
        """Process raw detections into UIElements with IDs and mapped coordinates.

        Args:
            detections: List of detection dictionaries from OmniParser
                Each should have: bbox, confidence, text/description, element_type

        Returns:
            Tuple of (elements_list, element_map)
        """
        self.reset_counter()
        elements = []
        element_map = {}

        for det in detections:
            bbox = det.get("bbox", [0, 0, 0, 0])
            center = self.bbox_to_center(bbox)
            page_center = self.image_center_to_page_center(center)

            element_id = self.assign_id()

            # Import here to avoid circular import
            from browser_agent.models.ui_element import UIElement, ElementType, classify_element_type

            # Determine element type
            text = det.get("text", "") or det.get("description", "")
            element_type = det.get("element_type")
            if element_type:
                if isinstance(element_type, str):
                    try:
                        element_type = ElementType(element_type)
                    except ValueError:
                        element_type = classify_element_type(text)
            else:
                element_type = classify_element_type(text)

            element = UIElement(
                element_id=element_id,
                element_type=element_type,
                bbox=bbox,
                center=center,
                page_center=page_center,
                text=det.get("text"),
                description=det.get("description"),
                confidence=det.get("confidence", 0.0),
                is_interactable=det.get("confidence", 0.0) > 0.3,
                attributes=det.get("attributes", {}),
            )

            elements.append(element)
            element_map[element_id] = element

        logger.debug(f"Processed {len(elements)} elements")
        return elements, element_map

    def find_element_at(
        self,
        x: float,
        y: float,
        elements: Dict[str, UIElement],
        coord_type: str = "page",
    ) -> Optional[str]:
        """Find element ID at given coordinates.

        Args:
            x: X coordinate
            y: Y coordinate
            elements: Dictionary mapping element_id to UIElement
            coord_type: "page" for CSS pixels, "image" for image pixels

        Returns:
            Element ID if found, None otherwise
        """
        for elem_id, elem in elements.items():
            if coord_type == "page":
                cx, cy = elem.page_center
            else:
                cx, cy = elem.center

            # Check if point is within element bounds
            half_width = elem.width / 2
            half_height = elem.height / 2

            if (
                abs(x - cx) <= half_width
                and abs(y - cy) <= half_height
            ):
                return elem_id

        return None

    def get_element_center(
        self,
        element_id: str,
        elements: Dict[str, UIElement],
        coord_type: str = "page",
    ) -> Optional[Tuple[float, float]]:
        """Get center coordinates for an element.

        Args:
            element_id: Element ID to look up
            elements: Dictionary mapping element_id to UIElement
            coord_type: "page" for CSS pixels, "image" for image pixels

        Returns:
            (cx, cy) coordinates or None if not found
        """
        element = elements.get(element_id)
        if not element:
            return None

        if coord_type == "page":
            return element.page_center
        return element.center
