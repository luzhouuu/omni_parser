"""Image processing utilities for browser automation agent."""

import base64
from pathlib import Path
from typing import Tuple, Optional
from datetime import datetime

from browser_agent.agent.configuration import SCREENSHOT_DIR
from browser_agent.utils.log_utils import get_logger

logger = get_logger(__name__)


def encode_image_base64(image_bytes: bytes) -> str:
    """Encode image bytes to base64 string.

    Args:
        image_bytes: Raw image bytes

    Returns:
        Base64 encoded string
    """
    return base64.b64encode(image_bytes).decode("utf-8")


def decode_image_base64(base64_string: str) -> bytes:
    """Decode base64 string to image bytes.

    Args:
        base64_string: Base64 encoded string

    Returns:
        Raw image bytes
    """
    return base64.b64decode(base64_string)


def save_screenshot(
    image_bytes: bytes,
    filename: Optional[str] = None,
    directory: Optional[Path] = None,
) -> str:
    """Save screenshot to file.

    Args:
        image_bytes: Raw image bytes (PNG format)
        filename: Optional filename (auto-generated if not provided)
        directory: Directory to save to (uses SCREENSHOT_DIR if not provided)

    Returns:
        Full path to saved file
    """
    save_dir = directory or SCREENSHOT_DIR
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"screenshot_{timestamp}.png"

    file_path = save_dir / filename

    with open(file_path, "wb") as f:
        f.write(image_bytes)

    logger.debug(f"Screenshot saved to: {file_path}")
    return str(file_path)


def load_screenshot(file_path: str) -> Tuple[bytes, str]:
    """Load screenshot from file.

    Args:
        file_path: Path to the screenshot file

    Returns:
        Tuple of (image_bytes, base64_string)
    """
    with open(file_path, "rb") as f:
        image_bytes = f.read()

    base64_string = encode_image_base64(image_bytes)
    return image_bytes, base64_string


def get_image_dimensions(image_bytes: bytes) -> Tuple[int, int]:
    """Get dimensions of a PNG image from its bytes.

    Args:
        image_bytes: Raw PNG image bytes

    Returns:
        Tuple of (width, height)
    """
    # PNG header: first 8 bytes are signature, then IHDR chunk
    # IHDR chunk: 4 bytes length + 4 bytes type + 4 bytes width + 4 bytes height
    # Width is at bytes 16-19, height is at bytes 20-23
    width = int.from_bytes(image_bytes[16:20], byteorder="big")
    height = int.from_bytes(image_bytes[20:24], byteorder="big")
    return width, height


def compute_image_hash(image_bytes: bytes) -> str:
    """Compute a simple hash for image comparison.

    Args:
        image_bytes: Raw image bytes

    Returns:
        Hexadecimal hash string
    """
    import hashlib

    return hashlib.md5(image_bytes).hexdigest()


def images_are_similar(
    image1_bytes: bytes,
    image2_bytes: bytes,
    threshold: float = 0.95,
) -> bool:
    """Compare two images for similarity.

    This is a simple hash-based comparison. For more sophisticated
    comparison, use OpenCV or similar libraries.

    Args:
        image1_bytes: First image bytes
        image2_bytes: Second image bytes
        threshold: Similarity threshold (not used in hash comparison)

    Returns:
        True if images are identical (same hash)
    """
    # Simple hash comparison for exact match
    hash1 = compute_image_hash(image1_bytes)
    hash2 = compute_image_hash(image2_bytes)
    return hash1 == hash2
