"""OmniParser CPU inference for Mac environments.

This module provides CPU-based inference for OmniParser models:
- Icon detection using YOLOv8
- Icon captioning using Florence-2

Reference: https://huggingface.co/microsoft/OmniParser-v2.0
"""

from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
from PIL import Image

from browser_agent.agent.configuration import (
    OMNIPARSER_WEIGHTS_DIR,
    OMNIPARSER_CONF_THRESHOLD,
    OMNIPARSER_IOU_THRESHOLD,
    OMNIPARSER_MAX_DETECTIONS,
)
from browser_agent.models.ui_element import UIElement, classify_element_type, ElementType
from browser_agent.utils.log_utils import get_logger

logger = get_logger(__name__)


class OmniParserCPU:
    """OmniParser for Mac CPU inference using YOLO and Florence models.

    This class implements the OmniParser pipeline:
    1. Icon detection using fine-tuned YOLOv8
    2. Icon captioning using fine-tuned Florence-2

    Performance note: CPU inference is slower than GPU but functional
    for development and testing on Mac machines.
    """

    def __init__(
        self,
        weights_dir: Optional[str] = None,
        conf_threshold: float = OMNIPARSER_CONF_THRESHOLD,
        iou_threshold: float = OMNIPARSER_IOU_THRESHOLD,
        max_detections: int = OMNIPARSER_MAX_DETECTIONS,
    ):
        """Initialize OmniParser.

        Args:
            weights_dir: Directory containing model weights
            conf_threshold: Confidence threshold for detection
            iou_threshold: IOU threshold for NMS
            max_detections: Maximum number of detections
        """
        self.weights_dir = Path(weights_dir or OMNIPARSER_WEIGHTS_DIR)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections

        # Lazy-loaded models
        self._detector = None
        self._captioner = None
        self._processor = None
        self._captioner_failed = False  # Flag to skip captioning after failure
        self._ocr_reader = None  # EasyOCR reader

        # Device configuration
        self.device = "cpu"

        logger.info(f"OmniParser initialized with weights_dir: {self.weights_dir}")

    def _check_weights(self) -> bool:
        """Check if model weights exist.

        Returns:
            True if weights exist, False otherwise
        """
        detector_path = self.weights_dir / "icon_detect" / "model.pt"
        captioner_path = self.weights_dir / "icon_caption_florence"

        if not detector_path.exists():
            logger.warning(f"Icon detector weights not found: {detector_path}")
            return False

        if not captioner_path.exists():
            logger.warning(f"Icon captioner weights not found: {captioner_path}")
            return False

        return True

    def load_detector(self):
        """Load YOLOv8 icon detector.

        Returns:
            Loaded YOLO model
        """
        if self._detector is not None:
            return self._detector

        try:
            from ultralytics import YOLO

            model_path = self.weights_dir / "icon_detect" / "model.pt"
            logger.info(f"Loading icon detector from: {model_path}")

            self._detector = YOLO(str(model_path))
            # Explicitly set device to CPU
            self._detector.to(self.device)

            logger.info("Icon detector loaded successfully")
            return self._detector

        except ImportError:
            logger.error("ultralytics not installed. Run: pip install ultralytics")
            raise
        except Exception as e:
            logger.error(f"Failed to load icon detector: {e}")
            raise

    def load_captioner(self) -> Tuple[Any, Any]:
        """Load Florence-2 captioner.

        Returns:
            Tuple of (model, processor)

        Raises:
            RuntimeError: If captioner previously failed to load
        """
        if self._captioner is not None:
            return self._captioner, self._processor

        # Skip if we already know captioner failed
        if self._captioner_failed:
            raise RuntimeError("Captioner previously failed to load, skipping")

        try:
            import os
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor

            # Disable SDPA to avoid compatibility issues with Florence-2
            os.environ["TRANSFORMERS_NO_SDPA"] = "1"

            model_path = self.weights_dir / "icon_caption_florence"
            logger.info(f"Loading icon captioner from: {model_path}")

            # Load processor
            self._processor = AutoProcessor.from_pretrained(
                str(model_path),
                trust_remote_code=True,
            )

            # Load model with CPU settings and disable SDPA
            self._captioner = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                torch_dtype=torch.float32,  # CPU uses float32
                trust_remote_code=True,
                device_map="cpu",
                attn_implementation="eager",  # Use eager attention instead of SDPA
            )

            logger.info("Icon captioner loaded successfully")
            return self._captioner, self._processor

        except ImportError:
            logger.error("transformers not installed. Run: pip install transformers")
            self._captioner_failed = True
            raise
        except Exception as e:
            logger.error(f"Failed to load icon captioner: {e}")
            self._captioner_failed = True
            raise

    def load_ocr_reader(self):
        """Load EasyOCR reader for text recognition.

        Returns:
            EasyOCR Reader instance
        """
        if self._ocr_reader is not None:
            return self._ocr_reader

        try:
            import easyocr

            logger.info("Loading EasyOCR reader (zh_ch, en)...")
            self._ocr_reader = easyocr.Reader(
                ['ch_sim', 'en'],
                gpu=False,
                verbose=False,
            )
            logger.info("EasyOCR reader loaded successfully")
            return self._ocr_reader

        except ImportError:
            logger.error("easyocr not installed. Run: pip install easyocr")
            raise
        except Exception as e:
            logger.error(f"Failed to load OCR reader: {e}")
            raise

    def ocr_region(
        self,
        image: Image.Image,
        bbox: List[float],
        padding: int = 2,
    ) -> str:
        """Extract text from a specific region using OCR.

        Args:
            image: PIL Image
            bbox: Bounding box [x1, y1, x2, y2]
            padding: Padding around the crop region

        Returns:
            Recognized text string
        """
        reader = self.load_ocr_reader()

        # Crop region with padding
        x1, y1, x2, y2 = bbox
        x1 = max(0, int(x1) - padding)
        y1 = max(0, int(y1) - padding)
        x2 = min(image.width, int(x2) + padding)
        y2 = min(image.height, int(y2) + padding)

        cropped = image.crop((x1, y1, x2, y2))
        cropped_np = np.array(cropped)

        # Run OCR
        results = reader.readtext(cropped_np, detail=0)

        # Join all detected text
        text = ' '.join(results).strip()
        return text if text else "UI element"

    def ocr_full_image(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Run OCR on full image to detect all text with positions.

        This is useful for finding text links that YOLO might miss.

        Args:
            image: PIL Image

        Returns:
            List of text detections with bbox and text
        """
        reader = self.load_ocr_reader()
        image_np = np.array(image)

        # Run OCR with detail=1 to get bounding boxes
        results = reader.readtext(image_np, detail=1)

        detections = []
        for (bbox_points, text, confidence) in results:
            if confidence < 0.3:  # Skip low confidence
                continue

            # Convert polygon to xyxy bbox
            xs = [p[0] for p in bbox_points]
            ys = [p[1] for p in bbox_points]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "text": text,
                "description": text,
                "confidence": confidence,
                "source": "ocr",
            })

        return detections

    def detect_icons(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Run icon detection on image.

        Args:
            image: Input image as numpy array (RGB)

        Returns:
            List of detection dictionaries with bbox, confidence, class_id
        """
        detector = self.load_detector()

        results = detector.predict(
            source=image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            max_det=self.max_detections,
            device=self.device,
            verbose=False,
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i in range(len(boxes)):
                box = boxes[i]
                detection = {
                    "bbox": box.xyxy[0].tolist(),  # [x1, y1, x2, y2]
                    "confidence": float(box.conf[0]),
                    "class_id": int(box.cls[0]) if box.cls is not None else 0,
                }
                detections.append(detection)

        logger.debug(f"Detected {len(detections)} icons")
        return detections

    def caption_region(
        self,
        image: Image.Image,
        bbox: List[float],
        padding: int = 5,
    ) -> str:
        """Generate caption for a specific region.

        Args:
            image: PIL Image
            bbox: Bounding box [x1, y1, x2, y2]
            padding: Padding around the crop region

        Returns:
            Generated caption string
        """
        import torch

        captioner, processor = self.load_captioner()

        # Crop region with padding
        x1, y1, x2, y2 = bbox
        x1 = max(0, int(x1) - padding)
        y1 = max(0, int(y1) - padding)
        x2 = min(image.width, int(x2) + padding)
        y2 = min(image.height, int(y2) + padding)

        cropped = image.crop((x1, y1, x2, y2))

        # Process with Florence
        prompt = "<CAPTION>"
        inputs = processor(
            text=prompt,
            images=cropped,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            generated_ids = captioner.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=50,
                num_beams=1,  # Greedy for speed
                use_cache=False,  # Disable KV cache for compatibility
            )

        caption = processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        # Clean up the caption
        caption = caption.replace("<CAPTION>", "").strip()
        return caption

    def parse_screenshot(
        self,
        image_path: str,
        caption_all: bool = False,
        use_ocr: bool = True,  # Use OCR by default (fast and accurate for text)
        ocr_full_scan: bool = True,  # Also scan full image for text links
    ) -> List[Dict[str, Any]]:
        """Full parsing pipeline: detect + OCR/caption.

        Args:
            image_path: Path to screenshot image
            caption_all: Whether to caption all detections with Florence (slow)
            use_ocr: Whether to use OCR for text recognition (fast, default True)
            ocr_full_scan: Also run full image OCR to find text links (default True)

        Returns:
            List of element dictionaries with bbox, text, confidence, etc.
        """
        logger.info(f"Parsing screenshot: {image_path}")

        # Load image
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)

        # Step 1: Detect all icons/elements with YOLO
        detections = self.detect_icons(image_np)
        logger.debug(f"YOLO detection: {len(detections)} elements found")

        # Step 2: Run full image OCR to find text elements YOLO might miss
        if use_ocr and ocr_full_scan:
            logger.info("Running full image OCR scan...")
            try:
                ocr_detections = self.ocr_full_image(image)
                logger.debug(f"OCR found {len(ocr_detections)} text elements")

                # Merge OCR detections, avoiding duplicates
                existing_bboxes = set()
                for det in detections:
                    bbox = det.get("bbox", [])
                    if bbox:
                        # Use center point for dedup
                        cx = (bbox[0] + bbox[2]) / 2
                        cy = (bbox[1] + bbox[3]) / 2
                        existing_bboxes.add((int(cx/20), int(cy/20)))  # Grid-based dedup

                added = 0
                for ocr_det in ocr_detections:
                    bbox = ocr_det.get("bbox", [])
                    if bbox:
                        cx = (bbox[0] + bbox[2]) / 2
                        cy = (bbox[1] + bbox[3]) / 2
                        key = (int(cx/20), int(cy/20))
                        if key not in existing_bboxes:
                            detections.append(ocr_det)
                            existing_bboxes.add(key)
                            added += 1

                logger.info(f"Added {added} text elements from OCR scan")
            except Exception as e:
                logger.warning(f"Full image OCR scan failed: {e}")

        # Step 3: Extract text for YOLO detections using OCR
        if use_ocr:
            logger.info("Using OCR for element text recognition...")
            for i, det in enumerate(detections):
                # Skip if already has text from OCR scan
                if det.get("source") == "ocr":
                    continue
                try:
                    text = self.ocr_region(image, det["bbox"])
                    det["text"] = text
                    det["description"] = text
                except Exception as e:
                    if i == 0:
                        logger.warning(f"OCR failed: {e}")
                    det["text"] = "UI element"
                    det["description"] = "UI element"

        elif caption_all and self._check_weights() and not self._captioner_failed:
            # Use Florence captioning (slow but more descriptive)
            captioned_count = 0
            for i, det in enumerate(detections):
                try:
                    caption = self.caption_region(image, det["bbox"])
                    det["text"] = caption
                    det["description"] = caption
                    captioned_count += 1
                except Exception as e:
                    if captioned_count == 0 and i == 0:
                        logger.warning(f"Captioning failed, falling back to placeholder: {e}")
                    det["text"] = "UI element"
                    det["description"] = "UI element"
                    if self._captioner_failed:
                        break

            if self._captioner_failed:
                for det in detections:
                    if "text" not in det:
                        det["text"] = "UI element"
                        det["description"] = "UI element"
        else:
            # No text extraction
            for det in detections:
                det["text"] = "UI element"
                det["description"] = "UI element"

        # Step 3: Classify element types based on captions
        for det in detections:
            det["element_type"] = classify_element_type(det.get("text", ""))

        logger.info(f"Parsing complete: {len(detections)} elements processed")
        return detections

    def parse_screenshot_to_elements(
        self,
        image_path: str,
        viewport_width: int = 1280,
        viewport_height: int = 720,
        dpr: float = 1.0,
        caption_all: bool = False,
        use_ocr: bool = True,  # Use OCR by default for fast text recognition
    ) -> Tuple[List[UIElement], Dict[str, UIElement]]:
        """Parse screenshot and return UIElement objects.

        This is a convenience method that combines parsing and element mapping.

        Args:
            image_path: Path to screenshot image
            viewport_width: Viewport width in CSS pixels
            viewport_height: Viewport height in CSS pixels
            dpr: Device pixel ratio
            caption_all: Whether to caption elements with Florence (slow)
            use_ocr: Whether to use OCR for text recognition (fast, default True)

        Returns:
            Tuple of (elements_list, element_map)
        """
        from browser_agent.core.element_mapper import ElementMapper

        # Get image dimensions
        image = Image.open(image_path)
        image_width, image_height = image.size

        # Parse screenshot with OCR enabled by default
        detections = self.parse_screenshot(image_path, caption_all=caption_all, use_ocr=use_ocr)

        # Map to UIElements
        mapper = ElementMapper(
            viewport_width=viewport_width,
            viewport_height=viewport_height,
            dpr=dpr,
            image_width=image_width,
            image_height=image_height,
        )

        return mapper.process_detections(detections)


class MockOmniParser:
    """Mock OmniParser for testing without model weights.

    This class provides a fallback when OmniParser weights are not available.
    It uses simple heuristics or returns empty results.
    """

    def __init__(self, **kwargs):
        """Initialize mock parser."""
        logger.warning("Using MockOmniParser - no actual parsing will occur")

    def detect_icons(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Return empty detections."""
        return []

    def caption_region(self, image: Image.Image, bbox: List[float]) -> str:
        """Return placeholder caption."""
        return "Mock element"

    def parse_screenshot(self, image_path: str, **kwargs) -> List[Dict[str, Any]]:
        """Return mock detections based on image size."""
        try:
            image = Image.open(image_path)
            width, height = image.size

            # Create some mock detections
            return [
                {
                    "bbox": [50, 50, 150, 100],
                    "confidence": 0.9,
                    "text": "Mock Button",
                    "description": "Mock Button",
                    "element_type": ElementType.BUTTON,
                },
                {
                    "bbox": [200, 50, 400, 90],
                    "confidence": 0.85,
                    "text": "Mock Input Field",
                    "description": "Mock Input Field",
                    "element_type": ElementType.INPUT,
                },
            ]
        except Exception:
            return []

    def parse_screenshot_to_elements(
        self,
        image_path: str,
        viewport_width: int = 1280,
        viewport_height: int = 720,
        dpr: float = 1.0,
        caption_all: bool = False,
        use_ocr: bool = True,
    ) -> Tuple[List[UIElement], Dict[str, UIElement]]:
        """Return mock elements."""
        from browser_agent.core.element_mapper import ElementMapper

        detections = self.parse_screenshot(image_path)
        mapper = ElementMapper(
            viewport_width=viewport_width,
            viewport_height=viewport_height,
            dpr=dpr,
        )
        return mapper.process_detections(detections)


def create_omni_parser(use_mock: bool = False, **kwargs) -> OmniParserCPU:
    """Factory function to create OmniParser instance.

    Args:
        use_mock: Force use of mock parser
        **kwargs: Arguments passed to OmniParserCPU

    Returns:
        OmniParser instance (real or mock)
    """
    if use_mock:
        return MockOmniParser(**kwargs)

    parser = OmniParserCPU(**kwargs)
    if not parser._check_weights():
        logger.warning("Model weights not found, falling back to MockOmniParser")
        return MockOmniParser(**kwargs)

    return parser
