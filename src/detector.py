"""
DocFraudDetector — Document Detection Module
Locates document regions in images using YOLOv8 with OpenCV contour fallback.

Author: Pranav Kashyap | IIIT Dharwad
"""

import cv2
import numpy as np
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


class DocumentDetector:
    """
    Detects and localizes documents in images.
    
    Primary method: YOLOv8-based detection (if ultralytics available)
    Fallback: OpenCV contour-based detection (always available)
    """

    def __init__(self, use_yolo: bool = True):
        self.use_yolo = use_yolo
        self.yolo_model = None
        
        if use_yolo:
            try:
                from ultralytics import YOLO
                self.yolo_model = YOLO(config.YOLO_MODEL)
                print(f"[DocumentDetector] YOLOv8 model loaded: {config.YOLO_MODEL}")
            except ImportError:
                print("[DocumentDetector] ultralytics not installed. Using contour fallback.")
                self.use_yolo = False
            except Exception as e:
                print(f"[DocumentDetector] YOLO load failed: {e}. Using contour fallback.")
                self.use_yolo = False

    def detect(self, image: np.ndarray) -> dict:
        """
        Detect document in image.
        
        Args:
            image: BGR numpy array
            
        Returns:
            dict with keys:
                - 'bbox': [x1, y1, x2, y2] bounding box
                - 'corners': [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] four corners
                - 'confidence': detection confidence (0-1)
                - 'method': 'yolo' or 'contour'
                - 'cropped': cropped document region
        """
        if self.use_yolo and self.yolo_model is not None:
            result = self._detect_yolo(image)
            if result is not None:
                return result
        
        return self._detect_contour(image)

    def _detect_yolo(self, image: np.ndarray) -> Optional[dict]:
        """Detect document using YOLOv8."""
        results = self.yolo_model(
            image,
            conf=config.YOLO_CONF_THRESHOLD,
            iou=config.YOLO_IOU_THRESHOLD,
            verbose=False,
        )
        
        if len(results) == 0 or len(results[0].boxes) == 0:
            return None
        
        # Get the largest detection (most likely the document)
        boxes = results[0].boxes
        areas = (boxes.xyxy[:, 2] - boxes.xyxy[:, 0]) * (boxes.xyxy[:, 3] - boxes.xyxy[:, 1])
        best_idx = areas.argmax().item()
        
        bbox = boxes.xyxy[best_idx].cpu().numpy().astype(int)
        conf = boxes.conf[best_idx].item()
        
        x1, y1, x2, y2 = bbox
        corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
        cropped = image[y1:y2, x1:x2].copy()
        
        return {
            "bbox": bbox.tolist(),
            "corners": corners.tolist(),
            "confidence": round(conf, 4),
            "method": "yolo",
            "cropped": cropped,
        }

    def _detect_contour(self, image: np.ndarray) -> dict:
        """
        Detect document using OpenCV contour detection.
        Robust fallback that works without any ML model.
        """
        h, w = image.shape[:2]
        
        # Preprocess
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, config.GAUSSIAN_BLUR_KERNEL, 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, config.CANNY_LOW, config.CANNY_HIGH)
        
        # Dilate to connect edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # Fallback: return the entire image
            return self._full_image_fallback(image)
        
        # Sort by area (largest first)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Find quadrilateral contours
        for contour in contours[:5]:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, config.CONTOUR_APPROX_EPSILON * peri, True)
            
            area = cv2.contourArea(approx)
            if area < 0.05 * h * w:
                continue  # Too small
            
            if len(approx) == 4:
                corners = self._order_corners(approx.reshape(4, 2).astype(np.float32))
                bbox = self._corners_to_bbox(corners)
                confidence = min(area / (h * w), 1.0)
                
                x1, y1, x2, y2 = [int(v) for v in bbox]
                cropped = image[y1:y2, x1:x2].copy()
                
                return {
                    "bbox": bbox,
                    "corners": corners.tolist(),
                    "confidence": round(confidence, 4),
                    "method": "contour",
                    "cropped": cropped,
                }
        
        # If no quadrilateral found, use largest contour's bounding rect
        largest = contours[0]
        x, y, cw, ch = cv2.boundingRect(largest)
        
        if cw * ch < 0.05 * h * w:
            return self._full_image_fallback(image)
        
        bbox = [x, y, x + cw, y + ch]
        corners = np.array(
            [[x, y], [x + cw, y], [x + cw, y + ch], [x, y + ch]], dtype=np.float32
        )
        cropped = image[y : y + ch, x : x + cw].copy()
        confidence = (cw * ch) / (h * w)
        
        return {
            "bbox": bbox,
            "corners": corners.tolist(),
            "confidence": round(confidence, 4),
            "method": "contour",
            "cropped": cropped,
        }

    def _order_corners(self, pts: np.ndarray) -> np.ndarray:
        """
        Order corners as: top-left, top-right, bottom-right, bottom-left.
        Uses sum and difference of coordinates.
        """
        rect = np.zeros((4, 2), dtype=np.float32)
        
        s = pts.sum(axis=1)
        d = np.diff(pts, axis=1).flatten()
        
        rect[0] = pts[np.argmin(s)]     # Top-left: smallest sum
        rect[2] = pts[np.argmax(s)]     # Bottom-right: largest sum
        rect[1] = pts[np.argmin(d)]     # Top-right: smallest difference
        rect[3] = pts[np.argmax(d)]     # Bottom-left: largest difference
        
        return rect

    def _corners_to_bbox(self, corners: np.ndarray) -> list:
        """Convert corners to bounding box [x1, y1, x2, y2]."""
        x_min = int(corners[:, 0].min())
        y_min = int(corners[:, 1].min())
        x_max = int(corners[:, 0].max())
        y_max = int(corners[:, 1].max())
        return [x_min, y_min, x_max, y_max]

    def _full_image_fallback(self, image: np.ndarray) -> dict:
        """Return the full image when no document is detected."""
        h, w = image.shape[:2]
        return {
            "bbox": [0, 0, w, h],
            "corners": [[0, 0], [w, 0], [w, h], [0, h]],
            "confidence": 0.0,
            "method": "fallback",
            "cropped": image.copy(),
        }


if __name__ == "__main__":
    # Quick test
    import sys
    from utils import load_image, draw_bounding_box, draw_corners, save_image
    
    detector = DocumentDetector(use_yolo=False)
    
    if len(sys.argv) > 1:
        img = load_image(sys.argv[1])
        result = detector.detect(img)
        print(f"Method: {result['method']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Bbox: {result['bbox']}")
        
        vis = draw_bounding_box(img, result["bbox"], f"Doc ({result['confidence']:.2f})")
        vis = draw_corners(vis, np.array(result["corners"]))
        save_image(vis, os.path.join(config.OUTPUT_DIR, "detection_result.jpg"))
        print(f"Saved visualization to {config.OUTPUT_DIR}/detection_result.jpg")
    else:
        print("Usage: python detector.py <image_path>")
