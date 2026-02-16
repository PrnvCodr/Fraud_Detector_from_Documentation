"""
DocFraudDetector — Perspective Rectification Module
Corrects perspective distortion in document images using OpenCV transforms.

Author: Pranav Kashyap | IIIT Dharwad
"""

import cv2
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


class DocumentRectifier:
    """
    Rectifies perspective-distorted document images.
    
    Uses OpenCV's getPerspectiveTransform and warpPerspective to produce
    a clean, top-down view of the document.
    """

    def __init__(
        self,
        output_width: int = config.RECTIFIED_WIDTH,
        output_height: int = config.RECTIFIED_HEIGHT,
    ):
        self.output_width = output_width
        self.output_height = output_height

    def rectify(self, image: np.ndarray, corners: np.ndarray = None) -> dict:
        """
        Rectify a document image.
        
        Args:
            image: BGR input image (full image or cropped document)
            corners: Optional [4, 2] array of corner points (TL, TR, BR, BL).
                     If None, corners are auto-detected.
        
        Returns:
            dict with keys:
                - 'rectified': rectified document image
                - 'corners_used': corners used for the transform
                - 'transform_matrix': 3x3 perspective transform matrix
                - 'method': 'provided' or 'auto_detected'
        """
        if corners is not None:
            corners = np.array(corners, dtype=np.float32)
            if corners.shape == (4, 2):
                corners = self._order_corners(corners)
                method = "provided"
            else:
                corners = self._detect_corners(image)
                method = "auto_detected"
        else:
            corners = self._detect_corners(image)
            method = "auto_detected"
        
        # Compute target dimensions from corners
        width, height = self._compute_target_dimensions(corners)
        
        # Use configured dimensions if they produce a reasonable aspect ratio
        if abs(width / max(height, 1) - self.output_width / self.output_height) < 1.0:
            width = self.output_width
            height = self.output_height
        
        # Define destination points
        dst_pts = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype=np.float32,
        )
        
        # Compute perspective transform
        M = cv2.getPerspectiveTransform(corners, dst_pts)
        
        # Apply warp
        rectified = cv2.warpPerspective(
            image, M, (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        
        return {
            "rectified": rectified,
            "corners_used": corners.tolist(),
            "transform_matrix": M.tolist(),
            "method": method,
        }

    def _detect_corners(self, image: np.ndarray) -> np.ndarray:
        """
        Auto-detect document corners using edge detection and contour finding.
        Falls back to image corners if detection fails.
        """
        h, w = image.shape[:2]
        
        # Preprocess
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, config.GAUSSIAN_BLUR_KERNEL, 0)
        
        # Adaptive thresholding for better edge detection
        edges = cv2.Canny(blurred, config.CANNY_LOW, config.CANNY_HIGH)
        
        # Dilate to connect edge segments
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return self._image_corners(w, h)
        
        # Get largest contour
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        for contour in contours[:5]:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            if len(approx) == 4 and cv2.contourArea(approx) > 0.1 * w * h:
                return self._order_corners(approx.reshape(4, 2).astype(np.float32))
        
        # Fallback: use the convex hull of the largest contour
        hull = cv2.convexHull(contours[0])
        if len(hull) >= 4:
            # Find the 4 extreme points
            return self._extreme_points(hull.reshape(-1, 2), w, h)
        
        return self._image_corners(w, h)

    def _order_corners(self, pts: np.ndarray) -> np.ndarray:
        """Order corners: top-left, top-right, bottom-right, bottom-left."""
        rect = np.zeros((4, 2), dtype=np.float32)
        
        s = pts.sum(axis=1)
        d = np.diff(pts, axis=1).flatten()
        
        rect[0] = pts[np.argmin(s)]     # Top-left
        rect[2] = pts[np.argmax(s)]     # Bottom-right
        rect[1] = pts[np.argmin(d)]     # Top-right
        rect[3] = pts[np.argmax(d)]     # Bottom-left
        
        return rect

    def _image_corners(self, w: int, h: int) -> np.ndarray:
        """Return image corners as fallback."""
        margin = 5
        return np.array(
            [
                [margin, margin],
                [w - margin, margin],
                [w - margin, h - margin],
                [margin, h - margin],
            ],
            dtype=np.float32,
        )

    def _extreme_points(self, points: np.ndarray, w: int, h: int) -> np.ndarray:
        """Find 4 extreme points from a set of points."""
        # Top-left: minimize x + y
        # Top-right: maximize x - y
        # Bottom-right: maximize x + y
        # Bottom-left: maximize y - x
        s = points.sum(axis=1)
        d = np.diff(points, axis=1).flatten()
        
        tl = points[np.argmin(s)]
        br = points[np.argmax(s)]
        tr = points[np.argmin(d)]
        bl = points[np.argmax(d)]
        
        return np.array([tl, tr, br, bl], dtype=np.float32)

    def _compute_target_dimensions(self, corners: np.ndarray) -> tuple:
        """Compute output dimensions based on corner positions."""
        tl, tr, br, bl = corners
        
        # Width: max of top edge and bottom edge
        width_top = np.linalg.norm(tr - tl)
        width_bottom = np.linalg.norm(br - bl)
        width = int(max(width_top, width_bottom))
        
        # Height: max of left edge and right edge
        height_left = np.linalg.norm(bl - tl)
        height_right = np.linalg.norm(br - tr)
        height = int(max(height_left, height_right))
        
        # Ensure minimum size
        width = max(width, 200)
        height = max(height, 150)
        
        return width, height

    def enhance_rectified(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance the rectified document for better readability.
        Applies contrast enhancement and sharpening.
        """
        # Convert to LAB for better contrast handling
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        
        enhanced = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Sharpen
        sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        enhanced = cv2.filter2D(enhanced, -1, sharpen_kernel)
        
        return enhanced


if __name__ == "__main__":
    from utils import load_image, draw_corners, save_image
    
    rectifier = DocumentRectifier()
    
    if len(sys.argv) > 1:
        img = load_image(sys.argv[1])
        result = rectifier.rectify(img)
        print(f"Method: {result['method']}")
        print(f"Output size: {result['rectified'].shape}")
        
        # Save results
        save_image(result["rectified"], os.path.join(config.OUTPUT_DIR, "rectified.jpg"))
        enhanced = rectifier.enhance_rectified(result["rectified"])
        save_image(enhanced, os.path.join(config.OUTPUT_DIR, "rectified_enhanced.jpg"))
        print(f"Saved rectified images to {config.OUTPUT_DIR}/")
    else:
        print("Usage: python rectifier.py <image_path>")
