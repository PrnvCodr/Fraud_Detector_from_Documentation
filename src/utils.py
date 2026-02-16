"""
DocFraudDetector — Image Utilities & Visualization
Provides helper functions for image loading, preprocessing, and result visualization.

Author: Pranav Kashyap | IIIT Dharwad
"""

import cv2
import numpy as np
from PIL import Image
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def load_image(image_path: str) -> np.ndarray:
    """Load an image from disk. Returns BGR numpy array."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    return img


def load_image_rgb(image_path: str) -> np.ndarray:
    """Load an image and convert to RGB."""
    img = load_image(image_path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert BGR image to RGB."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    """Convert RGB image to BGR."""
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def resize_image(image: np.ndarray, width: int = None, height: int = None) -> np.ndarray:
    """Resize image maintaining aspect ratio if only one dimension given."""
    h, w = image.shape[:2]
    
    if width and height:
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    elif width:
        ratio = width / w
        new_h = int(h * ratio)
        return cv2.resize(image, (width, new_h), interpolation=cv2.INTER_AREA)
    elif height:
        ratio = height / h
        new_w = int(w * ratio)
        return cv2.resize(image, (new_w, height), interpolation=cv2.INTER_AREA)
    return image


def preprocess_for_display(image: np.ndarray, max_width: int = 800) -> np.ndarray:
    """Resize large images for display purposes."""
    h, w = image.shape[:2]
    if w > max_width:
        return resize_image(image, width=max_width)
    return image.copy()


def draw_bounding_box(
    image: np.ndarray,
    bbox: tuple,
    label: str = "",
    color: tuple = (0, 255, 0),
    thickness: int = 2,
    font_scale: float = 0.7,
) -> np.ndarray:
    """Draw a bounding box with label on an image."""
    img = image.copy()
    x1, y1, x2, y2 = [int(v) for v in bbox]
    
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    
    if label:
        label_size, baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        y1_label = max(y1 - 10, label_size[1] + 10)
        cv2.rectangle(
            img,
            (x1, y1_label - label_size[1] - 10),
            (x1 + label_size[0] + 10, y1_label + baseline - 5),
            color,
            cv2.FILLED,
        )
        cv2.putText(
            img,
            label,
            (x1 + 5, y1_label - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness,
        )
    return img


def draw_corners(
    image: np.ndarray,
    corners: np.ndarray,
    color: tuple = (0, 0, 255),
    radius: int = 8,
) -> np.ndarray:
    """Draw detected corners on an image."""
    img = image.copy()
    for i, corner in enumerate(corners):
        pt = tuple(corner.astype(int).flatten())
        cv2.circle(img, pt, radius, color, -1)
        cv2.putText(
            img,
            str(i + 1),
            (pt[0] + 10, pt[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )
    return img


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Overlay a heatmap on an image."""
    if len(heatmap.shape) == 2:
        heatmap_normalized = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
        heatmap_colored = cv2.applyColorMap(
            heatmap_normalized.astype(np.uint8), colormap
        )
    else:
        heatmap_colored = heatmap
    
    # Resize heatmap to match image
    heatmap_resized = cv2.resize(
        heatmap_colored, (image.shape[1], image.shape[0])
    )
    
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_resized, alpha, 0)
    return overlay


def draw_ocr_results(
    image: np.ndarray,
    ocr_results: list,
    color: tuple = (255, 165, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw OCR bounding boxes and text on an image."""
    img = image.copy()
    
    for result in ocr_results:
        bbox = result.get("bbox", [])
        text = result.get("text", "")
        conf = result.get("confidence", 0)
        
        if len(bbox) == 4:
            pts = np.array(bbox, dtype=np.int32)
            if pts.ndim == 1:
                # [x1, y1, x2, y2] format
                x1, y1, x2, y2 = pts
                cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
                label_pos = (x1, y1 - 5)
            else:
                # Polygon format [[x1,y1], [x2,y2], ...]
                cv2.polylines(img, [pts], True, color, thickness)
                label_pos = tuple(pts[0])
            
            label = f"{text} ({conf:.2f})"
            cv2.putText(
                img,
                label,
                label_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
            )
    return img


def create_analysis_grid(images: dict, max_width: int = 1200) -> np.ndarray:
    """
    Create a grid visualization of analysis stages.
    
    Args:
        images: Dict of {stage_name: image} pairs
        max_width: Maximum width of the grid
    
    Returns:
        Grid image with labeled stages
    """
    stage_images = []
    n = len(images)
    
    if n == 0:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Calculate cell size
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    cell_w = max_width // cols
    cell_h = int(cell_w * 0.75)
    
    for name, img in images.items():
        # Resize to cell size
        resized = cv2.resize(img, (cell_w - 20, cell_h - 40))
        
        # Create cell with label
        cell = np.ones((cell_h, cell_w, 3), dtype=np.uint8) * 30  # Dark background
        
        # Place image centered
        y_offset = 35
        x_offset = 10
        cell[y_offset : y_offset + resized.shape[0], x_offset : x_offset + resized.shape[1]] = resized
        
        # Add label
        cv2.putText(
            cell, name, (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
        )
        stage_images.append(cell)
    
    # Pad to fill grid
    while len(stage_images) < rows * cols:
        stage_images.append(np.ones((cell_h, cell_w, 3), dtype=np.uint8) * 30)
    
    # Assemble grid
    grid_rows = []
    for r in range(rows):
        row_images = stage_images[r * cols : (r + 1) * cols]
        grid_rows.append(np.hstack(row_images))
    
    grid = np.vstack(grid_rows)
    return grid


def save_image(image: np.ndarray, path: str) -> str:
    """Save an image to disk."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    cv2.imwrite(path, image)
    return path


def compute_ela(image: np.ndarray, quality: int = 90, scale: int = 10) -> np.ndarray:
    """
    Compute Error Level Analysis (ELA) of an image.
    ELA highlights regions that were modified after the last JPEG compression.
    
    Args:
        image: Input BGR image
        quality: JPEG compression quality (lower = more aggressive)
        scale: Amplification factor for the difference
    
    Returns:
        ELA difference image (BGR)
    """
    # Encode to JPEG in memory
    encode_param = [cv2.IMWRITE_JPEG_QUALITY, quality]
    _, encoded = cv2.imencode(".jpg", image, encode_param)
    
    # Decode back
    recompressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    
    # Compute absolute difference and amplify
    diff = cv2.absdiff(image, recompressed)
    ela = np.clip(diff * scale, 0, 255).astype(np.uint8)
    
    return ela


def compute_noise_map(image: np.ndarray) -> np.ndarray:
    """
    Compute noise residual map using high-pass filtering.
    Tampered regions often have different noise characteristics.
    
    Returns:
        Single-channel noise map (uint8)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Apply median filter to get "clean" version
    denoised = cv2.medianBlur(gray, 5)
    
    # Noise = original - denoised
    noise = cv2.absdiff(gray, denoised)
    
    # Amplify and normalize
    noise_amplified = cv2.normalize(noise, None, 0, 255, cv2.NORM_MINMAX)
    
    return noise_amplified.astype(np.uint8)
