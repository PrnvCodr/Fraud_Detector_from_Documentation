"""
DocFraudDetector — Pipeline Orchestrator
Chains all modules into a unified end-to-end document analysis pipeline.

Author: Pranav Kashyap | IIIT Dharwad
"""

import cv2
import numpy as np
import json
import time
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.detector import DocumentDetector
from src.rectifier import DocumentRectifier
from src.tamper_detector import TamperDetector
from src.ocr_engine import OCREngine
from src.utils import (
    load_image, save_image, draw_bounding_box, draw_corners,
    overlay_heatmap, draw_ocr_results, create_analysis_grid,
    bgr_to_rgb,
)


class DocumentAnalysisPipeline:
    """
    End-to-end document fraud detection pipeline.
    
    Pipeline stages:
    1. Document Detection — locate the document in the image
    2. Perspective Rectification — correct rotation/tilt/perspective
    3. Tamper Detection — analyze for signs of forgery/manipulation
    4. OCR Extraction — extract and structure text content
    
    Each stage can be run independently or as part of the full pipeline.
    """

    def __init__(self, use_yolo: bool = True, verbose: bool = True):
        self.verbose = verbose
        self._log("Initializing DocumentAnalysisPipeline...")
        
        self.detector = DocumentDetector(use_yolo=use_yolo)
        self.rectifier = DocumentRectifier()
        self.tamper_detector = TamperDetector()
        self.ocr_engine = OCREngine()
        
        self._log("Pipeline ready.")

    def analyze(self, image_input, save_results: bool = True) -> dict:
        """
        Run the full analysis pipeline on an image.
        
        Args:
            image_input: str (file path) or np.ndarray (BGR image)
            save_results: whether to save visualizations to disk
            
        Returns:
            Comprehensive analysis result dict
        """
        start_time = time.time()
        
        # Load image
        if isinstance(image_input, str):
            image = load_image(image_input)
            image_path = image_input
        elif isinstance(image_input, np.ndarray):
            image = image_input
            image_path = "uploaded_image"
        else:
            raise ValueError("image_input must be a file path (str) or numpy array")
        
        self._log(f"Image loaded: {image.shape[1]}x{image.shape[0]}")
        
        result = {
            "timestamp": datetime.now().isoformat(),
            "image_path": image_path,
            "image_size": {"width": image.shape[1], "height": image.shape[0]},
            "stages": {},
            "visualizations": {},
        }
        
        # ─── Stage 1: Document Detection ───────────────────────────
        self._log("Stage 1: Document Detection...")
        t1 = time.time()
        
        detection_result = self.detector.detect(image)
        detection_time = time.time() - t1
        
        result["stages"]["detection"] = {
            "bbox": detection_result["bbox"],
            "corners": detection_result["corners"],
            "confidence": detection_result["confidence"],
            "method": detection_result["method"],
            "time_ms": round(detection_time * 1000, 1),
        }
        
        # Visualization
        vis_detection = draw_bounding_box(
            image.copy(),
            detection_result["bbox"],
            f"Document ({detection_result['confidence']:.2f})",
            color=(0, 255, 0),
        )
        corners_array = np.array(detection_result["corners"], dtype=np.float32)
        vis_detection = draw_corners(vis_detection, corners_array)
        result["visualizations"]["detection"] = vis_detection
        
        self._log(f"  → Method: {detection_result['method']}, "
                   f"Confidence: {detection_result['confidence']:.2f}, "
                   f"Time: {detection_time * 1000:.0f}ms")
        
        # ─── Stage 2: Perspective Rectification ─────────────────────
        self._log("Stage 2: Perspective Rectification...")
        t2 = time.time()
        
        cropped = detection_result["cropped"]
        rectification_result = self.rectifier.rectify(cropped)
        rectified = rectification_result["rectified"]
        
        # Also create enhanced version
        enhanced = self.rectifier.enhance_rectified(rectified)
        rectification_time = time.time() - t2
        
        result["stages"]["rectification"] = {
            "output_size": {
                "width": rectified.shape[1],
                "height": rectified.shape[0],
            },
            "method": rectification_result["method"],
            "time_ms": round(rectification_time * 1000, 1),
        }
        
        result["visualizations"]["rectified"] = rectified
        result["visualizations"]["enhanced"] = enhanced
        
        self._log(f"  → Output: {rectified.shape[1]}x{rectified.shape[0]}, "
                   f"Time: {rectification_time * 1000:.0f}ms")
        
        # ─── Stage 3: Tamper Detection ──────────────────────────────
        self._log("Stage 3: Tamper Detection...")
        t3 = time.time()
        
        tamper_result = self.tamper_detector.detect(rectified)
        tamper_time = time.time() - t3
        
        result["stages"]["tamper_detection"] = {
            "is_tampered": tamper_result["is_tampered"],
            "tamper_probability": tamper_result["tamper_probability"],
            "analysis_scores": {
                k: v.get("score", v.get("tamper_prob", 0))
                for k, v in tamper_result["analysis"].items()
            },
            "time_ms": round(tamper_time * 1000, 1),
        }
        
        # Tamper visualizations
        result["visualizations"]["ela"] = tamper_result["ela_image"]
        result["visualizations"]["heatmap"] = overlay_heatmap(
            rectified, tamper_result["heatmap"], alpha=0.4
        )
        
        self._log(f"  → Verdict: {'TAMPERED' if tamper_result['is_tampered'] else 'GENUINE'}, "
                   f"Probability: {tamper_result['tamper_probability']:.1%}, "
                   f"Time: {tamper_time * 1000:.0f}ms")
        
        # ─── Stage 4: OCR Extraction ────────────────────────────────
        self._log("Stage 4: OCR Extraction...")
        t4 = time.time()
        
        ocr_result = self.ocr_engine.extract(enhanced)
        ocr_time = time.time() - t4
        
        result["stages"]["ocr"] = {
            "raw_text": ocr_result["raw_text"],
            "structured_fields": ocr_result["structured_fields"],
            "word_count": ocr_result["word_count"],
            "confidence_avg": ocr_result["confidence_avg"],
            "num_detections": len(ocr_result["detections"]),
            "time_ms": round(ocr_time * 1000, 1),
        }
        
        # OCR visualization
        vis_ocr = draw_ocr_results(rectified.copy(), ocr_result["detections"])
        result["visualizations"]["ocr"] = vis_ocr
        
        self._log(f"  → Words: {ocr_result['word_count']}, "
                   f"Confidence: {ocr_result['confidence_avg']:.1%}, "
                   f"Time: {ocr_time * 1000:.0f}ms")
        
        # ─── Summary ────────────────────────────────────────────────
        total_time = time.time() - start_time
        
        result["summary"] = {
            "total_time_ms": round(total_time * 1000, 1),
            "is_tampered": tamper_result["is_tampered"],
            "tamper_probability": tamper_result["tamper_probability"],
            "text_extracted": ocr_result["word_count"] > 0,
            "document_detected": detection_result["confidence"] > 0,
            "verdict": "⚠️ TAMPERED" if tamper_result["is_tampered"] else "✅ GENUINE",
        }
        
        self._log(f"\n{'=' * 50}")
        self._log(f"  ANALYSIS COMPLETE")
        self._log(f"  Verdict: {result['summary']['verdict']}")
        self._log(f"  Total time: {total_time * 1000:.0f}ms")
        self._log(f"{'=' * 50}")
        
        # Save results
        if save_results:
            self._save_results(result)
        
        return result

    def _save_results(self, result: dict):
        """Save all visualizations and JSON report."""
        output_dir = config.OUTPUT_DIR
        
        # Save visualizations
        for name, img in result["visualizations"].items():
            if isinstance(img, np.ndarray):
                path = os.path.join(output_dir, f"{name}.jpg")
                save_image(img, path)
        
        # Create and save analysis grid
        grid_images = {}
        stage_names = {
            "detection": "1. Detection",
            "rectified": "2. Rectified",
            "ela": "3. ELA Analysis",
            "heatmap": "4. Tamper Heatmap",
            "ocr": "5. OCR Results",
            "enhanced": "6. Enhanced",
        }
        
        for key, label in stage_names.items():
            if key in result["visualizations"]:
                grid_images[label] = result["visualizations"][key]
        
        if grid_images:
            grid = create_analysis_grid(grid_images)
            save_image(grid, os.path.join(output_dir, "analysis_grid.jpg"))
        
        # Save JSON report (without image arrays)
        json_result = {
            k: v for k, v in result.items() if k != "visualizations"
        }
        
        json_path = os.path.join(output_dir, "analysis_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_result, f, indent=2, ensure_ascii=False, default=str)
        
        self._log(f"Results saved to {output_dir}/")

    def get_json_report(self, result: dict) -> str:
        """Get JSON-serializable report (without image arrays)."""
        json_result = {
            k: v for k, v in result.items() if k != "visualizations"
        }
        return json.dumps(json_result, indent=2, ensure_ascii=False, default=str)

    def _log(self, message: str):
        """Print log message if verbose mode is on."""
        if self.verbose:
            print(f"[Pipeline] {message}")


if __name__ == "__main__":
    pipeline = DocumentAnalysisPipeline(use_yolo=False)
    
    if len(sys.argv) > 1:
        result = pipeline.analyze(sys.argv[1])
        print(f"\n{pipeline.get_json_report(result)}")
    else:
        print("Usage: python pipeline.py <image_path>")
        print("\nQuick test with synthetic image...")
        
        # Create a simple test document image
        test_img = np.ones((400, 600, 3), dtype=np.uint8) * 240
        cv2.putText(test_img, "IDENTITY CARD", (150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        cv2.putText(test_img, "Name: John Doe", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        cv2.putText(test_img, "DOB: 15/03/1995", (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        cv2.putText(test_img, "ID No: ABCDE1234F", (50, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        cv2.putText(test_img, "Address: 123 Main St", (50, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        cv2.rectangle(test_img, (10, 10), (590, 390), (0, 0, 0), 2)
        
        # Add a photo placeholder
        cv2.rectangle(test_img, (420, 80), (560, 250), (128, 128, 128), -1)
        cv2.putText(test_img, "PHOTO", (445, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        test_path = os.path.join(config.SAMPLE_IMAGES_DIR, "test_doc.png")
        save_image(test_img, test_path)
        print(f"Created test image: {test_path}")
        
        result = pipeline.analyze(test_path)
