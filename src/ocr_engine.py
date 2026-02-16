"""
DocFraudDetector — OCR Engine Module
Extracts and structures text from document images using EasyOCR.

Author: Pranav Kashyap | IIIT Dharwad
"""

import cv2
import numpy as np
import re
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


class OCREngine:
    """
    Extracts text from document images and structures it into fields.
    
    Uses EasyOCR for robust multilingual text recognition,
    with regex-based field extraction for structured documents (ID cards, etc.).
    """

    def __init__(self, languages: list = None):
        self.languages = languages or config.OCR_LANGUAGES
        self.reader = None
        self._init_reader()

    def _init_reader(self):
        """Initialize EasyOCR reader."""
        try:
            import easyocr
            self.reader = easyocr.Reader(
                self.languages,
                gpu=config.OCR_GPU,
                verbose=False,
            )
            print(f"[OCREngine] EasyOCR initialized. Languages: {self.languages}")
        except ImportError:
            print("[OCREngine] EasyOCR not installed. OCR will be unavailable.")
        except Exception as e:
            print(f"[OCREngine] Failed to initialize EasyOCR: {e}")

    def extract(self, image: np.ndarray) -> dict:
        """
        Extract text from document image.
        
        Args:
            image: BGR numpy array (preferably rectified)
            
        Returns:
            dict with keys:
                - 'raw_text': full extracted text as string
                - 'detections': list of {bbox, text, confidence} dicts
                - 'structured_fields': dict of extracted fields (name, dob, etc.)
                - 'confidence_avg': average OCR confidence
                - 'word_count': total words detected
        """
        if self.reader is None:
            return self._empty_result("EasyOCR not available")
        
        # Preprocess for better OCR
        preprocessed = self._preprocess(image)
        
        # Run OCR
        try:
            results = self.reader.readtext(preprocessed)
        except Exception as e:
            return self._empty_result(f"OCR failed: {e}")
        
        if not results:
            return self._empty_result("No text detected")
        
        # Parse results
        detections = []
        texts = []
        confidences = []
        
        for detection in results:
            bbox, text, conf = detection
            
            if conf < config.OCR_CONFIDENCE_THRESHOLD:
                continue
            
            # Convert bbox to serializable format
            bbox_serializable = [[int(p[0]), int(p[1])] for p in bbox]
            
            detections.append({
                "bbox": bbox_serializable,
                "text": text.strip(),
                "confidence": round(float(conf), 4),
            })
            texts.append(text.strip())
            confidences.append(float(conf))
        
        # Build raw text
        raw_text = " ".join(texts)
        
        # Extract structured fields
        structured_fields = self._extract_fields(raw_text, texts)
        
        # Compute average confidence
        avg_conf = np.mean(confidences) if confidences else 0.0
        
        return {
            "raw_text": raw_text,
            "detections": detections,
            "structured_fields": structured_fields,
            "confidence_avg": round(float(avg_conf), 4),
            "word_count": len(raw_text.split()),
        }

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better OCR accuracy.
        Applies denoising, contrast enhancement, and binarization.
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
        
        # CLAHE for contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # Adaptive thresholding for clean binarization
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Convert back to 3-channel (EasyOCR works with both)
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    def _extract_fields(self, raw_text: str, text_lines: list) -> dict:
        """
        Extract structured fields from raw text using regex patterns.
        
        Handles common document fields: name, DOB, ID number, address, gender.
        """
        fields = {}
        full_text = raw_text.lower()
        
        for field_name, pattern in config.FIELD_PATTERNS.items():
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                fields[field_name] = {
                    "value": value,
                    "confidence": "high" if len(value) > 2 else "low",
                }
        
        # Try to extract any numbers that look like IDs (fallback)
        if "id_number" not in fields:
            id_patterns = [
                r'\b\d{4}\s?\d{4}\s?\d{4}\b',          # Aadhaar: XXXX XXXX XXXX
                r'\b[A-Z]{5}\d{4}[A-Z]\b',              # PAN: ABCDE1234F
                r'\b[A-Z]{2}\d{2}\s?\d{8,}\b',          # Passport-like
                r'\b\d{2}/\d{2}/\d{4,}\b',              # Date-like numbers
            ]
            
            for pattern in id_patterns:
                match = re.search(pattern, raw_text, re.IGNORECASE)
                if match:
                    fields["id_number"] = {
                        "value": match.group(0),
                        "confidence": "medium",
                    }
                    break
        
        # Extract all capitalized words as potential names
        if "name" not in fields:
            name_candidates = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', raw_text)
            if name_candidates:
                # Take the longest match (most likely full name)
                best_name = max(name_candidates, key=len)
                fields["name"] = {
                    "value": best_name,
                    "confidence": "medium",
                }
        
        return fields

    def _empty_result(self, reason: str) -> dict:
        """Return an empty result when OCR fails."""
        return {
            "raw_text": "",
            "detections": [],
            "structured_fields": {},
            "confidence_avg": 0.0,
            "word_count": 0,
            "error": reason,
        }

    def extract_from_regions(self, image: np.ndarray, regions: list) -> list:
        """
        Extract text from specific regions of interest.
        
        Args:
            image: Full document image
            regions: List of [x1, y1, x2, y2] bounding boxes
        
        Returns:
            List of extraction results, one per region
        """
        results = []
        for region in regions:
            x1, y1, x2, y2 = [int(v) for v in region]
            crop = image[y1:y2, x1:x2]
            
            if crop.size == 0:
                results.append(self._empty_result("Empty region"))
                continue
            
            result = self.extract(crop)
            result["region"] = [x1, y1, x2, y2]
            results.append(result)
        
        return results


if __name__ == "__main__":
    from utils import load_image, draw_ocr_results, save_image
    
    engine = OCREngine()
    
    if len(sys.argv) > 1:
        img = load_image(sys.argv[1])
        result = engine.extract(img)
        
        print(f"Words detected: {result['word_count']}")
        print(f"Average confidence: {result['confidence_avg']:.2%}")
        print(f"\nRaw text:\n{result['raw_text']}")
        print(f"\nStructured fields:")
        for field, data in result["structured_fields"].items():
            print(f"  {field}: {data['value']} (confidence: {data['confidence']})")
        
        # Visualize
        vis = draw_ocr_results(img, result["detections"])
        save_image(vis, os.path.join(config.OUTPUT_DIR, "ocr_result.jpg"))
        print(f"\nSaved OCR visualization to {config.OUTPUT_DIR}/ocr_result.jpg")
    else:
        print("Usage: python ocr_engine.py <image_path>")
