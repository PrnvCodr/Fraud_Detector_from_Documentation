"""
DocFraudDetector — Tamper Detection Module
Detects document tampering using EfficientNet CNN + Error Level Analysis + Noise Analysis.

Author: Pranav Kashyap | IIIT Dharwad
"""

import cv2
import numpy as np
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.utils import compute_ela, compute_noise_map


class TamperDetector:
    """
    Multi-technique document tamper detection system.
    
    Combines:
    1. EfficientNet-B0 CNN classifier (if trained model available)
    2. Error Level Analysis (ELA) — detects JPEG re-compression artifacts
    3. Noise consistency analysis — detects spliced regions with different noise
    4. Edge density analysis — detects unnatural boundaries from cut-paste
    
    Each technique produces a score; the final verdict is a weighted combination.
    """

    def __init__(self, model_path: str = None):
        self.model = None
        self.device = config.DEVICE
        self.model_loaded = False
        
        # Try loading trained model
        model_path = model_path or config.TAMPER_CHECKPOINT
        if os.path.exists(model_path):
            self._load_model(model_path)
        else:
            print("[TamperDetector] No trained model found. Using forensic analysis only.")

    def _load_model(self, model_path: str):
        """Load pre-trained EfficientNet tamper detection model."""
        try:
            import torch
            import timm
            
            self.model = timm.create_model(
                config.TAMPER_MODEL_NAME,
                pretrained=False,
                num_classes=config.TAMPER_NUM_CLASSES,
            )
            
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            self.model_loaded = True
            print(f"[TamperDetector] Model loaded from {model_path}")
        except Exception as e:
            print(f"[TamperDetector] Failed to load model: {e}. Using forensic analysis.")
            self.model_loaded = False

    def detect(self, image: np.ndarray) -> dict:
        """
        Analyze document image for tampering.
        
        Args:
            image: BGR numpy array of rectified document
            
        Returns:
            dict with keys:
                - 'is_tampered': bool
                - 'tamper_probability': float (0-1)
                - 'analysis': dict of individual analysis results
                - 'ela_image': ELA visualization (BGR)
                - 'noise_map': Noise analysis map
                - 'heatmap': Combined tamper heatmap
                - 'details': human-readable analysis summary
        """
        results = {}
        scores = []
        weights = []
        
        # 1. CNN prediction (if model available)
        if self.model_loaded:
            cnn_result = self._cnn_predict(image)
            results["cnn"] = cnn_result
            scores.append(cnn_result["tamper_prob"])
            weights.append(0.5)  # Highest weight for learned model
        
        # 2. Error Level Analysis
        ela_result = self._ela_analysis(image)
        results["ela"] = ela_result
        scores.append(ela_result["score"])
        weights.append(0.25 if self.model_loaded else 0.4)
        
        # 3. Noise consistency analysis
        noise_result = self._noise_analysis(image)
        results["noise"] = noise_result
        scores.append(noise_result["score"])
        weights.append(0.15 if self.model_loaded else 0.35)
        
        # 4. Edge density analysis
        edge_result = self._edge_density_analysis(image)
        results["edge"] = edge_result
        scores.append(edge_result["score"])
        weights.append(0.10 if self.model_loaded else 0.25)
        
        # Compute weighted tamper probability
        weights = np.array(weights)
        weights /= weights.sum()  # Normalize
        tamper_probability = float(np.dot(scores, weights))
        
        is_tampered = tamper_probability > config.TAMPER_THRESHOLD
        
        # Generate combined heatmap
        heatmap = self._generate_heatmap(image, ela_result, noise_result)
        
        # Build details string
        details = self._build_details(results, tamper_probability, is_tampered)
        
        return {
            "is_tampered": is_tampered,
            "tamper_probability": round(tamper_probability, 4),
            "analysis": results,
            "ela_image": ela_result["ela_image"],
            "noise_map": noise_result["noise_map"],
            "heatmap": heatmap,
            "details": details,
        }

    def _cnn_predict(self, image: np.ndarray) -> dict:
        """Run CNN model prediction."""
        import torch
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(config.TAMPER_INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = transform(img_rgb).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(tensor)
            probs = torch.softmax(output, dim=1)
            tamper_prob = probs[0, 1].item()  # Class 1 = tampered
        
        return {
            "tamper_prob": tamper_prob,
            "genuine_prob": 1.0 - tamper_prob,
            "prediction": "tampered" if tamper_prob > 0.5 else "genuine",
        }

    def _ela_analysis(self, image: np.ndarray) -> dict:
        """
        Error Level Analysis.
        Tampered regions show higher error levels after re-compression,
        since they were saved at a different compression quality.
        """
        ela_image = compute_ela(image, config.ELA_QUALITY, config.ELA_SCALE)
        
        # Convert to grayscale for scoring
        ela_gray = cv2.cvtColor(ela_image, cv2.COLOR_BGR2GRAY)
        
        # Compute statistics
        mean_ela = np.mean(ela_gray)
        std_ela = np.std(ela_gray)
        max_ela = np.max(ela_gray)
        
        # High variance in ELA suggests tampering
        # Score: normalize std to 0-1 range (empirically, std > 30 is suspicious)
        score = min(std_ela / 50.0, 1.0)
        
        # Check for localized high-ELA regions (indicates splicing)
        threshold = mean_ela + 2 * std_ela
        high_ela_mask = (ela_gray > threshold).astype(np.float32)
        high_ela_ratio = np.mean(high_ela_mask)
        
        # If high ELA is concentrated (not uniform), more likely tampered
        if 0.01 < high_ela_ratio < 0.3:
            score = min(score * 1.5, 1.0)
        
        return {
            "score": float(score),
            "mean_ela": float(mean_ela),
            "std_ela": float(std_ela),
            "max_ela": float(max_ela),
            "high_ela_ratio": float(high_ela_ratio),
            "ela_image": ela_image,
        }

    def _noise_analysis(self, image: np.ndarray) -> dict:
        """
        Noise consistency analysis.
        Natural images have uniform noise; spliced regions introduce
        noise pattern discontinuities.
        """
        noise_map = compute_noise_map(image)
        
        # Divide image into blocks and check noise consistency
        h, w = noise_map.shape
        block_size = 32
        block_stds = []
        
        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = noise_map[y : y + block_size, x : x + block_size]
                block_stds.append(np.std(block))
        
        block_stds = np.array(block_stds)
        
        if len(block_stds) == 0:
            return {"score": 0.0, "noise_map": noise_map, "noise_variance": 0.0}
        
        # High variance in block noise levels suggests tampering
        noise_variance = np.std(block_stds)
        mean_noise = np.mean(block_stds)
        
        # Coefficient of variation for noise
        cv = noise_variance / max(mean_noise, 1e-6)
        
        # Score based on coefficient of variation
        score = min(cv / 1.5, 1.0)
        
        return {
            "score": float(score),
            "noise_map": noise_map,
            "noise_variance": float(noise_variance),
            "noise_cv": float(cv),
            "block_noise_mean": float(mean_noise),
        }

    def _edge_density_analysis(self, image: np.ndarray) -> dict:
        """
        Edge density analysis.
        Copy-paste tampering often introduces unnatural edge boundaries
        where the pasted region meets the original.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Divide into blocks and check edge density distribution
        h, w = edges.shape
        block_size = 32
        densities = []
        
        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = edges[y : y + block_size, x : x + block_size]
                density = np.mean(block) / 255.0
                densities.append(density)
        
        densities = np.array(densities)
        
        if len(densities) == 0:
            return {"score": 0.0, "edge_density_std": 0.0}
        
        # High variance in edge density can indicate tampering
        edge_std = np.std(densities)
        edge_mean = np.mean(densities)
        
        # Very high or very uneven edge density suggests manipulation
        score = min(edge_std / 0.15, 1.0)
        
        return {
            "score": float(score),
            "edge_density_mean": float(edge_mean),
            "edge_density_std": float(edge_std),
        }

    def _generate_heatmap(self, image: np.ndarray, ela_result: dict, noise_result: dict) -> np.ndarray:
        """Generate a combined tamper likelihood heatmap."""
        h, w = image.shape[:2]
        
        # ELA contribution
        ela_gray = cv2.cvtColor(ela_result["ela_image"], cv2.COLOR_BGR2GRAY).astype(np.float32)
        ela_norm = cv2.normalize(ela_gray, None, 0, 1, cv2.NORM_MINMAX)
        ela_resized = cv2.resize(ela_norm, (w, h))
        
        # Noise contribution
        noise_resized = cv2.resize(
            noise_result["noise_map"].astype(np.float32), (w, h)
        )
        noise_norm = cv2.normalize(noise_resized, None, 0, 1, cv2.NORM_MINMAX)
        
        # Combine
        combined = 0.6 * ela_resized + 0.4 * noise_norm
        combined = cv2.normalize(combined, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Apply colormap
        heatmap = cv2.applyColorMap(combined, cv2.COLORMAP_JET)
        
        return heatmap

    def _build_details(self, analysis: dict, probability: float, is_tampered: bool) -> str:
        """Build human-readable analysis summary."""
        lines = []
        lines.append(f"{'=' * 50}")
        lines.append(f"  TAMPER DETECTION REPORT")
        lines.append(f"{'=' * 50}")
        lines.append(f"  Verdict: {'⚠️  TAMPERED' if is_tampered else '✅ GENUINE'}")
        lines.append(f"  Tamper Probability: {probability:.1%}")
        lines.append(f"{'─' * 50}")
        
        if "cnn" in analysis:
            cnn = analysis["cnn"]
            lines.append(f"  [CNN Model]     {cnn['prediction'].upper():>10s}  ({cnn['tamper_prob']:.1%})")
        
        ela = analysis["ela"]
        lines.append(f"  [ELA Analysis]  Score: {ela['score']:.3f}  (mean={ela['mean_ela']:.1f}, std={ela['std_ela']:.1f})")
        
        noise = analysis["noise"]
        lines.append(f"  [Noise Analysis] Score: {noise['score']:.3f}  (CV={noise.get('noise_cv', 0):.3f})")
        
        edge = analysis["edge"]
        lines.append(f"  [Edge Density]  Score: {edge['score']:.3f}  (std={edge['edge_density_std']:.4f})")
        
        lines.append(f"{'=' * 50}")
        
        return "\n".join(lines)


if __name__ == "__main__":
    from utils import load_image, overlay_heatmap, save_image
    
    detector = TamperDetector()
    
    if len(sys.argv) > 1:
        img = load_image(sys.argv[1])
        result = detector.detect(img)
        
        print(result["details"])
        
        # Save visualizations
        save_image(result["ela_image"], os.path.join(config.OUTPUT_DIR, "ela_analysis.jpg"))
        
        overlay = overlay_heatmap(img, result["heatmap"])
        save_image(overlay, os.path.join(config.OUTPUT_DIR, "tamper_heatmap.jpg"))
        
        print(f"\nSaved analysis images to {config.OUTPUT_DIR}/")
    else:
        print("Usage: python tamper_detector.py <image_path>")
