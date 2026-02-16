"""
DocFraudDetector — Synthetic Document Image Generator
Generates realistic synthetic document images with controlled tampering for training.

Key features:
- Creates genuine document-like images with structured fields
- Applies various tamper types: text replacement, font mismatch, copy-paste,
  blur injection, noise injection
- Adds realistic augmentations: lighting, perspective, compression artifacts
- Mirrors BigVision's approach of synthetic data generation (Case Study #6)

Author: Pranav Kashyap | IIIT Dharwad
"""

import cv2
import numpy as np
import os
import sys
import random
import json
from typing import Tuple
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


class SyntheticDocumentGenerator:
    """
    Generates synthetic document images for training the tamper detection model.
    
    Produces two categories:
    - Genuine: clean document images with realistic content
    - Tampered: documents with intentional modifications
    """

    # Font variations for document text
    FONTS = [
        cv2.FONT_HERSHEY_SIMPLEX,
        cv2.FONT_HERSHEY_DUPLEX,
        cv2.FONT_HERSHEY_COMPLEX,
        cv2.FONT_HERSHEY_TRIPLEX,
    ]
    
    # Sample names for document generation
    FIRST_NAMES = [
        "Rahul", "Priya", "Amit", "Sneha", "Vikram", "Ananya", "Rohan",
        "Kavitha", "Arjun", "Meera", "Sanjay", "Divya", "Karthik", "Neha",
        "Pranav", "Lakshmi", "Ravi", "Pooja", "Suresh", "Anjali",
    ]
    
    LAST_NAMES = [
        "Sharma", "Patel", "Kumar", "Singh", "Gupta", "Reddy", "Nair",
        "Joshi", "Deshmukh", "Pillai", "Iyer", "Verma", "Mishra",
        "Kashyap", "Doddamani", "Hegde", "Kulkarni", "Rao", "Shetty",
    ]
    
    CITIES = [
        "Bangalore", "Mumbai", "Delhi", "Hyderabad", "Chennai", "Pune",
        "Kolkata", "Ahmedabad", "Dharwad", "Hubli", "Mysore", "Mangalore",
    ]
    
    STATES = [
        "Karnataka", "Maharashtra", "Tamil Nadu", "Telangana", "Kerala",
        "Gujarat", "Rajasthan", "West Bengal", "Uttar Pradesh",
    ]
    
    DOC_TYPES = ["IDENTITY CARD", "PAN CARD", "VOTER ID", "LICENSE", "EMPLOYEE ID"]

    def __init__(self, output_dir: str = None):
        self.output_dir = output_dir or config.SYNTHETIC_DIR
        self.genuine_dir = os.path.join(self.output_dir, "genuine")
        self.tampered_dir = os.path.join(self.output_dir, "tampered")
        
        os.makedirs(self.genuine_dir, exist_ok=True)
        os.makedirs(self.tampered_dir, exist_ok=True)

    def generate_dataset(
        self,
        num_genuine: int = config.SYNTHETIC_NUM_GENUINE,
        num_tampered: int = config.SYNTHETIC_NUM_TAMPERED,
    ) -> dict:
        """
        Generate a complete training dataset.
        
        Returns:
            dict with dataset statistics and file paths
        """
        print(f"Generating {num_genuine} genuine + {num_tampered} tampered images...")
        
        genuine_paths = []
        tampered_paths = []
        
        # Generate genuine documents
        print("Generating genuine documents...")
        for i in tqdm(range(num_genuine), desc="Genuine"):
            img = self._create_document()
            img = self._apply_realistic_augmentation(img)
            
            path = os.path.join(self.genuine_dir, f"genuine_{i:04d}.png")
            cv2.imwrite(path, img)
            genuine_paths.append(path)
        
        # Generate tampered documents
        print("Generating tampered documents...")
        for i in tqdm(range(num_tampered), desc="Tampered"):
            img = self._create_document()
            img = self._apply_tamper(img)
            img = self._apply_realistic_augmentation(img)
            
            path = os.path.join(self.tampered_dir, f"tampered_{i:04d}.png")
            cv2.imwrite(path, img)
            tampered_paths.append(path)
        
        # Save dataset metadata
        metadata = {
            "total_images": num_genuine + num_tampered,
            "num_genuine": num_genuine,
            "num_tampered": num_tampered,
            "genuine_dir": self.genuine_dir,
            "tampered_dir": self.tampered_dir,
        }
        
        meta_path = os.path.join(self.output_dir, "dataset_metadata.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nDataset generated!")
        print(f"  Genuine: {num_genuine} images → {self.genuine_dir}")
        print(f"  Tampered: {num_tampered} images → {self.tampered_dir}")
        print(f"  Metadata: {meta_path}")
        
        return metadata

    def _create_document(self) -> np.ndarray:
        """Create a realistic-looking synthetic document image."""
        w, h = config.SYNTHETIC_IMAGE_SIZE
        
        # Random background (slightly off-white, varying tones)
        bg_color = random.randint(230, 250)
        img = np.ones((h, w, 3), dtype=np.uint8) * bg_color
        
        # Add subtle paper texture
        noise = np.random.normal(0, 3, (h, w, 3)).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Document type
        doc_type = random.choice(self.DOC_TYPES)
        
        # Color scheme
        header_color = random.choice([
            (139, 69, 19),   # Brown
            (0, 0, 128),     # Navy
            (0, 100, 0),     # Dark Green
            (128, 0, 0),     # Maroon
        ])
        text_color = (0, 0, 0)
        accent_color = (
            random.randint(0, 100),
            random.randint(0, 100),
            random.randint(100, 200),
        )
        
        font = random.choice(self.FONTS)
        
        # ── Header section ──
        # Header background
        cv2.rectangle(img, (0, 0), (w, 60), header_color, -1)
        cv2.putText(img, doc_type, (w // 2 - 120, 42),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)
        
        # ── Border ──
        cv2.rectangle(img, (5, 5), (w - 5, h - 5), header_color, 2)
        cv2.rectangle(img, (8, 8), (w - 8, h - 8), accent_color, 1)
        
        # ── Photo placeholder ──
        photo_x, photo_y = w - 170, 80
        photo_w, photo_h = 140, 170
        cv2.rectangle(img, (photo_x, photo_y), (photo_x + photo_w, photo_y + photo_h),
                      (180, 180, 180), -1)
        cv2.rectangle(img, (photo_x, photo_y), (photo_x + photo_w, photo_y + photo_h),
                      header_color, 2)
        cv2.putText(img, "PHOTO", (photo_x + 30, photo_y + 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
        
        # ── Generate random person data ──
        name = f"{random.choice(self.FIRST_NAMES)} {random.choice(self.LAST_NAMES)}"
        dob = f"{random.randint(1, 28):02d}/{random.randint(1, 12):02d}/{random.randint(1980, 2005)}"
        gender = random.choice(["Male", "Female"])
        id_num = self._generate_id_number()
        address = f"{random.randint(1, 999)}, {random.choice(['MG Road', 'Station Rd', 'Gandhi Nagar', 'Nehru St', 'Park Avenue'])}"
        city = random.choice(self.CITIES)
        state = random.choice(self.STATES)
        pin = f"{random.randint(400000, 600000)}"
        
        # ── Document fields ──
        y_start = 90
        line_height = 35
        label_x = 25
        value_x = 180
        
        fields = [
            ("Name:", name),
            ("Date of Birth:", dob),
            ("Gender:", gender),
            ("ID Number:", id_num),
            ("Address:", address),
            (f"", f"{city}, {state} - {pin}"),
        ]
        
        for i, (label, value) in enumerate(fields):
            y = y_start + i * line_height
            if label:
                cv2.putText(img, label, (label_x, y), font, 0.5, accent_color, 1)
            cv2.putText(img, value, (value_x, y), font, 0.55, text_color, 1)
        
        # ── Decorative elements ──
        # Watermark-like pattern
        for y_pos in range(70, h - 30, 80):
            alpha = 0.05
            overlay = img.copy()
            cv2.putText(overlay, "GOVERNMENT OF INDIA", (50, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, accent_color, 1)
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        
        # Bottom bar
        cv2.rectangle(img, (0, h - 40), (w, h), header_color, -1)
        cv2.putText(img, f"Issued: {random.randint(2015, 2024)}", (20, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(img, "Valid for 10 years", (w - 200, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Signature line
        cv2.line(img, (25, h - 80), (200, h - 80), text_color, 1)
        cv2.putText(img, "Authorized Signatory", (35, h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
        
        return img

    def _generate_id_number(self) -> str:
        """Generate a realistic-looking ID number."""
        formats = [
            lambda: f"{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}" * 5 + f"{random.randint(1000, 9999)}" + random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
            lambda: f"{random.randint(1000, 9999)} {random.randint(1000, 9999)} {random.randint(1000, 9999)}",
            lambda: f"DL-{random.randint(10, 99)}{random.randint(20100000, 20249999)}",
            lambda: f"IN{random.randint(100000, 999999)}{random.choice('ABCDEFG')}",
        ]
        return random.choice(formats)()

    def _apply_tamper(self, image: np.ndarray) -> np.ndarray:
        """Apply a random tampering operation to the image."""
        tamper_type = random.choices(
            list(config.TAMPER_TYPES.keys()),
            weights=list(config.TAMPER_TYPES.values()),
        )[0]
        
        tamper_fn = {
            "text_replacement": self._tamper_text_replacement,
            "font_mismatch": self._tamper_font_mismatch,
            "copy_paste": self._tamper_copy_paste,
            "blur_injection": self._tamper_blur_injection,
            "noise_injection": self._tamper_noise_injection,
        }
        
        return tamper_fn[tamper_type](image)

    def _tamper_text_replacement(self, image: np.ndarray) -> np.ndarray:
        """Replace text in a region with different text."""
        img = image.copy()
        h, w = img.shape[:2]
        
        # Pick a random region
        x1 = random.randint(150, w - 200)
        y1 = random.randint(70, h - 100)
        x2 = x1 + random.randint(100, 200)
        y2 = y1 + random.randint(20, 35)
        
        # White out the region
        cv2.rectangle(img, (x1, y1 - 15), (x2, y2), (240, 240, 240), -1)
        
        # Write new text with slightly different shade
        new_text = f"{random.choice(self.FIRST_NAMES)} {random.choice(self.LAST_NAMES)}"
        text_color = (random.randint(0, 30), random.randint(0, 30), random.randint(0, 30))
        cv2.putText(img, new_text, (x1, y2 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 1)
        
        return img

    def _tamper_font_mismatch(self, image: np.ndarray) -> np.ndarray:
        """Add text with a mismatched font (inconsistent typography)."""
        img = image.copy()
        h, w = img.shape[:2]
        
        x = random.randint(150, w - 250)
        y = random.randint(80, h - 80)
        
        # Use a different font and scale
        font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX  # Very different from document fonts
        scale = random.uniform(0.5, 0.8)
        
        cv2.rectangle(img, (x, y - 20), (x + 180, y + 5), (238, 238, 238), -1)
        cv2.putText(img, random.choice(self.FIRST_NAMES), (x, y),
                    font, scale, (0, 0, 0), 1)
        
        return img

    def _tamper_copy_paste(self, image: np.ndarray) -> np.ndarray:
        """Copy a region and paste it elsewhere (cloning)."""
        img = image.copy()
        h, w = img.shape[:2]
        
        # Source region
        rw = random.randint(50, 120)
        rh = random.randint(20, 50)
        sx = random.randint(20, w - rw - 20)
        sy = random.randint(60, h - rh - 60)
        
        # Copy source
        source = img[sy : sy + rh, sx : sx + rw].copy()
        
        # Paste at a different location
        dx = random.randint(20, w - rw - 20)
        dy = random.randint(60, h - rh - 60)
        
        img[dy : dy + rh, dx : dx + rw] = source
        
        return img

    def _tamper_blur_injection(self, image: np.ndarray) -> np.ndarray:
        """Apply selective blur to a region (hiding original content)."""
        img = image.copy()
        h, w = img.shape[:2]
        
        # Random region
        rw = random.randint(80, 200)
        rh = random.randint(25, 60)
        x = random.randint(20, w - rw - 20)
        y = random.randint(60, h - rh - 60)
        
        # Apply strong blur to this region
        region = img[y : y + rh, x : x + rw]
        kernel_size = random.choice([15, 21, 25])
        blurred = cv2.GaussianBlur(region, (kernel_size, kernel_size), 0)
        img[y : y + rh, x : x + rw] = blurred
        
        return img

    def _tamper_noise_injection(self, image: np.ndarray) -> np.ndarray:
        """Inject noise into a specific region."""
        img = image.copy()
        h, w = img.shape[:2]
        
        # Random region
        rw = random.randint(80, 200)
        rh = random.randint(25, 60)
        x = random.randint(20, w - rw - 20)
        y = random.randint(60, h - rh - 60)
        
        # Add noise to this region
        region = img[y : y + rh, x : x + rw].astype(np.int16)
        noise = np.random.normal(0, random.randint(20, 50), region.shape).astype(np.int16)
        noisy = np.clip(region + noise, 0, 255).astype(np.uint8)
        img[y : y + rh, x : x + rw] = noisy
        
        return img

    def _apply_realistic_augmentation(self, image: np.ndarray) -> np.ndarray:
        """
        Apply realistic augmentations to simulate real-world capture conditions.
        """
        img = image.copy()
        
        # Random brightness adjustment
        if random.random() < 0.5:
            factor = random.uniform(0.7, 1.3)
            img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        
        # Random slight rotation
        if random.random() < 0.3:
            angle = random.uniform(-5, 5)
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        
        # Random JPEG compression artifacts
        if random.random() < 0.4:
            quality = random.randint(50, 90)
            _, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
            img = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        
        # Random slight perspective distortion
        if random.random() < 0.3:
            img = self._random_perspective(img)
        
        # Random shadow
        if random.random() < 0.2:
            img = self._add_shadow(img)
        
        return img

    def _random_perspective(self, image: np.ndarray) -> np.ndarray:
        """Apply slight random perspective transform."""
        h, w = image.shape[:2]
        margin = int(min(w, h) * 0.05)
        
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = np.float32([
            [random.randint(0, margin), random.randint(0, margin)],
            [w - random.randint(0, margin), random.randint(0, margin)],
            [w - random.randint(0, margin), h - random.randint(0, margin)],
            [random.randint(0, margin), h - random.randint(0, margin)],
        ])
        
        M = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    def _add_shadow(self, image: np.ndarray) -> np.ndarray:
        """Add a realistic shadow gradient."""
        h, w = image.shape[:2]
        shadow = np.ones_like(image, dtype=np.float32)
        
        # Random gradient direction
        if random.random() < 0.5:
            for i in range(w):
                shadow[:, i, :] = 0.6 + 0.4 * (i / w)
        else:
            for i in range(h):
                shadow[i, :, :] = 0.6 + 0.4 * (i / h)
        
        result = (image.astype(np.float32) * shadow).astype(np.uint8)
        return result


def generate_sample_images():
    """Generate a small set of sample images for testing."""
    generator = SyntheticDocumentGenerator(
        output_dir=config.SAMPLE_IMAGES_DIR
    )
    
    # Create sample genuine and tampered images
    os.makedirs(os.path.join(config.SAMPLE_IMAGES_DIR, "genuine"), exist_ok=True)
    os.makedirs(os.path.join(config.SAMPLE_IMAGES_DIR, "tampered"), exist_ok=True)
    
    print("Generating sample test images...")
    
    for i in range(5):
        # Genuine
        img = generator._create_document()
        path = os.path.join(config.SAMPLE_IMAGES_DIR, f"genuine/sample_genuine_{i}.png")
        cv2.imwrite(path, img)
        
        # Tampered
        img = generator._create_document()
        img = generator._apply_tamper(img)
        path = os.path.join(config.SAMPLE_IMAGES_DIR, f"tampered/sample_tampered_{i}.png")
        cv2.imwrite(path, img)
    
    # Also create one main test document
    test_img = generator._create_document()
    cv2.imwrite(os.path.join(config.SAMPLE_IMAGES_DIR, "test_doc.png"), test_img)
    
    print(f"Sample images saved to {config.SAMPLE_IMAGES_DIR}/")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Synthetic Document Generator")
    parser.add_argument("--samples-only", action="store_true",
                       help="Generate only a few sample images for testing")
    parser.add_argument("--num-genuine", type=int, default=config.SYNTHETIC_NUM_GENUINE)
    parser.add_argument("--num-tampered", type=int, default=config.SYNTHETIC_NUM_TAMPERED)
    
    args = parser.parse_args()
    
    if args.samples_only:
        generate_sample_images()
    else:
        generator = SyntheticDocumentGenerator()
        generator.generate_dataset(args.num_genuine, args.num_tampered)
