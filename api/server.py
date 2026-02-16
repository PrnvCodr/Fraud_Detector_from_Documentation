"""
DocFraudDetector — FastAPI REST API
Provides HTTP endpoints for document fraud analysis.

Endpoints:
- POST /analyze      — Upload an image for full analysis
- GET  /health       — Health check
- GET  /docs         — Swagger UI (auto-generated)

Author: Pranav Kashyap | IIIT Dharwad
"""

import io
import os
import sys
import time
import base64
import numpy as np
import cv2
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.pipeline import DocumentAnalysisPipeline

# ═══════════════════════════════════════════════════════════════════
# App Initialization
# ═══════════════════════════════════════════════════════════════════

app = FastAPI(
    title="DocFraudDetector API",
    description=(
        "End-to-end document fraud & tamper detection API.\n\n"
        "Upload a document image to receive:\n"
        "- Document detection & localization\n"
        "- Perspective rectification\n"
        "- Tamper/fraud analysis with ELA & noise forensics\n"
        "- OCR text extraction with field parsing\n\n"
        "Built by **Pranav Kashyap** | IIIT Dharwad"
    ),
    version="1.0.0",
    contact={
        "name": "Pranav Kashyap",
        "url": "https://github.com/pranav-kashyap",
    },
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pipeline (lazy loading)
pipeline: Optional[DocumentAnalysisPipeline] = None


def get_pipeline() -> DocumentAnalysisPipeline:
    global pipeline
    if pipeline is None:
        pipeline = DocumentAnalysisPipeline(use_yolo=False, verbose=False)
    return pipeline


# ═══════════════════════════════════════════════════════════════════
# Response Models
# ═══════════════════════════════════════════════════════════════════

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    device: str
    version: str


class AnalysisStage(BaseModel):
    pass  # Dynamic content


# ═══════════════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════════════

def image_to_base64(image: np.ndarray, format: str = ".jpg") -> str:
    """Encode image as base64 string."""
    _, buffer = cv2.imencode(format, image)
    return base64.b64encode(buffer).decode("utf-8")


def read_upload_image(file_bytes: bytes) -> np.ndarray:
    """Convert uploaded file bytes to OpenCV image."""
    nparr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    return image


# ═══════════════════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════════════════

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        device=config.DEVICE,
        version="1.0.0",
    )


@app.post("/analyze")
async def analyze_document(
    file: UploadFile = File(..., description="Document image to analyze"),
    include_images: bool = True,
):
    """
    Analyze a document image for fraud/tampering.
    
    Upload a document image (JPG, PNG) to receive:
    - Document detection results
    - Perspective-corrected image
    - Tamper analysis with probability score
    - OCR text extraction
    
    Set `include_images=false` to exclude base64 images from response.
    """
    # Validate file
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"File must be an image. Got: {file.content_type}"
        )
    
    # Read and validate image
    contents = await file.read()
    
    if len(contents) > config.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max: {config.MAX_FILE_SIZE // (1024*1024)}MB"
        )
    
    image = read_upload_image(contents)
    
    # Run analysis pipeline
    pipe = get_pipeline()
    
    try:
        result = pipe.analyze(image, save_results=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    
    # Build response
    response = {
        "status": "success",
        "filename": file.filename,
        "summary": result["summary"],
        "stages": result["stages"],
    }
    
    # Include base64 images if requested
    if include_images:
        response["images"] = {}
        for name, img in result["visualizations"].items():
            if isinstance(img, np.ndarray):
                response["images"][name] = image_to_base64(img)
    
    return JSONResponse(content=response)


@app.get("/")
async def root():
    """API info page."""
    return {
        "name": "DocFraudDetector API",
        "version": "1.0.0",
        "author": "Pranav Kashyap | IIIT Dharwad",
        "description": "Document Fraud & Tamper Detection System",
        "endpoints": {
            "/analyze": "POST - Upload document image for analysis",
            "/health": "GET - Health check",
            "/docs": "GET - Swagger API documentation",
        },
    }


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 50)
    print("  DocFraudDetector API Server")
    print("=" * 50)
    print(f"  Host: {config.API_HOST}")
    print(f"  Port: {config.API_PORT}")
    print(f"  Docs: http://localhost:{config.API_PORT}/docs")
    print("=" * 50)
    
    uvicorn.run(
        "server:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=True,
    )
