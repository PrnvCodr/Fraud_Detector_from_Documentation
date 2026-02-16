"""
DocFraudDetector — Streamlit Web Demo
Interactive web-based demo showing step-by-step document analysis.

Author: Pranav Kashyap | IIIT Dharwad
"""

import streamlit as st
import cv2
import numpy as np
import os
import sys
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.pipeline import DocumentAnalysisPipeline
from src.utils import bgr_to_rgb


# ═══════════════════════════════════════════════════════════════════
# Page Configuration
# ═══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="DocFraudDetector — Document Fraud Analysis",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
    }
    .verdict-genuine {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        color: white;
    }
    .verdict-tampered {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        color: white;
    }
    .stage-header {
        color: #1E88E5;
        border-bottom: 2px solid #1E88E5;
        padding-bottom: 0.3rem;
    }
    .info-box {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1E88E5;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# Initialize Pipeline (cached)
# ═══════════════════════════════════════════════════════════════════

@st.cache_resource
def load_pipeline():
    """Load the analysis pipeline (cached for reuse)."""
    return DocumentAnalysisPipeline(use_yolo=False, verbose=False)


# ═══════════════════════════════════════════════════════════════════
# Main App
# ═══════════════════════════════════════════════════════════════════

def main():
    # Header
    st.markdown('<div class="main-header">🔍 DocFraudDetector</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">AI-Powered Document Fraud & Tamper Detection System</div>',
        unsafe_allow_html=True,
    )
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/security-checked.png", width=80)
        st.markdown("### 🛡️ About")
        st.markdown(
            "**DocFraudDetector** analyzes document images for signs of "
            "tampering or fraud using a multi-stage AI pipeline."
        )
        
        st.markdown("---")
        st.markdown("### 📋 Pipeline Stages")
        st.markdown("""
        1. 📄 **Document Detection**
        2. 📐 **Perspective Rectification**
        3. 🔬 **Tamper Analysis** (ELA + Noise)
        4. 🔤 **OCR Text Extraction**
        """)
        
        st.markdown("---")
        st.markdown("### ⚙️ Settings")
        show_detailed = st.checkbox("Show detailed metrics", value=True)
        show_json = st.checkbox("Show JSON report", value=False)
        
        st.markdown("---")
        st.markdown(
            "**Built by** [Pranav Kashyap](https://github.com/pranav-kashyap)  \n"
            "IIIT Dharwad | BigVision Internship Project"
        )
    
    # ── File Upload ──
    st.markdown("### 📤 Upload Document Image")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a document image",
            type=["jpg", "jpeg", "png", "bmp", "tiff"],
            help="Upload a photo of an ID card, PAN card, passport, or any document.",
        )
    
    with col2:
        # Sample images selector
        st.markdown("**Or try a sample:**")
        sample_dir = config.SAMPLE_IMAGES_DIR
        
        sample_options = ["— Select —"]
        if os.path.exists(sample_dir):
            for root_dir, dirs, files in os.walk(sample_dir):
                for f in files:
                    if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                        rel_path = os.path.relpath(os.path.join(root_dir, f), sample_dir)
                        sample_options.append(rel_path)
        
        selected_sample = st.selectbox("Sample images", sample_options)
    
    # Determine which image to use
    image = None
    image_name = ""
    
    if uploaded_file is not None:
        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image_name = uploaded_file.name
    elif selected_sample != "— Select —":
        sample_path = os.path.join(sample_dir, selected_sample)
        if os.path.exists(sample_path):
            image = cv2.imread(sample_path)
            image_name = selected_sample
    
    if image is None:
        st.info("👆 Upload a document image or select a sample to begin analysis.")
        
        # Show architecture info
        with st.expander("🏗️ System Architecture", expanded=True):
            st.markdown("""
            ```
            Input Image
                ↓
            ┌─────────────────────────┐
            │  1. Document Detection  │  YOLOv8 / OpenCV Contours
            │     (Localization)      │
            └──────────┬──────────────┘
                       ↓
            ┌─────────────────────────┐
            │  2. Perspective         │  OpenCV getPerspectiveTransform
            │     Rectification       │  + warpPerspective
            └──────────┬──────────────┘
                       ↓
            ┌─────────────────────────┐
            │  3. Tamper Detection    │  EfficientNet-B0 CNN
            │     • ELA Analysis      │  + Error Level Analysis
            │     • Noise Analysis    │  + Noise Consistency
            │     • Edge Density      │  + Edge Density Check
            └──────────┬──────────────┘
                       ↓
            ┌─────────────────────────┐
            │  4. OCR Extraction      │  EasyOCR (EN + HI)
            │     • Field Parsing     │  + Regex Patterns
            │     • Structured Output │
            └──────────┬──────────────┘
                       ↓
                 JSON Report
            ```
            """)
        return
    
    # ── Run Analysis ──
    st.markdown("---")
    
    # Show original image
    st.markdown(f"### 📸 Input: `{image_name}`")
    st.image(bgr_to_rgb(image), caption="Original Image", use_container_width=True)
    
    # Analyze button
    if st.button("🚀 Run Full Analysis", type="primary", use_container_width=True):
        pipeline = load_pipeline()
        
        with st.spinner("🔍 Analyzing document..."):
            progress_bar = st.progress(0, text="Initializing...")
            
            start = time.time()
            result = pipeline.analyze(image, save_results=False)
            elapsed = time.time() - start
            
            progress_bar.progress(100, text="Complete!")
        
        # ── Display Results ──
        st.markdown("---")
        
        # Verdict Banner
        summary = result["summary"]
        if summary["is_tampered"]:
            st.markdown(
                '<div class="verdict-tampered">⚠️ DOCUMENT APPEARS TAMPERED</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="verdict-genuine">✅ DOCUMENT APPEARS GENUINE</div>',
                unsafe_allow_html=True,
            )
        
        st.write("")
        
        # Metrics Row
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Tamper Probability", f"{summary['tamper_probability']:.1%}")
        m2.metric("OCR Words", result["stages"]["ocr"]["word_count"])
        m3.metric("Detection Confidence", f"{result['stages']['detection']['confidence']:.2f}")
        m4.metric("Total Time", f"{summary['total_time_ms']:.0f}ms")
        
        # ── Stage-by-Stage Results ──
        st.markdown("---")
        st.markdown("## 📊 Stage-by-Stage Analysis")
        
        # Stage 1: Detection
        with st.expander("📄 Stage 1: Document Detection", expanded=True):
            det = result["stages"]["detection"]
            c1, c2 = st.columns([2, 1])
            with c1:
                if "detection" in result["visualizations"]:
                    st.image(
                        bgr_to_rgb(result["visualizations"]["detection"]),
                        caption="Detected Document Region",
                        use_container_width=True,
                    )
            with c2:
                st.markdown(f"**Method:** `{det['method']}`")
                st.markdown(f"**Confidence:** `{det['confidence']:.4f}`")
                st.markdown(f"**Bounding Box:** `{det['bbox']}`")
                st.markdown(f"**Time:** `{det['time_ms']}ms`")
        
        # Stage 2: Rectification
        with st.expander("📐 Stage 2: Perspective Rectification", expanded=True):
            rect = result["stages"]["rectification"]
            c1, c2 = st.columns(2)
            with c1:
                if "rectified" in result["visualizations"]:
                    st.image(
                        bgr_to_rgb(result["visualizations"]["rectified"]),
                        caption="Rectified Document",
                        use_container_width=True,
                    )
            with c2:
                if "enhanced" in result["visualizations"]:
                    st.image(
                        bgr_to_rgb(result["visualizations"]["enhanced"]),
                        caption="Enhanced (CLAHE + Sharpened)",
                        use_container_width=True,
                    )
            st.markdown(f"**Output Size:** `{rect['output_size']['width']}×{rect['output_size']['height']}`  |  "
                       f"**Method:** `{rect['method']}`  |  **Time:** `{rect['time_ms']}ms`")
        
        # Stage 3: Tamper Detection
        with st.expander("🔬 Stage 3: Tamper Detection", expanded=True):
            tamper = result["stages"]["tamper_detection"]
            
            c1, c2 = st.columns(2)
            with c1:
                if "ela" in result["visualizations"]:
                    st.image(
                        bgr_to_rgb(result["visualizations"]["ela"]),
                        caption="Error Level Analysis (ELA)",
                        use_container_width=True,
                    )
            with c2:
                if "heatmap" in result["visualizations"]:
                    st.image(
                        bgr_to_rgb(result["visualizations"]["heatmap"]),
                        caption="Tamper Likelihood Heatmap",
                        use_container_width=True,
                    )
            
            if show_detailed:
                st.markdown("**Individual Analysis Scores:**")
                scores = tamper["analysis_scores"]
                score_cols = st.columns(len(scores))
                for col, (name, score) in zip(score_cols, scores.items()):
                    col.metric(name.upper(), f"{score:.3f}")
            
            st.markdown(f"**Verdict:** `{'TAMPERED' if tamper['is_tampered'] else 'GENUINE'}`  |  "
                       f"**Probability:** `{tamper['tamper_probability']:.1%}`  |  "
                       f"**Time:** `{tamper['time_ms']}ms`")
        
        # Stage 4: OCR
        with st.expander("🔤 Stage 4: OCR Text Extraction", expanded=True):
            ocr = result["stages"]["ocr"]
            
            c1, c2 = st.columns([2, 1])
            with c1:
                if "ocr" in result["visualizations"]:
                    st.image(
                        bgr_to_rgb(result["visualizations"]["ocr"]),
                        caption="OCR Detections",
                        use_container_width=True,
                    )
            with c2:
                st.markdown("**Extracted Text:**")
                if ocr["raw_text"]:
                    st.code(ocr["raw_text"], language=None)
                else:
                    st.warning("No text extracted (EasyOCR may not be installed)")
                
                if ocr["structured_fields"]:
                    st.markdown("**Structured Fields:**")
                    for field, data in ocr["structured_fields"].items():
                        st.markdown(f"- **{field.title()}:** {data['value']} "
                                  f"({data['confidence']})")
            
            st.markdown(f"**Words:** `{ocr['word_count']}`  |  "
                       f"**Confidence:** `{ocr['confidence_avg']:.1%}`  |  "
                       f"**Detections:** `{ocr['num_detections']}`  |  "
                       f"**Time:** `{ocr['time_ms']}ms`")
        
        # JSON Report
        if show_json:
            with st.expander("📋 Full JSON Report"):
                json_report = pipeline.get_json_report(result)
                st.json(json.loads(json_report))
        
        # Download JSON
        json_str = pipeline.get_json_report(result)
        st.download_button(
            "📥 Download JSON Report",
            data=json_str,
            file_name=f"analysis_{image_name.split('.')[0]}.json",
            mime="application/json",
        )


if __name__ == "__main__":
    main()
