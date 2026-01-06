from __future__ import annotations
from io import BytesIO
from typing import Literal
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
import raster_contrast_enhancement as rce

# --- TYPE DEFINITIONS ---
# Added new methods from the lecture slides
EnhancementMethod = Literal[
    "Histogram Equalization",
    "Contrast Stretching",
    "Gamma Correction",
    "Logarithmic Transform",
    "Negative Image",
    "Thresholding",
    "Bit-Plane Slicing"
]

# --- CORE LOGIC WRAPPER ---
def enhance_image(image: Image.Image, method: EnhancementMethod, **kwargs) -> Image.Image:
    # Convert PIL to Numpy
    if image.mode == "L":
        arr = np.asarray(image)
        is_gray = True
    else:
        rgb = image.convert("RGB")
        arr = np.asarray(rgb)
        is_gray = False

    # Apply Techniques
    if method == "Histogram Equalization":
        out = rce.global_histogram_equalization(arr)
        
    elif method == "Contrast Stretching":
        out = rce.linear_contrast_stretching(arr)
        
    elif method == "Gamma Correction":
        # Slide 15-21: Power Law
        gamma_val = kwargs.get("gamma", 1.0)
        out = rce.gamma_correction(arr, gamma=gamma_val)
        
    elif method == "Logarithmic Transform":
        # Slide 12-14: Log Transform
        out = rce.logarithmic_transformation(arr)
        
    elif method == "Negative Image":
        # Slide 6-7: Negative
        out = rce.image_negative(arr)
        
    elif method == "Thresholding":
        # Slide 8-9: Thresholding
        thresh = kwargs.get("threshold", 128)
        out = rce.thresholding(arr, threshold=thresh)
        
    elif method == "Bit-Plane Slicing":
        # Project Title Requirement: Bit Techniques
        plane = kwargs.get("bit_plane", 7)
        # Shift bit to position 0, isolate it, then scale to 255
        out = ((arr >> plane) & 1) * 255
        # Ensure output is uint8
        out = out.astype(np.uint8)
        
    else:
        # Fallback
        out = arr

    # Convert back to PIL
    mode = "L" if is_gray else "RGB"
    return Image.fromarray(out, mode=mode)

# --- HELPER FUNCTIONS ---
def _open_image(uploaded_file) -> Image.Image:
    image = Image.open(BytesIO(uploaded_file.getvalue()))
    image = ImageOps.exif_transpose(image)
    return image

def _pil_to_numpy_for_hist(image: Image.Image) -> np.ndarray:
    if image.mode == "L":
        return np.asarray(image)
    return np.asarray(image.convert("RGB"))

def _plot_histogram(image: Image.Image, title: str) -> plt.Figure:
    histograms = rce.histogram_data(_pil_to_numpy_for_hist(image))
    fig, ax = plt.subplots(figsize=(6.2, 2.6), dpi=160)
    for h in histograms:
        ax.plot(h.bins, h.counts, color=h.color, linewidth=1.3, alpha=0.9, label=h.label)
    ax.set_title(title, fontsize=11, pad=8)
    ax.set_xlim(0, 255)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    fig.tight_layout()
    return fig

def _inject_css() -> None:
    st.markdown("""
        <style>
          .block-container { padding-top: 2rem; }
          .app-title { font-size: 1.65rem; font-weight: 650; }
          .panel { border: 1px solid rgba(49, 51, 63, 0.15); border-radius: 14px; padding: 1rem; }
        </style>
        """, unsafe_allow_html=True)

# --- MAIN APP ---
def main() -> None:
    st.set_page_config(page_title="Raster Bit Contrast Enhancement", layout="wide")
    _inject_css()

    st.markdown('<div class="app-title">Contrast Enhancement (Raster Bit Techniques)</div>', unsafe_allow_html=True)

    # Sidebar Controls
    with st.sidebar:
        st.header("Control Panel")
        uploaded = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg", "bmp"])
        
        # Updated Options to match Lecture Slides
        method = st.selectbox(
            "Enhancement Technique",
            options=[
                "Histogram Equalization", 
                "Contrast Stretching", 
                "Gamma Correction", 
                "Logarithmic Transform", 
                "Negative Image", 
                "Thresholding",
                "Bit-Plane Slicing"
            ]
        )

        # Dynamic Controls based on selection
        params = {}
        if method == "Gamma Correction":
            params["gamma"] = st.slider("Gamma Value (γ)", 0.1, 3.0, 1.2, 0.1, help="< 1: Brighter | > 1: Darker")
        
        elif method == "Bit-Plane Slicing":
            st.info("Extracts a specific bit-layer from the image pixels.")
            params["bit_plane"] = st.slider("Bit Plane", 0, 7, 7, help="7 = MSB (Structure), 0 = LSB (Noise)")
            
        elif method == "Thresholding":
            st.info("Pixels > Threshold = White (255), else Black (0).")
            params["threshold"] = st.slider("Threshold Value", 0, 255, 128)

        elif method == "Logarithmic Transform":
            st.info("Expands the values of dark pixels. Useful for high dynamic range images.")

        elif method == "Negative Image":
            st.info("Inverts pixel values (s = 255 - r). Useful for medical images.")

    # Main Area
    if uploaded is None:
        st.info("👈 Please upload an image to start.")
        return

    original = _open_image(uploaded)
    
    # Process Image
    try:
        processed = enhance_image(original.copy(), method, **params)
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return

    # Display Columns
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Original")
        st.image(original, use_container_width=True)
        st.pyplot(_plot_histogram(original, "Original Histogram"))
    
    with col2:
        st.markdown(f"### Processed ({method})")
        st.image(processed, use_container_width=True)
        st.pyplot(_plot_histogram(processed, "Processed Histogram"))

if __name__ == "__main__":
    main()