from __future__ import annotations

from io import BytesIO
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image, ImageOps

import raster_contrast_enhancement as rce


EnhancementMethod = Literal[
    "Histogram Equalization",
    "Contrast Stretching",
    "Gamma Correction",
]

def enhance_image(image: Image.Image, method: EnhancementMethod) -> Image.Image:
    if image.mode == "L":
        arr = np.asarray(image)
        if method == "Histogram Equalization":
            out = rce.global_histogram_equalization(arr)
        elif method == "Contrast Stretching":
            out = rce.linear_contrast_stretching(arr)
        else:
            out = rce.gamma_correction(arr, gamma=1.2)
        return Image.fromarray(out, mode="L")

    rgb = image.convert("RGB")
    arr = np.asarray(rgb)
    if method == "Histogram Equalization":
        out = rce.global_histogram_equalization(arr)
    elif method == "Contrast Stretching":
        out = rce.linear_contrast_stretching(arr)
    else:
        out = rce.gamma_correction(arr, gamma=1.2)
    return Image.fromarray(out, mode="RGB")


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
    ax.set_xlabel("Intensity", fontsize=9)
    ax.set_ylabel("Frequency", fontsize=9)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def _inject_css() -> None:
    st.markdown(
        """
        <style>
          .block-container { padding-top: 2rem; padding-bottom: 2rem; }
          .app-title { font-size: 1.65rem; font-weight: 650; margin-bottom: 0.25rem; }
          .app-subtitle { color: #5b6472; margin-top: 0; }
          .panel {
            border: 1px solid rgba(49, 51, 63, 0.15);
            border-radius: 14px;
            padding: 1rem 1rem 0.75rem 1rem;
            background: rgba(255, 255, 255, 0.02);
          }
          .panel h3 { margin-top: 0.2rem; margin-bottom: 0.85rem; font-size: 1.15rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Contrast Enhancement Using Raster Bit Techniques",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_css()

    st.markdown('<div class="app-title">Contrast Enhancement Using Raster Bit Techniques</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="app-subtitle">Upload an image and compare the original and enhanced outputs with histograms.</p>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Control Panel")
        uploaded = st.file_uploader(
            "Upload Image (JPG, PNG, TIFF, BMP, WebP)",
            type=["jpg", "jpeg", "png", "tif", "tiff", "bmp", "webp"],
        )
        method: EnhancementMethod = st.selectbox(
            "Enhancement Technique",
            options=[
                "Histogram Equalization",
                "Contrast Stretching",
                "Gamma Correction",
            ],
            index=0,
        )

    if uploaded is None:
        st.info("Upload an image from the sidebar to begin.")
        col_a, col_b = st.columns(2, gap="large")
        with col_a:
            st.markdown('<div class="panel"><h3>Original Image</h3></div>', unsafe_allow_html=True)
        with col_b:
            st.markdown('<div class="panel"><h3>Processed Image</h3></div>', unsafe_allow_html=True)
        return

    original = _open_image(uploaded)
    processed = enhance_image(original.copy(), method)

    col_left, col_right = st.columns(2, gap="large")

    with col_left:
        st.markdown('<div class="panel"><h3>Original Image</h3></div>', unsafe_allow_html=True)
        st.image(original, use_container_width=True)
        st.pyplot(_plot_histogram(original, "Original Histogram"), use_container_width=True)

    with col_right:
        st.markdown('<div class="panel"><h3>Processed Image</h3></div>', unsafe_allow_html=True)
        st.image(processed, use_container_width=True)
        st.pyplot(_plot_histogram(processed, "Processed Histogram"), use_container_width=True)


if __name__ == "__main__":
    main()
