from __future__ import annotations
from dataclasses import dataclass
import cv2
import numpy as np

@dataclass(frozen=True)
class HistogramData:
    bins: np.ndarray
    counts: np.ndarray
    label: str
    color: str

# --- UTILITIES ---
def _ensure_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    if np.issubdtype(image.dtype, np.floating):
        return (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8) if image.max() <= 1.0 else np.clip(image, 0, 255).astype(np.uint8)
    return np.clip(image, 0, 255).astype(np.uint8)

def histogram_data(image: np.ndarray) -> list[HistogramData]:
    image_u8 = _ensure_uint8(image)
    if image_u8.ndim == 2:
        counts, bin_edges = np.histogram(image_u8.ravel(), bins=256, range=(0, 255))
        return [HistogramData(bins=bin_edges[:-1], counts=counts, label="Gray", color="#4c566a")]
    
    colors = ["#d62728", "#2ca02c", "#1f77b4"]
    labels = ["Red", "Green", "Blue"]
    output = []
    for i, (label, color) in enumerate(zip(labels, colors)):
        counts, bin_edges = np.histogram(image_u8[:, :, i].ravel(), bins=256, range=(0, 255))
        output.append(HistogramData(bins=bin_edges[:-1], counts=counts, label=label, color=color))
    return output

# --- LECTURE SPECIFIC ALGORITHMS ---

def image_negative(image: np.ndarray) -> np.ndarray:
    """
    Slide 6: Negative Image.
    Formula: s = L - 1 - r (where L=256)
    Useful for enhancing white/grey detail in dark regions[cite: 53].
    """
    image_u8 = _ensure_uint8(image)
    # 255 - r is the standard negative transform
    return 255 - image_u8

def thresholding(image: np.ndarray, threshold: int) -> np.ndarray:
    """
    Slide 8: Thresholding.
    Formula: s = 1.0 if r > threshold else 0.0
    Useful for segmentation/isolating objects[cite: 75].
    """
    image_u8 = _ensure_uint8(image)
    # Create binary mask: > threshold becomes 255 (White), else 0 (Black)
    _, binary = cv2.threshold(image_u8, threshold, 255, cv2.THRESH_BINARY)
    return binary

def logarithmic_transformation(image: np.ndarray) -> np.ndarray:
    """
    Slide 12: Logarithmic Transformation.
    Formula: s = c * log(1 + r)
    Expands dark pixels. Useful for Fourier Spectrums[cite: 159].
    """
    image_u8 = _ensure_uint8(image)
    # Convert to float for log calculation
    img_float = image_u8.astype(np.float32)
    
    # c is a scaling constant to ensure result fits in 0-255
    # c = 255 / log(1 + max_input_value)
    c = 255 / np.log(1 + np.max(img_float))
    
    log_image = c * np.log(1 + img_float)
    return np.clip(log_image, 0, 255).astype(np.uint8)

def gamma_correction(image: np.ndarray, gamma: float) -> np.ndarray:
    """
    Slide 15: Power Law (Gamma).
    Formula: s = c * r^gamma[cite: 185].
    """
    if gamma <= 0: return image
    image_u8 = _ensure_uint8(image)
    # Build Lookup Table (LUT) for speed
    lut = np.array([((i / 255.0) ** gamma) * 255.0 for i in range(256)], dtype=np.float32)
    lut_u8 = np.clip(lut, 0, 255).astype(np.uint8)
    return cv2.LUT(image_u8, lut_u8)

def global_histogram_equalization(image: np.ndarray) -> np.ndarray:
    image_u8 = _ensure_uint8(image)
    if image_u8.ndim == 2:
        return cv2.equalizeHist(image_u8)
    ycrcb = cv2.cvtColor(image_u8, cv2.COLOR_RGB2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)

def linear_contrast_stretching(image: np.ndarray) -> np.ndarray:
    image_u8 = _ensure_uint8(image)
    if image_u8.ndim == 2:
        imin, imax = image_u8.min(), image_u8.max()
        if imax == imin: return image_u8
        return ((image_u8 - imin) * (255.0 / (imax - imin))).astype(np.uint8)
    
    out = image_u8.copy()
    for i in range(3):
        c = image_u8[:, :, i]
        imin, imax = c.min(), c.max()
        if imax > imin:
            out[:, :, i] = ((c - imin) * (255.0 / (imax - imin))).astype(np.uint8)
    return out