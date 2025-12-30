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


def _ensure_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image

    if np.issubdtype(image.dtype, np.floating):
        maxv = float(np.nanmax(image)) if image.size else 0.0
        if maxv <= 1.0:
            scaled = image * 255.0
        else:
            scaled = image
        clipped = np.clip(scaled, 0.0, 255.0)
        return clipped.astype(np.uint8)

    maxv = int(image.max()) if image.size else 0
    minv = int(image.min()) if image.size else 0
    if maxv <= 255 and minv >= 0:
        clipped = np.clip(image, 0, 255)
        return clipped.astype(np.uint8)

    if maxv == minv:
        return np.zeros(image.shape, dtype=np.uint8)

    scaled = (image.astype(np.float32) - float(minv)) * (255.0 / float(maxv - minv))
    return np.clip(scaled, 0, 255).astype(np.uint8)


def _validate_image_shape(image: np.ndarray) -> None:
    if image.ndim == 2:
        return
    if image.ndim == 3 and image.shape[2] == 3:
        return
    raise ValueError("Expected grayscale (H,W) or RGB color (H,W,3) image.")


def histogram_data(image: np.ndarray) -> list[HistogramData]:
    image_u8 = _ensure_uint8(image)
    _validate_image_shape(image_u8)

    if image_u8.ndim == 2:
        counts, bin_edges = np.histogram(image_u8.ravel(), bins=256, range=(0, 255))
        return [
            HistogramData(
                bins=bin_edges[:-1],
                counts=counts,
                label="Gray",
                color="#4c566a",
            )
        ]

    colors = ["#d62728", "#2ca02c", "#1f77b4"]
    labels = ["Red", "Green", "Blue"]
    output: list[HistogramData] = []
    for channel_index, (label, color) in enumerate(zip(labels, colors, strict=True)):
        values = image_u8[:, :, channel_index].ravel()
        counts, bin_edges = np.histogram(values, bins=256, range=(0, 255))
        output.append(HistogramData(bins=bin_edges[:-1], counts=counts, label=label, color=color))
    return output


def global_histogram_equalization(image: np.ndarray) -> np.ndarray:
    """
    Raster Bit Technique:
    This operation enhances contrast by directly re-mapping pixel intensity values
    (0-255) using the image histogram. It works at the raster (pixel) level by
    modifying intensity values for individual pixels.
    """

    image_u8 = _ensure_uint8(image)
    _validate_image_shape(image_u8)

    if image_u8.ndim == 2:
        return cv2.equalizeHist(image_u8)

    ycrcb = cv2.cvtColor(image_u8, cv2.COLOR_RGB2YCrCb)
    y = ycrcb[:, :, 0]
    y_eq = cv2.equalizeHist(y)
    ycrcb[:, :, 0] = y_eq
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


def linear_contrast_stretching(image: np.ndarray) -> np.ndarray:
    """
    Raster Bit Technique:
    This operation linearly re-scales pixel values based on min/max intensity,
    directly updating each pixel's intensity in the raster grid.
    """

    image_u8 = _ensure_uint8(image)
    _validate_image_shape(image_u8)

    if image_u8.ndim == 2:
        imin = int(image_u8.min())
        imax = int(image_u8.max())
        if imax == imin:
            return image_u8.copy()
        scaled = (image_u8.astype(np.float32) - imin) * (255.0 / (imax - imin))
        return np.clip(scaled, 0, 255).astype(np.uint8)

    out = np.empty_like(image_u8)
    for c in range(3):
        channel = image_u8[:, :, c]
        imin = int(channel.min())
        imax = int(channel.max())
        if imax == imin:
            out[:, :, c] = channel
            continue
        scaled = (channel.astype(np.float32) - imin) * (255.0 / (imax - imin))
        out[:, :, c] = np.clip(scaled, 0, 255).astype(np.uint8)
    return out


def gamma_correction(image: np.ndarray, gamma: float) -> np.ndarray:
    """
    Raster Bit Technique:
    Gamma correction applies a non-linear transform to each pixel intensity value:
    output = 255 * (input/255) ** gamma. This is a per-pixel raster operation.
    """

    if gamma <= 0:
        raise ValueError("gamma must be > 0")

    image_u8 = _ensure_uint8(image)
    _validate_image_shape(image_u8)

    lut = np.array([((i / 255.0) ** gamma) * 255.0 for i in range(256)], dtype=np.float32)
    lut_u8 = np.clip(lut, 0, 255).astype(np.uint8)
    return cv2.LUT(image_u8, lut_u8)
