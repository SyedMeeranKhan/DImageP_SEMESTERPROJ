# Contrast Enhancement Using Raster Bit Techniques

Web-based UI for a university image processing project focused on raster (pixel-wise) contrast enhancement methods.

## What this project does

The enhancement techniques are *raster-based* (“Raster Bit Techniques”) because they operate directly on the image raster by transforming individual pixel intensity values (0–255) for each pixel/channel.

Implemented techniques:
- Global Histogram Equalization
- Linear Contrast Stretching
- Gamma Correction
- Logarithmic Transform
- Negative Image
- Thresholding
- Bit-Plane Slicing

## Project structure

- [app.py]: Streamlit UI (upload + side-by-side original/processed + histograms)
- [raster_contrast_enhancement.py]: Processing backend (OpenCV + NumPy)

## Requirements

- Python 3.10+ recommended
- Node.js + npm (used only to bootstrap Python dependencies)

Install dependencies:
```bash
npm i
```

## Run the app

```bash
npm run start
```

Open the local URL printed in the terminal.

## Notes (developer)

- Image input formats supported in the UI: JPG/JPEG, PNG, TIFF/TIF, BMP, WebP.
- Grayscale and RGB are both supported:
  - Grayscale images are processed directly.
  - Color images are treated as RGB; histogram equalization is applied on the luminance channel (Y) in YCrCb to reduce color shifts.
- **Gamma Correction**: Now includes a dynamic slider in the sidebar to adjust $\gamma$.
- **Thresholding**: Interactive slider to set the cutoff value (0-255).
- **Bit-Plane Slicing**: Slider to select the specific bit plane (0-7).
