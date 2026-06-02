# pycrack — Intelligent Crack Image Analysis System

A desktop GUI application for quantitative analysis of crack patterns in 2D images. Designed for soil desiccation crack research, materials surface inspection, and geological fracture characterization.

## Features

### Binarization (7 methods)
| Method | Type | Best for |
|--------|------|----------|
| Sauvola (Recommended) | Local adaptive | Uneven illumination, textured backgrounds |
| Niblack | Local adaptive | Fine low-contrast crack networks |
| Otsu | Global automatic | Bimodal histograms with uniform lighting |
| Triangle | Global automatic | Asymmetric pixel distributions |
| Global Threshold | Manual | Full user control |
| Adaptive Mean | Local adaptive | Gradual illumination changes |
| Adaptive Gaussian | Local adaptive | Sharp illumination gradients |

Auto mode analyzes image statistics (contrast, illumination uniformity, edge density, texture complexity, histogram bimodality) and selects the optimal algorithm and parameters automatically.

### Denoising
- Median Filter — preserves crack edges while removing salt-and-pepper noise
- Gaussian Filter — soft smoothing for high-frequency noise
- Mean Filter — uniform blur for low-resolution images
- None — passthrough for already clean binary masks

### Morphological Analysis
- **Skeleton extraction** with spur pruning — topological thinning to 1-pixel-wide crack centerlines
- **Junction detection** — 4-connectivity analysis identifies branch points from crossing cracks
- **Crack segmentation** — watershed-based labeling assigns each crack a unique ID

### Quantitative Metrics
| Metric | Method |
|--------|--------|
| Area Ratio | Crack pixel count / total ROI area × 100% |
| Crack Length | Euclidean skeleton traversal (accurate, not pixel-count) |
| Average Width | Area / Length (validated against distance-transform mean) |
| Maximum Width | Maximum distance transform value along the crack mask |
| Fractal Dimension | Box-counting method with R² goodness-of-fit |
| Junction Count | Number of branch points (≥3 neighbors, 4-connectivity verified) |

### Orientation & Angles
- **Rose diagram** — crack segment orientation distribution weighted by segment length
- **Junction angle analysis** — interactive window to click junction points and measure branch angles with arc visualization

### Soil Clod Analysis
- Inverts crack mask to identify soil polygons
- Computes area, perimeter, equivalent diameter, and shape factor for each clod
- Summary statistics: count, area ratio, mean/median/min/max area, clods per unit area

### Crack Network Connectivity
Based on graph-theoretic analysis of the crack skeleton:
- Junction count (J), endpoint count (E), segment count (S), component count (C)
- Connectivity Index CI = (J − E + S) / C (CIAS 2024 aligned)
- Network density D = 2S / J(J−1)
- Euler characteristic x = C − S + J

### Junction Classification
Each junction is classified into one of four types:
- **T-type** (3 branches, one angle ≈ 180°) — secondary crack meets primary
- **Y-type** (3 branches, all angles < 165°) — synchronous cracking
- **X-type** (4 branches) — two cracks cross
- **Multi-type** (5+ branches) — complex intersection

### Calibration & ROI
- **Physical scale calibration** — measure a known distance on the image, all metrics convert to physical units (mm, μm, etc.)
- **Circular ROI** — restrict analysis to a user-defined circular region
- **Rectangular ROI** — batch processing supports crop rectangles

### Manual Editing
- Brush-based binary mask editing — paint to add or erase crack pixels
- Undo/redo support

### Batch Processing
- Process all images in a folder with consistent parameters
- Configurable output directory and export formats
- Per-image and summary CSV/Excel output

### Export
| Format | Content |
|--------|---------|
| Excel (.xlsx) | Analysis summary + per-segment metrics |
| CSV (.csv) | Tabular data for further processing |
| PDF (.pdf) | Full report with images, metrics, and tables |
| Binary Image (.png) | Processed binary mask |

## Installation

### Prerequisites
- Python 3.9 or later
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/your-username/crack-analysis.git
cd crack-analysis

# Install dependencies
pip install -r requirements.txt
```

### Launch

```bash
python -m PyCrack.main
```

Or from within the project root:

```bash
python PyPyCrack/main.py
```

## Usage Workflow

### 1. Load Image
Click **Open Image** or use the side panel. Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`.

### 2. Calibrate Scale (optional)
Click **Calibrate Scale** → draw a line on a known reference distance → enter the physical length and unit. All subsequent metrics will be reported in physical units.

### 3. Select ROI (optional)
Click **Select ROI** → draw a circle on the image to restrict analysis to a circular region. Useful for excluding edge artifacts or focusing on a specific area.

### 4. Adjust Parameters
- **Algorithm**: Select from the dropdown. Sauvola is recommended for most soil crack images.
- **Window Size** (slider): Larger windows handle uneven illumination better but lose fine detail. Smaller windows preserve thin cracks but amplify noise.
- **k parameter** (slider): Sensitivity. Lower values detect more cracks (include noise); higher values are more conservative (miss thin cracks).
- **Denoising**: Median filter with kernel size 3 is a good default.
- **Auto mode**: Toggle on to let the system analyze the image and set optimal parameters automatically. Moving any slider manually disables auto mode.

### 5. Analyze
Use the left-side rail buttons or the **Analyze** dropdown menu:

| Button | Analysis |
|--------|----------|
| A | Area Ratio |
| L | Skeleton Length |
| W | Average/Max Width |
| S | Crack Segmentation (IDs each crack) |
| R | Orientation Rose Diagram |
| F | Fractal Dimension |
| ∠ | Junction Angles (interactive) |
| C | Soil Clod Analysis |
| N | Connectivity Graph |
| J | Junction Types |

Results appear in the right-side panel and log window (Ctrl+L).

### 6. Manual Editing (optional)
Click **Manual Edit** → paint on the binary mask to add (left-click) or remove (right-click) crack pixels. Useful for touching up misclassified regions.

### 7. Export
Click **Export** → choose format (Excel, CSV, PDF, or binary image).

## Batch Processing

1. Click **Batch Process**
2. Select any sample image from the target folder
3. Configure batch parameters in the setup window:
   - **Input directory**: auto-detected from sample image location
   - **Output directory**: where results are saved
   - **Processing parameters**: inherited from current UI settings
   - **ROI options**: crop rectangle or circular mask applied to all images
   - **Export options**: choose which formats to save per image
4. Click **Start** to begin processing

Batch processing uses multi-threading — the UI remains responsive during processing. A progress window shows per-image status.

## Analysis Methods — Technical Details

### Skeleton Length Calculation
Length is computed by tracing each connected skeleton component from endpoint to endpoint using 8-connectivity traversal. Diagonal steps contribute √2; orthogonal steps contribute 1. This is more accurate than simple pixel counting, which overestimates length by ~5-15%.

### Width Calculation
Width is derived from the Euclidean distance transform. At each skeleton pixel, the distance to the nearest crack boundary is read from the precomputed distance map. Local width = 2 × distance (radius to diameter). Average width is the mean of these local widths across all skeleton pixels.

### Fractal Dimension (Box-Counting)
The image is recursively divided into boxes of sizes [2, 3, 4, 6, 8, 12, 16, 32, 64]. For each size, the number of boxes containing at least one crack pixel is counted. The negative slope of log(counts) vs log(sizes) gives the fractal dimension D. Typical values: D ≈ 1.0 (linear cracks), D ≈ 1.5-1.8 (complex crack networks), D ≈ 2.0 (space-filling patterns).

### Junction Detection
A pixel is a junction if:
1. It has ≥ 3 skeleton neighbors in the 3×3 window
2. Removing it from the local 3×3 subgraph produces ≥ 3 disconnected components (4-connectivity)

Spurs shorter than 5 pixels are pruned before detection to reduce noise.

### Connectivity Index (CI)
Following CIAS 2024 recommendations: CI = (J − E + S) / C, where J=junctions, E=endpoints, S=segments, C=components. Normalized per connected component for comparability across images.

## Project Structure

```
PyCrack/
├── __init__.py           # Package marker
├── config.py             # Global constants (thresholds, theme)
├── main.py               # GUI application entry point
├── image_processing.py   # Binarization, denoising, skeleton, junction detection, auto-parameter
├── analysis.py           # Metrics, segmentation, fractal dim, connectivity, junction classification
├── visualization.py      # Plots, charts, image display, zoom/pan
├── export.py             # Excel/CSV/PDF/binary image export
├── calibration.py        # Physical scale calibration window
├── circular_region.py    # Circular ROI selection window
├── manual_edit.py        # Manual binary mask editing window
├── batch_processing.py   # Batch processing engine
├── batch_setup.py        # Batch setup configuration window
├── batch_options.py      # Batch export options and utilities
└── requirements.txt      # Python dependencies
```

## Configuration

Edit `config.py` to adjust global defaults:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `THEME_NAME` | `"cosmo"` | ttkbootstrap theme |
| `MIN_CRACK_AREA` | `20` | Minimum crack area (px²) for noise filtering |
| `MIN_CLOD_AREA` | `50` | Minimum soil clod area (px²) |
| `JUNCTION_THRESH` | `12` | Junction detection sensitivity |
| `DEFAULT_AUTO_MODE` | `True` | Enable auto parameter recommendation on startup |

## Dependencies

| Package | Minimum Version | Purpose |
|---------|----------------|---------|
| OpenCV (opencv-python) | 4.8.0 | Image processing, thresholding, morphology, distance transform |
| NumPy | 1.24.0 | Array operations, statistics |
| Pandas | 2.0.0 | Data export, batch summary tables |
| scikit-image | 0.21.0 | Skeletonization, Sauvola/Niblack thresholding, connected components |
| SciPy | 1.10.0 | Scientific computation |
| ttkbootstrap | 1.10.0 | Modern themed GUI widgets |
| Pillow | 9.5.0 | Image I/O, PIL rendering for PDF export |
| Matplotlib | 3.7.0 | Rose diagrams, histograms, fractal plots |
| ReportLab | 4.0.0 | PDF report generation |
| openpyxl | 3.1.0 | Excel file export |

## License

This project is provided for academic and research purposes. See LICENSE file for details.

## Citation

If you use this software in your research, please cite the repository.

---

**pycrack Q** — Quantitative Crack Morphology Analysis
