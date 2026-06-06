# PyCrackQ

PyCrackQ is a Python-based desktop GUI application for quantitative analysis of soil desiccation-crack images. It provides an integrated workflow for crack segmentation, skeleton extraction, morphological measurement, branch-level analysis, junction classification, soil-clod analysis, batch processing, and result export.

## Installation

```bash
git clone https://github.com/DZ-LT1/PyCrackQ.git
cd PyCrackQ
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

### Basic workflow

1. **Open Image** — loads `.jpg`, `.jpeg`, `.png`, `.bmp`, or `.tif` files.
2. **Calibrate Scale** (optional) — draw a line on a known reference distance and enter its physical length and unit. Metrics are then reported in physical units.
3. **Select ROI** (optional) — draw a circular region to restrict analysis.
4. **Adjust Parameters** — choose a binarization method (Sauvola, Niblack, Otsu, Triangle, Global Threshold, Adaptive Mean, Adaptive Gaussian), window size, sensitivity, and denoising filter. Auto mode analyzes image statistics and sets parameters automatically.
5. **Analyze** — use the rail buttons to run individual analyses.
6. **Export** — save results as Excel, CSV, PDF, or binary image.

### Analysis modules

| Button | Analysis |
|--------|----------|
| A | Area Ratio |
| L | Skeleton Length |
| W | Average/Max Width |
| S | Crack Segmentation |
| R | Orientation Rose Diagram |
| F | Fractal Dimension (box-counting) |
| ∠ | Junction Angles (interactive) |
| C | Soil Clod Analysis |
| N | Connectivity Graph |
| J | Junction Types (T/Y/X/Multi) |

### Batch processing

Click **Batch Process**, select a sample image from the target folder, configure output directory and export formats, then click **Start**. Parameters are inherited from the current UI settings.

### Manual editing

Click **Manual Edit** to toggle between repair (white) and erase (black) brush modes on the binary mask. Brush size is adjustable from 1 to 20 px.

## Analysis methods

**Skeleton length** is computed by tracing each connected skeleton component using 8-connectivity traversal. Diagonal steps contribute √2, orthogonal steps contribute 1.

**Width** is derived from the Euclidean distance transform. At each skeleton pixel, local width = 2 × distance to the nearest crack boundary.

**Fractal dimension** uses box-counting with box sizes [2, 3, 4, 6, 8, 12, 16, 32, 64]. The negative slope of log(counts) vs log(sizes) gives D.

**Junction detection** identifies pixels with ≥ 3 skeleton neighbors in the 3×3 window, where removing the pixel from the local subgraph produces ≥ 3 disconnected components (4-connectivity). Spurs shorter than 5 pixels are pruned before detection.

**Connectivity index**: CI = (J − E + S) / C, where J = junctions, E = endpoints, S = segments, C = components.

**Junction classification**: each junction is classified as T-type (3 branches, one angle ≈ 180°), Y-type (3 branches, all angles < 165°), X-type (4 branches), or Multi-type (5+ branches).

## Project structure

```
PyCrackQ/
├── main.py               # GUI entry point
├── config.py              # Global constants
├── image_processing.py    # Binarization, denoising, skeleton, junction detection
├── analysis.py            # Metrics, segmentation, fractal dimension, connectivity
├── visualization.py       # Plots, charts, image display
├── export.py              # Excel/CSV/PDF/image export
├── calibration.py         # Physical scale calibration
├── circular_region.py     # Circular ROI selection
├── manual_edit.py         # Binary mask editing
├── batch_processing.py    # Batch processing engine
├── batch_setup.py         # Batch setup window
├── batch_options.py       # Batch export utilities
└── requirements.txt       # Dependencies
```

## Configuration

Edit `config.py` to adjust defaults:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `THEME_NAME` | `"cosmo"` | ttkbootstrap theme |
| `MIN_CRACK_AREA` | `20` | Minimum crack area (px²) |
| `MIN_CLOD_AREA` | `50` | Minimum soil clod area (px²) |
| `JUNCTION_THRESH` | `12` | Junction detection sensitivity |
| `DEFAULT_AUTO_MODE` | `True` | Auto parameter mode on startup |

## Dependencies

| Package | Min Version |
|---------|-------------|
| opencv-python | 4.8.0 |
| numpy | 1.24.0 |
| pandas | 2.0.0 |
| scikit-image | 0.21.0 |
| scipy | 1.10.0 |
| ttkbootstrap | 1.10.0 |
| Pillow | 9.5.0 |
| matplotlib | 3.7.0 |
| reportlab | 4.0.0 |
| openpyxl | 3.1.0 |

## License

This project is provided for academic and research purposes. See LICENSE for details.

## Citation

If you use PyCrackQ in your research, please cite the related SoftwareX paper and this repository.

```bibtex
@software{pycrackq_2026,
  title = {PyCrackQ: A Python-Based Software for Quantitative Analysis of Soil Desiccation-Crack Images},
  author = {Zhang, Bo and Hu, Feiyang and Zhang, Dexuan and Chen, Xu and Zhu, Zhehao and Duishanalieva Aisin},
  year = {2026},
  version = {1.0.1},
  url = {https://github.com/DZ-LT1/PyCrackQ}
}
