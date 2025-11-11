# Map-Generation-From-Overlapping-Microscopy-Images-Using-Stitching-methods
Oficial implementation for the paper "Map Generation From Overlapping Microscopy Images Using Stitching methods"

# Patch Stitching & Quality Metrics Pipeline

End-to-end pipeline to:
1) prepare and tile large slides,  
2) stitch patches back into whole-slide images with three matchers (KAZE, SuperPoint+LightGlue, SIFT+LightGlue),  
3) geometrically rectify the stitched mosaic, and  
4) score seam quality (SSIM / Mixed Quality / Phase Correlation) and export results as JSON.

> This README documents the script you shared (as-is). Save it (e.g., as `stitch_pipeline.py`) in your repo root and follow the instructions below.

---

## Features

- **Environment bootstrap**
  - Activates `conda` env `segformer3d`
  - Clones and installs **LightGlue** in editable mode
  - Installs extra deps (`scikit-image>=0.22`, `matplotlib`, `kornia`)

- **Data preparation**
  - Optional **pad** with black borders
  - **Patch extraction** on a configurable grid with **random overlap** and **random translational offsets**
  - Optional **rename+reorder** via a filename mapping

- **Stitching back**
  - Classical **KAZE** (OpenCV)
  - **SuperPoint + LightGlue** (DL)
  - **SIFT + LightGlue** (DL)
  - Robust **RANSAC** homography + vectorized blending

- **Geometric rectification**
  - Auto-detect a purple quadrant and **warp convex quad → rectangle** (homography)

- **Quality metrics**
  - **Seam SSIM** on a K×K stitched layout
  - **Improved scan metric**: α·(intensity SSIM) + β·(edge SSIM) − γ·(grad-diff)
  - **Phase-correlation seam score** with shift penalty
  - Aggregates written to `output.json`

---

## Repo structure (suggested)

```
.
├── stitch_pipeline.py          # the provided script (this repo's main file)
├── Dataset_full/
│   └── images/                 # put your input images here
└── README.md
```

---

## Requirements

- Linux/macOS recommended (script uses `bash` to source conda)
- Python 3.8+ (tested with PyTorch + CUDA if available)
- Conda (Anaconda/Miniconda)
- GPU optional but **recommended** for SuperPoint/LightGlue

### Python dependencies (installed automatically by the script)
`torch`, `torchvision`, `opencv-python`, `numpy`, `Pillow`, `matplotlib`, `tqdm`, `scikit-image>=0.22`, `kornia`, plus **LightGlue** (editable install from GitHub)

---

## Quickstart

1) **Create/activate** the conda environment (name **must** be `segformer3d` or edit the script):

```bash
conda create -y -n segformer3d python=3.10
conda activate segformer3d

# (recommended) install torch matching your CUDA, e.g.:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

2) **Place your input images** in:

```
Dataset_full/images/
```

Accepted extensions: `.png`, `.jpg`, `.jpeg`, `.tiff`, `.bmp`, `.svs`

3) **Run the pipeline**

```bash
python stitch_pipeline.py
```

The script:
- changes directory to `/work/SebiButa/` (edit this if needed),
- installs LightGlue + deps,
- runs the **full pipeline** into:

```
Dataset_full/images_result_full/
```

---

## Configuration

Edit these constants near the top of the script:

```python
# Where to read & write
INPUT_FOLDER  = "Dataset_full/images"
OUTPUT_FOLDER = "Dataset_full/images_result_full"

# Target working dir (repo root)
target_dir = "/work/SebiButa/"  # <- change to your repo path
```

Tunable arguments inside functions (search in file):

- `pad_images(..., padding_fraction=0.2)`
- `split_image_into_patches(..., grid_size=(5,5))` (random overlaps ≈ 20%)
- `LightGlue_matrix_scan_diagonal_matching(..., percentage_of_image_used=1.0, pair_extractor="SuperPoint"|"SIFT")`
- `KAZE_matching(..., kaze_thresholding=0.7, ransac_thresholding=100)`
- `compute_stitched_ssim(..., seam_width=50)`
- `compute_improved_scan_ssim(..., patch_h=200, patch_w=200, stride=50, α=0.2, β=0.4, γ=0.4)`
- `compute_phasecorr_scan(..., patch_h=200, patch_w=200, stride=50, lam=1.0)`
- Purple-mask thresholds in `extract_corners_function` (HSV) if your slides differ

> ⚠️ Patch extraction includes randomness (overlaps, offsets). For reproducibility, add your own seeding where random is used.

---

## Output layout

```
Dataset_full/images_result_full/
├── padded_images/
├── patches/
│   ├── <image_name>_patches/                # raw grid patches
│   └── <image_name>_reordered_patches/     # renamed by name_map (if applicable)
├── KAZE/
│   └── <image_name>/
│       ├── 0000-XXXX_KAZE_0.7_1.0.jpg
│       └── homography_0000-0024_KAZE_0.7_1.0.jpg
├── SuperPoint/
│   └── <image_name>/
│       ├── 0000-XXXX_SuperPoint_1.0.jpg
│       └── homography_0000-0024_SuperPoint_1.0.jpg
├── SIFT/
│   └── <image_name>/
│       ├── 0000-XXXX_SIFT_1.0.jpg
│       └── homography_0000-0024_SIFT_1.0.jpg
└── output.json                              # metrics summary
```

**`output.json` structure (per image):**
```json
[
  {
    "Image": "MY_SLIDE",
    "KAZE": {
      "SSIM_score": 0.83,
      "MQ_score": 0.71,
      "Phase_Correlation_score": 0.65
    },
    "SuperPoint": {
      "SSIM_score": 0.87,
      "MQ_score": 0.76,
      "Phase_Correlation_score": 0.69
    },
    "SIFT": {
      "SSIM_score": 0.85,
      "MQ_score": 0.74,
      "Phase_Correlation_score": 0.67
    }
  }
]
```

Values are `-1` if a stitched/homography image was missing.

---

## How it works (high-level)

1) **Padding & Patching**
   - Adds thin black borders (optional) and splits each input into a `5×5` grid with randomized overlap/offsets, then optionally renames via `name_map`.

2) **Pairwise Stitching**
   - **KAZE path** (OpenCV): KAZE → BFMatcher+ratio → RANSAC → warp & blend.
   - **DL paths**: SuperPoint or SIFT (via LightGlue’s wrappers) → LightGlue matching → RANSAC → warp & blend.
   - For DL routes, a **diagonal traversal** of the grid with local sub-window matching helps robustness.

3) **Rectification**
   - Detects a purple tissue quadrilateral in HSV, orders corners, and warps convex quad to a clean rectangle.

4) **Scoring**
   - **Seam SSIM** on K×K grid interfaces.
   - **Improved scan metric** combining intensity SSIM, edge SSIM, and gradient difference.
   - **Phase correlation** across seams with shift penalty.
   - Aggregates and writes **`output.json`**.

---

## Tips & Troubleshooting

- **Change paths**: Update `target_dir` (`/work/SebiButa/`) and IO folders for your machine.
- **Large slides**: The script sets `PIL.Image.MAX_IMAGE_PIXELS = 933120000`. Ensure you have enough RAM/VRAM.
- **CUDA OOM**: Reduce image scale (see `read_image` and `resize_torch`), decrease `grid_size`, or use CPU matchers (KAZE).
- **Corner detection fails**: Tweak HSV purple thresholds in `extract_corners_function`.
- **Different grids**: Adjust `grid_size` and the `diagonal_submatrix` / `poz_list` logic accordingly.
- **Reproducibility**: Add `random.seed()` and `np.random.seed()` if you need deterministic overlaps/offsets.

---

## Acknowledgements

- **[LightGlue](https://github.com/cvg/LightGlue)**
- **OpenCV**, **PyTorch**, **Kornia**, **scikit-image**, **Matplotlib**

---

## License

Add your project’s license here (e.g., MIT). Note that LightGlue and other dependencies have their own licenses—review them before distribution.

---

## Citation

If you use **LightGlue**, please cite the authors as per their repository instructions. For your own paper/project, include an appropriate citation to this repo and any datasets used.
