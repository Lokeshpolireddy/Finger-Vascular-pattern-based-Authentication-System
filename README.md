# Finger Vein Classification

This project implements a Single Stream ResNet-18 Classification Model for finger vein recognition. The approach treats the problem as a standard image classification task, pooling 6 versions of each image (from the subject's left/right index, middle, and ring fingers) into a single dataset.

The model relies entirely on **Classification** (Softmax) for verification, taking a single raw image as input to predict the unique finger identity out of 600 possible classes (SDUMLA dataset).

# SDUMLA Dataset Enhancement Versions (v1 - v6)

This document serves as the formal reference outlining the progressive computer vision enhancement pipelines applied to generate the six variant iterations of the SDUMLA-HMT finger vein dataset. These variants were developed to empirically test baseline neural network recognition capability against increasingly aggressive background elimination and high-contrast vesselness isolation strategies.

## Dataset Structure & Statistics
**Total Images per Variant:** 3,816 `.png` files (106 Subjects x 6 Fingers x 6 Captures)

Each of the six dataset variations (e.g., `SDUMLA-HMT_Preprocessed_v1` through `v6`) shares the exact identical nested directory structure, mapped from the raw dataset conventions to facilitate automated PyTorch `ImageFolder` or custom PyTorch `Dataset` parsing:

```text
SDUMLA-HMT_Preprocessed_v[X]/
├── 001/                        # Subject ID (001 - 106)
│   ├── L_Fore/                 # Left Index Finger
│   │   ├── index_1.png         # Capture 1
│   │   ├── ...
│   │   └── index_6.png         # Capture 6
│   ├── L_Middle/               # Left Middle Finger
│   │   ├── middle_1.png
│   │   └── ...
│   ├── L_Ring/                 # Left Ring Finger
│   ├── R_Fore/                 # Right Index Finger
│   ├── R_Middle/               # Right Middle Finger
│   └── R_Ring/                 # Right Ring Finger
├── 002/
...
└── 106/
```

## Dynamic Pre-Processing Constraint (Universal 224x72 Crop)
**UPDATE:** To ensure an exact, apples-to-apples comparison across all six variants, Versions `v1` through `v4` were retroactively regenerated to use the exact same dynamic 224x72 mathematical Region-of-Interest extraction as `v5` and `v6`.

**Advanced Mathematical ROI Crop**:
- Image normalization -> Guassian Blur `(9,9)`.
- **Horizontal Projection:** `mean(projection) * 0.9` -> Yields dynamic Y-axis bounds.
- **Vertical Projection:** `mean(projection) * 9.0` -> Yields dynamic X-axis bounds.
- **Boundary Trimming:** 15% deterministic chop from the computed Top and Bottom boundaries to aggressively destroy lamp reflections.
- Resize to rigid `224x72` aspect-ratio geometry.

---

## 🔹 Version 1: `SDUMLA-HMT_Preprocessed_v1`
**Objective:** Baseline contrast enhancement via histogram equalization.

**Pipeline sequence:**
1. Advanced Mathematical ROI Crop (224x72)
2. Normalization (Linear scaling from 0 to 255)
3. CLAHE (Contrast Limited Adaptive Histogram Equalization)
   - `clipLimit` = 3.0
   - `tileGridSize` = (8,8)

---

## 🔹 Version 2: `SDUMLA-HMT_Preprocessed_v2`
**Objective:** Eliminate large-scale illumination bias and sensor hot-spots.

**Pipeline sequence:**
1. Base: `V1` Images (Normalized + CLAHE)
2. Gaussian Blur to estimate illumination background
   - Kernel size = `(51, 51)`
3. Subtraction `(Background - Original)`
4. Normalization (0-255 scale)

---

## 🔹 Version 3: `SDUMLA-HMT_Preprocessed_v3`
**Objective:** Topographical Vesselness mapping.

**Pipeline sequence:**
1. Base: `V2` Images (Normalized + CLAHE + Illumination Correction)
2. Float32 conversion (image / 255.0)
3. Multi-scale Frangi filter (Hessian-based ridge detection)
   - `sigmas` = range(2, 5)
   - `black_ridges` = False
4. Re-normalization strictly to 0-255 scale (`uint8`)

---

## 🔹 Version 4: `SDUMLA-HMT_Preprocessed_v4`
**Objective:** Darken false-positives and tighten the dynamic range gradient.

**Pipeline sequence:**
1. Base: `V3` Images (Frangi output)
2. Gamma Correction
   - `gamma` = 0.7

---

## 🔹 Version 5: `SDUMLA-HMT_Preprocessed_v5`
**Objective:** Combine the perfect mathematical bounds with a research-grade feature sequence (fixing the order of operations from v2).

**Pipeline sequence:**
1. Advanced Mathematical ROI Crop (From raw SDUMLA, 224x72)
2. Illumination Normalization (Pre-CLAHE)
   - Gaussian Blur Estimate `(61, 61)`
   - Weighted blend and intensity shift to 128.
3. CLAHE
   - `clipLimit` = 2.0 (softer than v1 to prevent noise amplification)
   - `tileGridSize` = (8,8)
4. Gamma Correction
   - `gamma` = 0.7

---

## 🔹 Version 6: `SDUMLA-HMT_Preprocessed_v6`
**Objective:** Construct a purely isolated Black Background / Glowing Veins map.

**Pipeline sequence:**
1. Base: `V5` Processed Images (Advanced ROI + Illumination/CLAHE/Gamma)
2. Complete Image Inversion `bitwise_not()` (Veins glow, background darkens)
3. Adaptive Binarization
   - `cv2.adaptiveThreshold`
   - block size = 31
   - C = -5
4. Frangi Filter (Applied to the pure Binary Mask)
   - `sigmas` = range(2, 5)
   - `black_ridges` = False
5. Normalization back to `0-255` uint8.

## Features
- **Data Loaders & Transform**: Customizable loaders for handling finger vein images, automatically inferring class mapping based on directory structure.
- **Model**: Pre-trained ResNet-18 acting as a backbone, fine-tuned on finger vein images with a custom 512-dimensional embedding layer.
- **Evaluation**: Calculates Test Loss, Test Accuracy, and generates Precision, Recall, F1-scores, and comprehensive ROC curves containing Equal Error Rate (EER) analysis at optimal Thresholds.
- **Support for Two Datasets**:
  - SDUMLA-HMT Database
  - MMCBNU_6000 Database
- Automated train/test splits (80%-20%) per class.

## Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd "Finger Vascular pattern based Authentication System"
   ```

2. **Install requirements:**
   Install dependencies via pip:
   ```bash
   pip install -r requirements.txt
   ```

3. **Data Configuration:**
   - Update `BASE_DIR_ROOT` and `WORK_DIR_ROOT` in the notebook to match where the datasets are stored.
   - Adjust `USE_SDUMLA=True/False` to toggle between the SDUMLA and MMCBNU databases within the second code cell of the notebook.

## Usage
The entire workflow is self-contained within the `finger_vein_authentication_resnet18.ipynb` notebook.
- Open the notebook using Jupyter:
  ```bash
  jupyter notebook finger_vein_authentication_resnet18.ipynb
  ```
- Run the cells sequentially to build the dataset, instantiate the ResNet-18 model, train for 10 epochs, and evaluate the final test metrics.

## Logs & Results
During training, multiple files will be created in an experiments folder under `WORK_DIR_ROOT`:
- `training_log.csv`: Epoch-by-epoch tracking of Train/Test Accuracy and Loss.
- `best_classification_model.pth`: Saved model weights of the best performing epoch.
- `final_classification_model.pth`: Saved model weights from the last epoch.
- `roc_curve_full_data.csv`: Output of false acceptance/rejection metrics data.
