# Brain MRI Tumor Classification 
> *Multi-class classification using HOG + LBP feature fusion with SVM*

---

## Objective

Classify brain MRI scans into **one of four categories** using classical machine learning  (without any deep learning or pre-trained models) <br>

**Brain Tumor** — [Kaggle Hackathon Link](https://www.kaggle.com/competitions/datasprint)

| Class | Description |
|---|---|
| `glioma_tumor` | MRI scans showing glioma tumors |
| `meningioma_tumor` | MRI scans showing meningioma tumors |
| `pituitary_tumor` | MRI scans showing pituitary tumors |
| `no_tumor` | MRI scans of healthy brains |

---

## Dataset

| Metric | Value |
|---|---|
| Training images | **2,007** (across 4 classes) |
| Test images | **863** (unlabeled, flat folder) |
| Format | Grayscale PNG / JPG, varying resolutions |
| Class balance | ~70% tumor classes, ~30% no_tumor |

> **Imbalanced dataset** - tumor classes make up ~70% of training data, with varying distribution across glioma, meningioma, and pituitary types. Labels are embedded directly in filenames (e.g. `00001_image(45)_glioma_tumor.png`).

---

## Methodology

### Step 1 - Label Extraction from Filenames

Labels were parsed directly from filenames by matching known class suffixes. No separate label file was required.

```
00001_gg(399)_glioma_tumor.png  →  glioma_tumor
```

### Step 2 - Preprocessing

- All images **resized to 128×128 pixels**
- Converted to **grayscale**
- **Normalized to [0, 1]** for consistent feature extraction

### Step 3 - Feature Engineering: HOG + LBP Fusion

Two complementary feature descriptors were combined into a single **8,260-dimensional feature vector**.

**HOG (Histogram of Oriented Gradients)** -> captures edge and shape structure
- 9 orientations
- 8×8 pixels per cell
- 2×2 cells per block
- L2-Hys normalization

**LBP (Local Binary Pattern)** - captures local texture patterns
- P=8, R=1, uniform method
- Applied over a 4×4 spatial grid

> HOG and LBP are *complementary* -> HOG captures global shape/edge structure while LBP captures fine-grained local texture. Their fusion produces a richer feature representation for MRI classification.

**Combined feature vector: HOG + LBP = 8,260 dims**

### Step 4 - Data Augmentation

Horizontal flips were applied to **training images only** (not validation), doubling the effective training size.

| Stage | Count |
|---|---|
| Original train split | 1,605 |
| After horizontal flip | 3,210 |
| **Final (all 2,007 + 2,007 flipped)** | **4,014 samples** |

### Step 5 - Dimensionality Reduction: PCA

- `StandardScaler` applied before PCA *(required for SVM)*
- Top **300 principal components** retained
- **Explained variance retained: 69.54%**

> Reduces the 8,260-dim feature space significantly while preserving the most discriminative structure, speeding up SVM training without sacrificing meaningful signal.

### Step 6 - Class Imbalance Handling

- `class_weight="balanced"` applied to SVM -> penalizes minority class misclassifications proportionally
- **Stratified 80/20 train/validation split** -> ensures equal class representation across both sets

### Step 7 - Model Selection: SVM with GridSearchCV

SVM with RBF kernel tuned via **5-fold stratified cross-validation**.

**Search space:**

| Hyperparameter | Values searched |
|---|---|
| C | {1, 10, 50, 100} |
| gamma | {scale, auto, 0.001, 0.005} |

**Best parameters found:**

| Hyperparameter | Best value |
|---|---|
| C | **10** |
| gamma | **scale** |

> **Final model: SVM · RBF kernel · C=10 · gamma=scale**

---

## Results

*Evaluated on validation set - 402 original held-out images*

| Metric | Score |
|---|---|
| **Accuracy** | **0.9129** |
| **Precision** *(weighted)* | **0.9126** |
| **F1 Score** *(weighted)* | **0.9124** |

---

## Repository Structure
```
Brain MRI Tumor Classification
├── NOTICE
├── LICENSE 
├── README.md 
├── report.txt                              -> Whole Approach Report
├── submission.csv                          -> 863-row prediction file with `image_id` and `label` columns
└── brain-mri-tumor-classification.ipynb    -> Full notebook - preprocessing, feature extraction, training, and evaluation
```

---
