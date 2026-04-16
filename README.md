# Decode the Beat: Spotify Genre Classification 
> *Advanced Multi-Class Classification Workflow using Class-Balanced Voting Ensemble*

---

## Objective

Build a robust machine learning classification model to accurately predict the **`track_genre`** of Spotify songs based on distinct audio features and metadata (without AutoML tools, external APIs, or pre-trained models).

**DataSprint - Data Science Club, NIST University** — [Kaggle Hackathon Link](https://www.kaggle.com/competitions/spotify-track-genre-classification-challenge)

| Attributes | Description |
|---|---|
| `track_genre` | Target categorical variable (e.g., pop, rock, classical) |
| `danceability`, `energy`, `tempo` | Core acoustic and rhythmic features |
| `acousticness`, `liveness`, `valence` | Tonal and mood indicators |
| `duration_ms`, `time_signature` | Track structural and length data |

---

## Dataset

| Metric | Value |
|---|---|
| Training tracks | **84,800** |
| Test tracks | **34,200** (unlabeled flat `test.csv`) |
| Format | `.csv` (Tabular numeric and categorical data) |

> **Features Overview**: The datasets (`train.csv`, `test.csv`) contain numerous acoustic vectors representing the track's sound profile alongside entity metadata (e.g., `artists`). The target `track_genre` is only present in the train split.

---

## Methodology

### Step 1 - Preprocessing
- Missing numeric and categorical data were handled independently utilizing median/mode-based imputation.
- Executed **Frequency Encoding** explicitly targeting the `artists` feature.
- **Normalization:** Applied `StandardScaler` to ensure features contributed uniformly across internal distance metrics and split criteria.

### Step 2 - Feature Engineering: Acoustic Intersections
Derived meaningful new dimensions from raw inputs to amplify predictive signals.

- Synthesized interaction terms representing acoustic dynamics (e.g., `energy_x_danceability`).
- Identified `mood_divergence` metrics through overlapping feature spaces.

> Enhancing the base dimensional space allowed tree-based algorithms to discover richer splits early in their decision paths.

### Step 3 - Model Architecture: Voting Ensemble Setup
Tackled complex, overlapping music genre clusters by developing a **Class-Balanced Voting Ensemble**, leveraging complementary tree structures.

**LightGBM (Gradient Boosting)** -> efficiently fits deep non-linear gradients to accurately identify dense clusters.
**Extra Trees (Extremely Randomized Trees)** -> mitigates overfitting by introducing extreme variance during feature partitioning.

> Ensemble voting captures both deep contextual gradient boundaries and generalized random boundaries for stable, generalized performance.

### Step 4 - Validation: Out-of-Fold (OOF) 

Adopted a robust Cross-Validation architecture avoiding traditional validation holdouts. 
Stratified Out-of-Fold logic evaluated metrics over the entire dataset, serving as a highly reliable generalized estimator for the unseen 34,200 test tracks.

---

## Results

*Evaluated across the entire dataset via Out-of-Fold (OOF) validation strategy*

| Metric | Score |
|---|---|
| **Accuracy Assessment** | **0.3321** |
| **Precision Aggregate** | **0.3291** |
| **F1 Score Evaluation** | **0.3276** |

---

## Repository Structure
```
Spotify Track Genre Classification
├── NOTICE
├── LICENSE 
├── README.md                               -> High-level summary (This file)
├── data/                                   -> Provided datasets (`train.csv`, `test.csv`)
├── report.txt                              -> Detailed approach and methodology report
├── submission.csv                          -> Final Kaggle prediction submission (34,200 rows)
└── track_genre_classification.ipynb        -> Full Jupyter Notebook - preprocessing, training, evaluation
```

---
