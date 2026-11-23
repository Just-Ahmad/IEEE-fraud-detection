# IEEE-CIS E-Commerce Fraud Detection

End-to-end machine learning pipeline for detecting fraudulent online transactions on the **IEEE-CIS Fraud Detection** dataset (Vesta Corporation) using **LightGBM** and time-aware cross-validation.

## Overview

The dataset spans ~6 months of e-commerce transactions (~590,540 records) with severe class imbalance (**3.5% fraud rate**), combining transactional metadata (`train_transaction`) and digital device/network footprints (`train_identity`).

Key challenges addressed in this pipeline:
- **Temporal Drift & Leakage Prevention:** Avoiding future-to-past leakage by validating strictly across non-overlapping calendar months rather than random splits.
- **Native Missing-Value & Categorical Handling:** Preserving informative `NaN` patterns (missing identity/distance fields) and casting string/object fields (`StringDtype` / `object`) to native LightGBM `category` dtypes.
- **Imbalanced Threshold Optimization:** Tuning probability cutoffs using Precision-Recall curves to balance fraud recall against false-positive alert volume.

---

## Requirements

- **Python** 3.10+
- **`lightgbm`** — gradient-boosted decision tree modeling with native categorical support
- **`scikit-learn`** — `GroupKFold` cross-validation and evaluation metrics (`roc_auc_score`, `recall_score`, `precision_recall_curve`)
- **`pandas`** — tabular data merging, memory downcasting, and time/string feature engineering *(automatically installs `numpy`)*

### Installation

Using **uv**:
```bash
uv pip install lightgbm scikit-learn pandas
