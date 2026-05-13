# Features Directory

Outputs from `notebooks/features_and_lr.ipynb` (Yuzhao's pipeline).
All models in this project MUST load these files instead of regenerating
features or re-splitting data, to ensure cross-model comparisons are strict.

## File inventory

| File | Description |
|---|---|
| `X_meta_q1.npz` | (124146, 115) — Numeric + multi-hot meta features, Q1 |
| `X_meta_q2.npz` | (30621, 115) — Same feature space, Q2 subset |
| `X_tfidf_q1.npz` | (124146, 2000) — TF-IDF unigrams + bigrams, Q1 |
| `X_tfidf_q2.npz` | (30621, 2000) — Same vocabulary, Q2 subset |
| `y_q1.npy` | Binary labels: 1 = popular (top 25% by estimated owners) |
| `y_q2.npy` | Binary labels: 1 = highly_positive (≥80% positive ratio) |
| `train_idx_q1.npy` / `test_idx_q1.npy` | Shared 80/20 split for Q1 |
| `train_idx_q2.npy` / `test_idx_q2.npy` | Shared 80/20 split for Q2 |
| `appID_q1.npy` / `appID_q2.npy` | Steam app IDs (for error analysis traceability) |
| `feature_names_meta.txt` | Human-readable names for X_meta columns |
| `tfidf_vocabulary.txt` | Vocabulary of the TF-IDF vectoriser |
| `lr_results.csv` | Logistic Regression baseline results |

## How to load (copy-paste)

```python
from scipy.sparse import load_npz, hstack
import numpy as np

FEAT = "outputs/features"

# Features
X_meta_q1  = load_npz(f"{FEAT}/X_meta_q1.npz")
X_tfidf_q1 = load_npz(f"{FEAT}/X_tfidf_q1.npz")
X_both_q1  = hstack([X_meta_q1, X_tfidf_q1]).tocsr()   # 2115 features
y_q1 = np.load(f"{FEAT}/y_q1.npy")

# Shared split — DO NOT call train_test_split() again
train_idx = np.load(f"{FEAT}/train_idx_q1.npy")
test_idx  = np.load(f"{FEAT}/test_idx_q1.npy")

X_train, X_test = X_both_q1[train_idx], X_both_q1[test_idx]
y_train, y_test = y_q1[train_idx],      y_q1[test_idx]
```

Replace `_q1` with `_q2` for the Q2 task.

## Project-wide conventions

- `random_state = 42` (baked into split)
- `test_size = 0.2`, `stratify = y` (baked into split)
- Primary metric: **macro-F1** (Q1 is 1:4 imbalanced)
- Also report: accuracy, precision+, recall+, confusion matrix

## Logistic Regression baseline (for comparison)

| Experiment | best C | Macro-F1 |
|---|---:|---:|
| Q1 \| meta only | 1.0 | 0.7226 |
| Q1 \| text only | 10.0 | 0.6041 |
| Q1 \| meta + text | 0.1 | 0.7261 |
| Q2 \| meta only | 10.0 | 0.6487 |
| Q2 \| text only | 1.0 | 0.6216 |
| Q2 \| meta + text | 1.0 | 0.6685 |

Other models (SVM, DT, RF, NN) should beat these numbers, or explain why.