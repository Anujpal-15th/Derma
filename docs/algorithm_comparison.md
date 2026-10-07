# Algorithm Comparison by Project Phase

## Phase 1: Data source
| Option | Pros | Cons | Decision |
|---|---|---|---|
| **DermaMNIST** (MedMNIST v2) | 19 MB, official splits, CC BY-NC, laptop-trainable | 28 px, image-level split, no metadata | **Chosen** for the runnable prototype |
| HAM10000 full-res | Real resolution, lesion IDs allow grouped splits | 2.6 GB, needs a CNN/GPU to benefit | Next step |
| SCIN / PAD-UFES-20 | Real smartphone photos, metadata, skin tone | Large, label mapping work, licence steps | Target for full DERMAFUSION |

## Phase 2: Preprocessing and features
| Option | Notes | Decision |
|---|---|---|
| Raw pixels (2,352) | Simple and lossless at 28 px | Used |
| Colour histograms (48) | Captures lesion colour, cheap | Used (concatenated) |
| StandardScaler + PCA(100) | Speeds up LR/MLP/HGB, denoises; fit on train only inside the pipeline | Used for LR, HGB, MLP |
| HOG / LBP texture | Adds texture, needs scikit-image | Skipped; small expected gain at 28 px |
| CNN embeddings | Best features, needs torch | Next step |

**Imbalance handling:** `class_weight="balanced"` (LR, HGB) or `balanced_subsample` (RF). The MLP has no class-weight option, which explains its lower balanced accuracy. Focal loss and oversampling are deferred to the deep-learning phase.

## Phase 3: Models (test-set results, seed 42, reproducible)
Selection is by **validation** macro-F1. Test numbers are reported once.

| Model | Role | Rationale | Val macro-F1 | Test macro-F1 | Test bal-acc | Test acc | Test ROC-AUC | Test ECE | Fit (s) |
|---|---|---|---|---|---|---|---|---|---|
| Dummy (prior) | Sanity floor | Shows what accuracy alone hides | 0.115 | 0.115 | 0.143 | **0.669** | 0.500 | 0.001 | 0 |
| Logistic regression | Baseline | Linear, interpretable, class-weighted | 0.343 | 0.363 | **0.493** | 0.545 | 0.846 | 0.093 | 4 |
| Random forest | Baseline | Non-linear, robust, no scaling | 0.248 | 0.259 | 0.241 | 0.710 | 0.897 | 0.035 | 12 |
| HistGradientBoosting | Advanced | Strong tabular learner, early stopping | 0.406 | 0.419 | 0.431 | 0.665 | 0.871 | 0.042 | 14 |
| **MLP (256-128)** | Advanced | Non-linear, learns pixel interactions | **0.475** | **0.427** | 0.394 | **0.727** | **0.898** | 0.124 | 4 |

### Calibration and uncertainty (added after review loop 1)
Temperature T is fitted on **validation** NLL and applied to test. Argmax is unchanged, so F1 and accuracy do not move.

| Model | T | Test ECE raw → scaled | Test macro-F1 95% bootstrap CI |
|---|---|---|---|
| MLP | 1.80 | 0.124 → **0.023** | 0.374 – 0.470 |
| HistGB | 0.85 | 0.042 → 0.036 | 0.379 – 0.457 |
| LogReg | 1.46 | 0.093 → 0.028 | 0.335 – 0.392 |
| Random forest | 0.85 | 0.035 → 0.037 | 0.235 – 0.283 |

Exact McNemar test, MLP vs HistGB on test correctness: p < 0.0001.

### Reading the table
- **The MLP and HistGB are statistically tied on macro-F1** (the CIs overlap almost completely). The MLP is selected because it leads on validation. It is significantly more *accurate* (McNemar p < 0.0001), but that advantage comes mostly from the majority class. HistGB has better balanced accuracy (0.431 vs 0.394).
- **The MLP is over-confident** (T = 1.80, raw ECE 0.124). After temperature scaling it becomes the best-calibrated model (0.023). The API serves the scaled probabilities.
- **Logistic regression has the best balanced accuracy** (0.493) because class weighting buys minority recall at the cost of majority accuracy. Prefer it if missing a rare class is costlier than a false alarm.
- **Random forest** has an AUC as good as the MLP (0.897) but collapses toward the majority class (macro-F1 0.26). It is a textbook case of why neither accuracy nor AUC should be the headline metric.
- **The MLP is the only model without class balancing** (`MLPClassifier` has no `class_weight`), so the comparison is not perfectly like-for-like. Oversampling for the MLP is a next step.
- *External reference:* MedMNIST v2 reports CNN baselines on DermaMNIST at roughly AUC 0.90–0.92 and accuracy 0.72–0.77. Those numbers come from different models and protocols, so they are context only, not a head-to-head comparison.

## Phase 4: Evaluation metrics
| Metric | Why | Caveat |
|---|---|---|
| Macro-F1 (primary) | Weights all 7 classes equally | Noisy for tiny classes (dermatofibroma n≈23 in test) |
| Balanced accuracy | Mean per-class recall | Ignores precision |
| ROC-AUC (OvR macro) | Threshold-free ranking quality | Can be high while argmax predictions are poor (see RF) |
| ECE (15 bins) | Calibration (Guo et al. 2017) | Bin-dependent; the dummy gets ~0 trivially |
| Accuracy | Context only | Misleading under imbalance |

## Phase 5: Tracking and serving
| Option | Decision | Note |
|---|---|---|
| MLflow, SQLite backend + Model Registry | **Chosen** | The file store is deprecated in MLflow 3.x; the registry needs a DB backend |
| W&B / Neptune | Rejected | Hosted accounts, not fully local |
| FastAPI + uvicorn | **Chosen** | Pydantic validation, auto OpenAPI docs |
| Flask / BentoML / `mlflow models serve` | Rejected | Fewer validation features, or less control over the error contract |
